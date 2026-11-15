"""Budget-isolated RNA-FM Q/V weight adapters and duplex regression head.

Weight parametrization is intentional: RNA-FM's fused attention reads projection
weights directly, bypassing nn.Linear.forward hooks. No base tensor is trainable.
"""

import hashlib
import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils import parametrize

from oligo_engine.contracts import Observation, Prediction


class LowRankWeight(nn.Module):
    """Zero-initialized residual on a frozen projection weight."""

    def __init__(self, rows: int, cols: int, rank: int, alpha: float):
        super().__init__()
        if rank < 1 or rank > min(rows, cols) or not math.isfinite(alpha):
            raise ValueError("Invalid LoRA rank/alpha")
        self.a = nn.Parameter(torch.empty(rank, cols))
        self.b = nn.Parameter(torch.zeros(rows, rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        return weight + self.scale * (self.b @ self.a)


def inject_qv_adapters(backbone: nn.Module, rank: int, alpha: float) -> list[str]:
    """Freeze backbone, then adapt genuine RNA-FM Q/V linear projections."""
    if any(parametrize.is_parametrized(m) for m in backbone.modules()):
        raise ValueError("Expected fresh base model, found existing parametrization")
    targets = [
        (name, module)
        for name, module in backbone.named_modules()
        if name.endswith(("self_attn.q_proj", "self_attn.v_proj"))
    ]
    if not targets or any(not isinstance(m, nn.Linear) for _, m in targets):
        raise ValueError("No compatible RNA-FM Q/V attention projections")
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    for _, module in targets:
        adapter = LowRankWeight(module.out_features, module.in_features, rank, alpha)
        adapter.to(device=module.weight.device, dtype=module.weight.dtype)
        parametrize.register_parametrization(module, "weight", adapter)
    return [name for name, _ in targets]


def tensor_digest(values: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        data = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(data.shape), data.dtype)).encode())
        digest.update(data.numpy().tobytes())
    return digest.hexdigest()


class RNAFMLoRAPredictor:
    """Fixed-epoch diagnostic adapter. Every fit resets to the verified base.

    The model factory must return a fresh backbone and alphabet. Sequence decoding
    is supplied by the verified encoder's label-free HELM parser. No oracle or
    evaluator is accepted, and all normalization uses the supplied fit labels.
    """

    def __init__(
        self,
        model_factory: Callable,
        sequence_pairs: Callable[[Sequence[Observation]], Sequence[tuple[str, str]]],
        base_identity: dict,
        seed: int = 0,
        rank: int = 4,
        alpha: float = 8.0,
        epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 1e-3,
        hidden_size: int = 640,
        layer: int = 12,
        device: str = "cpu",
        adapt_backbone: bool = True,
    ):
        if epochs < 1 or batch_size < 1 or learning_rate <= 0:
            raise ValueError("Positive epochs, batch size and learning rate required")
        if not base_identity.get("checkpoint_sha256"):
            raise ValueError("Verified base checkpoint SHA256 is required")
        self.factory, self.sequence_pairs = model_factory, sequence_pairs
        self._config = {
            "name": "rnafm_qv_lora_regression",
            "base": json.loads(json.dumps(base_identity)),
            "seed": seed,
            "rank": rank,
            "alpha": alpha,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "hidden_size": hidden_size,
            "layer": layer,
            "device": device,
            "pooling": "nucleotide_mean_each_strand_then_RNA1_RNA2_concatenation",
            "adapt_backbone": adapt_backbone,
            "early_stopping": False,
            "uncertainty": None,
        }
        self._fitted = False

    @property
    def config(self) -> dict:
        import copy

        return copy.deepcopy(self._config)

    def _reset(self) -> None:
        cfg = self._config
        torch.manual_seed(cfg["seed"])
        self.backbone, self.alphabet = self.factory()
        self.backbone.to(cfg["device"])
        if cfg["adapt_backbone"]:
            self.targets = inject_qv_adapters(self.backbone, cfg["rank"], cfg["alpha"])
        else:
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)
            self.targets = []
        # Use the same head initialization in frozen and adapted comparisons.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(cfg["seed"] + 10000)
            self.head = nn.Linear(2 * cfg["hidden_size"], 1).to(cfg["device"])
        # Disable dropout while preserving autograd through the adapters.
        self.backbone.eval()
        self.converter = self.alphabet.get_batch_converter()
        self._fitted = False

    def _features(self, observations: Sequence[Observation]) -> torch.Tensor:
        pairs = self.sequence_pairs(observations)
        if len(pairs) != len(observations) or any(len(p) != 2 for p in pairs):
            raise ValueError("Expected aligned RNA1/RNA2 strand pairs")
        strands = [seq for pair in pairs for seq in pair]
        if any(not seq or set(seq) - set("ACGU") for seq in strands):
            raise ValueError("Expected nonempty natural RNA strands")
        _, _, tokens = self.converter([(str(i), seq) for i, seq in enumerate(strands)])
        tokens = tokens.to(self._config["device"])
        layer = self._config["layer"]
        embeddings = self.backbone(tokens, repr_layers=[layer])["representations"][
            layer
        ]
        pooled = torch.stack(
            [embeddings[i, 1 : len(seq) + 1].mean(0) for i, seq in enumerate(strands)]
        )
        return pooled.reshape(len(observations), 2 * self._config["hidden_size"])

    def _trainable(self) -> dict[str, torch.Tensor]:
        return {
            **{
                f"backbone.{name}": value
                for name, value in self.backbone.named_parameters()
                if value.requires_grad
            },
            **{f"head.{name}": value for name, value in self.head.named_parameters()},
        }

    def fit(
        self, observations: tuple[Observation, ...], labels: tuple[float, ...]
    ) -> None:
        if not observations or len(observations) != len(labels):
            raise ValueError("Nonempty aligned training rows and labels required")
        if len({r.sample_id for r in observations}) != len(observations):
            raise ValueError("Training IDs must be unique")
        if any(not math.isfinite(y) for y in labels):
            raise ValueError("Training labels must be finite")
        # Canonical order makes fit independent of acquisition batch ordering.
        order = sorted(
            range(len(observations)), key=lambda i: observations[i].sample_id
        )
        observations = tuple(observations[i] for i in order)
        labels = tuple(labels[i] for i in order)
        self._reset()
        cfg = self._config
        y = torch.tensor(labels, dtype=torch.float32, device=cfg["device"])
        self.y_mean = float(y.mean())
        self.y_scale = max(float(y.std(unbiased=False)), 1e-6)
        y = (y - self.y_mean) / self.y_scale
        frozen = {
            name: p
            for name, p in self.backbone.named_parameters()
            if not p.requires_grad
        }
        before = tensor_digest(frozen)
        adapter_before = tensor_digest(
            {k: v for k, v in self._trainable().items() if k.startswith("backbone.")}
        )
        optimizer = torch.optim.AdamW(
            list(self._trainable().values()), lr=cfg["learning_rate"], weight_decay=0
        )
        generator = torch.Generator().manual_seed(cfg["seed"])
        losses = []
        for _ in range(cfg["epochs"]):
            permutation = torch.randperm(
                len(observations), generator=generator
            ).tolist()
            total = 0.0
            for start in range(0, len(permutation), cfg["batch_size"]):
                indices = permutation[start : start + cfg["batch_size"]]
                optimizer.zero_grad(set_to_none=True)
                pred = self.head(
                    self._features(tuple(observations[i] for i in indices))
                ).flatten()
                loss = ((pred - y[indices]) ** 2).mean()
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite adaptation loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(self._trainable().values()), 1.0)
                optimizer.step()
                total += float(loss.detach()) * len(indices)
            losses.append(total / len(observations))
        if tensor_digest(frozen) != before:
            raise RuntimeError("Frozen base weights changed during adaptation")
        self.fit_audit = {
            "train_ids": [r.sample_id for r in observations],
            "labels_used": len(labels),
            "loss_by_epoch": losses,
            "base_unchanged": True,
            "frozen_tensor_sha256": before,
            "adapted_modules": self.targets,
            "trainable_parameters": sum(p.numel() for p in self._trainable().values()),
            "adapter_changed": adapter_before
            != tensor_digest(
                {
                    k: v
                    for k, v in self._trainable().items()
                    if k.startswith("backbone.")
                }
            ),
        }
        self._fitted = True

    def predict(self, observations: tuple[Observation, ...]) -> tuple[Prediction, ...]:
        if not self._fitted:
            raise RuntimeError("fit or load_adapter must precede prediction")
        output = []
        with torch.no_grad():
            for start in range(0, len(observations), self._config["batch_size"]):
                pred = self.head(
                    self._features(
                        observations[start : start + self._config["batch_size"]]
                    )
                ).flatten()
                output.extend(
                    Prediction(float(p) * self.y_scale + self.y_mean) for p in pred
                )
        return tuple(output)

    def save_adapter(self, path: Path) -> str:
        if not self._fitted:
            raise RuntimeError("Cannot save an unfitted model")
        if path.exists():
            raise ValueError("Adapter output already exists")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": self.config,
                "state": {k: v.detach().cpu() for k, v in self._trainable().items()},
                "y_mean": self.y_mean,
                "y_scale": self.y_scale,
                "fit_audit": self.fit_audit,
            },
            path,
        )
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def load_adapter(self, path: Path, expected_sha256: str) -> None:
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
            raise ValueError("Adapter checksum mismatch")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["config"] != self.config:
            raise ValueError("Adapter/base identity or training configuration mismatch")
        self._reset()
        parameters = self._trainable()
        if parameters.keys() != payload["state"].keys():
            raise ValueError("Adapter tensor schema mismatch")
        with torch.no_grad():
            for key, value in parameters.items():
                value.copy_(payload["state"][key])
        self.y_mean, self.y_scale = payload["y_mean"], payload["y_scale"]
        self.fit_audit = payload["fit_audit"]
        self._fitted = True
