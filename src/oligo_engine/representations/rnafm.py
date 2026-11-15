"""Pinned RNA-FM duplex representation; no learned preprocessing or fallbacks."""

import copy
import hashlib
import json
import subprocess
import tempfile
from argparse import Namespace
from pathlib import Path
from typing import Sequence

import numpy as np

from oligo_engine.contracts import Observation

SOURCE_REVISION = "348951516e0963d22bbb33b3c9fc18c89081d38e"
CHECKPOINT_SHA256 = "5b5d7d87b37c291ef42c140ef9edf7aea29f255fa2a4fd435f776c52e93d5e99"
CHECKPOINT_URL = (
    "https://huggingface.co/cuhkaih/rnafm/resolve/main/RNA-FM_pretrained.pth"
)
UPSTREAM_REVISION = "97f5b9f58d9e445a8ecb0218037af7465c3708c0"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source(package_path: str, expected: str) -> Path:
    root = Path(package_path).resolve().parent
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    if result.stdout.strip() != expected:
        raise ValueError("Imported source does not match the pinned revision")
    dirty = subprocess.run(
        ["git", "-C", str(root), "diff", "--name-only", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    if dirty.stdout.strip():
        raise ValueError("Pinned dependency checkout contains tracked modifications")
    return root


def duplex_sequences(
    observations: Sequence[Observation],
) -> tuple[tuple[str, str], ...]:
    """Return RNA1/RNA2 natural-base projections, preserving source strand order."""
    import oligogym
    from oligogym.helm import helm2xna

    verify_source(oligogym.__file__, UPSTREAM_REVISION)
    pairs = []
    for row in observations:
        try:
            polymers = helm2xna(row.sequence).polymers
        except Exception as error:
            raise ValueError(f"{row.sample_id}: invalid HELM") from error
        if set(polymers) != {"RNA1", "RNA2"}:
            raise ValueError(f"{row.sample_id}: expected exactly RNA1 and RNA2")
        pair = []
        for name in ("RNA1", "RNA2"):
            bases = polymers[name]["base"].split(".")
            if not bases or any(base not in set("ACGTU") for base in bases):
                raise ValueError(f"{row.sample_id}: unsupported or missing base")
            sequence = "".join(bases).replace("T", "U")
            if not 1 <= len(sequence) <= 1022:
                raise ValueError(f"{row.sample_id}: unsupported strand length")
            pair.append(sequence)
        pairs.append(tuple(pair))
    return tuple(pairs)


class FrozenRNAFM:
    """Two separately encoded strand means, concatenated as 1280 float32 values."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cpu",
        cache_dir: str | Path | None = None,
    ):
        self._initialize(Path(checkpoint_path), device, cache_dir, None)

    @classmethod
    def random_initialized(
        cls,
        checkpoint_path: str | Path,
        seed: int = 0,
        device: str = "cpu",
        cache_dir: str | Path | None = None,
    ):
        """Architecture-matched random weights; explicitly not pretrained features."""
        if type(seed) is not int:
            raise ValueError("Random control seed must be an integer")
        instance = cls.__new__(cls)
        instance._initialize(Path(checkpoint_path), device, cache_dir, seed)
        return instance

    def _initialize(self, checkpoint_path, device, cache_dir, random_seed):
        import fm
        import torch

        source = verify_source(fm.__file__, SOURCE_REVISION)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        actual_hash = file_sha256(checkpoint_path)
        if actual_hash != CHECKPOINT_SHA256:
            raise ValueError(
                "Checkpoint SHA-256 differs from official published weights"
            )
        # Official checkpoint stores argparse.Namespace. No unrestricted pickle load.
        with torch.serialization.safe_globals([Namespace]):
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0 if random_seed is None else random_seed)
            if random_seed is None:
                model, alphabet = fm.pretrained.load_model_and_alphabet_core(
                    "RNA-FM_pretrained", checkpoint, theme="rna"
                )
            else:
                model, alphabet, _ = fm.pretrained._load_model_and_alphabet_core_v1(
                    checkpoint, theme="rna"
                )
        del checkpoint
        if model.args.layers != 12 or model.args.embed_dim != 640:
            raise ValueError("Unexpected RNA-FM architecture")
        self.model, self.alphabet = model.to(device).eval(), alphabet
        self.model.requires_grad_(False)
        self.batch_converter = alphabet.get_batch_converter()
        self.device = device
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._config = {
            "name": "frozen_rnafm" if random_seed is None else "random_rnafm_control",
            "pretrained": random_seed is None,
            "random_seed": random_seed,
            "source_revision": SOURCE_REVISION,
            "checkpoint_sha256": actual_hash,
            "checkpoint_source": CHECKPOINT_URL,
            "oligogym_revision": UPSTREAM_REVISION,
            "representation_schema": 1,
            "layer": 12,
            "n_features": 1280,
            "pooling": "mean_nucleotide_tokens_per_strand_then_concatenate",
            "strands": ["RNA1", "RNA2"],
            "natural_base_projection": "T_to_U",
            "chemical_features": "omitted; original HELM retained in observations",
            "normalization": "none",
            "device": device,
            "dtype": "float32",
            "torch_version": torch.__version__,
            "fm_version": fm.__version__,
            "source_files_sha256": {
                str(p.relative_to(source)): file_sha256(p)
                for p in sorted(source.rglob("*.py"))
            },
        }
        self._identity = hashlib.sha256(
            json.dumps(self._config, sort_keys=True).encode()
        ).hexdigest()
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def config(self) -> dict:
        return copy.deepcopy(self._config)

    def _cache_path(self, sequence: str) -> Path | None:
        if self.cache_dir is None:
            return None
        key = hashlib.sha256((self._identity + sequence).encode()).hexdigest()
        return self.cache_dir / f"{key}.npz"

    def _load_cache(self, sequence: str) -> np.ndarray | None:
        path = self._cache_path(sequence)
        if path is None or not path.exists():
            return None
        with np.load(path, allow_pickle=False) as archive:
            vector = archive["vector"]
            if (
                str(archive["identity"]) != self._identity
                or str(archive["sequence_sha256"])
                != hashlib.sha256(sequence.encode()).hexdigest()
                or str(archive["vector_sha256"])
                != hashlib.sha256(vector.tobytes()).hexdigest()
            ):
                raise ValueError("Embedding cache identity or content hash mismatch")
        self._validate_vector(vector)
        return vector

    @staticmethod
    def _validate_vector(vector: np.ndarray) -> None:
        if (
            vector.shape != (1280,)
            or vector.dtype != np.float32
            or not np.isfinite(vector).all()
        ):
            raise ValueError("Invalid RNA-FM representation shape, dtype or values")

    def _save_cache(self, sequence: str, vector: np.ndarray) -> None:
        path = self._cache_path(sequence)
        if path is None:
            return
        with tempfile.NamedTemporaryFile(
            dir=path.parent, suffix=".npz", delete=False
        ) as file:
            temporary = Path(file.name)
            np.savez_compressed(
                file,
                vector=vector,
                identity=self._identity,
                sequence_sha256=hashlib.sha256(sequence.encode()).hexdigest(),
                vector_sha256=hashlib.sha256(vector.tobytes()).hexdigest(),
            )
        temporary.replace(path)

    def encode(self, observations: Sequence[Observation]) -> np.ndarray:
        import torch

        pairs = duplex_sequences(observations)
        if not pairs:
            return np.empty((0, 1280), dtype=np.float32)
        rows = []
        self.model.eval()
        for row, pair in zip(observations, pairs, strict=True):
            vector = self._load_cache(row.sequence)
            if vector is None:
                _, _, tokens = self.batch_converter(list(zip(("RNA1", "RNA2"), pair)))
                with torch.inference_mode():
                    output = self.model(tokens.to(self.device), repr_layers=[12])
                    representations = output["representations"][12]
                    if representations.shape != (2, tokens.shape[1], 640):
                        raise ValueError("Unexpected RNA-FM hidden-state shape")
                    # Token index 0 is BOS; padding/EOS are never pooled.
                    means = [
                        representations[i, 1 : len(seq) + 1].mean(0)
                        for i, seq in enumerate(pair)
                    ]
                    vector = (
                        torch.cat(means).cpu().numpy().astype(np.float32, copy=True)
                    )
                self._validate_vector(vector)
                self._save_cache(row.sequence, vector)
            rows.append(vector)
        return np.stack(rows)
