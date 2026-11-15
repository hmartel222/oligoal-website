"""Budgeted adaptation versus tuned Ridge with train-internal epoch selection."""

import argparse
import gc
import hashlib
import resource
import shutil
import time
import tomllib
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit

from oligo_engine.audit_upstream import audit_checkout
from oligo_engine.contracts import Observation
from oligo_engine.data.catalog import Catalog
from oligo_engine.evaluation.runner import environment_identity
from oligo_engine.huesken_learning import official_split
from oligo_engine.models.rnafm_heads import CachedRepresentation, RepresentationRidge
from oligo_engine.models.rnafm_lora import RNAFMLoRAPredictor
from oligo_engine.representations.rnafm import FrozenRNAFM, duplex_sequences
from oligo_engine.rnafm_study import (
    materialize_representations,
    nested_order,
    write_json,
)

ROOT = Path(__file__).resolve().parents[2]
CAVEATS = [
    "Measured Huesken development study, not independent biological confirmation.",
    "Fifty to 200 measured labels only; no broad label-efficiency claim.",
    "Repeated official nucleobase holdouts overlap and share target genes.",
    "Neural epoch selection uses an acquired-only validation partition.",
    "Frozen head uses cached frozen features; identical head/optimizer settings.",
    "Classical references are fit separately; no full-pool LoRA reference exists.",
    "No therapeutic or combined efficacy/safety claim.",
]


class CachedFrozenHead(RNAFMLoRAPredictor):
    """Equivalent frozen-backbone head optimization without repeated encoding."""

    def __init__(self, representation: CachedRepresentation, *args, **kwargs) -> None:
        self.representation = representation
        super().__init__(*args, **kwargs)
        if self.config["adapt_backbone"]:
            raise ValueError("Cached features cannot train adapters")

    def _features(self, observations: tuple[Observation, ...]) -> torch.Tensor:
        return torch.tensor(
            self.representation.encode(observations),
            dtype=torch.float32,
            device=self.config["device"],
        )


def validation_indices(
    records: tuple[Observation, ...], seed: int, fraction: float
) -> tuple:
    groups = [row.target_id for row in records]
    grouped = all(groups) and len(set(groups)) >= 2
    splitter = (
        GroupShuffleSplit(1, test_size=fraction, random_state=seed)
        if grouped
        else ShuffleSplit(1, test_size=fraction, random_state=seed)
    )
    fit, valid = next(
        splitter.split(np.arange(len(records)), groups=groups if grouped else None)
    )
    return fit, valid, "target_group" if grouped else "row_fallback"


def fit_selected_neural(
    records: tuple[Observation, ...],
    labels: tuple[float, ...],
    seed: int,
    config: dict,
    factory: Callable,
) -> tuple[RNAFMLoRAPredictor, dict]:
    """factory(epochs) cannot receive unrevealed or held-out labels."""
    fit, valid, method = validation_indices(
        records, seed, config["inner_validation_fraction"]
    )
    scores = []
    for epochs in config["epoch_candidates"]:
        model = factory(epochs)
        model.fit(tuple(records[i] for i in fit), tuple(labels[i] for i in fit))
        predictions = model.predict(tuple(records[i] for i in valid))
        rmse = float(
            np.sqrt(
                np.mean(
                    [
                        (p.mean - labels[i]) ** 2
                        for p, i in zip(predictions, valid, strict=True)
                    ]
                )
            )
        )
        scores.append(
            {"epochs": epochs, "validation_rmse": rmse, "fit_audit": model.fit_audit}
        )
        del model
        gc.collect()
    chosen = min(scores, key=lambda row: row["validation_rmse"])["epochs"]
    model = factory(chosen)
    model.fit(records, labels)
    return model, {
        "validation": method,
        "fit_ids": [records[i].sample_id for i in fit],
        "validation_ids": [records[i].sample_id for i in valid],
        "epoch_candidates": scores,
        "selected_epochs": chosen,
        "refit_all_acquired": True,
        "fit_audit": model.fit_audit,
    }


def run_seed(
    loaded, split, representations, config, neural_factory, progress, check_time
):
    order = nested_order(split, split.seed, config["budgets"][0])
    rows = {row.sample_id: row for row in loaded.dataset.observations}
    test = tuple(rows[i] for i in split.test_ids)
    pending, refs = {}, {}
    for arm in config["arms"]:
        oracle, evaluator, reference = loaded.capabilities(split)
        pending[arm] = []
        previous = 0
        for budget in config["budgets"]:
            check_time()
            progress(arm, budget)
            batch = order[previous:budget]
            oracle.reveal(batch)
            ids = tuple(sorted(oracle.revealed_ids))
            records = tuple(rows[i] for i in ids)
            labels = oracle.acquired_labels(ids)
            started = time.perf_counter()
            if arm.endswith("ridge"):
                rep = representations[
                    "onehot" if arm == "onehot_ridge" else "frozen_rnafm"
                ]
                model = RepresentationRidge(
                    rep,
                    split.seed,
                    tuple(config["ridge_alphas"]),
                    config["ridge_folds"],
                )
                model.fit(records, labels)
                fit = model.fit_record
            else:
                model, fit = fit_selected_neural(
                    records,
                    labels,
                    split.seed,
                    config,
                    lambda e: neural_factory(arm, split.seed, e),
                )
            predictions = model.predict(test)
            entry = {
                "labels_used": budget,
                "acquired_ids": list(batch),
                "training_ids": list(ids),
                "model": model.config,
                "fit": fit,
                "fit_predict_seconds": time.perf_counter() - started,
                "peak_process_rss_bytes": resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss,
            }
            pending[arm].append((entry, predictions))
            previous = budget
            del model
            gc.collect()
        if arm.endswith("ridge"):
            check_time()
            ids = tuple(sorted(split.train_ids))
            model = RepresentationRidge(
                rep, split.seed, tuple(config["ridge_alphas"]), config["ridge_folds"]
            )
            model.fit(tuple(rows[i] for i in ids), reference.reveal(ids))
            refs[arm] = (
                {
                    "labels_used": len(ids),
                    "model": model.config,
                    "fit": model.fit_record,
                },
                model.predict(test),
            )
            del model
            gc.collect()
    # All model/epoch/alpha choices at every budget are fixed before evaluation.
    for arm, points in pending.items():
        for entry, predictions in points:
            entry["metrics"] = asdict(evaluator.score(predictions))
    for entry, predictions in refs.values():
        entry["metrics"] = asdict(evaluator.score(predictions))
    return {
        "seed": split.seed,
        "split": asdict(split),
        "arms": {
            arm: [entry for entry, _ in points] for arm, points in pending.items()
        },
        "full_pool_references": {arm: entry for arm, (entry, _) in refs.items()},
        "test_scoring": "after_all_arms_and_budgets_frozen",
    }


def run_study(args: argparse.Namespace) -> dict:
    config = tomllib.loads(args.config.read_text())
    if config["arms"] != ["onehot_ridge", "frozen_ridge", "frozen_head", "lora_head"]:
        raise ValueError("Expected four predeclared arms")
    if config["budgets"] != [50, 100, 200] or len(set(config["seeds"])) != 5:
        raise ValueError("Expected bounded five-seed protocol")
    if args.output.exists():
        raise ValueError("Output exists; preserve all earlier attempts")
    torch.set_num_threads(1)
    audit = audit_checkout(args.checkout.resolve())
    loaded = Catalog().load("Huesken")
    name = Catalog().resolve("Huesken")["data_file"]
    if (
        loaded.dataset.content_sha256
        != audit["files_sha256"][f"oligogym/resources/pkg_dataset/{name}"]
    ):
        raise ValueError("Source dataset mismatch")
    effective = dict(config)
    if args.smoke:
        effective["seeds"] = config["seeds"][:1]
        effective["budgets"] = config["budgets"][:1]
    args.output.mkdir(parents=True)
    identity = environment_identity()
    for path in (ROOT / "src/oligo_engine").rglob("*.py"):
        destination = args.output / "source" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    shutil.copyfile(args.config, args.output / "config.toml")
    suite = {
        "status": "running",
        "mode": "smoke" if args.smoke else "development",
        "config": config,
        "effective": effective,
        "environment": identity,
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "dataset": loaded.public_metadata(),
        "runs": [],
        "caveats": CAVEATS,
    }
    started = time.perf_counter()
    write_json(args.output / "suite.json", suite)

    def check_time():
        if time.perf_counter() - started > config["max_runtime_seconds"]:
            raise RuntimeError("Predeclared compute cap exceeded; incomplete result")

    try:
        representations = materialize_representations(
            loaded.dataset,
            {"representations": ["onehot", "frozen_rnafm"]},
            args.checkpoint,
            args.cache,
            "cpu",
        )
        probe = FrozenRNAFM(args.checkpoint)
        base = probe.config
        del probe

        def backbone_factory():
            encoder = FrozenRNAFM(args.checkpoint)
            return encoder.model, encoder.alphabet

        def neural_factory(arm, seed, epochs):
            kwargs = dict(
                model_factory=backbone_factory,
                sequence_pairs=duplex_sequences,
                base_identity=base,
                seed=seed,
                rank=config["rank"],
                alpha=config["lora_alpha"],
                epochs=epochs,
                batch_size=config["batch_size"],
                learning_rate=config["learning_rate"],
                adapt_backbone=arm == "lora_head",
            )
            return (
                RNAFMLoRAPredictor(**kwargs)
                if arm == "lora_head"
                else CachedFrozenHead(representations["frozen_rnafm"], **kwargs)
            )

        for seed in effective["seeds"]:
            split = official_split(loaded.dataset, seed, args.checkout.resolve())

            def progress(arm, budget):
                suite["current"] = {"seed": seed, "arm": arm, "budget": budget}
                write_json(args.output / "suite.json", suite)
                print(f"seed={seed} arm={arm} budget={budget}", flush=True)

            run = run_seed(
                loaded,
                split,
                representations,
                effective,
                neural_factory,
                progress,
                check_time,
            )
            path = args.output / f"seed-{seed}.json"
            write_json(path, run)
            suite["runs"].append(
                {
                    "seed": seed,
                    "file": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
            write_json(args.output / "suite.json", suite)
        if environment_identity()["source_sha256"] != identity["source_sha256"]:
            raise RuntimeError("Executed source changed")
        suite.update(status="complete", runtime_seconds=time.perf_counter() - started)
        suite.pop("current", None)
        write_json(args.output / "suite.json", suite)
        from oligo_engine.reporting.rnafm_lora_study import report_study

        report_study(args.output)
    except Exception as error:
        suite.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(args.output / "suite.json", suite)
        raise
    return suite


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/rnafm-lora-comparison.toml")
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/rnafm-checkpoint/RNA-FM_pretrained.pth"),
    )
    parser.add_argument(
        "--checkout", type=Path, default=Path("/private/tmp/oligoal-upstream-97f5b9f")
    )
    parser.add_argument("--cache", type=Path, default=Path("artifacts/rnafm-cache"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    run_study(parser.parse_args())


if __name__ == "__main__":
    main()
