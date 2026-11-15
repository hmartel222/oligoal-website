"""Matched low-label Ridge curves on one-hot and verified RNA-FM representations."""

import argparse
import hashlib
import json
import shutil
import tomllib
from dataclasses import asdict
from pathlib import Path
from random import Random
from time import perf_counter
from typing import Callable

from oligo_engine.audit_upstream import audit_checkout
from oligo_engine.contracts import Dataset, Prediction, SplitManifest
from oligo_engine.data.audit import audit_split
from oligo_engine.data.catalog import Catalog, LoadedDataset
from oligo_engine.data.splits import make_split
from oligo_engine.evaluation.runner import environment_identity, validate_predictions
from oligo_engine.features.upstream import OneHotBaseFeaturizer
from oligo_engine.huesken_learning import official_split
from oligo_engine.models.rnafm_heads import CachedRepresentation, RepresentationRidge

ROOT = Path(__file__).resolve().parents[2]
CAVEATS = (
    "Measured diagnostic replay; expanded historical reproduction unresolved.",
    "Huesken informed prior development; holdouts are not biological replications.",
    "Frozen RNA-FM natural-base projection discards modification identities.",
    "RNA1/RNA2 names do not establish guide/passenger orientation.",
    "Extraction is label-free; head scaling/tuning use acquired labels only.",
    "Prediction curves establish neither therapeutic nor acquisition benefit.",
)
EXECUTED_FILES = (
    "src/oligo_engine/rnafm_study.py",
    "src/oligo_engine/models/rnafm_heads.py",
    "src/oligo_engine/reporting/rnafm.py",
    "src/oligo_engine/representations/rnafm.py",
    "src/oligo_engine/features/upstream.py",
    "src/oligo_engine/huesken_learning.py",
    "src/oligo_engine/audit_upstream.py",
    "src/oligo_engine/contracts.py",
    "src/oligo_engine/data/catalog.py",
    "src/oligo_engine/data/oracle.py",
    "src/oligo_engine/data/audit.py",
    "src/oligo_engine/data/splits.py",
    "src/oligo_engine/evaluation/runner.py",
    "src/oligo_engine/evaluation/metrics.py",
)


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_hashes() -> dict:
    return {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in EXECUTED_FILES
        if (ROOT / name).exists()
    }


def nested_order(split: SplitManifest, seed: int, initial: int) -> tuple[str, ...]:
    """Initial sample and subsequent order use separate, outcome-independent RNGs."""
    if not 1 <= initial <= len(split.train_ids):
        raise ValueError("Invalid initial budget")
    first = tuple(Random(seed).sample(list(split.train_ids), initial))
    remaining = [i for i in split.train_ids if i not in set(first)]
    Random(f"rnafm-random-acquisition:{seed}").shuffle(remaining)
    return first + tuple(remaining)


def run_seed(
    loaded: LoadedDataset,
    split: SplitManifest,
    representations: dict[str, CachedRepresentation],
    budgets: tuple[int, ...],
    alphas: tuple[float, ...],
    folds: int,
    progress: Callable[[str, int], None] | None = None,
) -> dict:
    """All representations see identical reveals; evaluation follows all fits."""
    split.validate(loaded.dataset)
    if (
        not budgets
        or tuple(sorted(set(budgets))) != budgets
        or any(type(n) is not int or n < folds for n in budgets)
        or budgets[-1] >= len(split.train_ids)
    ):
        raise ValueError("Distinct increasing budgets must precede full-pool reference")
    rows = {r.sample_id: r for r in loaded.dataset.observations}
    order = nested_order(split, split.seed, budgets[0])
    test_rows = tuple(rows[i] for i in split.test_ids)
    pending: dict[str, list[tuple[dict, tuple[Prediction, ...]]]] = {}
    reference_pending = {}
    evaluator = None
    for name, representation in representations.items():
        oracle, evaluator, reference = loaded.capabilities(split)
        pending[name] = []
        previous = 0
        for budget in budgets:
            if progress:
                progress(name, budget)
            batch = order[previous:budget]
            oracle.reveal(batch)
            ids = tuple(sorted(oracle.revealed_ids))
            model = RepresentationRidge(representation, split.seed, alphas, folds)
            started = perf_counter()
            model.fit(tuple(rows[i] for i in ids), oracle.acquired_labels(ids))
            predictions = model.predict(test_rows)
            validate_predictions(predictions, len(test_rows))
            pending[name].append(
                (
                    {
                        "labels_used": budget,
                        "acquired_ids": list(batch),
                        "model": model.config,
                        "fit": model.fit_record,
                        "fit_predict_seconds": perf_counter() - started,
                    },
                    predictions,
                )
            )
            previous = budget
        if progress:
            progress(name, len(split.train_ids))
        ids = tuple(sorted(split.train_ids))
        values = reference.reveal(ids)
        model = RepresentationRidge(representation, split.seed, alphas, folds)
        started = perf_counter()
        model.fit(tuple(rows[i] for i in ids), values)
        predictions = model.predict(test_rows)
        validate_predictions(predictions, len(test_rows))
        reference_pending[name] = (
            {
                "labels_used": len(ids),
                "model": model.config,
                "fit": model.fit_record,
                "fit_predict_seconds": perf_counter() - started,
            },
            predictions,
        )
    # No test label is scored until all models, alphas and representations are fixed.
    assert evaluator is not None
    runs = {}
    for name, checkpoints in pending.items():
        runs[name] = {
            "checkpoints": [
                {**record, "metrics": asdict(evaluator.score(predictions))}
                for record, predictions in checkpoints
            ],
            "full_reference": {
                **reference_pending[name][0],
                "metrics": asdict(evaluator.score(reference_pending[name][1])),
            },
        }
    return {
        "seed": split.seed,
        "split": asdict(split),
        "initial_ids": list(order[: budgets[0]]),
        "acquisition": "nested uniform random, common across representations",
        "data_audit": audit_split(loaded.dataset, split),
        "representations": runs,
        "label_budget_note": "validation reuses counted rows; reference separate",
    }


def validate_config(config: dict) -> None:
    if config["dataset"] != "Huesken" or config["purpose"] != "diagnostic":
        raise ValueError("Only independent measured Huesken diagnostic is supported")
    seeds = config["seeds"]
    if (
        not seeds
        or any(type(s) is not int for s in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("Seeds must be distinct integers")
    if config["budgets"] != [50, 100, 200, 400, 800]:
        raise ValueError("Frozen prediction budgets differ")
    if config["split"] not in ("nucleobase", "target"):
        raise ValueError("Unknown split protocol")
    if config["representations"] not in (
        ["onehot", "frozen_rnafm"],
        ["onehot", "frozen_rnafm", "random_rnafm"],
    ):
        raise ValueError(
            "Require matched one-hot/frozen comparison and optional random control"
        )
    if not 0 < config["pearson_target"] < 1:
        raise ValueError("Shared absolute target must be in (0,1)")


def materialize_representations(
    dataset: Dataset, config: dict, checkpoint: Path, cache: Path, device: str
) -> dict[str, CachedRepresentation]:
    import gc

    from oligo_engine.representations.rnafm import FrozenRNAFM

    observations = dataset.observations
    encoder = OneHotBaseFeaturizer()
    outputs = {
        "onehot": CachedRepresentation(
            observations, encoder.transform(observations), encoder.config
        )
    }
    for name in config["representations"][1:]:
        if name == "random_rnafm":
            encoder = FrozenRNAFM.random_initialized(
                checkpoint,
                seed=config["random_encoder_seed"],
                device=device,
                cache_dir=cache,
            )
        else:
            encoder = FrozenRNAFM(checkpoint, device=device, cache_dir=cache)
        values = encoder.encode(observations)
        outputs[name] = CachedRepresentation(observations, values, encoder.config)
        del encoder, values
        gc.collect()
    return outputs


def run_study(
    config_path: Path,
    checkout: Path,
    checkpoint: Path,
    cache: Path,
    output: Path,
    device: str = "cpu",
    smoke: bool = False,
) -> dict:
    config = tomllib.loads(config_path.read_text())
    validate_config(config)
    if output.exists():
        raise ValueError("Output exists; earlier attempts must be preserved")
    audit = audit_checkout(checkout.resolve())
    catalog = Catalog()
    loaded = catalog.load("Huesken")
    source_name = catalog.resolve("Huesken")["data_file"]
    if (
        loaded.dataset.content_sha256
        != audit["files_sha256"][f"oligogym/resources/pkg_dataset/{source_name}"]
    ):
        raise ValueError("Dataset does not match pinned source")
    seeds = config["seeds"][:1] if smoke else config["seeds"]
    budgets = tuple(config["budgets"][:2] if smoke else config["budgets"])
    splits = [
        official_split(loaded.dataset, seed, checkout.resolve())
        if config["split"] == "nucleobase"
        else make_split(loaded.dataset, "target", seed)
        for seed in seeds
    ]
    if any(len(split.train_ids) <= budgets[-1] for split in splits):
        raise ValueError("Frozen budget must be smaller than each eligible pool")
    output.mkdir(parents=True)
    hashes = source_hashes()
    for name in hashes:
        dest = output / "source" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
    shutil.copyfile(config_path, output / "config.toml")
    suite = {
        "status": "running",
        "stage": "smoke" if smoke else "development",
        "provenance": "measured",
        "purpose": "diagnostic",
        "config": config,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "dataset": loaded.public_metadata(),
        "source_audit": audit,
        "environment": environment_identity(),
        "executed_source_sha256": hashes,
        "seeds": seeds,
        "budgets": budgets,
        "splits": [asdict(s) for s in splits],
        "runs": [],
        "caveats": CAVEATS,
        "pretraining_control_present": "random_rnafm" in config["representations"],
    }
    write_json(output / "suite.json", suite)
    started = perf_counter()
    try:
        representations = materialize_representations(
            loaded.dataset, config, checkpoint, cache, device
        )
        suite["representation_configs"] = {
            k: v.config for k, v in representations.items()
        }
        suite["extraction_seconds"] = perf_counter() - started
        write_json(output / "suite.json", suite)
        for split in splits:

            def progress(name: str, budget: int) -> None:
                suite["current"] = {
                    "seed": split.seed,
                    "representation": name,
                    "labels": budget,
                }
                write_json(output / "suite.json", suite)
                print(
                    f"seed={split.seed} representation={name} labels={budget}",
                    flush=True,
                )

            result = run_seed(
                loaded,
                split,
                representations,
                budgets,
                tuple(config["alphas"]),
                config["inner_folds"],
                progress,
            )
            path = output / f"seed-{split.seed}.json"
            write_json(path, result)
            suite["runs"].append(
                {
                    "seed": split.seed,
                    "file": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
            write_json(output / "suite.json", suite)
        if source_hashes() != hashes:
            raise ValueError("Executed source changed during study")
        suite["status"] = "complete"
        suite["runtime_seconds"] = perf_counter() - started
        suite.pop("current", None)
        write_json(output / "suite.json", suite)
        from oligo_engine.reporting.rnafm import report_study

        report_study(output)
    except Exception as error:
        suite.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(output / "suite.json", suite)
        raise
    return suite


def run_geometry_study(
    config_path: Path,
    checkout: Path,
    checkpoint: Path,
    cache: Path,
    output: Path,
    device: str = "cpu",
    smoke: bool = False,
) -> dict:
    """Hold one-hot Ridge fixed while varying label-free acquisition geometry."""
    from dataclasses import replace

    from oligo_engine.acquisition.pool_diversity import DiversityAcquisition
    from oligo_engine.contracts import RunConfig
    from oligo_engine.evaluation.runner import run_campaign

    config = tomllib.loads(config_path.read_text())
    validate_config(config)
    if output.exists():
        raise ValueError("Output exists; preserve prior attempts")
    audit = audit_checkout(checkout.resolve())
    loaded = Catalog().load("Huesken")
    source_name = Catalog().resolve("Huesken")["data_file"]
    if (
        loaded.dataset.content_sha256
        != audit["files_sha256"][f"oligogym/resources/pkg_dataset/{source_name}"]
    ):
        raise ValueError("Dataset does not match pinned source")
    seeds = config["seeds"][:1] if smoke else config["seeds"]
    budgets = tuple(config["budgets"][:2] if smoke else config["budgets"])
    splits = [
        official_split(loaded.dataset, seed, checkout.resolve())
        if config["split"] == "nucleobase"
        else make_split(loaded.dataset, "target", seed)
        for seed in seeds
    ]
    if any(len(split.train_ids) <= budgets[-1] for split in splits):
        raise ValueError("Budgets exceed eligible pool")
    hashes = source_hashes()
    output.mkdir(parents=True)
    for name in hashes:
        dest = output / "source" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
    shutil.copyfile(config_path, output / "config.toml")
    suite = {
        "status": "running",
        "stage": "smoke" if smoke else "development",
        "study": "fixed_onehot_ridge_acquisition_geometry",
        "config": config,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "source_audit": audit,
        "executed_source_sha256": hashes,
        "seeds": seeds,
        "budgets": budgets,
        "runs": [],
        "environment": environment_identity(),
        "dataset": loaded.public_metadata(),
        "caveats": CAVEATS[:5]
        + (
            "Fixed-predictor geometry with overlapping development holdouts.",
            "Prediction results preceded this untuned geometry comparison.",
        ),
    }
    write_json(output / "suite.json", suite)
    references = {}
    by_id = {row.sample_id: row for row in loaded.dataset.observations}
    try:
        representations = materialize_representations(
            loaded.dataset, config, checkpoint, cache, device
        )
        suite["representation_configs"] = {
            k: v.config for k, v in representations.items()
        }
        for split in splits:
            records = tuple(by_id[i] for i in split.train_ids)
            order = nested_order(split, split.seed, budgets[0])

            class NestedRandom:
                @property
                def config(self) -> dict:
                    return {
                        "name": "random",
                        "order": "same_nested_prediction_study_order",
                    }

                def choose(self, eligible, labeled, budget, rng, predictor):
                    selected = order[len(labeled) : len(labeled) + budget]
                    if not set(selected) <= {row.sample_id for row in eligible}:
                        raise ValueError("Random prefix differs from current pool")
                    return selected

            policies = {"random": NestedRandom()}
            for name, representation in representations.items():
                policies[name + "_diversity"] = DiversityAcquisition(
                    records,
                    representation.encode(records),
                    {
                        "representation": name,
                        "representation_config": representation.config,
                        "normalization": "per_row_l2",
                        "distance": "euclidean",
                        "tie_break": "sample_id_lexicographic",
                    },
                )
            for name, policy in policies.items():
                oracle, evaluator, reference = loaded.capabilities(split)
                fit_records = []

                class RecordedRidge(RepresentationRidge):
                    def fit(self, observations, labels):
                        super().fit(observations, labels)
                        fit_records.append(self.fit_record)

                def factory():
                    return RecordedRidge(
                        representations["onehot"],
                        split.seed,
                        tuple(config["alphas"]),
                        config["inner_folds"],
                    )

                def progress(phase, count):
                    suite["current"] = {
                        "seed": split.seed,
                        "policy": name,
                        "phase": phase,
                        "labels": count,
                    }
                    write_json(output / "suite.json", suite)
                    print(
                        f"seed={split.seed} policy={name} {phase} labels={count}",
                        flush=True,
                    )

                result = run_campaign(
                    loaded.dataset,
                    split,
                    oracle,
                    evaluator,
                    factory,
                    policy,
                    RunConfig(
                        split.seed,
                        budgets[0],
                        50,
                        budgets[-1],
                        config["pearson_target"],
                        "diagnostic",
                    ),
                    reference,
                    reference_cache=references,
                    progress=progress,
                    checkpoint_labels=budgets,
                )
                result = replace(
                    result,
                    caveats=suite["caveats"],
                    execution={
                        **result.execution,
                        "acquisition_policy": name,
                        "fit_records": fit_records,
                        "fixed_prediction_representation": "onehot",
                    },
                )
                path = output / f"{name}-{split.seed}.json"
                write_json(path, asdict(result))
                suite["runs"].append(
                    {
                        "seed": split.seed,
                        "policy": name,
                        "file": path.name,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                )
                write_json(output / "suite.json", suite)
        if source_hashes() != hashes:
            raise ValueError("Executed source changed during study")
        suite.update(status="complete")
        suite.pop("current", None)
        write_json(output / "suite.json", suite)
        from oligo_engine.reporting.rnafm import report_geometry

        report_geometry(output)
    except Exception as error:
        suite.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(output / "suite.json", suite)
        raise
    return suite


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/rnafm-prediction.toml")
    )
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--cache", type=Path, default=Path("artifacts/rnafm-embeddings")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--geometry", action="store_true")
    args = parser.parse_args()
    runner = run_geometry_study if args.geometry else run_study
    runner(
        args.config,
        args.checkout,
        args.checkpoint,
        args.cache,
        args.output,
        args.device,
        args.smoke,
    )


if __name__ == "__main__":
    main()
