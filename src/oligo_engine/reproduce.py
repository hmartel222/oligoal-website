"""Rerun one frozen published configuration using the pinned upstream wrappers."""

import argparse
import ast
import csv
import hashlib
import importlib.util
import inspect
import json
import random
import tomllib
from collections.abc import Callable
from dataclasses import asdict
from importlib.metadata import distributions, version
from math import isfinite
from pathlib import Path
from statistics import fmean, stdev
from time import perf_counter
from types import ModuleType

from oligo_engine.audit_upstream import PINNED_REVISION, audit_checkout
from oligo_engine.data.catalog import Catalog
from oligo_engine.evaluation.metrics import regression_metrics
from oligo_engine.evaluation.runner import environment_identity
from oligo_engine.native_cache import native_monomer_cache


def execution_mode(config: dict) -> str:
    """Preserve legacy reruns and reject unsupported script configurations."""
    mode = config.get("execution_mode", "direct_wrappers")
    if mode not in {
        "direct_wrappers",
        "benchmark_model_factory",
        "benchmark_helpers",
    }:
        raise ValueError("Unknown baseline execution_mode")
    seeds = config["seeds"]
    if (
        not seeds
        or any(type(seed) is not int or seed < 0 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("Baseline seeds must be distinct nonnegative integers")
    if mode == "benchmark_helpers" and config["split"] != {
        "strategy": "nucleobase",
        "test_size": 0.2,
        "n_clusters": 8,
        "kmer": True,
        "kmer_max": 3,
        "strand": "RNA1",
    }:
        raise ValueError("Benchmark helpers require the audited default split config")
    return mode


def load_benchmark(checkout: Path) -> ModuleType:
    """Load helpers only after reproduce() has audited the unchanged checkout."""
    source = checkout / "benchmarks/train_model.py"
    spec = importlib.util.spec_from_file_location("oligoal_pinned_benchmark", source)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load the audited benchmark training script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_baseline_model(
    config: dict,
    seed: int,
    input_dim: int,
    model_class: type,
    benchmark: ModuleType | None,
) -> tuple[object, dict]:
    """Record effective parameters; CSV arguments alone do not specify training."""
    mode = execution_mode(config)
    requested = {k: v for k, v in config["model"].items() if k != "name"}
    forest = config["model"]["name"] == "RandomForestModel"
    neural = config["model"]["name"] in {"CNN", "MLP"}
    if neural and mode != "benchmark_helpers":
        raise ValueError("Neural reproduction requires audited benchmark helpers")
    overrides = {"random_state": seed, "n_jobs": 2} if forest else {}
    ignored = {}
    if mode == "direct_wrappers":
        model = model_class(**requested, **overrides)
    else:
        if benchmark is None:
            raise ValueError("Benchmark execution requires audited training helpers")
        script_args = {**requested, "input_dim": input_dim}
        accepted = inspect.signature(model_class).parameters
        ignored = {k: v for k, v in script_args.items() if k not in accepted}
        model = benchmark.prepare_model(
            {"model": config["model"]["name"], "model_args": script_args}
        )
        if mode == "benchmark_model_factory":
            # Controlled argument-only comparison retains the previous RF RNG/jobs.
            model.model.set_params(**overrides)
        else:
            overrides = {}
    if neural:
        constructor = inspect.signature(model_class).bind(
            **requested, input_dim=input_dim
        )
        constructor.apply_defaults()
        effective = dict(constructor.arguments)
    else:
        effective = model.model.get_params(deep=False)
    return model, {
        "execution_mode": mode,
        "requested_model_args": requested,
        "arguments_filtered_by_benchmark": ignored,
        "explicit_estimator_overrides": overrides,
        "effective_estimator_params": serialize_parameters(effective),
    }


def serialize_parameters(value: object) -> object:
    """Preserve XGBoost's NaN sentinel without emitting nonstandard JSON."""
    if isinstance(value, float) and not isfinite(value):
        return {"nonfinite_float": str(value)}
    if isinstance(value, dict):
        return {key: serialize_parameters(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [serialize_parameters(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise ValueError(f"Unsupported model parameter type: {type(value).__name__}")


def published_reference(checkout: Path, config: dict) -> dict:
    values = []
    model_args = {k: v for k, v in config["model"].items() if k != "name"}
    features = {k: v for k, v in config["featurizer"].items() if k != "name"}
    with (checkout / "benchmarks/oligogym_benchmarks.csv").open() as file:
        for row in csv.DictReader(file):
            if (
                row["dataset"] == config["benchmark_alias"]
                and row["cross_validation"] == config["split"]["strategy"]
                and row["model"] == config["model"]["name"]
                and row["featurizer"] == config["featurizer"]["name"]
                and ast.literal_eval(row["model_args"]) == model_args
                and ast.literal_eval(row["featurizer_args"]) == features
                and row["test_pearson_correlation"]
            ):
                values.append(float(row["test_pearson_correlation"]))
    return {"n": len(values), "mean": fmean(values) if values else None}


def reproduce(
    checkout: Path,
    config: dict,
    catalog: Catalog | None = None,
    on_run: Callable[[dict], None] | None = None,
) -> dict:
    audit = audit_checkout(checkout)
    with native_monomer_cache(config.get("performance", {})) as performance:
        result = _reproduce(checkout, config, catalog, on_run, audit)
        result["performance_execution"] = performance
    return result


def _reproduce(
    checkout: Path,
    config: dict,
    catalog: Catalog | None,
    on_run: Callable[[dict], None] | None,
    audit: dict,
) -> dict:
    mode = execution_mode(config)
    benchmark = load_benchmark(checkout) if mode != "direct_wrappers" else None
    catalog = catalog or Catalog()
    entry = catalog.resolve(config["loader_key"])
    if entry["benchmark_alias"] != config["benchmark_alias"]:
        raise ValueError("Configured dataset and historical benchmark alias differ")
    source_data = catalog.verify_file(entry["data_file"])
    source_meta = catalog.verify_file(entry["metadata_file"])
    digest = catalog.manifest["files"][entry["data_file"]]["sha256"]
    if (
        audit["files_sha256"][f"oligogym/resources/pkg_dataset/{entry['data_file']}"]
        != digest
    ):
        raise ValueError("Vendored data differ from the audited upstream release")
    if config["upstream_revision"] != PINNED_REVISION:
        raise ValueError("Config and audited source revision differ")
    # Imports must resolve to the source that was audited, not another installation.
    import numpy as np
    import oligogym.data.dataset_downloader as downloader_module
    import pandas as pd
    import torch
    from oligogym.data import Dataset as UpstreamDataset
    from oligogym.features import KMersCounts, OneHotEncoder
    from oligogym.models import (
        CNN,
        MLP,
        LinearModel,
        RandomForestModel,
        XGBoostModel,
    )

    expected = (checkout / "oligogym/data/dataset_downloader.py").resolve()
    if Path(downloader_module.__file__).resolve() != expected:
        raise ValueError("Installed OligoGym differs from audited checkout")
    if version("scikit-learn") != "1.4.0":
        raise ValueError("Baseline requires pinned upstream scikit-learn 1.4.0")
    linear = config["model"]["name"] == "LinearModel"
    neural = config["model"]["name"] in {"CNN", "MLP"}
    model_classes = {
        "LinearModel": LinearModel,
        "RandomForestModel": RandomForestModel,
        "XGBoostModel": XGBoostModel,
        "CNN": CNN,
        "MLP": MLP,
    }
    if config["model"]["name"] not in model_classes:
        raise ValueError("Unsupported upstream reproduction model")
    feature_classes = {"KMersCounts": KMersCounts, "OneHotEncoder": OneHotEncoder}
    if config["featurizer"]["name"] not in feature_classes:
        raise ValueError("Supported upstream features: KMersCounts, OneHotEncoder")
    reference = published_reference(checkout, config)
    if (
        "reference_mean_pearson" in config
        and config["reference_mean_pearson"] != reference["mean"]
    ):
        raise ValueError("Configured reference does not match the upstream CSV")
    data = UpstreamDataset(json.loads(source_meta.read_text()))
    data.build(dataframe=pd.read_csv(source_data, compression="gzip"))
    if mode == "benchmark_helpers":
        original = benchmark.download_data_with_retries(
            {"dataset": entry["key"]}, max_retries=1
        )
        if not original.data.equals(data.data):
            raise ValueError("Native downloader differs from the vendored data")
        data = original
    if config["split"]["strategy"] == "target" and (
        pd.isna(data.targets).any() or len(set(data.targets)) < 2
    ):
        raise ValueError("Target split requires known targets and at least two groups")
    ids = [f"{digest[:16]}:{entry['name']}:{i}" for i in range(len(data.x))]
    results = []
    for seed in config["seeds"]:
        started = perf_counter()
        random.seed(seed)
        np.random.seed(seed)
        if neural:
            torch.manual_seed(seed)
        print(f"seed={seed} phase=splitting rows={len(data.x)}", flush=True)
        args = dict(config["split"])
        strategy = args.pop("strategy")
        if mode == "benchmark_helpers":
            split = benchmark.prepare_data(data, split_strategy=strategy)
        else:
            split = data.split(strategy, random_state=seed, return_index=True, **args)
        xtrain, xtest, ytrain, ytest, train_idx, test_idx = split
        if set(xtrain) & set(xtest):
            raise ValueError("Exact sequence overlap across official split")
        feature_args = dict(config["featurizer"])
        feature_args.pop("name")
        print(f"seed={seed} phase=featurizing train_rows={len(xtrain)}", flush=True)
        if mode == "benchmark_helpers":
            script_config = {
                "model": config["model"]["name"],
                "featurizer": config["featurizer"]["name"],
                "featurizer_args": feature_args,
            }
            featurizer = benchmark.prepare_featurizer(script_config)
            features_train, features_test = benchmark.featurize(
                xtrain, xtest, featurizer, script_config
            )
        else:
            featurizer = feature_classes[config["featurizer"]["name"]](**feature_args)
            features_train = featurizer.fit_transform(xtrain)
            features_test = featurizer.transform(xtest)
            if len(features_train.shape) == 3:
                features_train = features_train.reshape(len(xtrain), -1)
                features_test = features_test.reshape(len(xtest), -1)
        model, model_execution = prepare_baseline_model(
            config,
            seed,
            features_train.shape[-1],
            model_classes[config["model"]["name"]],
            benchmark,
        )
        print(f"seed={seed} phase=training model={config['model']['name']}", flush=True)
        training_execution = None
        if neural:
            fit_args = dict(config.get("training", {}))
            trainer_args = dict(config.get("trainer", {}))
            defaults = {
                key: parameter.default
                for key, parameter in inspect.signature(model.fit).parameters.items()
                if parameter.default is not inspect.Parameter.empty
                and key not in {"X_val", "y_val"}
            }
            model.fit(features_train, ytrain, **fit_args, **trainer_args)
            training_execution = {
                "effective_fit_settings": serialize_parameters(
                    {**defaults, **fit_args}
                ),
                "declared_trainer_settings": trainer_args,
                "completed_epochs": int(model.trainer.current_epoch),
                "torch_seed": seed,
                "inner_validation": {
                    "source": "upstream LightningModel.fit",
                    "test_size": 0.2,
                    "random_state": 42,
                    "uses_outer_test_labels": False,
                },
            }
        else:
            model.fit(features_train, ytrain)
        predictions = np.asarray(model.predict(features_test)).reshape(-1)
        metrics = regression_metrics(tuple(ytest), tuple(predictions))
        results.append(
            {
                "seed": seed,
                "train_ids": [ids[i] for i in train_idx],
                "test_ids": [ids[i] for i in test_idx],
                "train_rows": len(train_idx),
                "test_rows": len(test_idx),
                "metrics": asdict(metrics),
                "model_execution": model_execution,
                "training_execution": training_execution,
                "feature_shape_train": list(features_train.shape),
                "feature_shape_test": list(features_test.shape),
                "prediction_sha256": hashlib.sha256(
                    np.asarray(predictions, dtype="<f8").tobytes()
                ).hexdigest(),
                "linear_diagnostics": {
                    "feature_count": features_train.shape[1],
                    "rank": int(model.model.rank_),
                    "smallest_singular_value": float(model.model.singular_[-1]),
                    "largest_singular_value": float(model.model.singular_[0]),
                    "max_abs_coefficient": float(np.max(np.abs(model.model.coef_))),
                }
                if linear and hasattr(model.model, "rank_")
                else None,
                "runtime_seconds": perf_counter() - started,
            }
        )
        if on_run:
            on_run(results[-1])
        print(
            f"seed={seed} test_rows={len(test_idx)} Pearson={metrics.pearson_r}",
            flush=True,
        )
    values = [r["metrics"]["pearson_r"] for r in results]
    defined = [v for v in values if v is not None]
    mean = fmean(defined) if defined else None
    delta = (
        mean - reference["mean"]
        if mean is not None and reference["mean"] is not None
        else None
    )
    return {
        "status": "measured_baseline_rerun_with_historical_discrepancies",
        "source_audit": audit,
        "config": config,
        "dataset": entry,
        "published_configuration_reference": reference,
        "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest(),
        "environment": environment_identity(),
        "installed_distribution_versions": {
            dist.metadata["Name"]: dist.version
            for dist in sorted(
                distributions(), key=lambda dist: dist.metadata["Name"].lower()
            )
        },
        "versions": {
            name: version(name)
            for name in (
                "numpy",
                "pandas",
                "scikit-learn",
                "scipy",
                "torch",
                "catboost",
                "torch-geometric",
                "torchinfo",
                "pytorch-lightning",
                "xgboost",
            )
        },
        "runs": results,
        "mean_pearson_r": mean,
        "sample_sd_pearson_r": stdev(defined) if len(defined) > 1 else None,
        "n_defined_seeds": len(defined),
        "delta_from_published_configuration": delta,
        "within_numeric_tolerance": (
            abs(delta) <= config["tolerance"] if delta is not None else None
        ),
        "gate_1_exact_reproduction": False,
        "caveats": [
            "Dataset-specific alias verified from the pinned catalog/Analysis.ipynb.",
            "Upstream benchmark script does not record deterministic split seeds.",
            "Three missing imports were installed separately; see versions.",
            "CSV model arguments may be dropped by upstream prepare_model.",
            f"Execution mode: {mode}; inspect effective estimator params per seed.",
            "Our seed schedule is explicit; historical seeds remain unavailable.",
            "The selected model/configuration is explicit; no local model selection.",
            "Neural CPU/deterministic/logging choices are declared in trainer config.",
            "No active-learning comparison has been performed by this command.",
            *entry["caveats"],
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/reproduce-linear.toml")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dataset", help="Any catalog key/name/alias; derives its own reference"
    )
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; preserve previous results")
    config = tomllib.loads(args.config.read_text())
    catalog = Catalog(args.data_root) if args.data_root else Catalog()
    if args.dataset:
        entry = catalog.resolve(args.dataset)
        config.update(loader_key=entry["key"], benchmark_alias=entry["benchmark_alias"])
        config.pop("reference_mean_pearson", None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    progress_path = args.output.with_suffix(".progress.json")
    if progress_path.exists():
        parser.error("Progress output exists; preserve previous attempts")
    progress = {"status": "running", "config": config, "runs": []}

    def write_progress(run: dict | None = None) -> None:
        if run is not None:
            progress["runs"].append(run)
        temporary = progress_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(progress, allow_nan=False) + "\n")
        temporary.replace(progress_path)

    write_progress()
    try:
        result = reproduce(args.checkout, config, catalog, on_run=write_progress)
    except Exception as error:
        progress.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_progress()
        raise
    with args.output.open("x") as file:
        file.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    progress["status"] = "complete"
    write_progress()
    print(f"Mean Pearson={result['mean_pearson_r']}; exact Gate 1 remains open.")


if __name__ == "__main__":
    main()
