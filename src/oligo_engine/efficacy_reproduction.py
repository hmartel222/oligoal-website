"""Freeze and execute published best nucleobase baselines on all efficacy datasets."""

import argparse
import ast
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import tomllib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import fmean, stdev

from oligo_engine.audit_upstream import PINNED_REVISION, audit_checkout
from oligo_engine.data.catalog import Catalog

ROOT = Path(__file__).resolve().parents[2]
TRAINING = {
    "val_split": True,
    "max_epochs": 100,
    "early_stopping": True,
    "early_stopping_kwargs": {"patience": 5},
    "batch_size": 32,
    "learning_rate": 0.001,
    "weight_decay": 0.01,
    "verbose": False,
}
TRAINER = {
    "accelerator": "cpu",
    "devices": 1,
    "deterministic": True,
    "enable_checkpointing": False,
    "logger": False,
    "enable_progress_bar": False,
    "enable_model_summary": False,
}


def select_best(rows: list[dict], aliases: set[str]) -> dict:
    """Use released results only; require five finite folds for every candidate."""
    import math

    groups = defaultdict(list)
    arguments = {}
    for row in rows:
        if row["dataset"] not in aliases or row["cross_validation"] != "nucleobase":
            continue
        model_args = ast.literal_eval(row["model_args"])
        feature_args = ast.literal_eval(row["featurizer_args"])
        identity = json.dumps(
            [row["dataset"], row["model"], model_args, row["featurizer"], feature_args],
            sort_keys=True,
        )
        text = row["test_pearson_correlation"].strip()
        value = float(text) if text else float("nan")
        groups[identity].append(value)
        arguments[identity] = {
            "model": {"name": row["model"], **model_args},
            "featurizer": {"name": row["featurizer"], **feature_args},
            "alias": row["dataset"],
        }
    winners = {}
    for alias in sorted(aliases):
        candidates = [
            (fmean(values), identity, values)
            for identity, values in groups.items()
            if arguments[identity]["alias"] == alias
            and len(values) == 5
            and all(math.isfinite(value) for value in values)
        ]
        if not candidates:
            raise ValueError(f"No complete five-fold published baseline: {alias}")
        # Lexicographic tie-breaking is independent of any local model result.
        mean, identity, values = sorted(candidates, key=lambda c: (-c[0], c[1]))[0]
        winners[alias] = {
            **arguments[identity],
            "reference_mean_pearson": mean,
            "reference_sample_sd": stdev(values),
            "reference_values": values,
        }
    return winners


def toml_scalar(value: object) -> str:
    if isinstance(value, dict):
        return (
            "{ "
            + ", ".join(f"{key} = {toml_scalar(item)}" for key, item in value.items())
            + " }"
        )
    return json.dumps(value)


def config_text(config: dict) -> str:
    lines = ["# Frozen from the released benchmark before local best-model reruns."]
    for key, value in config.items():
        if not isinstance(value, dict):
            lines.append(f"{key} = {toml_scalar(value)}")
    for key, value in config.items():
        if isinstance(value, dict):
            lines.extend(["", f"[{key}]"])
            lines.extend(
                f"{name} = {toml_scalar(item)}" for name, item in value.items()
            )
    return "\n".join(lines) + "\n"


def freeze(checkout: Path, destination: Path, catalog: Catalog) -> dict:
    audit = audit_checkout(checkout)
    if destination.exists():
        raise ValueError("Configuration destination exists; preserve frozen settings")
    entries = [row for row in catalog.entries if row["collection"] == "Activity"]
    aliases = {row["benchmark_alias"] for row in entries}
    if len(entries) != 7 or len(aliases) != 7:
        raise ValueError("Expected all seven released efficacy datasets")
    with (checkout / "benchmarks/oligogym_benchmarks.csv").open() as file:
        winners = select_best(list(csv.DictReader(file)), aliases)
    destination.mkdir(parents=True)
    frozen = []
    for entry in sorted(entries, key=lambda row: row["key"]):
        winner = winners[entry["benchmark_alias"]]
        config = {
            "upstream_revision": PINNED_REVISION,
            "loader_key": entry["key"],
            "benchmark_alias": entry["benchmark_alias"],
            "seeds": [0, 1, 2, 3, 4],
            "reference_mean_pearson": winner["reference_mean_pearson"],
            "tolerance": 0.03,
            "execution_mode": "benchmark_helpers",
            "split": {
                "strategy": "nucleobase",
                "test_size": 0.2,
                "n_clusters": 8,
                "kmer": True,
                "kmer_max": 3,
                "strand": "RNA1",
            },
            "featurizer": winner["featurizer"],
            "model": winner["model"],
            "performance": {
                "cache_native_monomers": True,
                "max_entries": 262144,
            },
        }
        if winner["model"]["name"] in {"CNN", "MLP"}:
            config.update(training=TRAINING, trainer=TRAINER)
        content = config_text(config)
        filename = entry["key"].replace(" ", "-") + ".toml"
        (destination / filename).write_text(content)
        frozen.append(
            {
                "dataset": entry,
                "file": filename,
                "sha256": hashlib.sha256(content.encode()).hexdigest(),
                "published": winner,
            }
        )
    manifest = {
        "source_revision": PINNED_REVISION,
        "benchmark_sha256": audit["files_sha256"]["benchmarks/oligogym_benchmarks.csv"],
        "selection": "highest released mean test Pearson, five nucleobase repeats",
        "local_results_used_for_selection": False,
        "datasets": frozen,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def interpreter_path(path: Path) -> Path:
    """Keep the virtualenv executable path; resolving its symlink loses the env."""
    return path.expanduser().absolute()


def run_suite(configs: Path, checkout: Path, python: Path, output: Path) -> dict:
    configs, checkout, output = (path.resolve() for path in (configs, checkout, output))
    python = interpreter_path(python)
    if output.exists():
        raise ValueError("Suite output exists; preserve previous experiments")
    audit = audit_checkout(checkout)
    manifest = json.loads((configs / "manifest.json").read_text())
    expected = {
        row["key"]: row for row in Catalog().entries if row["collection"] == "Activity"
    }
    selected = [row["dataset"]["key"] for row in manifest["datasets"]]
    if len(selected) != 7 or set(selected) != set(expected):
        raise ValueError("Suite must include each of the seven efficacy datasets once")
    if (
        manifest["source_revision"] != PINNED_REVISION
        or manifest["benchmark_sha256"]
        != audit["files_sha256"]["benchmarks/oligogym_benchmarks.csv"]
    ):
        raise ValueError("Frozen configurations differ from audited benchmark source")
    for row in manifest["datasets"]:
        if (
            Path(row["file"]).name != row["file"]
            or row["dataset"] != expected[row["dataset"]["key"]]
        ):
            raise ValueError("Frozen dataset identity or configuration path differs")
        if (
            hashlib.sha256((configs / row["file"]).read_bytes()).hexdigest()
            != row["sha256"]
        ):
            raise ValueError("Frozen configuration was modified")
        settings = tomllib.loads((configs / row["file"]).read_text())
        if (
            settings["seeds"] != [0, 1, 2, 3, 4]
            or settings["execution_mode"] != "benchmark_helpers"
            or settings["loader_key"] != row["dataset"]["key"]
        ):
            raise ValueError("Suite requires the frozen five-seed helper protocol")
    output.mkdir(parents=True)
    suite = {"status": "running", "manifest": manifest, "datasets": {}}

    def save() -> None:
        temporary = output / "suite.tmp"
        temporary.write_text(json.dumps(suite, indent=2, allow_nan=False) + "\n")
        temporary.replace(output / "suite.json")

    def worker(row: dict) -> dict:
        key = row["dataset"]["key"]
        print(f"Starting published efficacy baseline: {key}", flush=True)
        environment = dict(os.environ)
        environment.update(
            PYTHONPATH=str(ROOT / "src"),
            MPLCONFIGDIR=str(Path(tempfile.gettempdir()) / "oligoal-mpl"),
            LOKY_MAX_CPU_COUNT="2",
            OMP_NUM_THREADS="2",
            OPENBLAS_NUM_THREADS="2",
            MKL_NUM_THREADS="2",
            PYTHONHASHSEED="0",
        )
        filename = Path(row["file"]).stem
        result_path = output / f"{filename}.json"
        with (output / f"{filename}.log").open("w") as log:
            process = subprocess.run(
                [
                    str(python),
                    "-m",
                    "oligo_engine.reproduce",
                    "--checkout",
                    str(checkout),
                    "--config",
                    str(configs / row["file"]),
                    "--output",
                    str(result_path),
                ],
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                cwd=ROOT,
            )
        if process.returncode != 0:
            return {"status": "failed", "returncode": process.returncode}
        result = json.loads(result_path.read_text())
        if [run["seed"] for run in result["runs"]] != [0, 1, 2, 3, 4]:
            raise ValueError("Incomplete baseline seed ledger")
        return {
            "status": "complete",
            "mean_pearson_r": result["mean_pearson_r"],
            "sample_sd_pearson_r": result["sample_sd_pearson_r"],
            "reference": result["published_configuration_reference"],
            "delta": result["delta_from_published_configuration"],
            "within_numeric_tolerance": result["within_numeric_tolerance"],
            "n_defined_seeds": result["n_defined_seeds"],
            "artifact": str(result_path),
        }

    # Start the largest dataset first; process isolation bounds memory across models.
    priority = {
        key: index
        for index, key in enumerate(
            [
                "Sherwood",
                "Huesken",
                "Ichihara",
                "Shmushkovich",
                "siRNAmod",
                "OpenASO",
                "ASOptimizer",
            ]
        )
    }
    rows = sorted(manifest["datasets"], key=lambda row: priority[row["dataset"]["key"]])
    suite["datasets"] = {row["dataset"]["key"]: {"status": "scheduled"} for row in rows}
    save()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(worker, row): row["dataset"]["key"] for row in rows}
        for future in as_completed(futures):
            key = futures[future]
            try:
                suite["datasets"][key] = future.result()
            except Exception as error:
                suite["datasets"][key] = {"status": "failed", "error": str(error)}
            print(f"Finished {key}: {suite['datasets'][key]}", flush=True)
            save()
    suite["status"] = (
        "complete"
        if all(r["status"] == "complete" for r in suite["datasets"].values())
        else "failed"
    )
    suite["all_published_means_within_tolerance"] = all(
        row.get("within_numeric_tolerance", False) for row in suite["datasets"].values()
    )
    suite["gate_1_exact_historical_reproduction"] = False
    save()
    return suite


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--configs", type=Path, default=Path("configs/gate1-efficacy"))
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.freeze:
        manifest = freeze(args.checkout, args.configs, Catalog())
        print(f"Frozen {len(manifest['datasets'])} published efficacy baselines")
        return
    if args.output is None:
        parser.error("--output is required for execution")
    result = run_suite(args.configs, args.checkout, args.python, args.output)
    if result["status"] != "complete":
        parser.exit(
            1, "One or more baseline runs failed; inspect suite.json and logs\n"
        )
    print(
        "All seven efficacy baseline executions complete; inspect numerical agreement."
    )


if __name__ == "__main__":
    main()
