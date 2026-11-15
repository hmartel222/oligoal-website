"""Inspect a clean pinned source checkout, without importing scientific packages."""

import argparse
import csv
import gzip
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from statistics import fmean

PINNED_REVISION = "97f5b9f58d9e445a8ecb0218037af7465c3708c0"


def audit_checkout(checkout: Path) -> dict:
    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(checkout), *args],
            text=True,
            stderr=subprocess.PIPE,
        ).strip()

    if git("rev-parse", "HEAD") != PINNED_REVISION:
        raise ValueError(f"Expected OligoGym revision {PINNED_REVISION}")
    if git("status", "--porcelain", "--untracked-files=all"):
        raise ValueError("Upstream source checkout must be clean")
    registry = checkout / "oligogym/resources/pkg_dataset/all_datasets_info.csv.gz"
    benchmark = checkout / "benchmarks/oligogym_benchmarks.csv"
    with gzip.open(registry, "rt") as file:
        rows = list(csv.DictReader(file))
    huesken = next(row for row in rows if row["key"] == "Huesken")
    if huesken["name"] != "Ichihara_2007_1":
        raise ValueError("Unexpected registry mapping")
    keys = {r["key"].lower() for r in rows}
    notebook = json.loads((checkout / "benchmarks/Analysis.ipynb").read_text())
    alias_verified = any(
        '"siRNA1": "Huesken"' in "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    if not alias_verified:
        raise ValueError("Expected historical alias mapping in Analysis.ipynb")
    groups: dict[tuple[str, ...], list[float]] = defaultdict(list)
    total = 0
    with benchmark.open() as file:
        for row in csv.DictReader(file):
            total += 1
            if (
                row["dataset"] == "siRNA1"
                and row["cross_validation"] == "nucleobase"
                and row["test_pearson_correlation"]
            ):
                key = tuple(
                    row[k]
                    for k in ("model", "featurizer", "featurizer_args", "model_args")
                )
                groups[key].append(float(row["test_pearson_correlation"]))
    best, values = max(groups.items(), key=lambda item: fmean(item[1]))
    files = [
        "pyproject.toml",
        "poetry.lock",
        "oligogym/split.py",
        "oligogym/features.py",
        "oligogym/models.py",
        "oligogym/data/dataset_downloader.py",
        "oligogym/data/data.py",
        "benchmarks/train_model.py",
        "benchmarks/config.yaml",
        "benchmarks/Analysis.ipynb",
        "benchmarks/oligogym_benchmarks.csv",
        "oligogym/resources/pkg_dataset/all_datasets_info.csv.gz",
        "oligogym/resources/pkg_dataset/Ichihara_2007_1_processed.csv.gz",
        "oligogym/resources/pkg_dataset/Ichihara_2007_1_processed.json",
    ]
    files = sorted(
        set(files)
        | {
            f"oligogym/resources/pkg_dataset/{row['name']}_processed.{suffix}"
            for row in rows
            for suffix in ("csv.gz", "json")
        }
    )
    return {
        "source_revision": PINNED_REVISION,
        "status": "source_audit_only_gate_1_open",
        "source_url": f"https://github.com/Roche/OligoGym/tree/{PINNED_REVISION}",
        "huesken_registry": {
            k: huesken[k] for k in ("key", "name", "size", "num_targets")
        },
        "sirna1_is_valid_loader_key": "sirna1" in keys,
        "historical_alias_mapping": "siRNA1 -> Huesken; Analysis.ipynb SIRNA_MAP",
        "registry": [
            {k: r[k] for k in ("key", "name", "size", "num_targets", "desc")}
            for r in rows
        ],
        "benchmark_rows": total,
        "sirna1_nucleobase_reference": {
            "model": best[0],
            "featurizer": best[1],
            "featurizer_args_repr": best[2],
            "model_args_repr": best[3],
            "n_valid_rows": len(values),
            "mean_pearson_r": fmean(values),
            "meaning": "upstream CSV aggregation; not a reproduced result",
        },
        "files_sha256": {
            path: hashlib.sha256((checkout / path).read_bytes()).hexdigest()
            for path in files
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = audit_checkout(args.checkout)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as file:
        file.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Source audit written to {args.output}; Gate 1 remains open.")


if __name__ == "__main__":
    main()
