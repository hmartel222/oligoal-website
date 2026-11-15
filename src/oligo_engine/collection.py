"""Validate the complete collection or run independent measured diagnostic replays."""

import argparse
import hashlib
import json
import tomllib
from dataclasses import asdict, replace
from pathlib import Path

from oligo_engine.acquisition.random import RandomAcquisition
from oligo_engine.acquisition.uncertainty import UncertaintyAcquisition
from oligo_engine.contracts import RunConfig
from oligo_engine.data.catalog import DEFAULT_DATA_ROOT, Catalog
from oligo_engine.data.splits import UnsupportedSplit, make_split
from oligo_engine.evaluation.runner import environment_identity, run_campaign
from oligo_engine.models.forest import HelmFeatureCache, HelmHashForest
from oligo_engine.reporting.artifacts import write_artifacts


def effective_budget(seed: int, budget: dict, train_rows: int) -> RunConfig:
    requested = RunConfig(
        seed=seed,
        purpose="diagnostic",
        pearson_threshold=None,
        **budget,
    )
    maximum = min(train_rows, requested.max_labels)
    return replace(
        requested,
        max_labels=maximum,
        initial_labels=min(requested.initial_labels, maximum),
    )


def run_collection(catalog: Catalog, config: dict, output: Path) -> dict:
    if config.get("purpose") != "diagnostic":
        raise ValueError("Collection runs must explicitly declare purpose='diagnostic'")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty; previous runs are preserved")
    seeds = config["seeds"]
    if (
        not seeds
        or any(type(s) is not int for s in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("Seeds must be distinct integers")
    policies = config.get("policies", ["random"])
    if (
        not policies
        or len(set(policies)) != len(policies)
        or set(policies) - {"random", "uncertainty"}
    ):
        raise ValueError("Policies must be unique random/uncertainty names")
    selector = config["dataset"]
    if isinstance(selector, list):
        entries = tuple(catalog.resolve(name) for name in selector)
        if not entries or len({e["name"] for e in entries}) != len(entries):
            raise ValueError("Dataset list must be nonempty and unique")
    else:
        entries = catalog.entries if selector == "all" else (catalog.resolve(selector),)
    output.mkdir(parents=True, exist_ok=True)
    suite = {
        "status": "running",
        "purpose": "diagnostic",
        "config": config,
        "source_revision": catalog.manifest["upstream_revision"],
        "datasets": [
            {"name": e["name"], "key": e["key"], "runs": [], "status": "pending"}
            for e in entries
        ],
        "environment": environment_identity(),
        "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest(),
        "caveat": "Engineering replays only; Gate 1 open; no pooled scientific score.",
    }

    def save_suite():
        temporary = output / "suite.json.tmp"
        temporary.write_text(json.dumps(suite, indent=2, allow_nan=False) + "\n")
        temporary.replace(output / "suite.json")

    save_suite()
    for entry, record in zip(entries, suite["datasets"], strict=True):
        record["status"] = "running"
        save_suite()
        try:
            loaded = catalog.load(
                entry["name"],
                max_rows=config.get("max_rows"),
                selection_seed=config.get("selection_seed", 2026),
            )
            prepared = [
                (seed, make_split(loaded.dataset, seed=seed, **config["split"]))
                for seed in seeds
            ]
            print(
                f"{entry['key']}: loaded "
                f"{len(loaded.dataset.observations)} source rows",
                flush=True,
            )
            feature_cache = (
                HelmFeatureCache(
                    loaded.dataset.observations,
                    config.get("model", {}).get("n_features", 512),
                )
                if config.get("cache_features", False)
                else None
            )
            reference_cache = {} if config.get("reuse_full_reference", False) else None
            for policy_name in policies:
                runs = []
                for seed, split in prepared:
                    oracle, evaluator, reference = loaded.capabilities(split)
                    budget = effective_budget(
                        seed, config["budget"], len(split.train_ids)
                    )

                    def factory(seed=seed):
                        return HelmHashForest(
                            seed=seed,
                            feature_cache=feature_cache,
                            **config.get("model", {}),
                        )

                    def progress(phase, labels):
                        record["current"] = {
                            "seed": seed,
                            "policy": policy_name,
                            "phase": phase,
                            "labels": labels,
                        }
                        save_suite()
                        print(
                            f"{entry['key']} / {policy_name} / seed={seed}: "
                            f"{phase}, labels={labels}",
                            flush=True,
                        )

                    policy = (
                        RandomAcquisition()
                        if policy_name == "random"
                        else UncertaintyAcquisition()
                    )
                    run = run_campaign(
                        loaded.dataset,
                        split,
                        oracle,
                        evaluator,
                        factory,
                        policy,
                        budget,
                        reference,
                        reference_cache=reference_cache,
                        progress=progress,
                    )
                    runs.append(run)
                    directory = output / entry["name"] / policy_name
                    directory.mkdir(parents=True, exist_ok=True)
                    with (directory / f"run-seed-{seed}.json").open("x") as file:
                        file.write(
                            json.dumps(asdict(run), indent=2, allow_nan=False) + "\n"
                        )
                    record["runs"].append(
                        {
                            "policy": policy_name,
                            "seed": seed,
                            "status": run.status,
                            "effective_budget": asdict(budget),
                            "train_rows": len(split.train_ids),
                            "test_rows": len(split.test_ids),
                        }
                    )
                    save_suite()
                    print(
                        f"{entry['key']} / {policy_name} / seed={seed}: "
                        f"{len(loaded.dataset.observations)} rows, "
                        f"{budget.max_labels} labels; diagnostic complete",
                        flush=True,
                    )
                write_artifacts(
                    tuple(runs),
                    output / entry["name"] / policy_name,
                    allow_completed_runs=True,
                )
            record["status"] = "complete"
            record.pop("current", None)
        except UnsupportedSplit as error:
            record.update(status="unsupported_split", reason=str(error))
        except Exception as error:
            # Preserve successful datasets and a machine-readable failure ledger.
            record.update(status="failed", reason=f"{type(error).__name__}: {error}")
        save_suite()
    suite["status"] = (
        "complete"
        if all(r["status"] == "complete" for r in suite["datasets"])
        else "incomplete"
    )
    save_suite()
    return suite


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--list-datasets", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        catalog = Catalog(args.data_root)
        if args.list_datasets:
            for entry in catalog.entries:
                print(
                    f"{entry['key']} | {entry['name']} | "
                    f"{entry['modality']} | {entry['rows']} rows"
                )
            return
        if args.validate:
            report = catalog.validate_all()
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open("x") as file:
                    file.write(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report, indent=2))
            return
        if args.config is None or args.output is None:
            parser.error(
                "Specify --validate, --list-datasets, or --config and --output"
            )
        result = run_collection(
            catalog,
            tomllib.loads(args.config.read_text()),
            args.output,
        )
        if result["status"] != "complete":
            parser.exit(
                1, f"Collection run incomplete; inspect {args.output / 'suite.json'}\n"
            )
    except (ValueError, OSError, KeyError, ModuleNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
