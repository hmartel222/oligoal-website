"""Run the frozen Huesken CNN random-versus-diversity measured-label pilot."""

import argparse
import copy
import hashlib
import json
import shutil
import tomllib
from dataclasses import asdict, replace
from importlib.metadata import version
from pathlib import Path

from oligo_engine.acquisition.pool_diversity import DiversityAcquisition
from oligo_engine.acquisition.random import RandomAcquisition
from oligo_engine.audit_upstream import PINNED_REVISION, audit_checkout
from oligo_engine.contracts import RunConfig, SplitManifest
from oligo_engine.data.catalog import DEFAULT_DATA_ROOT, Catalog
from oligo_engine.evaluation.runner import environment_identity, run_campaign
from oligo_engine.models.upstream_cnn import UpstreamCNN
from oligo_engine.native_cache import native_monomer_cache
from oligo_engine.reporting.artifacts import write_artifacts

ROOT = Path(__file__).resolve().parents[2]
BASELINE_ARTIFACT = ROOT / "artifacts/gate1-efficacy-20261114-fixed-env/Huesken.json"
CAVEATS = (
    "Measured Huesken siRNA efficacy pilot using the source-pinned one-hot CNN.",
    "Explicit diagnostic purpose; seven-dataset/historical Gate 1 remains unresolved.",
    "Diversity is label-blind coverage selection, not efficacy-feedback adaptation.",
    "Official nucleobase holdouts; no unseen-target or clinical performance claim.",
    "Predictive means only; no calibrated uncertainty estimate.",
    "Full-reference models are freshly fitted with this study's canonical fit order.",
    "Thresholds and test metrics are reporting only, never policy/stopping feedback.",
)


def load_config(path: Path) -> tuple[dict, dict]:
    config = tomllib.loads(path.read_text())
    if (
        config["dataset"] != "Huesken"
        or config["purpose"] != "diagnostic"
        or config["seeds"] != [0, 1, 2, 3, 4]
        or config["policies"] != ["random", "diversity"]
        or config["budget"]
        != {"initial_labels": 50, "batch_size": 50, "max_labels": 500}
        or config["upstream"]["revision"] != PINNED_REVISION
    ):
        raise ValueError("Expected the frozen Huesken diagnostic pilot configuration")
    baseline_path = ROOT / config["upstream"]["baseline_config"]
    if (
        hashlib.sha256(baseline_path.read_bytes()).hexdigest()
        != config["upstream"]["baseline_config_sha256"]
    ):
        raise ValueError("Frozen native CNN baseline configuration hash differs")
    baseline = tomllib.loads(baseline_path.read_text())
    if config["split"] != {
        **baseline["split"],
        "seed_mode": "numpy_global_before_each_split",
    }:
        raise ValueError("Pilot split settings differ from the audited default split")
    if config["analysis"] != {
        "primary": "paired_normalized_pearson_aulc_diversity_minus_random",
        "band_method": "sample_standard_deviation",
        "full_reference_absolute_r_tolerance": 0.03,
        "reference_must_be_positive": True,
        "label_fraction_criterion": 0.5,
        "threshold_is_reporting_only": True,
    } or config["execution"] != {
        "fit_order": "sample_id_lexicographic",
        "refit_full_reference": True,
        "reuse_matching_full_reference_between_policies": True,
        "test_scoring": "deferred_until_campaign_acquisition_complete",
    }:
        raise ValueError("Analysis or execution differs from frozen pilot protocol")
    return config, baseline


def official_split(dataset, seed: int, checkout: Path) -> SplitManifest:
    """Regenerate native-helper splits without accessing any measured labels."""
    import numpy as np
    import oligogym.split as upstream_split

    if Path(upstream_split.__file__).resolve() != checkout / "oligogym/split.py":
        raise ValueError("Split implementation differs from audited checkout")
    if version("scikit-learn") != "1.4.0":
        raise ValueError("Official split requires scikit-learn 1.4.0")
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        result = upstream_split.nucleobase_split(
            np.asarray([row.sequence for row in dataset.observations]),
            np.zeros(len(dataset.observations)),
            test_size=0.2,
            return_index=True,
            kmer_max=3,
            kmer=True,
            strand="RNA1",
            n_clusters=8,
        )
    finally:
        np.random.set_state(state)
    if len(result) != 6:
        raise ValueError("Unexpected pinned splitter return schema")
    ids = [row.sample_id for row in dataset.observations]
    split = SplitManifest(
        "nucleobase",
        seed,
        tuple(ids[index] for index in result[-2]),
        tuple(ids[index] for index in result[-1]),
        (
            ("source_revision", PINNED_REVISION),
            ("seed_mode", "numpy_global_before_each_split"),
            ("numpy_seed", str(seed)),
            ("kmeans_random_state", "None"),
            ("n_clusters", "8"),
            ("kmer_max", "3"),
            ("construction_labels", "all_dummy_zero; no measured labels accessed"),
        ),
    )
    split.validate(dataset)
    return split


def verify_previous_splits(splits: tuple[SplitManifest, ...], path: Path) -> dict:
    if not path.exists():
        return {"status": "baseline_artifact_unavailable_regenerated_from_source"}
    previous = json.loads(path.read_text())
    if previous["source_audit"]["source_revision"] != PINNED_REVISION:
        raise ValueError("Previous baseline source revision differs")
    by_seed = {run["seed"]: run for run in previous["runs"]}
    for split in splits:
        other = by_seed[split.seed]
        if (
            list(split.train_ids) != other["train_ids"]
            or list(split.test_ids) != other["test_ids"]
        ):
            raise ValueError("Official split IDs differ from completed Huesken rerun")
    return {
        "status": "exact_ordered_split_ids_match",
        "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "seeds_checked": [split.seed for split in splits],
        "artifact": str(path),
    }


def make_diversity(dataset, split: SplitManifest, settings: dict):
    from oligogym.features import KMersCounts

    if settings != {
        "featurizer": "KMersCounts",
        "k": [1, 2, 3],
        "modification_abundance": False,
        "strands": "all",
        "normalization": "per_row_l2",
        "distance": "euclidean",
        "selection": "greedy_maximum_minimum_distance_to_labeled_and_batch",
        "tie_break": "sample_id_lexicographic",
        "representation_fit_scope": "eligible_training_pool_sequences_only",
    }:
        raise ValueError("Unexpected diversity representation or acquisition settings")
    by_id = {row.sample_id: row for row in dataset.observations}
    records = tuple(by_id[identity] for identity in split.train_ids)
    encoder = KMersCounts(k=[1, 2, 3], modification_abundance=False, strands=None)
    embedding = encoder.fit_transform([row.sequence for row in records])
    return DiversityAcquisition(records, embedding, settings)


def archive_source(output: Path, config_path: Path, baseline_path: Path) -> dict:
    destination = output / "source"
    files = list((ROOT / "src/oligo_engine").rglob("*.py"))
    files.extend(
        [config_path, baseline_path, ROOT / "scripts/report_huesken_learning.py"]
    )
    manifest = {}
    for path in files:
        relative = path.relative_to(ROOT)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        manifest[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def run_study(
    checkout: Path,
    config_path: Path,
    output: Path,
    data_root: Path = DEFAULT_DATA_ROOT,
    smoke: bool = False,
    previous_baseline: Path = BASELINE_ARTIFACT,
) -> dict:
    config_path, checkout = config_path.resolve(), checkout.resolve()
    config, baseline = load_config(config_path)
    audit = audit_checkout(checkout)
    if output.exists():
        raise ValueError("Study output exists; preserve previous attempts")
    catalog = Catalog(data_root)
    loaded = catalog.load("Huesken")
    if len(loaded.dataset.observations) != 2431:
        raise ValueError("Huesken study must use all 2431 released observations")
    source_name = catalog.resolve("Huesken")["data_file"]
    if audit["files_sha256"][f"oligogym/resources/pkg_dataset/{source_name}"] != (
        loaded.dataset.content_sha256
    ):
        raise ValueError("Catalog data differ from audited native release")
    effective_seeds = [0] if smoke else config["seeds"]
    budget = copy.deepcopy(config["budget"])
    if smoke:
        budget["max_labels"] = 100
    output.mkdir(parents=True)
    state = {
        "status": "running",
        "study": "huesken_cnn_random_vs_label_blind_diversity",
        "mode": "smoke" if smoke else "full",
        "purpose": "diagnostic",
        "config": config,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "effective_seeds": effective_seeds,
        "effective_budget": budget,
        "source_audit": audit,
        "environment": environment_identity(),
        "source_snapshot": archive_source(
            output, config_path, ROOT / config["upstream"]["baseline_config"]
        ),
        "dataset": loaded.public_metadata(),
        "caveats": CAVEATS,
        "runs": [],
    }

    def save() -> None:
        temporary = output / "suite.tmp"
        temporary.write_text(json.dumps(state, indent=2, allow_nan=False) + "\n")
        temporary.replace(output / "suite.json")

    save()
    try:
        with native_monomer_cache(baseline.get("performance", {})) as performance:
            splits = tuple(
                official_split(loaded.dataset, seed, checkout)
                for seed in effective_seeds
            )
            state["splits"] = [asdict(split) for split in splits]
            state["previous_split_verification"] = verify_previous_splits(
                splits, previous_baseline
            )
            save()
            references = {}
            runs_by_policy = {name: [] for name in config["policies"]}
            for split in splits:
                policies = {
                    "random": RandomAcquisition(),
                    "diversity": make_diversity(
                        loaded.dataset, split, config["diversity"]
                    ),
                }
                for policy_name in config["policies"]:
                    oracle, evaluator, reference_oracle = loaded.capabilities(split)
                    run_config = RunConfig(
                        seed=split.seed,
                        purpose="diagnostic",
                        pearson_threshold=None,
                        **budget,
                    )

                    def factory(seed=split.seed):
                        return UpstreamCNN(checkout, seed, baseline)

                    def progress(phase, labels):
                        state["current"] = {
                            "seed": split.seed,
                            "policy": policy_name,
                            "phase": phase,
                            "labels": labels,
                        }
                        save()
                        print(
                            f"seed={split.seed} policy={policy_name} "
                            f"phase={phase} labels={labels}",
                            flush=True,
                        )

                    run = run_campaign(
                        loaded.dataset,
                        split,
                        oracle,
                        evaluator,
                        factory,
                        policies[policy_name],
                        run_config,
                        reference_oracle,
                        reference_cache=references,
                        progress=progress,
                    )
                    run = replace(run, caveats=CAVEATS)
                    directory = output / policy_name
                    directory.mkdir(exist_ok=True)
                    path = directory / f"run-seed-{split.seed}.json"
                    with path.open("x") as file:
                        file.write(
                            json.dumps(asdict(run), indent=2, allow_nan=False) + "\n"
                        )
                    runs_by_policy[policy_name].append(run)
                    state["runs"].append(
                        {"policy": policy_name, "seed": split.seed, "path": str(path)}
                    )
                    save()
            for policy_name, runs in runs_by_policy.items():
                write_artifacts(
                    tuple(runs), output / policy_name, allow_completed_runs=True
                )
            state["performance_execution"] = performance
        state["status"] = "complete"
        state.pop("current", None)
        final_source = environment_identity()["source_sha256"]
        if final_source != state["environment"]["source_sha256"]:
            raise ValueError("Source changed during the frozen study")
        audit_checkout(checkout)
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise
    save()
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/huesken-learning.toml")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--previous-baseline", type=Path, default=BASELINE_ARTIFACT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    result = run_study(
        args.checkout,
        args.config,
        args.output,
        args.data_root,
        args.smoke,
        args.previous_baseline,
    )
    print(f"Completed {len(result['runs'])} measured diagnostic campaigns.")


if __name__ == "__main__":
    main()
