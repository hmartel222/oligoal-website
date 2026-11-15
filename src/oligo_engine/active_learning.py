"""Four-policy Huesken development study with acquisition-budgeted GP selection."""

import argparse
import hashlib
import json
import shutil
import tomllib
from dataclasses import asdict, replace
from pathlib import Path
from random import Random
from time import perf_counter

from oligo_engine.acquisition.hybrid import HybridAcquisition
from oligo_engine.acquisition.pool_diversity import DiversityAcquisition
from oligo_engine.acquisition.random import RandomAcquisition
from oligo_engine.acquisition.uncertainty import UncertaintyAcquisition
from oligo_engine.contracts import RunConfig
from oligo_engine.data.catalog import Catalog
from oligo_engine.evaluation.runner import environment_identity, run_campaign
from oligo_engine.features.upstream import OneHotBaseFeaturizer
from oligo_engine.huesken_learning import official_split
from oligo_engine.models.baseline import KERNEL_GRID, OneHotGaussianProcess

ROOT = Path(__file__).resolve().parents[2]
POLICIES = ("random", "diversity", "uncertainty", "hybrid")
CAVEATS = (
    "Measured Huesken development study; prior dataset-level exploration disclosed.",
    "Seven-dataset and exact historical reproduction remain unresolved.",
    "GP latent uncertainty is uncalibrated; benefit must be measured.",
    "Official nucleobase holdout does not establish unseen-gene generalization.",
    "Kernel selection consumes only the counted initial labels.",
    "No therapeutic validation, pooled safety claim or guaranteed label savings.",
)


class DiagnosticPolicy:
    """Log training-pool prediction spread without accessing any labels."""

    def __init__(self, policy):
        self.policy = policy
        self.rounds = []

    @property
    def config(self) -> dict:
        return self.policy.config

    def choose(self, eligible, labeled, budget, rng, predictor):
        import numpy as np

        started = perf_counter()
        values = np.array([p.uncertainty for p in predictor.predict(eligible)])
        selected = self.policy.choose(eligible, labeled, budget, rng, predictor)
        self.rounds.append(
            {
                "labels_before": len(labeled),
                "batch_size": budget,
                "candidate_count": len(eligible),
                "latent_sd_mean": float(values.mean()),
                "latent_sd_min": float(values.min()),
                "latent_sd_max": float(values.max()),
                "latent_sd_cv": float(values.std() / values.mean())
                if values.mean() > 0
                else None,
                "selection_and_diagnostic_seconds": perf_counter() - started,
            }
        )
        return selected


def save_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def schedule(pool_size: int, smoke: bool = False) -> tuple[int, ...]:
    if pool_size < 100:
        raise ValueError("Huesken protocol requires at least 100 eligible rows")
    if smoke:
        return (50, 100)
    if pool_size < 1000:
        raise ValueError("Pool cannot support the fixed primary 50-1000 interval")
    return tuple(
        sorted(
            set(range(50, 1001, 50))
            | {n for n in (1250, 1500, pool_size) if n <= pool_size}
        )
    )


def select_kernel(observations, labels, seed: int) -> dict:
    """Three inner folds; no access to any label outside initial acquisition."""
    import numpy as np
    from sklearn.model_selection import GroupKFold, KFold

    if len(observations) != 50 or len(labels) != 50:
        raise ValueError("Selection requires exactly the counted initial 50 labels")
    groups = [row.target_id for row in observations]
    grouped = all(groups) and len(set(groups)) >= 3
    splitter = GroupKFold(3) if grouped else KFold(3, shuffle=True, random_state=seed)
    folds = list(splitter.split(np.arange(50), groups=groups if grouped else None))
    scores = []
    for family, length in KERNEL_GRID:
        errors = []
        for train, valid in folds:
            predictor = OneHotGaussianProcess(seed, family, length)
            predictor.fit(
                tuple(observations[i] for i in train), tuple(labels[i] for i in train)
            )
            predictions = predictor.predict(tuple(observations[i] for i in valid))
            errors.append(
                float(
                    np.sqrt(
                        np.mean(
                            [
                                (p.mean - labels[i]) ** 2
                                for p, i in zip(predictions, valid, strict=True)
                            ]
                        )
                    )
                )
            )
        scores.append(
            {
                "family": family,
                "length_scale": length,
                "fold_rmse": errors,
                "mean_rmse": float(np.mean(errors)),
            }
        )
    winner = min(scores, key=lambda row: row["mean_rmse"])
    return {
        "selected": {key: winner[key] for key in ("family", "length_scale")},
        "criterion": "mean three-fold RMSE; grid order breaks ties",
        "validation": "target_group" if grouped else "shuffled_row_fallback",
        "initial_ids": [row.sample_id for row in observations],
        "folds": [
            {
                "train_ids": [observations[i].sample_id for i in train],
                "validation_ids": [observations[i].sample_id for i in valid],
            }
            for train, valid in folds
        ],
        "scores": scores,
    }


def run_study(config_path: Path, checkout: Path, output: Path, smoke: bool) -> dict:
    config = tomllib.loads(config_path.read_text())
    if config["dataset"] != "Huesken" or config["policies"] != list(POLICIES):
        raise ValueError("Expected Huesken and the frozen four-policy comparison")
    seeds = config["seeds"]
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(type(s) is not int for s in seeds)
    ):
        raise ValueError("Seeds must be distinct integers")
    if output.exists():
        raise ValueError("Output exists; preserve earlier attempts")
    from oligo_engine.audit_upstream import audit_checkout

    audit = audit_checkout(checkout.resolve())
    loaded = Catalog().load("Huesken")
    source_name = Catalog().resolve("Huesken")["data_file"]
    if audit["files_sha256"][f"oligogym/resources/pkg_dataset/{source_name}"] != (
        loaded.dataset.content_sha256
    ):
        raise ValueError("Dataset differs from pinned source")
    identity = environment_identity()
    seeds = seeds[:1] if smoke else seeds
    splits = [
        official_split(loaded.dataset, seed, checkout.resolve()) for seed in seeds
    ]
    schedules = {s.seed: schedule(len(s.train_ids), smoke) for s in splits}
    output.mkdir(parents=True)
    for path in (ROOT / "src/oligo_engine").rglob("*.py"):
        destination = output / "source" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    shutil.copyfile(config_path, output / "config.toml")
    state = {
        "status": "running",
        "mode": "smoke" if smoke else "development",
        "config": config,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "environment": identity,
        "source_audit": audit,
        "splits": [asdict(s) for s in splits],
        "seeds": seeds,
        "schedules": schedules,
        "kernel_selection": {},
        "runs": [],
        "caveats": CAVEATS,
    }
    save_json(output / "suite.json", state)
    references = {}
    by_id = {row.sample_id: row for row in loaded.dataset.observations}
    try:
        for split in splits:
            initial = tuple(Random(split.seed).sample(list(split.train_ids), 50))
            selection_oracle, _, _ = loaded.capabilities(split)
            selection_oracle.reveal(initial)
            ordered = tuple(sorted(initial))
            selection = select_kernel(
                tuple(by_id[i] for i in ordered),
                selection_oracle.acquired_labels(ordered),
                split.seed,
            )
            state["kernel_selection"][split.seed] = selection
            save_json(output / "suite.json", state)
            records = tuple(by_id[i] for i in split.train_ids)
            features = OneHotBaseFeaturizer().transform(records)
            diversity = DiversityAcquisition(
                records,
                features,
                {
                    "representation": "fixed_onehot_base_42x5",
                    "normalization": "per_row_l2",
                    "fit_scope": "stateless",
                    "tie_break": "sample_id_lexicographic",
                },
            )
            policies = {
                "random": RandomAcquisition(),
                "diversity": diversity,
                "uncertainty": UncertaintyAcquisition(),
                "hybrid": HybridAcquisition(diversity),
            }
            for name, policy in policies.items():
                policy = DiagnosticPolicy(policy)
                oracle, evaluator, reference = loaded.capabilities(split)

                def factory():
                    return OneHotGaussianProcess(split.seed, **selection["selected"])

                def progress(phase, labels):
                    state["current"] = {
                        "seed": split.seed,
                        "policy": name,
                        "phase": phase,
                        "labels": labels,
                    }
                    save_json(output / "suite.json", state)
                    print(
                        f"seed={split.seed} policy={name} {phase} labels={labels}",
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
                        50,
                        50,
                        schedules[split.seed][-1],
                        None,
                        "diagnostic",
                    ),
                    reference,
                    references,
                    progress,
                    checkpoint_labels=schedules[split.seed],
                )
                result = replace(
                    result,
                    caveats=CAVEATS,
                    execution={
                        **result.execution,
                        "kernel_selection": selection,
                        "selection_labels_in_campaign_budget": 50,
                        "acquisition_diagnostics": policy.rounds,
                    },
                )
                if result.initial_ids != initial:
                    raise ValueError("Kernel selection and campaign initial IDs differ")
                path = output / f"{name}-{split.seed}.json"
                save_json(path, asdict(result))
                state["runs"].append(
                    {
                        "seed": split.seed,
                        "policy": name,
                        "file": path.name,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                )
                save_json(output / "suite.json", state)
        if environment_identity()["source_sha256"] != identity["source_sha256"]:
            raise ValueError("Executed source changed during study")
        state["status"] = "complete"
        state.pop("current", None)
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        save_json(output / "suite.json", state)
        raise
    save_json(output / "suite.json", state)
    from oligo_engine.reporting.active_learning import report_study

    report_study(output)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/active-learning-development.toml")
    )
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run_study(args.config, args.checkout, args.output, args.smoke)


if __name__ == "__main__":
    main()
