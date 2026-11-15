"""Paired reporting for fixed-budget, measured RNA-FM representation studies."""

import hashlib
import json
from pathlib import Path
from statistics import fmean, stdev


def summarize(values: list[float | None]) -> dict:
    valid = [v for v in values if v is not None]
    return {
        "n": len(valid),
        "mean": fmean(valid) if valid else None,
        "sample_sd": stdev(valid) if len(valid) > 1 else None,
    }


def audit_fit(fit: dict, expected: set[str]) -> None:
    if set(fit["training_ids"]) != expected or len(fit["training_ids"]) != len(
        expected
    ):
        raise ValueError("Fit records differ from counted acquired IDs")
    validation_ids = []
    for fold in fit["folds"]:
        train, valid = set(fold["train_ids"]), set(fold["validation_ids"])
        if train & valid or train | valid != expected:
            raise ValueError("Inner fold violates acquired-label boundary")
        validation_ids.extend(fold["validation_ids"])
    if len(validation_ids) != len(expected) or set(validation_ids) != expected:
        raise ValueError("Validation folds must partition acquired records")


def audit_seed(run: dict, budgets: list[int], names: list[str]) -> None:
    if set(run["representations"]) != set(names):
        raise ValueError("Representations differ from declared comparison")
    train = set(run["split"]["train_ids"])
    if train & set(run["split"]["test_ids"]):
        raise ValueError("Train and test IDs overlap")
    baseline_batches = None
    for name in names:
        result = run["representations"][name]
        points = result["checkpoints"]
        if [p["labels_used"] for p in points] != budgets:
            raise ValueError("Learning budgets differ")
        batches = [p["acquired_ids"] for p in points]
        if baseline_batches is None:
            baseline_batches = batches
        elif batches != baseline_batches:
            raise ValueError("Representations did not receive identical labeled pools")
        if batches[0] != run["initial_ids"]:
            raise ValueError("Initial labels differ from recorded sample")
        acquired = []
        for point in points:
            acquired.extend(point["acquired_ids"])
            if len(acquired) != point["labels_used"] or len(set(acquired)) != len(
                acquired
            ):
                raise ValueError("Duplicate or miscounted acquisition")
            if not set(acquired) <= train:
                raise ValueError("Held-out ID entered acquisition")
            audit_fit(point["fit"], set(acquired))
        reference = result["full_reference"]
        if reference["labels_used"] != len(train):
            raise ValueError("Reference label count differs from eligible pool")
        audit_fit(reference["fit"], train)


def aulc(points: list[dict]) -> float | None:
    if len(points) < 2 or any(p["metrics"]["pearson_r"] is None for p in points):
        return None
    return sum(
        (b["labels_used"] - a["labels_used"])
        * (a["metrics"]["pearson_r"] + b["metrics"]["pearson_r"])
        / 2
        for a, b in zip(points, points[1:])
    ) / (points[-1]["labels_used"] - points[0]["labels_used"])


def threshold(result: dict, target: float) -> dict:
    for point in result["checkpoints"]:
        value = point["metrics"]["pearson_r"]
        if value is not None and value >= target:
            return {
                "status": "reached",
                "labels": point["labels_used"],
                "source": "budget_checkpoint",
            }
    reference = result["full_reference"]
    value = reference["metrics"]["pearson_r"]
    if value is not None and value >= target:
        return {
            "status": "reached_at_full_reference",
            "labels": reference["labels_used"],
            "source": "separate_full_pool_reference",
        }
    return {"status": "not_reached", "labels": None, "source": None}


def report_study(root: Path) -> dict:
    suite = json.loads((root / "suite.json").read_text())
    if suite["status"] != "complete":
        raise ValueError("Study must complete before reporting")
    if len(suite["runs"]) != len(suite["seeds"]):
        raise ValueError("Incomplete seed collection")
    names = suite["config"]["representations"]
    target = suite["config"]["pearson_target"]
    runs = []
    seen = set()
    for entry in suite["runs"]:
        path = root / entry["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Seed artifact changed")
        run = json.loads(path.read_text())
        if run["seed"] != entry["seed"] or run["seed"] in seen:
            raise ValueError("Duplicate or mismatched seed")
        audit_seed(run, suite["budgets"], names)
        seen.add(run["seed"])
        runs.append(run)
    if seen != set(suite["seeds"]):
        raise ValueError("Seed set differs")
    paired = []
    curves = {}
    references = {}
    for name in names:
        curves[name] = [
            {
                "labels": count,
                **{
                    metric: summarize(
                        [
                            run["representations"][name]["checkpoints"][index][
                                "metrics"
                            ][metric]
                            for run in runs
                        ]
                    )
                    for metric in ("pearson_r", "rmse", "mae")
                },
            }
            for index, count in enumerate(suite["budgets"])
        ]
        references[name] = {
            metric: summarize(
                [
                    run["representations"][name]["full_reference"]["metrics"][metric]
                    for run in runs
                ]
            )
            for metric in ("pearson_r", "rmse", "mae")
        }
    for run in runs:
        baseline = run["representations"]["onehot"]
        baseline_area = aulc(baseline["checkpoints"])
        baseline_count = threshold(baseline, target)["labels"]
        random_area = (
            aulc(run["representations"]["random_rnafm"]["checkpoints"])
            if "random_rnafm" in run["representations"]
            else None
        )
        for name in names:
            result = run["representations"][name]
            area = aulc(result["checkpoints"])
            efficiency = threshold(result, target)
            count = efficiency["labels"]
            paired.append(
                {
                    "seed": run["seed"],
                    "representation": name,
                    "aulc": area,
                    "aulc_minus_onehot": area - baseline_area
                    if area is not None and baseline_area is not None
                    else None,
                    "aulc_minus_random_encoder": area - random_area
                    if area is not None and random_area is not None
                    else None,
                    "shared_target": target,
                    "efficiency": efficiency,
                    "label_savings_vs_onehot": 1 - count / baseline_count
                    if count is not None and baseline_count is not None
                    else None,
                }
            )
    report = {
        "status": "complete_measured_diagnostic",
        "stage": suite["stage"],
        "comparison": "matched Ridge heads, nested random acquisition",
        "primary": "frozen RNA-FM minus one-hot normalized Pearson AULC",
        "aulc_interval": [suite["budgets"][0], suite["budgets"][-1]],
        "curves": curves,
        "full_references": references,
        "paired": paired,
        "summaries": {
            name: {
                "aulc_difference": summarize(
                    [
                        p["aulc_minus_onehot"]
                        for p in paired
                        if p["representation"] == name
                    ]
                ),
                "label_savings": summarize(
                    [
                        p["label_savings_vs_onehot"]
                        for p in paired
                        if p["representation"] == name
                    ]
                ),
                "threshold_status_counts": {
                    status: sum(
                        p["efficiency"]["status"] == status
                        for p in paired
                        if p["representation"] == name
                    )
                    for status in (
                        "reached",
                        "reached_at_full_reference",
                        "not_reached",
                    )
                },
            }
            for name in names
        },
        "pretraining_control_present": suite["pretraining_control_present"],
        "dispersion": "sample SD, overlapping holdouts; not confidence intervals",
        "caveats": suite["caveats"],
    }
    (root / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, metric in zip(axes, ("pearson_r", "rmse", "mae"), strict=True):
        for name, points in curves.items():
            x = [p["labels"] for p in points]
            means = np.array([p[metric]["mean"] for p in points], dtype=float)
            (line,) = ax.plot(x, means, marker="o", label=name)
            spreads = [p[metric]["sample_sd"] for p in points]
            if all(value is not None for value in spreads):
                sd = np.asarray(spreads)
                ax.fill_between(
                    x, means - sd, means + sd, color=line.get_color(), alpha=0.12
                )
            reference = references[name][metric]["mean"]
            if reference is not None:
                ax.axhline(reference, color=line.get_color(), linestyle="--", alpha=0.7)
        ax.set(xlabel="Measured training labels", ylabel=metric)
        ax.legend(fontsize=8)
    fig.suptitle(
        f"Huesken representations: {suite['stage']}, n={len(runs)}; "
        "dashed = full-pool means"
    )
    fig.savefig(root / "learning-curves.png", dpi=160)
    fig.savefig(root / "learning-curves.svg")
    plt.close(fig)
    return report


def report_geometry(root: Path) -> dict:
    """Audit fixed-predictor geometry while allowing acquired IDs to differ."""
    suite = json.loads((root / "suite.json").read_text())
    if suite["status"] != "complete":
        raise ValueError("Study must complete before reporting")
    names = ["random"] + [
        name + "_diversity" for name in suite["config"]["representations"]
    ]
    if len(suite["runs"]) != len(suite["seeds"]) * len(names):
        raise ValueError("Missing acquisition campaigns")
    runs = {}
    for entry in suite["runs"]:
        path = root / entry["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Campaign hash differs")
        run = json.loads(path.read_text())
        key = entry["seed"], entry["policy"]
        if (
            key in runs
            or run["config"]["seed"] != key[0]
            or run["execution"]["acquisition_policy"] != key[1]
        ):
            raise ValueError("Duplicate or mismatched campaign")
        if [p["labels_used"] for p in run["checkpoints"]] != suite["budgets"]:
            raise ValueError("Budget schedule differs")
        if run["initial_ids"] != run["checkpoints"][0]["acquired_ids"]:
            raise ValueError("Initial acquisition differs")
        train = set(run["split"]["train_ids"])
        if train & set(run["split"]["test_ids"]):
            raise ValueError("Train/test overlap")
        fits = run["execution"]["fit_records"]
        expected_fits = len(suite["budgets"]) + (
            not run["execution"]["full_reference_reused"]
        )
        if len(fits) != expected_fits:
            raise ValueError("Unexpected fit count")
        acquired = []
        for index, point in enumerate(run["checkpoints"]):
            acquired.extend(point["acquired_ids"])
            if (
                len(acquired) != point["labels_used"]
                or len(set(acquired)) != len(acquired)
                or not set(acquired) <= train
            ):
                raise ValueError("Invalid label reveal ledger")
            audit_fit(fits[index], set(acquired))
        if not run["execution"]["full_reference_reused"]:
            audit_fit(fits[-1], train)
        runs[key] = run
    curves = {}
    paired = []
    for seed in suite["seeds"]:
        random = runs[seed, "random"]
        raw = runs[seed, "onehot_diversity"]
        random_area, raw_area = aulc(random["checkpoints"]), aulc(raw["checkpoints"])
        for name in names:
            run = runs[seed, name]
            for field in (
                "dataset",
                "split",
                "config",
                "model",
                "initial_ids",
                "full_pool_metrics",
                "full_pool_labels",
            ):
                if run[field] != random[field]:
                    raise ValueError(f"Fixed-predictor pairing differs: {field}")
            area = aulc(run["checkpoints"])
            points = []
            selected, raw_selected = set(), set()
            for point, raw_point in zip(
                run["checkpoints"], raw["checkpoints"], strict=True
            ):
                selected.update(point["acquired_ids"])
                raw_selected.update(raw_point["acquired_ids"])
                points.append(
                    {
                        "labels": point["labels_used"],
                        "acquired_pool_jaccard_vs_onehot_diversity": len(
                            selected & raw_selected
                        )
                        / len(selected | raw_selected),
                    }
                )
            paired.append(
                {
                    "seed": seed,
                    "policy": name,
                    "aulc": area,
                    "aulc_minus_random": area - random_area
                    if area is not None and random_area is not None
                    else None,
                    "aulc_minus_onehot_diversity": area - raw_area
                    if area is not None and raw_area is not None
                    else None,
                    "labels_to_shared_threshold": run["labels_to_threshold"],
                    "threshold_status": run["threshold_status"],
                    "selection_overlap": points,
                }
            )
    for name in names:
        curves[name] = [
            {
                "labels": budget,
                **{
                    metric: summarize(
                        [
                            runs[seed, name]["checkpoints"][index]["metrics"][metric]
                            for seed in suite["seeds"]
                        ]
                    )
                    for metric in ("pearson_r", "rmse", "mae")
                },
            }
            for index, budget in enumerate(suite["budgets"])
        ]
    references = {
        metric: summarize(
            [
                runs[seed, "random"]["full_pool_metrics"][metric]
                for seed in suite["seeds"]
            ]
        )
        for metric in ("pearson_r", "rmse", "mae")
    }
    report = {
        "status": "complete_measured_diagnostic",
        "stage": suite["stage"],
        "primary": "frozen-RNAFM minus one-hot diversity: normalized Pearson AULC",
        "fixed_predictor": "onehot plus acquired-only tuned Ridge",
        "interval": [suite["budgets"][0], suite["budgets"][-1]],
        "paired": paired,
        "curves": curves,
        "full_reference": references,
        "summaries": {
            name: {
                "aulc_minus_random": summarize(
                    [p["aulc_minus_random"] for p in paired if p["policy"] == name]
                ),
                "aulc_minus_onehot_diversity": summarize(
                    [
                        p["aulc_minus_onehot_diversity"]
                        for p in paired
                        if p["policy"] == name
                    ]
                ),
                "threshold_reached_count": sum(
                    p["threshold_status"] == "reached"
                    for p in paired
                    if p["policy"] == name
                ),
            }
            for name in names
        },
        "dispersion": "sample SD; overlapping holdouts, not confidence intervals",
        "caveats": suite["caveats"],
    }
    (root / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, metric in zip(axes, ("pearson_r", "rmse", "mae"), strict=True):
        for name, points in curves.items():
            x = [p["labels"] for p in points]
            y = np.asarray([p[metric]["mean"] for p in points], dtype=float)
            (line,) = ax.plot(x, y, marker="o", label=name)
            spread = [p[metric]["sample_sd"] for p in points]
            if all(v is not None for v in spread):
                ax.fill_between(
                    x,
                    y - np.array(spread),
                    y + np.array(spread),
                    alpha=0.12,
                    color=line.get_color(),
                )
        if references[metric]["mean"] is not None:
            ax.axhline(
                references[metric]["mean"],
                color="black",
                linestyle="--",
                label="full-pool reference",
            )
        ax.set(xlabel="Measured labels", ylabel=metric)
        ax.legend(fontsize=7)
    fig.suptitle(
        f"Fixed one-hot Ridge; acquisition geometry; n={len(suite['seeds'])}; "
        "bands: sample SD"
    )
    fig.savefig(root / "learning-curves.png", dpi=160)
    plt.close(fig)
    return report
