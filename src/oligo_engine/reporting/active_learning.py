"""Validate paired ledgers and report exact-checkpoint label efficiency."""

import hashlib
import json
from pathlib import Path
from statistics import fmean, stdev


def stats(values: list) -> dict:
    valid = [v for v in values if v is not None]
    return {
        "n": len(valid),
        "mean": fmean(valid) if valid else None,
        "sample_sd": stdev(valid) if len(valid) > 1 else None,
    }


def area(points: list, end: int) -> float | None:
    points = [p for p in points if p["labels_used"] <= end]
    if len(points) < 2 or points[-1]["labels_used"] != end:
        return None
    if any(p["metrics"]["pearson_r"] is None for p in points):
        return None
    return sum(
        (b["labels_used"] - a["labels_used"])
        * (a["metrics"]["pearson_r"] + b["metrics"]["pearson_r"])
        / 2
        for a, b in zip(points, points[1:])
    ) / (end - points[0]["labels_used"])


def threshold(run: dict) -> dict:
    reference = run["full_pool_metrics"]["pearson_r"]
    if reference is None or reference <= 0.03:
        return {"status": "undefined_or_uninformative_reference", "labels": None}
    target = reference - 0.03
    count = next(
        (
            p["labels_used"]
            for p in run["checkpoints"]
            if p["metrics"]["pearson_r"] is not None
            and p["metrics"]["pearson_r"] >= target
        ),
        None,
    )
    return {
        "status": "reached" if count is not None else "not_reached",
        "threshold_r": target,
        "labels": count,
    }


def audit_run(run: dict, expected: list[int]) -> None:
    if [p["labels_used"] for p in run["checkpoints"]] != expected:
        raise ValueError("Checkpoint schedule differs")
    selected = []
    train = set(run["split"]["train_ids"])
    if train & set(run["split"]["test_ids"]):
        raise ValueError("Partitions overlap")
    for point in run["checkpoints"]:
        selected.extend(point["acquired_ids"])
        if len(selected) != point["labels_used"] or len(set(selected)) != len(selected):
            raise ValueError("Duplicate or miscounted labels")
        if not set(selected) <= train:
            raise ValueError("Non-training acquisition")
    if run["checkpoints"][0]["acquired_ids"] != run["initial_ids"]:
        raise ValueError("Initial IDs differ")
    selection = run["execution"]["kernel_selection"]
    if set(selection["initial_ids"]) != set(run["initial_ids"]):
        raise ValueError("Kernel selection used different labels")
    for fold in selection["folds"]:
        fit, valid = set(fold["train_ids"]), set(fold["validation_ids"])
        if fit & valid or fit | valid != set(run["initial_ids"]):
            raise ValueError("Kernel selection folds violate initial label boundary")
    if expected[-1] == len(train) and (
        run["checkpoints"][-1]["metrics"] != run["full_pool_metrics"]
    ):
        raise ValueError("Terminal full-pool model differs from reference")


def report_study(root: Path) -> dict:
    suite = json.loads((root / "suite.json").read_text())
    if suite["status"] != "complete":
        raise ValueError("Cannot summarize an incomplete study")
    policies = suite["config"]["policies"]
    if len(suite["runs"]) != len(suite["seeds"]) * len(policies):
        raise ValueError("Missing campaigns")
    runs = {}
    for entry in suite["runs"]:
        path = root / entry["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Run hash differs")
        run = json.loads(path.read_text())
        key = entry["seed"], entry["policy"]
        if (
            key in runs
            or run["config"]["seed"] != key[0]
            or run["policy"]["name"] != key[1]
        ):
            raise ValueError("Duplicate or mismatched ledger entry")
        if run["environment"]["source_sha256"] != suite["environment"]["source_sha256"]:
            raise ValueError("Source changed between runs")
        audit_run(run, suite["schedules"][str(entry["seed"])])
        runs[key] = run
    end = 100 if suite["mode"] == "smoke" else 1000
    pairs = []
    for seed in suite["seeds"]:
        random = runs[seed, "random"]
        random_count = threshold(random)["labels"]
        for policy in policies:
            run = runs[seed, policy]
            for key in (
                "dataset",
                "split",
                "config",
                "model",
                "initial_ids",
                "full_pool_metrics",
                "full_pool_labels",
            ):
                if run[key] != random[key]:
                    raise ValueError(f"Unmatched pair: {key}")
            a, b = area(random["checkpoints"], end), area(run["checkpoints"], end)
            efficiency = threshold(run)
            count = efficiency["labels"]
            pairs.append(
                {
                    "seed": seed,
                    "policy": policy,
                    "aulc": b,
                    "aulc_minus_random": b - a
                    if a is not None and b is not None
                    else None,
                    "efficiency": efficiency,
                    "savings_vs_random": 1 - count / random_count
                    if count is not None and random_count is not None
                    else None,
                }
            )
    curves = {}
    for policy in policies:
        curves[policy] = []
        for count in range(50, end + 1, 50):
            points = [
                next(
                    p
                    for p in runs[seed, policy]["checkpoints"]
                    if p["labels_used"] == count
                )
                for seed in suite["seeds"]
            ]
            curves[policy].append(
                {
                    "labels": count,
                    **{
                        metric: stats([p["metrics"][metric] for p in points])
                        for metric in ("pearson_r", "rmse", "mae")
                    },
                }
            )
    report = {
        "mode": suite["mode"],
        "primary": "hybrid minus random Pearson AULC",
        "interval": [50, end],
        "paired_results": pairs,
        "curves": curves,
        "policy_comparisons": {
            policy: {
                "aulc_difference": stats(
                    [p["aulc_minus_random"] for p in pairs if p["policy"] == policy]
                ),
                "label_savings": stats(
                    [p["savings_vs_random"] for p in pairs if p["policy"] == policy]
                ),
                "threshold_status_counts": {
                    status: sum(
                        p["efficiency"]["status"] == status
                        for p in pairs
                        if p["policy"] == policy
                    )
                    for status in (
                        "reached",
                        "not_reached",
                        "undefined_or_uninformative_reference",
                    )
                },
            }
            for policy in policies
        },
        "full_reference": stats(
            [
                runs[s, "random"]["full_pool_metrics"]["pearson_r"]
                for s in suite["seeds"]
            ]
        ),
        "practical_savings_target": 0.20,
        "dispersion": "sample SD; overlapping holdouts, no independence claim",
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
        for policy, points in curves.items():
            x = [p["labels"] for p in points]
            y = np.array([p[metric]["mean"] for p in points], dtype=float)
            ax.plot(x, y, label=policy, marker=".")
            sd = [p[metric]["sample_sd"] for p in points]
            if all(v is not None for v in sd):
                ax.fill_between(x, y - np.array(sd), y + np.array(sd), alpha=0.12)
        reference = stats(
            [runs[s, "random"]["full_pool_metrics"][metric] for s in suite["seeds"]]
        )["mean"]
        if reference is not None:
            ax.axhline(reference, linestyle="--", color="black", label="full-pool mean")
        ax.set(xlabel="Measured labels", ylabel=metric)
        ax.legend(fontsize=7)
    fig.suptitle(
        f"Huesken GP {suite['mode']} | n={len(suite['seeds'])} | bands: sample SD"
    )
    fig.savefig(root / "learning-curves.png", dpi=160)
    plt.close(fig)
    return report
