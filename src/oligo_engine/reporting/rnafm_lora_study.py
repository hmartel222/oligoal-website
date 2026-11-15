"""Audited four-arm low-label adaptation comparisons; no censored savings claims."""

import hashlib
import json
from pathlib import Path
from statistics import fmean, stdev


def summarize(values: list) -> dict:
    valid = [v for v in values if v is not None]
    return {
        "n": len(valid),
        "mean": fmean(valid) if valid else None,
        "sample_sd": stdev(valid) if len(valid) > 1 else None,
    }


def area(points: list[dict]) -> float | None:
    if len(points) < 2 or any(p["metrics"]["pearson_r"] is None for p in points):
        return None
    return sum(
        (b["labels_used"] - a["labels_used"])
        * (a["metrics"]["pearson_r"] + b["metrics"]["pearson_r"])
        / 2
        for a, b in zip(points, points[1:])
    ) / (points[-1]["labels_used"] - points[0]["labels_used"])


def audit_run(run: dict, config: dict) -> None:
    train, test = set(run["split"]["train_ids"]), set(run["split"]["test_ids"])
    if train & test:
        raise ValueError("Training/test overlap")
    if set(run["arms"]) != set(config["arms"]):
        raise ValueError("Missing arm")
    baseline = run["arms"]["onehot_ridge"]
    for arm, points in run["arms"].items():
        if [p["labels_used"] for p in points] != config["budgets"]:
            raise ValueError("Budget mismatch")
        acquired = []
        for point, other in zip(points, baseline, strict=True):
            acquired.extend(point["acquired_ids"])
            if len(acquired) != point["labels_used"] or len(set(acquired)) != len(
                acquired
            ):
                raise ValueError("Label accounting differs")
            if not set(acquired) <= train or set(point["training_ids"]) != set(
                acquired
            ):
                raise ValueError("Training includes non-acquired labels")
            if (
                point["acquired_ids"] != other["acquired_ids"]
                or point["training_ids"] != other["training_ids"]
            ):
                raise ValueError("Unmatched label sets")
            fit = point["fit"]
            if arm.endswith("ridge"):
                if set(fit["training_ids"]) != set(acquired):
                    raise ValueError("Ridge fit IDs differ")
                folds = fit["folds"]
            else:
                folds = [
                    {
                        "train_ids": fit["fit_ids"],
                        "validation_ids": fit["validation_ids"],
                    }
                ]
                if fit["selected_epochs"] not in config["epoch_candidates"]:
                    raise ValueError("Unknown epoch schedule")
                if set(fit["fit_audit"]["train_ids"]) != set(acquired):
                    raise ValueError("Neural final refit differs")
                if not fit["fit_audit"]["base_unchanged"]:
                    raise ValueError("Frozen base changed")
                if arm == "lora_head" and not fit["fit_audit"]["adapter_changed"]:
                    raise ValueError("Adapter did not change")
                for candidate in fit["epoch_candidates"]:
                    if set(candidate["fit_audit"]["train_ids"]) != set(fit["fit_ids"]):
                        raise ValueError(
                            "Neural candidate trained on validation labels"
                        )
                winner = min(
                    fit["epoch_candidates"], key=lambda c: c["validation_rmse"]
                )["epochs"]
                if fit["selected_epochs"] != winner:
                    raise ValueError("Epoch winner differs from validation scores")
            for fold in folds:
                a, b = set(fold["train_ids"]), set(fold["validation_ids"])
                if a & b or a | b != set(acquired):
                    raise ValueError("Internal validation label boundary violated")
    if run["test_scoring"] != "after_all_arms_and_budgets_frozen":
        raise ValueError("Unexpected evaluation protocol")
    for reference in run["full_pool_references"].values():
        if (
            reference["labels_used"] != len(train)
            or set(reference["fit"]["training_ids"]) != train
        ):
            raise ValueError("Full reference includes wrong labels")


def report_study(root: Path) -> dict:
    suite = json.loads((root / "suite.json").read_text())
    cfg = suite["effective"]
    if suite["status"] != "complete" or len(suite["runs"]) != len(cfg["seeds"]):
        raise ValueError("Study is incomplete")
    runs = []
    for entry in suite["runs"]:
        path = root / entry["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Run hash differs")
        run = json.loads(path.read_text())
        audit_run(run, cfg)
        runs.append(run)
    if sorted(r["seed"] for r in runs) != sorted(cfg["seeds"]):
        raise ValueError("Seed schedule differs")
    curves = {
        arm: [
            {
                "labels": budget,
                **{
                    metric: summarize(
                        [r["arms"][arm][j]["metrics"][metric] for r in runs]
                    )
                    for metric in ("pearson_r", "rmse", "mae")
                },
            }
            for j, budget in enumerate(cfg["budgets"])
        ]
        for arm in cfg["arms"]
    }
    pairs = []
    for run in runs:
        values = {arm: area(points) for arm, points in run["arms"].items()}
        reaches = {
            arm: next(
                (
                    p["labels_used"]
                    for p in points
                    if p["metrics"]["pearson_r"] is not None
                    and p["metrics"]["pearson_r"] >= cfg["pearson_target"]
                ),
                None,
            )
            for arm, points in run["arms"].items()
        }
        pairs.append(
            {
                "seed": run["seed"],
                "aulc": values,
                "lora_minus": {
                    arm: values["lora_head"] - values[arm]
                    if values["lora_head"] is not None and values[arm] is not None
                    else None
                    for arm in cfg["arms"]
                    if arm != "lora_head"
                },
                "labels_to_shared_target": {
                    arm: {
                        "status": "reached" if count is not None else "not_reached",
                        "labels": count,
                    }
                    for arm, count in reaches.items()
                },
            }
        )
    report = {
        "mode": suite["mode"],
        "primary": "LoRA minus tuned one-hot Ridge normalized Pearson AULC",
        "interval": [cfg["budgets"][0], cfg["budgets"][-1]],
        "curves": curves,
        "paired_results": pairs,
        "paired_aulc_differences": {
            arm: summarize([p["lora_minus"][arm] for p in pairs])
            for arm in cfg["arms"]
            if arm != "lora_head"
        },
        "shared_pearson_target": cfg["pearson_target"],
        "full_pool_references": {
            arm: {
                metric: summarize(
                    [r["full_pool_references"][arm]["metrics"][metric] for r in runs]
                )
                for metric in ("pearson_r", "rmse", "mae")
            }
            for arm in ("onehot_ridge", "frozen_ridge")
        },
        "runtime_seconds": suite["runtime_seconds"],
        "caveats": suite["caveats"],
        "dispersion": "Sample SD; overlapping holdouts; not a confidence interval",
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
        for arm, points in curves.items():
            x = [p["labels"] for p in points]
            y = np.array([p[metric]["mean"] for p in points], dtype=float)
            ax.plot(x, y, label=arm, marker="o")
            sd = [p[metric]["sample_sd"] for p in points]
            if all(s is not None for s in sd):
                ax.fill_between(x, y - np.array(sd), y + np.array(sd), alpha=0.12)
        ax.set(xlabel="Measured labels (including validation)", ylabel=metric)
        ax.legend(fontsize=7)
    fig.suptitle(f"Huesken LoRA {suite['mode']}: {len(runs)} paired seeds; sample SD")
    fig.savefig(root / "learning-curves.png", dpi=170)
    plt.close(fig)
    return report
