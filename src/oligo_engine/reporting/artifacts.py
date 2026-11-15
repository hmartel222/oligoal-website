"""Portable outputs consumed by the future product surface."""

import csv
import json
from dataclasses import asdict
from pathlib import Path
from statistics import fmean, stdev

from oligo_engine.contracts import SCHEMA_VERSION, RunArtifact


def write_artifacts(
    runs: tuple[RunArtifact, ...], output: Path, allow_completed_runs: bool = False
) -> None:
    if not runs or len({r.config.seed for r in runs}) != len(runs):
        raise ValueError("Expected runs with distinct seeds")
    if any(
        r.provenance != runs[0].provenance
        or r.dataset != runs[0].dataset
        or {k: v for k, v in r.model.items() if k != "seed"}
        != {k: v for k, v in runs[0].model.items() if k != "seed"}
        or r.policy != runs[0].policy
        or r.config.pearson_threshold != runs[0].config.pearson_threshold
        or r.config.purpose != runs[0].config.purpose
        for r in runs
    ):
        raise ValueError(
            "Summary requires matched dataset/model/policy/purpose/threshold"
        )
    if output.exists() and any(output.iterdir()):
        allowed = {f"run-seed-{r.config.seed}.json": asdict(r) for r in runs}
        if not allow_completed_runs or any(
            p.name not in allowed
            or json.loads(p.read_text()) != json.loads(json.dumps(allowed[p.name]))
            for p in output.iterdir()
        ):
            raise ValueError(
                "Output directory must be empty; previous runs are preserved"
            )
    output.mkdir(parents=True, exist_ok=True)
    for run in runs:
        (output / f"run-seed-{run.config.seed}.json").write_text(
            json.dumps(asdict(run), indent=2, allow_nan=False) + "\n"
        )
    points: dict[int, list] = {}
    with (output / "metrics.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["provenance", "seed", "labels_used", "pearson_r", "rmse", "mae"]
        )
        for run in runs:
            for point in run.checkpoints:
                m = point.metrics
                writer.writerow(
                    [
                        run.provenance,
                        run.config.seed,
                        point.labels_used,
                        m.pearson_r,
                        m.rmse,
                        m.mae,
                    ]
                )
                points.setdefault(point.labels_used, []).append(m)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "provenance": runs[0].provenance,
        "dataset": runs[0].dataset,
        "policy": runs[0].policy,
        "caveats": list(runs[0].caveats),
        "seeds": [r.config.seed for r in runs],
        "band_method": "mean +/- sample standard deviation (not confidence interval)",
        "checkpoints": [],
        "full_pool_references": [
            {
                "seed": r.config.seed,
                "labels_used": r.full_pool_labels,
                "metrics": asdict(r.full_pool_metrics),
            }
            for r in runs
        ],
        "aggregation": "exact label counts only; no interpolation or pooled datasets",
    }
    for count, metrics in sorted(points.items()):
        point = {"labels_used": count, "run_count": len(metrics)}
        for name in ("pearson_r", "rmse", "mae"):
            values = [getattr(m, name) for m in metrics if getattr(m, name) is not None]
            point[name] = {
                "n": len(values),
                "mean": fmean(values) if values else None,
                "sample_sd": stdev(values) if len(values) > 1 else None,
            }
        summary["checkpoints"].append(point)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
