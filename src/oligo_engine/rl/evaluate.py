"""Separate policy partitions and deferred exact-assay evaluation boundary."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class PolicyPartitions:
    train: tuple[str, ...]
    validation: tuple[str, ...]
    final: tuple[str, ...]

    def __post_init__(self) -> None:
        all_ids = self.train + self.validation + self.final
        if not all((self.train, self.validation, self.final)):
            raise ValueError("Every policy partition must be nonempty")
        if len(set(all_ids)) != len(all_ids) or any(not x for x in all_ids):
            raise ValueError("Policy target partitions must be unique and disjoint")


def paired_summary(runs: list[dict], reference: dict) -> dict:
    """Surrogate-only comparison; does not accept or read assay outcomes."""
    if not runs:
        raise ValueError("No optimizer runs")
    keys = ("scorer_identity", "target_identity", "assay", "max_unique", "max_requests")
    for run in runs:
        if any(run[k] != runs[0][k] for k in keys):
            raise ValueError("Unpaired scorer/target/assay or budget")
    if any(reference[k] != runs[0][k] for k in keys[:3]):
        raise ValueError("Enumeration reference differs from experiment")
    optimum = reference["best_reward"]
    return {
        "status": "synthetic_fixture"
        if all(
            r["score"]["provenance"].startswith("synthetic")
            for run in runs
            for r in run["archive"]
        )
        else "surrogate_diagnostic",
        "reference_complete": reference["status"] == "space_exhausted",
        "reference_optimum": optimum,
        "results": [
            {
                "method": r["method"],
                "best_reward": r["best_reward"],
                "regret": (optimum - r["best_reward"])
                if reference["status"] == "space_exhausted"
                and optimum is not None
                and r["best_reward"] is not None
                else None,
                "unique_scores": r["unique_scores"],
                "requests": r["requests"],
                "head_calls": r["head_calls"],
                "feasible_fraction": r["feasible_fraction"],
            }
            for r in runs
        ],
    }


def deferred_assay_report(
    locked_shortlist: tuple[str, ...],
    measurements: dict[str, float],
) -> dict:
    """Call after optimization is locked; IDs must include exact assay context.

    Never match a novel candidate to a similar measured sequence or impute an
    absent outcome. This function is not imported by the environment/scorer.
    """
    if len(set(locked_shortlist)) != len(locked_shortlist):
        raise ValueError("Duplicate shortlist identities")
    observed = {
        key: measurements[key] for key in locked_shortlist if key in measurements
    }
    if any(not isfinite(v) for v in observed.values()):
        raise ValueError("Nonfinite measured outcome")
    return {
        "exact_assays_observed": len(observed),
        "missing": [key for key in locked_shortlist if key not in measurements],
        "measured_mean": sum(observed.values()) / len(observed) if observed else None,
    }
