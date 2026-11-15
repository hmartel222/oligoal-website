"""Freeze validation groups before drawing nested measured-label allocations."""

import hashlib
import json
import math
import random
from collections.abc import Sequence


def nested_budget_plan(
    records: Sequence[dict],
    budgets: Sequence[int],
    seed: int,
    validation_fraction: float = 0.2,
    excluded_ids: Sequence[str] = (),
) -> dict:
    """Plan IDs only; reject extra columns rather than ingest endpoint labels.

    Input records must already be an eligible training pool. Group identifiers must
    encode the caller's leakage boundary (e.g. connected patent/compound groups).
    Groups are assigned once; row samples on each side are nested across budgets.
    This allocates random prediction/transfer controls, not an acquisition policy.
    """
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    if not budgets or any(type(b) is not int or b < 2 for b in budgets):
        raise ValueError("budgets must contain integers of at least two")
    if list(budgets) != sorted(set(budgets)):
        raise ValueError("budgets must be strictly increasing")
    if not records:
        raise ValueError("eligible records cannot be empty")
    for record in records:
        if set(record) != {"id", "group"}:
            raise ValueError("records must contain only id and group; no labels")
        if any(not isinstance(v, str) or not v.strip() for v in record.values()):
            raise ValueError("id and group must be nonempty strings")
    canonical = sorted(records, key=lambda row: row["id"])
    ids = [row["id"] for row in canonical]
    if len(set(ids)) != len(ids):
        raise ValueError("observation IDs must be unique")
    if set(ids) & set(excluded_ids):
        raise ValueError("eligible pool includes excluded evaluation IDs")
    groups = sorted({row["group"] for row in canonical})
    if len(groups) < 2:
        raise ValueError("at least two groups are required")
    rng = random.Random(seed)
    rng.shuffle(groups)
    counts = {g: 0 for g in groups}
    for row in canonical:
        counts[row["group"]] += 1
    # One fixed group prefix closest to the requested row fraction. Never retry
    # seeds/groups to satisfy a budget or a measured outcome.
    prefix_counts, cumulative = [], 0
    for group in groups[:-1]:
        cumulative += counts[group]
        prefix_counts.append(cumulative)
    cut = min(
        range(1, len(groups)),
        key=lambda k: abs(prefix_counts[k - 1] / len(ids) - validation_fraction),
    )
    validation_groups = set(groups[:cut])
    fitting = [r["id"] for r in canonical if r["group"] not in validation_groups]
    validation = [r["id"] for r in canonical if r["group"] in validation_groups]
    rng.shuffle(fitting)
    rng.shuffle(validation)
    allocations = []
    for budget in budgets:
        n_validation = max(1, math.floor(budget * validation_fraction))
        n_fit = budget - n_validation
        if n_fit > len(fitting) or n_validation > len(validation):
            raise ValueError(
                f"budget {budget} exceeds the fixed partition capacity; "
                "reduce budgets or declare a different partition before outcomes"
            )
        allocations.append(
            {
                "labels_total": budget,
                "fit_ids": fitting[:n_fit],
                "validation_ids": validation[:n_validation],
            }
        )
    identity = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema_version": 1,
        "status": "allocation_only_no_training_or_results",
        "seed": seed,
        "pool_size": len(ids),
        "pool_metadata_sha256": identity,
        "validation_fraction_requested": validation_fraction,
        "fit_groups": sorted(set(groups) - validation_groups),
        "validation_groups": sorted(validation_groups),
        "fit_capacity": len(fitting),
        "validation_capacity": len(validation),
        "validation_labels_count_toward_budget": True,
        "allocations": allocations,
    }
