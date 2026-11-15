"""Finite shared-domain baselines with per-method caches and scoring receipts."""

from random import Random
from time import perf_counter

from oligo_engine.design.aso_space import ASODesignSpace
from oligo_engine.reward.scoring import BudgetExhausted, ScoringSession

METHODS = ("unchanged", "random", "greedy", "genetic", "exhaustive")


def summarize(session: ScoringSession, method: str, status: str) -> dict:
    receipt = session.receipt()
    eligible = [r for r in receipt["archive"] if r["score"]["reward"] is not None]
    ranked = sorted(
        eligible, key=lambda r: (-r["score"]["reward"], r["score"]["candidate_id"])
    )
    count = len(receipt["archive"])
    sequences = sorted({r["candidate"]["sequence"] for r in eligible})
    distances = [
        sum(a != b for a, b in zip(left, right, strict=True)) / len(left)
        for i, left in enumerate(sequences)
        for right in sequences[i + 1 :]
    ]
    best, curve = None, []
    for row in receipt["trace"]:
        if row["reward"] is not None:
            best = row["reward"] if best is None else max(best, row["reward"])
        curve.append(
            {
                "requests": row["request"],
                "unique_scores": row["unique"],
                "best_reward": best,
            }
        )
    return {
        "method": method,
        "status": status,
        "best_reward": ranked[0]["score"]["reward"] if ranked else None,
        "feasible_fraction": len(eligible) / count if count else None,
        "eligible_candidates": len(eligible),
        "unique_eligible_sequences": len(sequences),
        "mean_pairwise_sequence_distance": sum(distances) / len(distances)
        if distances
        else None,
        "best_reward_curve": curve,
        "shortlist": ranked[:5],
        **receipt,
    }


def search(
    space: ASODesignSpace,
    session: ScoringSession,
    method: str,
    seed: int = 0,
) -> dict:
    started = perf_counter()
    if method not in METHODS or session.target != space.target:
        raise ValueError("Unknown search method or mismatched target")
    rng = Random(seed)
    sites = set(space.reachable())
    visited: dict[int, float] = {}
    current = space.start
    status = "complete"

    def score(position: int) -> None:
        result = session.score(space.candidate(position))
        visited[position] = result.terminal_reward()

    try:
        score(current)
        while sites - visited.keys() and method != "unchanged":
            unseen = sites - visited.keys()
            if method == "random":
                proposed = rng.choice(sorted(unseen))
            elif method == "exhaustive":
                proposed = min(unseen)
            elif method == "greedy":
                neighbors = sorted({current - 1, current + 1} & unseen)
                if not neighbors:
                    break
                for position in neighbors:
                    score(position)
                best = max(neighbors + [current], key=lambda p: (visited[p], -p))
                if best == current:
                    break
                current = best
                continue
            else:
                # Steady-state integer genetic baseline: rank-selected parents,
                # midpoint crossover, local mutation, unseen fallback on exhaustion.
                elite = sorted(visited, key=lambda p: (-visited[p], p))[:4]
                parents = rng.choices(elite, k=2)
                proposed = (sum(parents) // 2) + rng.choice((-2, -1, 1, 2))
                if proposed not in unseen:
                    proposed = rng.choice(sorted(unseen))
            score(proposed)
    except BudgetExhausted:
        status = "budget_exhausted"
    if sites <= visited.keys():
        status = "space_exhausted"
    return {
        **summarize(session, method, status),
        "wall_seconds": perf_counter() - started,
    }
