"""Label-free diversity and uncertainty+diversity batch acquisition.

Both policies receive only label-free records. Diversity uses a stateless featurizer
passed at construction (never fitted), so its geometry cannot depend on any label, on
which rows were revealed, or on held-out data.

* ``DiversityAcquisition``: greedy farthest-point (k-center) selection. Each pick is
  the eligible record whose Euclidean distance to the nearest labeled-or-already-
  chosen record is largest.
* ``UncertaintyDiversityAcquisition``: two stages. Keep the ``pool_multiplier * budget``
  eligible records with the highest named predictor uncertainty, then pick the batch
  from that pool by the same farthest-point rule measured against labeled and chosen
  records. Uncertainty decides the candidate pool; diversity decides redundancy.

Ties break on stable sample ID, so replay is deterministic.
"""

from random import Random
from typing import Protocol

from oligo_engine.contracts import Observation, Predictor
from oligo_engine.evaluation.runner import validate_predictions

TIE_TOLERANCE = 1e-9


class Featurizer(Protocol):
    @property
    def config(self) -> dict: ...

    def transform(self, observations: tuple[Observation, ...]): ...


def _min_distance(candidates, references):
    """Distance from each candidate row to its nearest reference row."""
    import numpy as np

    sq_c = (candidates**2).sum(axis=1)[:, None]
    sq_r = (references**2).sum(axis=1)[None, :]
    squared = sq_c + sq_r - 2.0 * candidates @ references.T
    return np.sqrt(np.clip(squared, 0.0, None).min(axis=1))


def _farthest_point(candidates, references, budget: int, ids: tuple[str, ...]):
    """Greedy k-center indices into ``candidates`` with stable-ID tie breaks."""
    import numpy as np

    if budget > len(ids):
        raise ValueError("Budget exceeds the candidate pool")
    rank = np.empty(len(ids), dtype=int)
    rank[np.argsort(ids, kind="stable")] = np.arange(len(ids))
    distance = _min_distance(candidates, references)
    available = np.ones(len(ids), dtype=bool)
    chosen: list[int] = []
    for _ in range(budget):
        masked = np.where(available, distance, -np.inf)
        ties = np.flatnonzero(masked >= masked.max() - TIE_TOLERANCE)
        pick = int(ties[np.argmin(rank[ties])])
        chosen.append(pick)
        available[pick] = False
        step = np.sqrt(
            np.clip(((candidates - candidates[pick]) ** 2).sum(axis=1), 0.0, None)
        )
        distance = np.minimum(distance, step)
    return chosen


class DiversityAcquisition:
    def __init__(self, featurizer: Featurizer):
        self.featurizer = featurizer

    @property
    def config(self) -> dict:
        return {
            "name": "diversity_farthest_point",
            "distance": "euclidean",
            "tie_break": "stable_sample_id",
            "uses_labels": False,
            "uses_predictor": False,
            "featurizer": self.featurizer.config,
        }

    def choose(
        self,
        eligible: tuple[Observation, ...],
        labeled: tuple[Observation, ...],
        budget: int,
        rng: Random,
        predictor: Predictor,
    ) -> tuple[str, ...]:
        if not labeled:
            raise ValueError("Diversity acquisition needs an initial labeled set")
        ids = tuple(row.sample_id for row in eligible)
        picks = _farthest_point(
            self.featurizer.transform(eligible),
            self.featurizer.transform(labeled),
            budget,
            ids,
        )
        return tuple(ids[i] for i in picks)


class UncertaintyDiversityAcquisition:
    def __init__(self, featurizer: Featurizer, pool_multiplier: int = 4):
        if type(pool_multiplier) is not int or pool_multiplier < 1:
            raise ValueError("pool_multiplier must be a positive integer")
        self.featurizer = featurizer
        self.pool_multiplier = pool_multiplier

    @property
    def config(self) -> dict:
        return {
            "name": "uncertainty_then_diversity",
            "pool_multiplier": self.pool_multiplier,
            "stage_1": "top predictor uncertainty, tie-break stable_sample_id",
            "stage_2": "greedy farthest point within the stage-1 pool",
            "distance": "euclidean",
            "uses_labels": False,
            "featurizer": self.featurizer.config,
        }

    def choose(
        self,
        eligible: tuple[Observation, ...],
        labeled: tuple[Observation, ...],
        budget: int,
        rng: Random,
        predictor: Predictor,
    ) -> tuple[str, ...]:
        if not labeled:
            raise ValueError("Diversity acquisition needs an initial labeled set")
        predictions = predictor.predict(eligible)
        validate_predictions(predictions, len(eligible))
        if any(p.uncertainty is None for p in predictions):
            raise ValueError("Hybrid acquisition requires uncertainty estimates")
        ordered = sorted(
            range(len(eligible)),
            key=lambda i: (-predictions[i].uncertainty, eligible[i].sample_id),
        )
        pool = ordered[: min(len(eligible), self.pool_multiplier * budget)]
        pool_rows = tuple(eligible[i] for i in pool)
        ids = tuple(row.sample_id for row in pool_rows)
        picks = _farthest_point(
            self.featurizer.transform(pool_rows),
            self.featurizer.transform(labeled),
            budget,
            ids,
        )
        return tuple(ids[i] for i in picks)
