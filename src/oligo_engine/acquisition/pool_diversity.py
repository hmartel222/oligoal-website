"""Label-blind, greedy max-min sequence coverage on eligible-pool embeddings."""

import copy
from random import Random

from oligo_engine.contracts import Observation, Predictor


class DiversityAcquisition:
    """Farthest-first batch selection; receives neither oracle nor measured labels."""

    def __init__(
        self, observations: tuple[Observation, ...], embeddings, settings: dict
    ):
        import numpy as np

        self._ids = tuple(row.sample_id for row in observations)
        if not self._ids or len(set(self._ids)) != len(self._ids):
            raise ValueError("Diversity requires unique training-pool IDs")
        values = np.asarray(embeddings, dtype=float)
        if (
            values.ndim != 2
            or values.shape[0] != len(observations)
            or values.shape[1] < 1
            or not np.isfinite(values).all()
        ):
            raise ValueError("Diversity embeddings must be aligned and finite")
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        self._vectors = values / np.where(norms == 0, 1, norms)
        self._indices = {identity: i for i, identity in enumerate(self._ids)}
        self._config = {"name": "diversity", **copy.deepcopy(settings)}

    @property
    def config(self) -> dict:
        return copy.deepcopy(self._config)

    def choose(
        self,
        eligible: tuple[Observation, ...],
        labeled: tuple[Observation, ...],
        budget: int,
        rng: Random,
        predictor: Predictor,
    ) -> tuple[str, ...]:
        import numpy as np

        candidates = sorted(row.sample_id for row in eligible)
        centers = [row.sample_id for row in labeled]
        if (
            type(budget) is not int
            or not 1 <= budget <= len(candidates)
            or len(set(candidates)) != len(candidates)
            or len(set(centers)) != len(centers)
            or set(candidates) & set(centers)
            or not set(candidates + centers) <= self._indices.keys()
        ):
            raise ValueError("Invalid eligible/labeled IDs or diversity budget")
        points = self._vectors[[self._indices[identity] for identity in candidates]]
        if centers:
            reference = self._vectors[[self._indices[identity] for identity in centers]]
            distances = (
                (points * points).sum(axis=1)[:, None]
                + (reference * reference).sum(axis=1)[None, :]
                - 2 * points @ reference.T
            )
            minimum = np.maximum(distances, 0).min(axis=1)
        else:
            minimum = np.full(len(candidates), np.inf)
        selected = []
        for _ in range(budget):
            index = int(np.argmax(minimum))
            selected.append(candidates[index])
            update = ((points - points[index]) ** 2).sum(axis=1)
            minimum = np.minimum(minimum, update)
            for identity in selected:
                minimum[candidates.index(identity)] = -np.inf
        return tuple(selected)
