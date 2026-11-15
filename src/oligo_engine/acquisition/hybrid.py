"""Fixed half uncertainty, half coverage batch acquisition."""

from random import Random

from oligo_engine.acquisition.pool_diversity import DiversityAcquisition
from oligo_engine.acquisition.uncertainty import UncertaintyAcquisition
from oligo_engine.contracts import Observation, Predictor


class HybridAcquisition:
    def __init__(self, diversity: DiversityAcquisition):
        self.diversity = diversity

    @property
    def config(self) -> dict:
        return {
            "name": "hybrid",
            "uncertainty_fraction": 0.5,
            "odd_batch": "extra_slot_to_diversity",
            "coverage": self.diversity.config,
        }

    def choose(
        self,
        eligible: tuple[Observation, ...],
        labeled: tuple[Observation, ...],
        budget: int,
        rng: Random,
        predictor: Predictor,
    ) -> tuple[str, ...]:
        if type(budget) is not int or not 1 <= budget <= len(eligible):
            raise ValueError("Invalid hybrid budget")
        count = budget // 2
        selected = (
            UncertaintyAcquisition().choose(eligible, labeled, count, rng, predictor)
            if count
            else ()
        )
        chosen = set(selected)
        remaining = tuple(row for row in eligible if row.sample_id not in chosen)
        centers = labeled + tuple(row for row in eligible if row.sample_id in chosen)
        return selected + self.diversity.choose(
            remaining, centers, budget - count, rng, predictor
        )
