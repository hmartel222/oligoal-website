"""Uniform random acquisition without replacement."""

from random import Random

from oligo_engine.contracts import Observation, Predictor


class RandomAcquisition:
    @property
    def config(self) -> dict:
        return {"name": "random", "replacement": False}

    def choose(
        self,
        eligible: tuple[Observation, ...],
        labeled: tuple[Observation, ...],
        budget: int,
        rng: Random,
        predictor: Predictor,
    ) -> tuple[str, ...]:
        return tuple(rng.sample([row.sample_id for row in eligible], budget))
