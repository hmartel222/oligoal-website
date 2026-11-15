"""Batch uncertainty ranking using explicitly named predictor uncertainty."""

from random import Random

from oligo_engine.contracts import Observation, Predictor
from oligo_engine.evaluation.runner import validate_predictions


class UncertaintyAcquisition:
    @property
    def config(self) -> dict:
        return {"name": "uncertainty", "tie_break": "stable_sample_id"}

    def choose(
        self,
        eligible: tuple[Observation, ...],
        labeled: tuple[Observation, ...],
        budget: int,
        rng: Random,
        predictor: Predictor,
    ) -> tuple[str, ...]:
        predictions = predictor.predict(eligible)
        validate_predictions(predictions, len(eligible))
        if any(p.uncertainty is None for p in predictions):
            raise ValueError("Uncertainty acquisition requires uncertainty estimates")
        ordered = sorted(
            range(len(eligible)),
            key=lambda i: (-predictions[i].uncertainty, eligible[i].sample_id),
        )
        return tuple(eligible[i].sample_id for i in ordered[:budget])
