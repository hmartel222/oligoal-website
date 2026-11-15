"""Inference-only bridge for an already fitted, split-bound OligoGym endpoint."""

from typing import Protocol

from oligo_engine.design.aso_space import TargetContext, candidate_identity
from oligo_engine.design.candidates import Candidate
from oligo_engine.reward.scoring import AssayProfile, EndpointPrediction


class EndpointModel(Protocol):
    def predict(self, candidates: list[Candidate]) -> list[dict]: ...


class OligoGymInference:
    """Does not construct DesignEngine or call its all-label model factory."""

    def __init__(self, model: EndpointModel, endpoint_id: str) -> None:
        self._model, self._endpoint_id = model, endpoint_id

    def __call__(
        self, candidate: Candidate, target: TargetContext, assay: AssayProfile
    ) -> EndpointPrediction:
        candidate_identity(candidate, target)
        values = self._model.predict([candidate])
        if len(values) != 1 or values[0]["endpoint"] != self._endpoint_id:
            raise ValueError("OligoGym endpoint identity mismatch")
        result = values[0]
        return EndpointPrediction(
            float(result["value"]),
            result.get("uncertainty"),
            result.get("uncertainty_type", "none"),
            tuple(
                (key, float(result[key]))
                for key in (
                    "training_percentile",
                    "nearest_labelled_distance",
                    "novelty_percentile",
                )
                if key in result
            ),
        )
