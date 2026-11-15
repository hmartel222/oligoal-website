"""Small dependency-free predictor for contract tests only, not a baseline."""

from math import dist

from oligo_engine.contracts import Observation, Prediction


class FixtureNearestNeighbor:
    def __init__(self) -> None:
        self._training: tuple[Observation, ...] = ()
        self._labels: tuple[float, ...] = ()

    @property
    def config(self) -> dict:
        return {"name": "fixture_nearest_neighbor", "scientific_baseline": False}

    def fit(
        self, observations: tuple[Observation, ...], labels: tuple[float, ...]
    ) -> None:
        if not observations or len(observations) != len(labels):
            raise ValueError("Training records and labels must align and be nonempty")
        self._training, self._labels = observations, labels

    def predict(self, observations: tuple[Observation, ...]) -> tuple[Prediction, ...]:
        if not self._training:
            raise ValueError("Predictor has not been fitted")
        return tuple(
            Prediction(
                self._labels[
                    min(
                        range(len(self._training)),
                        key=lambda i: dist(row.features, self._training[i].features),
                    )
                ]
            )
            for row in observations
        )
