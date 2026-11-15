"""Matched, training-only Ridge heads over immutable label-free representations."""

import copy
import hashlib
from typing import Protocol

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler

from oligo_engine.contracts import Observation, Prediction


class Representation(Protocol):
    @property
    def config(self) -> dict: ...

    def encode(self, observations: tuple[Observation, ...]) -> np.ndarray: ...


class CachedRepresentation:
    """An immutable, ID-and-sequence-checked table; no learned preprocessing."""

    def __init__(
        self, observations: tuple[Observation, ...], values: np.ndarray, config: dict
    ) -> None:
        array = np.asarray(values, dtype=np.float64)
        if (
            not observations
            or array.ndim != 2
            or array.shape[0] != len(observations)
            or not array.shape[1]
            or not np.isfinite(array).all()
            or len({r.sample_id for r in observations}) != len(observations)
        ):
            raise ValueError("Representation requires unique aligned finite records")
        self._array = array.copy()
        self._array.flags.writeable = False
        self._index = {row.sample_id: i for i, row in enumerate(observations)}
        self._sequences = {row.sample_id: row.sequence for row in observations}
        self._config = {
            **copy.deepcopy(config),
            "feature_dimension": array.shape[1],
            "matrix_sha256": hashlib.sha256(self._array.tobytes()).hexdigest(),
            "stateless_extraction": True,
        }

    @property
    def config(self) -> dict:
        return copy.deepcopy(self._config)

    def encode(self, observations: tuple[Observation, ...]) -> np.ndarray:
        if not observations or any(
            self._sequences.get(row.sample_id) != row.sequence for row in observations
        ):
            raise ValueError("Unknown or changed representation record")
        return self._array[[self._index[row.sample_id] for row in observations]].copy()


class RepresentationRidge:
    """Rebuild scaler and Ridge using only each fit's revealed labels."""

    def __init__(
        self,
        representation: Representation,
        seed: int,
        alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 1000.0),
        folds: int = 3,
    ) -> None:
        if not alphas or any(not np.isfinite(a) or a <= 0 for a in alphas):
            raise ValueError("Ridge alphas must be finite positive values")
        if folds < 2:
            raise ValueError("At least two validation folds required")
        self.representation = representation
        self.seed = seed
        self.alphas = tuple(alphas)
        self.folds = folds
        self._model = None
        self._scaler = None
        self.fit_record: dict = {}

    @property
    def config(self) -> dict:
        return {
            "name": "representation_ridge",
            "representation": self.representation.config,
            "seed": self.seed,
            "alphas": list(self.alphas),
            "folds": self.folds,
            "selection_metric": "mean inner-fold RMSE",
            "selection_scope": "currently_revealed_training_labels_only",
            "scaler": "StandardScaler refit inside every inner fold",
            "solver": "cholesky",
            "uncertainty_type": None,
        }

    def fit(
        self, observations: tuple[Observation, ...], labels: tuple[float, ...]
    ) -> None:
        if len(observations) != len(labels) or len(labels) < self.folds:
            raise ValueError("Aligned training records must support validation folds")
        if len({r.sample_id for r in observations}) != len(observations):
            raise ValueError("Training IDs must be unique")
        y = np.asarray(labels, dtype=float)
        if not np.isfinite(y).all():
            raise ValueError("Training labels must be finite")
        x = self.representation.encode(observations)
        groups = [row.target_id for row in observations]
        grouped = all(groups) and len(set(groups)) >= self.folds
        splitter = (
            GroupKFold(self.folds)
            if grouped
            else KFold(self.folds, shuffle=True, random_state=self.seed)
        )
        folds = list(splitter.split(x, groups=groups if grouped else None))
        errors: list[list[float]] = [[] for _ in self.alphas]
        for train, valid in folds:
            scaler = StandardScaler().fit(x[train])
            training = scaler.transform(x[train])
            validation = scaler.transform(x[valid])
            for index, alpha in enumerate(self.alphas):
                model = Ridge(alpha=alpha, solver="cholesky").fit(training, y[train])
                errors[index].append(
                    float(np.sqrt(np.mean((model.predict(validation) - y[valid]) ** 2)))
                )
        means = [float(np.mean(values)) for values in errors]
        winner = int(np.argmin(means))
        self._scaler = StandardScaler().fit(x)
        self._model = Ridge(alpha=self.alphas[winner], solver="cholesky").fit(
            self._scaler.transform(x), y
        )
        self.fit_record = {
            "training_ids": [row.sample_id for row in observations],
            "validation": "target_group" if grouped else "shuffled_row_fallback",
            "folds": [
                {
                    "train_ids": [observations[i].sample_id for i in train],
                    "validation_ids": [observations[i].sample_id for i in valid],
                }
                for train, valid in folds
            ],
            "alpha_scores": [
                {"alpha": alpha, "fold_rmse": values, "mean_rmse": mean}
                for alpha, values, mean in zip(self.alphas, errors, means, strict=True)
            ],
            "selected_alpha": self.alphas[winner],
            "tie_break": "first alpha in frozen grid",
        }

    def predict(self, observations: tuple[Observation, ...]) -> tuple[Prediction, ...]:
        if self._model is None or self._scaler is None:
            raise RuntimeError("fit must precede predict")
        x = self._scaler.transform(self.representation.encode(observations))
        values = self._model.predict(x)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite Ridge prediction")
        return tuple(Prediction(float(value)) for value in values)
