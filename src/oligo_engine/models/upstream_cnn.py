"""Source-audited Huesken CNN adapter with acquired-row-only feature fitting."""

import copy
import random
from importlib.metadata import version
from math import isfinite
from pathlib import Path

from oligo_engine.audit_upstream import PINNED_REVISION, audit_checkout
from oligo_engine.contracts import Observation, Prediction


def _native_components(checkout: Path):
    audit_checkout(checkout)
    import numpy as np
    import oligogym.features as features
    import oligogym.models as models
    import torch

    if (
        Path(features.__file__).resolve() != checkout / "oligogym/features.py"
        or Path(models.__file__).resolve() != checkout / "oligogym/models.py"
    ):
        raise ValueError("Native model/feature imports differ from audited checkout")
    if version("scikit-learn") != "1.4.0":
        raise ValueError("Native Huesken study requires scikit-learn 1.4.0")
    return np, torch, features.OneHotEncoder, models.CNN


class UpstreamCNN:
    """Fresh native encoder/CNN per fit; validation uses acquired labels only."""

    def __init__(self, checkout: Path, seed: int, baseline_config: dict):
        self.checkout = checkout.resolve()
        self.seed = seed
        self.baseline = copy.deepcopy(baseline_config)
        if (
            self.baseline["upstream_revision"] != PINNED_REVISION
            or self.baseline["loader_key"] != "Huesken"
            or self.baseline["model"]["name"] != "CNN"
            or self.baseline["featurizer"]["name"] != "OneHotEncoder"
        ):
            raise ValueError("Adapter requires the frozen native Huesken CNN")
        self.encoder = None
        self.model = None
        self.fit_diagnostics = {}
        self._config = {
            "name": "upstream_one_hot_cnn",
            "seed": seed,
            "upstream_revision": PINNED_REVISION,
            "featurizer": self.baseline["featurizer"],
            "model": self.baseline["model"],
            "training": self.baseline["training"],
            "trainer": self.baseline["trainer"],
            "encoder_fit_scope": "currently_acquired_sequences_only",
            "uncertainty_type": None,
            "uses_assay_context": False,
        }

    @property
    def config(self) -> dict:
        return copy.deepcopy(self._config)

    def fit(
        self, observations: tuple[Observation, ...], labels: tuple[float, ...]
    ) -> None:
        if not observations or len(observations) != len(labels):
            raise ValueError("Acquired observations and labels must align")
        if len({row.sample_id for row in observations}) != len(observations):
            raise ValueError("Repeated IDs in CNN fit")
        if any(not isfinite(value) for value in labels):
            raise ValueError("CNN labels must be finite")
        np, torch, encoder_class, model_class = _native_components(self.checkout)
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        try:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            feature_args = {
                key: value
                for key, value in self.baseline["featurizer"].items()
                if key != "name"
            }
            self.encoder = encoder_class(**feature_args)
            features = self.encoder.fit_transform(
                [row.sequence for row in observations]
            )
            model_args = {
                key: value
                for key, value in self.baseline["model"].items()
                if key != "name"
            }
            self.model = model_class(input_dim=features.shape[-1], **model_args)
            self.model.fit(
                features,
                np.asarray(labels),
                **self.baseline["training"],
                **self.baseline["trainer"],
            )
            self.fit_diagnostics = {
                "sample_ids": [row.sample_id for row in observations],
                "input_shape": list(features.shape),
                "completed_epochs": int(self.model.trainer.current_epoch),
                "inner_validation_random_state": 42,
                "inner_validation_fraction": 0.2,
            }
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)

    def predict(self, observations: tuple[Observation, ...]) -> tuple[Prediction, ...]:
        if self.encoder is None or self.model is None:
            raise ValueError("CNN must be fitted before prediction")
        if not observations:
            return ()
        features = self.encoder.transform([row.sequence for row in observations])
        values = self.model.predict(features).reshape(-1)
        if len(values) != len(observations) or any(
            not isfinite(float(value)) for value in values
        ):
            raise ValueError("CNN predictions must be aligned and finite")
        return tuple(Prediction(float(value)) for value in values)
