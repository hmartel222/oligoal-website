"""One-hot baseline predictors for single-endpoint efficacy (Huesken first).

Both models sit behind the frozen ``Predictor`` contract: ``fit`` receives only the
revealed observations/labels, ``predict`` returns mean and an explicitly named,
uncalibrated uncertainty. Features are stateless (see features/upstream.py), so the
only learned state is the model itself, refit from scratch on every ``fit`` call.

Uncertainty semantics are deliberately different:

* GP: ``gp_latent_sd_uncalibrated`` is the posterior SD of the latent function with
  the fitted observation-noise variance removed. sklearn's ``return_std`` includes
  the WhiteKernel variance, a near-constant aleatoric floor that cannot rank
  candidates; it is exposed only through ``predict_components`` for diagnostics.
* RF: ``rf_tree_prediction_sd_uncalibrated`` is the spread across trees.
"""

from oligo_engine.contracts import Observation, Prediction
from oligo_engine.features.upstream import (
    OneHotBaseFeaturizer,
    verify_upstream_environment,
)

GP_LATENT_SD = "gp_latent_sd_uncalibrated"
RF_TREE_SD = "rf_tree_prediction_sd_uncalibrated"

# Preregistered kernel grid (scripts/select_gp_kernel.py). Each entry is
# (family, initial length scale); all are ConstantKernel * family + WhiteKernel with
# length-scale bounds (1, 100). Hyperparameters are refit by marginal likelihood on
# the labels revealed to each ``fit`` call only.
KERNEL_GRID = (
    ("rbf", 1.0),
    ("rbf", 4.0),
    ("rbf", 8.0),
    ("rbf", 16.0),
    ("matern15", 8.0),
    ("matern25", 8.0),
)
LENGTH_SCALE_BOUNDS = (1.0, 100.0)


def build_kernel(family: str, length_scale: float):
    from sklearn.gaussian_process.kernels import (
        RBF,
        ConstantKernel,
        Matern,
        WhiteKernel,
    )

    if family == "rbf":
        base = RBF(length_scale, LENGTH_SCALE_BOUNDS)
    elif family == "matern15":
        base = Matern(length_scale, LENGTH_SCALE_BOUNDS, nu=1.5)
    elif family == "matern25":
        base = Matern(length_scale, LENGTH_SCALE_BOUNDS, nu=2.5)
    else:
        raise ValueError(f"Unknown kernel family {family!r}")
    return ConstantKernel(1.0) * base + WhiteKernel(0.5)


def _check(observations: tuple[Observation, ...], labels=None) -> None:
    if not observations:
        raise ValueError("At least one observation is required")
    if labels is not None and len(labels) != len(observations):
        raise ValueError("Training records and revealed labels must align")


class OneHotRandomForest:
    """OligoGym one-hot (base) Random Forest; the Gate 1 anchor configuration."""

    def __init__(
        self,
        seed: int,
        n_estimators: int = 100,
        max_depth: int | None = 20,
        n_jobs: int = 2,
        featurizer: OneHotBaseFeaturizer | None = None,
    ):
        from oligogym.models import RandomForestModel

        self.environment = verify_upstream_environment()
        self.featurizer = featurizer or OneHotBaseFeaturizer()
        self.model = RandomForestModel(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=seed,
            n_jobs=n_jobs,
        )
        self._config = {
            "name": "onehot_random_forest",
            "seed": seed,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "n_jobs": n_jobs,
            **self.featurizer.config,
            "uncertainty_type": RF_TREE_SD,
            "uses_assay_context": False,
            "upstream": self.environment,
        }
        self._fitted = False

    @property
    def config(self) -> dict:
        return dict(self._config)

    def fit(
        self, observations: tuple[Observation, ...], labels: tuple[float, ...]
    ) -> None:
        _check(observations, labels)
        self.model.fit(self.featurizer.transform(observations), list(labels))
        self._fitted = True

    def predict(self, observations: tuple[Observation, ...]) -> tuple[Prediction, ...]:
        import numpy as np

        if not self._fitted:
            raise RuntimeError("fit must be called before predict")
        _check(observations)
        features = self.featurizer.transform(observations)
        trees = np.asarray([t.predict(features) for t in self.model.model.estimators_])
        means, spread = trees.mean(axis=0), trees.std(axis=0)
        return tuple(
            Prediction(float(m), float(s), RF_TREE_SD)
            for m, s in zip(means, spread, strict=True)
        )


class OneHotGaussianProcess:
    """OligoGym GP wrapper with a documented kernel and latent-variance uncertainty."""

    def __init__(
        self,
        seed: int,
        family: str = "rbf",
        length_scale: float = 4.0,
        n_restarts_optimizer: int = 0,
        featurizer: OneHotBaseFeaturizer | None = None,
    ):
        from oligogym.models import GaussianProcessModel

        self.environment = verify_upstream_environment()
        self.featurizer = featurizer or OneHotBaseFeaturizer()
        self.model = GaussianProcessModel(
            kernel=build_kernel(family, length_scale),
            normalize_y=True,
            n_restarts_optimizer=n_restarts_optimizer,
            random_state=seed,
        )
        self._config = {
            "name": "onehot_gaussian_process",
            "seed": seed,
            "kernel_family": family,
            "kernel_initial_length_scale": length_scale,
            "length_scale_bounds": list(LENGTH_SCALE_BOUNDS),
            "normalize_y": True,
            "n_restarts_optimizer": n_restarts_optimizer,
            **self.featurizer.config,
            "uncertainty_type": GP_LATENT_SD,
            "uncertainty_note": "latent-function SD, observation noise removed",
            "uses_assay_context": False,
            "upstream": self.environment,
        }
        self._fitted = False

    @property
    def config(self) -> dict:
        return dict(self._config)

    @property
    def fitted_kernel(self) -> str:
        if not self._fitted:
            raise RuntimeError("fit must be called first")
        return str(self.model.model.kernel_)

    def fit(
        self, observations: tuple[Observation, ...], labels: tuple[float, ...]
    ) -> None:
        import numpy as np

        _check(observations, labels)
        y = np.asarray(labels, dtype=float)
        if not np.all(np.isfinite(y)):
            raise ValueError("Revealed labels must be finite")
        self.model.fit(self.featurizer.transform(observations), y)
        self._fitted = True

    def predict_components(self, observations: tuple[Observation, ...]) -> dict:
        """Means, latent SD and total predictive SD in label units (diagnostics)."""
        import numpy as np

        if not self._fitted:
            raise RuntimeError("fit must be called before predict")
        _check(observations)
        gp = self.model.model
        mean, total_sd = gp.predict(
            self.featurizer.transform(observations), return_std=True
        )
        noise = [
            v for k, v in gp.kernel_.get_params().items() if k.endswith("noise_level")
        ]
        if len(noise) != 1:
            raise RuntimeError("Fitted kernel must contain exactly one WhiteKernel")
        noise_variance = float(noise[0]) * float(gp._y_train_std) ** 2
        latent_variance = np.clip(total_sd**2 - noise_variance, 0.0, None)
        return {
            "mean": np.asarray(mean, dtype=float).reshape(-1),
            "latent_sd": np.sqrt(latent_variance).reshape(-1),
            "total_sd": np.asarray(total_sd, dtype=float).reshape(-1),
            "noise_sd": float(np.sqrt(noise_variance)),
        }

    def predict(self, observations: tuple[Observation, ...]) -> tuple[Prediction, ...]:
        parts = self.predict_components(observations)
        return tuple(
            Prediction(float(m), float(s), GP_LATENT_SD)
            for m, s in zip(parts["mean"], parts["latent_sd"], strict=True)
        )
