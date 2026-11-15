"""Uniform diagnostic model supporting variable-length, chemically modified HELM."""

from oligo_engine.contracts import Observation, Prediction


class HelmFeatureCache:
    """Stateless transforms only: no labels, vocabulary fitting or normalization fit."""

    def __init__(self, observations: tuple[Observation, ...], n_features: int):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.n_features = n_features
        self.indices = {row.sample_id: i for i, row in enumerate(observations)}
        if len(self.indices) != len(observations):
            raise ValueError("Feature cache requires unique IDs")
        self.sequences = {row.sample_id: row.sequence for row in observations}
        encoder = HashingVectorizer(
            analyzer="char",
            ngram_range=(1, 3),
            n_features=n_features,
            lowercase=False,
            alternate_sign=False,
            norm="l2",
        )
        self.matrix = encoder.transform([row.sequence for row in observations]).tocsr()

    def transform(self, observations: tuple[Observation, ...]):
        if any(
            self.sequences.get(row.sample_id) != row.sequence for row in observations
        ):
            raise ValueError("Feature-cache ID/sequence mismatch")
        return self.matrix[[self.indices[row.sample_id] for row in observations]]


class HelmHashForest:
    """Stateless representation; this is not an official OligoGym featurizer."""

    def __init__(
        self,
        seed: int,
        n_estimators: int = 32,
        max_depth: int = 12,
        n_features: int = 512,
        prediction_batch_size: int = 2048,
        n_jobs: int = 1,
        feature_cache: HelmFeatureCache | None = None,
    ):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.feature_extraction.text import HashingVectorizer

        self.encoder = HashingVectorizer(
            analyzer="char",
            ngram_range=(1, 3),
            n_features=n_features,
            lowercase=False,
            alternate_sign=False,
            norm="l2",
        )
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=seed,
            n_jobs=n_jobs,
        )
        if feature_cache is not None and feature_cache.n_features != n_features:
            raise ValueError("Cached feature dimension differs from predictor")
        self.feature_cache = feature_cache
        if prediction_batch_size < 1:
            raise ValueError("prediction_batch_size must be positive")
        self.prediction_batch_size = prediction_batch_size
        self._config = {
            "name": "helm_hash_random_forest",
            "seed": seed,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "n_features": n_features,
            "ngram_range": [1, 3],
            "featurizer": "stateless_case_sensitive_HELM_character_hash",
            "official_oligogym_representation": False,
            "uncertainty_type": "tree_prediction_sd_uncalibrated",
            "uses_assay_context": False,
            "prediction_batch_size": prediction_batch_size,
            "n_jobs": n_jobs,
        }

    @property
    def config(self) -> dict:
        return dict(self._config)

    def fit(
        self, observations: tuple[Observation, ...], labels: tuple[float, ...]
    ) -> None:
        if not observations or len(observations) != len(labels):
            raise ValueError("Training records and revealed labels must align")
        features = self._features(observations)
        self.model.fit(features, labels)

    def _features(self, observations: tuple[Observation, ...]):
        if self.feature_cache is not None:
            return self.feature_cache.transform(observations)
        return self.encoder.transform([row.sequence for row in observations])

    def predict(self, observations: tuple[Observation, ...]) -> tuple[Prediction, ...]:
        import numpy as np
        from joblib import Parallel, delayed

        results = []
        with Parallel(n_jobs=self.model.n_jobs, prefer="threads") as parallel:
            for start in range(0, len(observations), self.prediction_batch_size):
                batch = observations[start : start + self.prediction_batch_size]
                features = self._features(batch).tocsr().astype(np.float32, copy=False)
                tree_predictions = np.asarray(
                    parallel(
                        delayed(tree.predict)(features)
                        for tree in self.model.estimators_
                    )
                )
                means = tree_predictions.mean(axis=0)
                spread = tree_predictions.std(axis=0)
                results.extend(
                    Prediction(
                        float(mean), float(sd), "tree_prediction_sd_uncalibrated"
                    )
                    for mean, sd in zip(means, spread, strict=True)
                )
        return tuple(results)
