"""Label-free contracts shared by infrastructure, predictors, policies and UI."""

from dataclasses import dataclass, field
from math import isfinite
from random import Random
from typing import Literal, Protocol

SCHEMA_VERSION = "0.2"


@dataclass(frozen=True)
class Observation:
    sample_id: str
    sequence: str
    target_id: str
    modality: str
    chemistry: str
    assay: str
    features: tuple[float, ...] = ()
    context: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Dataset:
    observations: tuple[Observation, ...]
    name: str
    source_revision: str
    content_sha256: str
    endpoint: str
    provenance: Literal["synthetic", "measured"]
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        ids = [row.sample_id for row in self.observations]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("Dataset must contain unique, nonempty observation IDs")
        if any(not row.sample_id or not row.sequence for row in self.observations):
            raise ValueError("Observation IDs and sequences must not be empty")
        if any(not isfinite(x) for row in self.observations for x in row.features):
            raise ValueError("Features must be finite")


@dataclass(frozen=True)
class SplitManifest:
    strategy: str
    seed: int
    train_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    parameters: tuple[tuple[str, str], ...] = ()

    def validate(self, dataset: Dataset) -> None:
        ids = self.train_ids + self.test_ids
        if not self.train_ids or not self.test_ids:
            raise ValueError("Both training and evaluation partitions are required")
        if len(ids) != len(set(ids)):
            raise ValueError("Split IDs overlap or contain duplicates")
        if set(ids) != {row.sample_id for row in dataset.observations}:
            raise ValueError("Split must partition the dataset exactly")
        train = set(self.train_ids)
        sequences = {
            row.sequence for row in dataset.observations if row.sample_id in train
        }
        if any(
            row.sequence in sequences
            for row in dataset.observations
            if row.sample_id not in train
        ):
            raise ValueError("Exact sequence duplicates cross the split")
        if self.strategy in ("target", "target_split"):
            train_targets = {
                row.target_id for row in dataset.observations if row.sample_id in train
            }
            if any(
                row.target_id in train_targets
                for row in dataset.observations
                if row.sample_id not in train
            ):
                raise ValueError("Target groups cross the target split")


@dataclass(frozen=True)
class Prediction:
    mean: float
    uncertainty: float | None = None
    uncertainty_type: str | None = None


class Predictor(Protocol):
    @property
    def config(self) -> dict: ...

    def fit(
        self, observations: tuple[Observation, ...], labels: tuple[float, ...]
    ) -> None: ...

    def predict(
        self, observations: tuple[Observation, ...]
    ) -> tuple[Prediction, ...]: ...


class AcquisitionPolicy(Protocol):
    @property
    def config(self) -> dict: ...

    def choose(
        self,
        eligible: tuple[Observation, ...],
        labeled: tuple[Observation, ...],
        budget: int,
        rng: Random,
        predictor: Predictor,
    ) -> tuple[str, ...]: ...


@dataclass(frozen=True)
class RunConfig:
    seed: int
    initial_labels: int
    batch_size: int
    max_labels: int
    pearson_threshold: float | None = 0.60
    purpose: Literal["research", "diagnostic"] = "research"

    def __post_init__(self) -> None:
        if type(self.seed) is not int:
            raise ValueError("Seed must be an integer")
        sizes = (self.initial_labels, self.batch_size, self.max_labels)
        if any(type(value) is not int or value < 1 for value in sizes):
            raise ValueError("Label budgets must be positive integers")
        if self.initial_labels > self.max_labels:
            raise ValueError("Initial labels exceed total label budget")
        if self.purpose not in ("research", "diagnostic"):
            raise ValueError("Run purpose must be research or diagnostic")
        if self.pearson_threshold is not None and not -1 <= self.pearson_threshold <= 1:
            raise ValueError("Pearson threshold must lie in [-1, 1]")


@dataclass(frozen=True)
class Metrics:
    pearson_r: float | None
    pearson_undefined_reason: str | None
    rmse: float
    mae: float


@dataclass(frozen=True)
class Checkpoint:
    labels_used: int
    acquired_ids: tuple[str, ...]
    metrics: Metrics


@dataclass(frozen=True)
class RunArtifact:
    schema_version: str
    status: str
    provenance: str
    dataset: dict
    split: SplitManifest
    config: RunConfig
    model: dict
    policy: dict
    initial_ids: tuple[str, ...]
    checkpoints: tuple[Checkpoint, ...]
    full_pool_labels: int
    full_pool_metrics: Metrics
    labels_to_threshold: int | None
    threshold_status: str
    normalized_pearson_aulc: float | None
    runtime_seconds: float
    environment: dict
    caveats: tuple[str, ...]
    execution: dict = field(default_factory=dict)
