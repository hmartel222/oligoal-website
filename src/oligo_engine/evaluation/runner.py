"""Budget-stopped replay; acquisition completes before held-out scoring."""

import hashlib
import json
import platform
import subprocess
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from math import isfinite
from pathlib import Path
from random import Random
from time import perf_counter

from oligo_engine.contracts import (
    SCHEMA_VERSION,
    AcquisitionPolicy,
    Checkpoint,
    Dataset,
    Prediction,
    Predictor,
    RunArtifact,
    RunConfig,
    SplitManifest,
)
from oligo_engine.data.audit import audit_split
from oligo_engine.data.oracle import LabelOracle
from oligo_engine.evaluation.metrics import HeldOutEvaluator


def environment_identity() -> dict:
    root = Path(__file__).resolve().parents[3]
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = "unknown", None
    packages = {}
    for name in ("numpy", "scipy", "scikit-learn"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            pass
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "code_revision": revision,
        "working_tree_dirty": dirty,
        "scientific_packages": packages,
        "source_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((root / "src" / "oligo_engine").rglob("*.py"))
        },
    }


def validate_predictions(predictions: tuple[Prediction, ...], size: int) -> None:
    if len(predictions) != size or any(not isfinite(p.mean) for p in predictions):
        raise ValueError("Predictions must align with inputs and have finite means")
    for p in predictions:
        if (p.uncertainty is None) != (p.uncertainty_type is None):
            raise ValueError("Uncertainty must have an explicit type")
        if p.uncertainty is not None and (
            not isfinite(p.uncertainty) or p.uncertainty < 0
        ):
            raise ValueError("Uncertainty must be finite and nonnegative")


def run_campaign(
    dataset: Dataset,
    split: SplitManifest,
    oracle: LabelOracle,
    evaluator: HeldOutEvaluator,
    predictor_factory: Callable[[], Predictor],
    policy: AcquisitionPolicy,
    config: RunConfig,
    reference_oracle: LabelOracle,
    reference_cache: dict | None = None,
    progress: Callable[[str, int], None] | None = None,
    checkpoint_labels: tuple[int, ...] | None = None,
) -> RunArtifact:
    """Evaluation owns both oracles; policies only receive label-free records."""
    started = perf_counter()
    split.validate(dataset)
    if config.max_labels > len(split.train_ids):
        raise ValueError("Label budget exceeds eligible training pool")
    if checkpoint_labels is not None and (
        not checkpoint_labels
        or any(type(n) is not int for n in checkpoint_labels)
        or tuple(sorted(set(checkpoint_labels))) != checkpoint_labels
        or checkpoint_labels[0] != config.initial_labels
        or checkpoint_labels[-1] != config.max_labels
    ):
        raise ValueError("Checkpoint schedule must increase from initial to maximum")
    if evaluator.ids != split.test_ids:
        raise ValueError("Evaluator ordering must match fixed test manifest")
    if oracle.revealed_ids or reference_oracle.revealed_ids:
        raise ValueError("Each run requires fresh campaign and reference oracles")
    if oracle is reference_oracle:
        raise ValueError("Reference must use an independent oracle")
    if any(
        set(o.eligible_ids) != set(split.train_ids) for o in (oracle, reference_oracle)
    ):
        raise ValueError("Both oracles must own exactly the manifest training pool")
    # No measured experiment may silently bypass source/baseline verification.
    if dataset.provenance != "synthetic" and config.purpose != "diagnostic":
        raise ValueError(
            "Measured research runs require the pending Gate 1 integration"
        )
    rows = {r.sample_id: r for r in dataset.observations}
    test = tuple(rows[i] for i in split.test_ids)
    initial = tuple(
        Random(config.seed).sample(list(split.train_ids), config.initial_labels)
    )
    rng = Random(f"acquisition:{config.seed}")
    oracle.reveal(initial)
    batch = initial
    pending = []
    model_config = None
    while True:
        labeled_ids = oracle.revealed_ids
        fitting_ids = tuple(sorted(labeled_ids))
        labeled = tuple(rows[i] for i in fitting_ids)
        predictor = predictor_factory()
        predictor.fit(labeled, oracle.acquired_labels(fitting_ids))
        if model_config is None:
            model_config = predictor.config
        elif model_config != predictor.config:
            raise ValueError("Predictor configuration changed across checkpoints")
        predictions = predictor.predict(test)
        validate_predictions(predictions, len(test))
        pending.append((len(labeled_ids), batch, predictions))
        if progress:
            progress("acquisition_checkpoint", len(labeled_ids))
        if len(labeled_ids) == config.max_labels:
            break
        acquired = set(labeled_ids)
        eligible = tuple(rows[i] for i in split.train_ids if i not in acquired)
        size = min(config.batch_size, config.max_labels - len(labeled_ids))
        if checkpoint_labels is not None:
            size = next(n for n in checkpoint_labels if n > len(labeled_ids)) - len(
                labeled_ids
            )
        batch = tuple(policy.choose(eligible, labeled, size, rng, predictor))
        valid_ids = {row.sample_id for row in eligible}
        if len(batch) != size or len(set(batch)) != size or not set(batch) <= valid_ids:
            raise ValueError("Policy must choose the requested unique eligible IDs")
        oracle.reveal(batch)

    # Full-pool fitting happens only after the acquisition campaign ends.
    key = hashlib.sha256(
        json.dumps(
            {
                "dataset": dataset.content_sha256,
                "name": dataset.name,
                "source_revision": dataset.source_revision,
                "endpoint": dataset.endpoint,
                "train": split.train_ids,
                "test": split.test_ids,
                "model": model_config,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    reused = reference_cache is not None and key in reference_cache
    if progress:
        progress(
            "full_reference_cached" if reused else "full_reference_fit",
            len(split.train_ids),
        )
    if reused:
        reference_oracle.reveal(split.train_ids)
        reference_predictions = reference_cache[key]
    else:
        reference = predictor_factory()
        reference_ids = tuple(sorted(split.train_ids))
        reference_labels = dict(
            zip(
                split.train_ids,
                reference_oracle.reveal(split.train_ids),
                strict=True,
            )
        )
        reference.fit(
            tuple(rows[i] for i in reference_ids),
            tuple(reference_labels[i] for i in reference_ids),
        )
        if reference.config != model_config:
            raise ValueError(
                "Full-pool reference must use the same predictor configuration"
            )
        reference_predictions = reference.predict(test)
        if reference_cache is not None:
            reference_cache[key] = reference_predictions
    validate_predictions(reference_predictions, len(test))
    checkpoints = tuple(
        Checkpoint(count, ids, evaluator.score(predictions))
        for count, ids, predictions in pending
    )
    reached = next(
        (
            c.labels_used
            for c in checkpoints
            if c.metrics.pearson_r is not None
            and config.pearson_threshold is not None
            and c.metrics.pearson_r >= config.pearson_threshold
        ),
        None,
    )
    area = None
    if len(checkpoints) > 1 and all(
        c.metrics.pearson_r is not None for c in checkpoints
    ):
        area = sum(
            (b.labels_used - a.labels_used)
            * (a.metrics.pearson_r + b.metrics.pearson_r)
            / 2
            for a, b in zip(checkpoints, checkpoints[1:])
        ) / (config.max_labels - config.initial_labels)
    identity = {
        key: value for key, value in vars(dataset).items() if key != "observations"
    }
    return RunArtifact(
        schema_version=SCHEMA_VERSION,
        status=(
            "complete_synthetic"
            if dataset.provenance == "synthetic"
            else "complete_measured_diagnostic"
        ),
        provenance=dataset.provenance,
        dataset=identity,
        split=split,
        config=config,
        model=model_config,
        policy=policy.config,
        initial_ids=initial,
        checkpoints=checkpoints,
        full_pool_labels=len(split.train_ids),
        full_pool_metrics=evaluator.score(reference_predictions),
        labels_to_threshold=reached,
        threshold_status=(
            "not_requested"
            if config.pearson_threshold is None
            else "reached"
            if reached is not None
            else "not_reached"
        ),
        normalized_pearson_aulc=area,
        runtime_seconds=perf_counter() - started,
        environment=environment_identity(),
        execution={
            "full_reference_reused": reused,
            "stateless_feature_cache": getattr(predictor, "feature_cache", None)
            is not None,
            "fit_order": "sample_id_lexicographic",
            "data_audit": audit_split(dataset, split),
            "checkpoint_labels": checkpoint_labels,
        },
        caveats=(
            "Synthetic contract fixture; not measured biological data or a finding.",
            "Gate 1 OligoGym baseline reproduction remains open.",
            "Fixture predictor is not a scientific baseline; no uncertainty estimate.",
        )
        if dataset.provenance == "synthetic"
        else (
            "Measured-data engineering diagnostic; not benchmark reproduction.",
            "Gate 1 remains open; no active-learning benefit or generalization claim.",
            "Dataset-specific endpoints and label processing; no pooled endpoints.",
            "Diagnostic hash representation and tree spread; no calibration claim.",
            "Assay context is preserved; this diagnostic model uses HELM only.",
            "Near-duplicate sequence-family leakage has not been eliminated.",
            *dataset.metadata.get("caveats", []),
        ),
    )
