"""Fail-closed scoring of candidates with frozen endpoint predictors.

This module is an interface boundary for design hypotheses. It does not train
models and never turns predictions into observations.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, is_dataclass
from math import isfinite
from typing import Literal, Protocol

Status = Literal["accepted", "rejected", "not_applicable", "error"]
Direction = Literal["higher_is_better", "lower_is_better"]


def _hash(value: object) -> str:
    if is_dataclass(value):
        value = asdict(value)
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class Candidate:
    """Label-free candidate representation supplied to the adapter."""

    sequence_or_helm: str
    modality: str
    chemistry: str
    target_context: tuple[tuple[str, str], ...] = ()
    assay_context: tuple[tuple[str, str], ...] = ()
    source: Literal["generated", "fixture", "imported"] = "fixture"
    candidate_id: str = ""
    parser_identity: str = "fixture-acgt-v1"

    def __post_init__(self) -> None:
        if not self.sequence_or_helm or not self.modality or not self.chemistry:
            raise ValueError("Candidate sequence, modality, and chemistry are required")
        if self.source not in ("generated", "fixture", "imported"):
            raise ValueError("Unsupported candidate source")
        keys = [key for key, _ in self.target_context + self.assay_context]
        if len(keys) != len(set(keys)) or any(
            not key or not value
            for key, value in self.target_context + self.assay_context
        ):
            raise ValueError("Duplicate or empty context")
        expected = self.identity
        if self.candidate_id and self.candidate_id != expected:
            raise ValueError("Candidate ID does not match canonical candidate content")
        object.__setattr__(self, "candidate_id", expected)

    @property
    def identity(self) -> str:
        return _hash(
            {
                "sequence_or_helm": self.sequence_or_helm,
                "modality": self.modality,
                "chemistry": self.chemistry,
                "target_context": self.target_context,
                "assay_context": self.assay_context,
                "source": self.source,
                "parser_identity": self.parser_identity,
            }
        )

    def context(self) -> dict[str, str]:
        return dict(self.target_context + self.assay_context)


@dataclass(frozen=True)
class EndpointMetadata:
    endpoint_id: str
    endpoint_kind: str
    modality: str
    direction: Direction
    units: str
    chemistry_scope: tuple[str, ...]
    required_context: tuple[str, ...] = ()
    label_range: tuple[float, float] | None = None
    normalization_range: tuple[float, float] = (0.0, 1.0)
    artifact_hash: str = ""
    training_hash: str = ""
    split_hash: str = ""
    uncertainty_type: str = "none"
    disagreement_group: str | None = None
    model_id: str = ""
    feature_schema: str = "fixture-acgt-v1"
    normalization_hash: str = "fixture-normalization-v1"
    parser_identity: str = "fixture-acgt-v1"

    def validate(self) -> None:
        if not all(
            (
                self.endpoint_id,
                self.endpoint_kind,
                self.modality,
                self.units,
                self.artifact_hash,
                self.training_hash,
                self.split_hash,
                self.parser_identity,
            )
        ):
            raise ValueError(
                "Endpoint manifest requires identity, units, and source hashes"
            )
        if self.direction not in ("higher_is_better", "lower_is_better"):
            raise ValueError("Invalid endpoint direction")
        if self.normalization_range != (0.0, 1.0):
            raise ValueError("Unsupported normalization range")
        if (
            self.label_range is None
            or not all(isfinite(value) for value in self.label_range)
            or self.label_range[0] >= self.label_range[1]
        ):
            raise ValueError("Frozen training label range is required")
        if not self.normalization_hash or not self.feature_schema:
            raise ValueError("Frozen normalization and feature identity are required")


@dataclass(frozen=True)
class RawPrediction:
    value: float
    units: str
    uncertainty: float | None = None
    uncertainty_type: str = "none"


class FrozenPredictor(Protocol):
    metadata: EndpointMetadata

    def predict(self, candidate: Candidate) -> RawPrediction: ...


@dataclass(frozen=True)
class MockPredictor:
    """Deterministic predictor for contract and integration tests."""

    metadata: EndpointMetadata
    value: float

    def predict(self, candidate: Candidate) -> RawPrediction:
        return RawPrediction(self.value, self.metadata.units)


@dataclass(frozen=True)
class EndpointScore:
    endpoint_id: str
    raw: RawPrediction | None
    normalized: float | None
    applicable: bool
    reason_codes: tuple[str, ...] = ()
    clipped_value: float | None = None
    model_id: str = ""


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    observed_value: str
    threshold: str
    reason: str = ""
    units: str = ""


class HardConstraint(Protocol):
    name: str

    def evaluate(
        self, candidate: Candidate, scores: tuple[EndpointScore, ...]
    ) -> GateResult: ...


@dataclass(frozen=True)
class LengthConstraint:
    minimum: int
    maximum: int
    name: str = "length"

    def evaluate(
        self, candidate: Candidate, scores: tuple[EndpointScore, ...]
    ) -> GateResult:
        length = len(candidate.sequence_or_helm)
        passed = self.minimum <= length <= self.maximum
        return GateResult(
            self.name,
            passed,
            str(length),
            f"{self.minimum}..{self.maximum}",
            "" if passed else "LENGTH_OUT_OF_RANGE",
            "bases",
        )


@dataclass(frozen=True)
class MaxEndpointConstraint:
    endpoint_id: str
    maximum: float
    units: str
    name: str = "maximum_endpoint"

    def evaluate(
        self, candidate: Candidate, scores: tuple[EndpointScore, ...]
    ) -> GateResult:
        values = [
            score.raw
            for score in scores
            if score.endpoint_id == self.endpoint_id
            and score.raw is not None
            and score.applicable
        ]
        if not values or any(value.units != self.units for value in values):
            return GateResult(
                self.name,
                False,
                "missing",
                str(self.maximum),
                "REQUIRED_ENDPOINT_UNAVAILABLE",
                self.units,
            )
        observed = max(value.value for value in values)
        passed = observed <= self.maximum
        return GateResult(
            self.name,
            passed,
            str(observed),
            str(self.maximum),
            "" if passed else "ENDPOINT_LIMIT_EXCEEDED",
            self.units,
        )


@dataclass(frozen=True)
class ComplementarityConstraint:
    name: str = "complementarity"

    def evaluate(
        self, candidate: Candidate, scores: tuple[EndpointScore, ...]
    ) -> GateResult:
        site = candidate.context().get("verified_target_site")
        valid = bool(site) and not set(site + candidate.sequence_or_helm) - set("ACGT")
        if valid:
            valid = (
                candidate.sequence_or_helm
                == site.translate(str.maketrans("ACGT", "TGCA"))[::-1]
            )
        return GateResult(
            self.name,
            bool(valid),
            candidate.sequence_or_helm,
            site or "missing",
            "" if valid else "COMPLEMENTARITY_UNVERIFIED",
            "bases",
        )


@dataclass(frozen=True)
class DisagreementGroup:
    name: str
    model_ids: tuple[str, ...]
    threshold: float
    disposition: Literal["reject", "penalize"] = "reject"
    penalty: float = 0.0


@dataclass(frozen=True)
class DisagreementResult:
    name: str
    members: tuple[tuple[str, float], ...]
    value: float | None
    threshold: float
    status: str
    penalty: float


@dataclass(frozen=True)
class NoveltyReference:
    sequences: tuple[str, ...]
    chemistry: str
    representation: str = "acgt-hamming-fraction-v1"

    def __post_init__(self) -> None:
        if not self.sequences or not self.chemistry:
            raise ValueError("Novelty reference must be nonempty")
        if len({len(sequence) for sequence in self.sequences}) != 1 or any(
            not sequence or set(sequence) - set("ACGT") for sequence in self.sequences
        ):
            raise ValueError("Invalid novelty reference sequences")
        if self.representation != "acgt-hamming-fraction-v1":
            raise ValueError("Unsupported novelty representation")

    @property
    def identity(self) -> str:
        return _hash(self)


@dataclass(frozen=True)
class NoveltyResult:
    reference_hash: str
    nearest_id: str | None
    distance: float | None
    threshold: float | None
    status: str
    penalty: float


@dataclass(frozen=True)
class RewardScore:
    candidate_id: str
    status: Status
    endpoint_scores: tuple[EndpointScore, ...]
    constraints: tuple[GateResult, ...]
    disagreement: float | None
    disagreement_penalty: float
    novelty_distance: float | None
    novelty_penalty: float
    reward: float | None
    reason_codes: tuple[str, ...]
    provenance: tuple[tuple[str, str], ...]
    disagreements: tuple[DisagreementResult, ...] = ()
    novelty: NoveltyResult | None = None
    contributions: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class RewardConfig:
    required_endpoints: tuple[str, ...]
    weights: tuple[tuple[str, float], ...]
    disagreement_threshold: float | None = None
    disagreement_penalty: float = 0.0
    novelty_threshold: float | None = None
    novelty_penalty: float = 0.0
    reward_formula_version: str = "reward-v1"
    optional_endpoints: tuple[str, ...] = ()
    disagreement_groups: tuple[DisagreementGroup, ...] = ()
    novelty_disposition: Literal["reject", "penalize"] = "reject"

    def __post_init__(self) -> None:
        if not self.required_endpoints:
            raise ValueError("At least one required endpoint is needed")
        if len(set(self.required_endpoints + self.optional_endpoints)) != (
            len(self.required_endpoints) + len(self.optional_endpoints)
        ):
            raise ValueError("Duplicate endpoint IDs")
        if len({key for key, _ in self.weights}) != len(self.weights):
            raise ValueError("Duplicate endpoint weights")
        weights = dict(self.weights)
        if set(weights) != set(self.required_endpoints):
            raise ValueError("Weights must cover exactly the required endpoints")
        if any(not isfinite(value) for value in weights.values()):
            raise ValueError("Reward weights must be finite")
        if any(
            not isfinite(value) or value < 0
            for value in (self.disagreement_penalty, self.novelty_penalty)
        ):
            raise ValueError("Penalties must be nonnegative")
        if self.novelty_disposition not in ("reject", "penalize"):
            raise ValueError("Invalid novelty disposition")
        if self.novelty_threshold is not None and (
            not isfinite(self.novelty_threshold) or not 0 <= self.novelty_threshold <= 1
        ):
            raise ValueError("Invalid novelty threshold")
        if self.disagreement_threshold is not None and (
            not isfinite(self.disagreement_threshold)
            or not 0 <= self.disagreement_threshold <= 1
        ):
            raise ValueError("Invalid disagreement threshold")
        if len({g.name for g in self.disagreement_groups}) != len(
            self.disagreement_groups
        ):
            raise ValueError("Duplicate disagreement group")
        for group in self.disagreement_groups:
            if (
                not group.name
                or len(group.model_ids) < 2
                or len(set(group.model_ids)) != len(group.model_ids)
                or group.disposition not in ("reject", "penalize")
                or not isfinite(group.threshold)
                or not 0 <= group.threshold <= 1
                or not isfinite(group.penalty)
                or group.penalty < 0
            ):
                raise ValueError("Invalid disagreement group")


class MultiEndpointReward:
    """Deterministic, fail-closed composition of frozen endpoint predictions."""

    def __init__(
        self,
        predictors: tuple[FrozenPredictor, ...],
        config: RewardConfig,
        novelty_reference: NoveltyReference | None = None,
        provenance: tuple[tuple[str, str], ...] = (),
        constraints: tuple[HardConstraint, ...] = (),
    ) -> None:
        self._predictors = {p.metadata.endpoint_id: p for p in predictors}
        self._config = config
        self._reference = novelty_reference
        self._provenance = provenance
        self._constraints = constraints
        self._models = predictors
        self.predictor_calls = 0
        identities = [p.metadata.model_id or p.metadata.endpoint_id for p in predictors]
        if len(set(identities)) != len(identities):
            raise ValueError("Predictor endpoint IDs must be unique")
        self._by_model = dict(zip(identities, predictors, strict=True))
        for predictor in predictors:
            predictor.metadata.validate()
        missing = set(config.required_endpoints) - set(self._predictors)
        if missing:
            raise ValueError(f"Missing required predictors: {sorted(missing)}")
        if config.novelty_threshold is not None and novelty_reference is None:
            raise ValueError("Novelty threshold requires reference")
        if len({c.name for c in constraints}) != len(constraints):
            raise ValueError("Constraint names must be unique")
        for constraint in constraints:
            if isinstance(constraint, LengthConstraint) and (
                constraint.minimum < 1 or constraint.minimum > constraint.maximum
            ):
                raise ValueError("Invalid length constraint")
            if isinstance(constraint, MaxEndpointConstraint) and (
                not isfinite(constraint.maximum) or not constraint.units
            ):
                raise ValueError("Invalid endpoint limit")
        for group in config.disagreement_groups:
            if set(group.model_ids) - set(self._by_model):
                raise ValueError("Missing disagreement model")
            metas = [self._by_model[id_].metadata for id_ in group.model_ids]
            signatures = [
                (
                    m.endpoint_id,
                    m.modality,
                    m.units,
                    m.direction,
                    m.chemistry_scope,
                    m.required_context,
                    m.label_range,
                    m.normalization_hash,
                    m.parser_identity,
                    m.feature_schema,
                )
                for m in metas
            ]
            if len(set(signatures)) != 1:
                raise ValueError("Incompatible disagreement models")

    def score(self, candidate: Candidate) -> RewardScore:
        endpoint_scores = tuple(
            self._score_endpoint(predictor, candidate)
            for predictor in self._models
            if predictor.metadata.endpoint_id
            in (self._config.required_endpoints + self._config.optional_endpoints)
        )
        reasons = tuple(
            reason
            for score in endpoint_scores
            if score.endpoint_id in self._config.required_endpoints
            for reason in score.reason_codes
        )
        for endpoint_id in self._config.required_endpoints:
            if not any(
                s.endpoint_id == endpoint_id and s.applicable for s in endpoint_scores
            ):
                reasons += ("REQUIRED_ENDPOINT_UNAVAILABLE",)
        constraints = tuple(
            constraint.evaluate(candidate, endpoint_scores)
            for constraint in self._constraints
        )
        reasons += tuple(
            g.reason or "HARD_CONSTRAINT_FAILED" for g in constraints if not g.passed
        )
        disagreements = tuple(
            self._disagreement(group, endpoint_scores)
            for group in self._config.disagreement_groups
        )
        reasons += tuple(
            "MODEL_DISAGREEMENT" for d in disagreements if d.status == "rejected"
        )
        novelty_result = self._novelty(candidate)
        if novelty_result and novelty_result.status in ("rejected", "unavailable"):
            reasons += (
                "NOVELTY_THRESHOLD"
                if novelty_result.status == "rejected"
                else "NOVELTY_UNAVAILABLE",
            )
        disagreement_penalty = sum(d.penalty for d in disagreements)
        novelty_penalty = novelty_result.penalty if novelty_result else 0.0
        status: Status = "rejected" if reasons else "accepted"
        reward = None
        contributions: tuple[tuple[str, float], ...] = ()
        if status == "accepted":
            weights = dict(self._config.weights)
            contributions = tuple(
                (endpoint_id, weights[endpoint_id] * sum(values) / len(values))
                for endpoint_id in self._config.required_endpoints
                if (
                    values := [
                        s.normalized
                        for s in endpoint_scores
                        if s.endpoint_id == endpoint_id and s.normalized is not None
                    ]
                )
            )
            contributions += tuple(
                (f"disagreement:{d.name}", -d.penalty) for d in disagreements
            )
            if novelty_result:
                contributions += (("novelty", -novelty_penalty),)
            reward = sum(value for _, value in contributions)
            if not isfinite(reward):
                reasons += ("INVALID_REWARD",)
                status = "rejected"
                reward = None
                contributions = ()
        return self._result(
            candidate,
            status,
            endpoint_scores,
            constraints,
            reasons,
            None,
            disagreement_penalty,
            novelty_result.distance if novelty_result else None,
            novelty_penalty,
            reward,
            disagreements,
            novelty_result,
            contributions,
        )

    def policy_reward(self, candidate: Candidate) -> float:
        """Give policies a scalar only for an accepted candidate."""
        result = self.score(candidate)
        if result.reward is None:
            raise ValueError(f"Candidate rejected: {result.reason_codes}")
        return result.reward

    def _score_endpoint(
        self, predictor: FrozenPredictor, candidate: Candidate
    ) -> EndpointScore:
        metadata = predictor.metadata
        endpoint_id = metadata.endpoint_id
        model_id = metadata.model_id or endpoint_id
        reasons: list[str] = []
        if candidate.modality != metadata.modality:
            reasons.append("MODALITY_MISMATCH")
        if candidate.chemistry not in metadata.chemistry_scope:
            reasons.append("CHEMISTRY_UNSUPPORTED")
        if candidate.parser_identity != metadata.parser_identity:
            reasons.append("PARSER_MISMATCH")
        if candidate.parser_identity == "fixture-acgt-v1" and (
            set(candidate.sequence_or_helm) - set("ACGT")
        ):
            reasons.append("PARTIAL_PARSE")
        if metadata.feature_schema != "fixture-acgt-v1":
            reasons.append("FEATURE_SCHEMA_MISMATCH")
        context = candidate.context()
        if any(field not in context for field in metadata.required_context):
            reasons.append("MISSING_CONTEXT")
        if reasons:
            return EndpointScore(
                endpoint_id, None, None, False, tuple(reasons), model_id=model_id
            )
        self.predictor_calls += 1
        try:
            raw = predictor.predict(candidate)
        except Exception:
            return EndpointScore(
                endpoint_id, None, None, False, ("PREDICTOR_ERROR",), model_id=model_id
            )
        if (
            raw.units != metadata.units
            or not isfinite(raw.value)
            or raw.uncertainty_type != metadata.uncertainty_type
            or (raw.uncertainty is None) != (raw.uncertainty_type == "none")
            or (
                raw.uncertainty is not None
                and (not isfinite(raw.uncertainty) or raw.uncertainty < 0)
            )
        ):
            return EndpointScore(
                endpoint_id,
                raw,
                None,
                False,
                ("INVALID_PREDICTION",),
                model_id=model_id,
            )
        value = raw.value
        lo, hi = metadata.label_range
        value = min(max(value, lo), hi)
        normalized = (value - lo) / (hi - lo)
        if metadata.direction == "lower_is_better":
            normalized = 1.0 - normalized
        return EndpointScore(
            endpoint_id, raw, normalized, True, clipped_value=value, model_id=model_id
        )

    @staticmethod
    def _disagreement(
        group: DisagreementGroup, scores: tuple[EndpointScore, ...]
    ) -> DisagreementResult:
        members = tuple(
            (s.model_id, s.normalized)
            for s in scores
            if s.model_id in group.model_ids and s.normalized is not None
        )
        if len(members) < 2:
            return DisagreementResult(
                group.name, members, None, group.threshold, "unavailable", 0.0
            )
        value = max(v for _, v in members) - min(v for _, v in members)
        exceeded = value > group.threshold
        status = group.disposition if exceeded else "passed"
        return DisagreementResult(
            group.name,
            members,
            value,
            group.threshold,
            "rejected" if status == "reject" else status,
            group.penalty if status == "penalize" and exceeded else 0.0,
        )

    def _novelty(self, candidate: Candidate) -> NoveltyResult | None:
        reference = self._reference
        if reference is None:
            return None
        threshold = self._config.novelty_threshold
        if (
            candidate.chemistry != reference.chemistry
            or set(candidate.sequence_or_helm) - set("ACGT")
            or len(candidate.sequence_or_helm) != len(reference.sequences[0])
        ):
            return NoveltyResult(
                reference.identity, None, None, threshold, "unavailable", 0.0
            )
        nearest = min(
            reference.sequences,
            key=lambda sequence: (
                sum(a != b for a, b in zip(sequence, candidate.sequence_or_helm)),
                sequence,
            ),
        )
        distance = sum(a != b for a, b in zip(nearest, candidate.sequence_or_helm))
        distance /= len(nearest)
        exceeded = threshold is not None and distance > threshold
        status = self._config.novelty_disposition if exceeded else "passed"
        return NoveltyResult(
            reference.identity,
            _hash(nearest),
            distance,
            threshold,
            "rejected" if status == "reject" else status,
            self._config.novelty_penalty if status == "penalize" and exceeded else 0.0,
        )

    def _result(
        self,
        candidate: Candidate,
        status: Status,
        endpoint_scores: tuple[EndpointScore, ...],
        constraints: tuple[GateResult, ...],
        reasons: tuple[str, ...],
        disagreement: float | None = None,
        disagreement_penalty: float = 0.0,
        novelty: float | None = None,
        novelty_penalty: float = 0.0,
        reward: float | None = None,
        disagreements: tuple[DisagreementResult, ...] = (),
        novelty_result: NoveltyResult | None = None,
        contributions: tuple[tuple[str, float], ...] = (),
    ) -> RewardScore:
        model_hash = _hash(
            tuple(
                sorted(
                    (
                        p.metadata.model_id or p.metadata.endpoint_id,
                        p.metadata.artifact_hash,
                        p.metadata.training_hash,
                        p.metadata.split_hash,
                        p.metadata.normalization_hash,
                    )
                    for p in self._models
                )
            )
        )
        provenance = self._provenance + (
            ("candidate_hash", candidate.candidate_id),
            ("config_hash", _hash(self._config)),
            ("model_hash", model_hash),
            ("reference_hash", self._reference.identity if self._reference else "none"),
            ("parser_identity", candidate.parser_identity),
            ("constraint_hash", _hash(tuple(repr(c) for c in self._constraints))),
        )
        return RewardScore(
            candidate.candidate_id,
            status,
            endpoint_scores,
            constraints,
            disagreement,
            disagreement_penalty,
            novelty,
            novelty_penalty,
            reward,
            reasons,
            provenance,
            disagreements,
            novelty_result,
            contributions,
        )
