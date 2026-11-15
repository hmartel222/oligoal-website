"""Production candidate boundary and budgeted frozen multi-endpoint inference.

The older design.reward module remains a compatible fixture contract. This module
uses the real molecular Candidate and explicit model roles instead of fixture ACGT.
"""

from dataclasses import asdict, dataclass
from math import isfinite
from time import perf_counter
from typing import Callable, Literal

from oligo_engine.design.aso_space import TargetContext, candidate_identity, identity
from oligo_engine.design.candidates import Candidate, build_helm


class BudgetExhausted(RuntimeError):
    """No uncharged inference may run after a scorer budget is exhausted."""


class ScorerError(RuntimeError):
    """Model/infrastructure failure; never an ordinary candidate rejection."""


@dataclass(frozen=True)
class AssayProfile:
    cell_line: str
    cell_line_species: str
    transfection_method: str
    dosage_nm: float
    treatment_period_hrs: float

    def __post_init__(self) -> None:
        if not all((self.cell_line, self.cell_line_species, self.transfection_method)):
            raise ValueError("Complete fixed assay context required")
        if any(
            not isfinite(v) or v <= 0
            for v in (self.dosage_nm, self.treatment_period_hrs)
        ):
            raise ValueError("Positive finite dose/time required")


@dataclass(frozen=True)
class EndpointManifest:
    endpoint_id: str
    model_id: str
    units: str
    chemistry: str
    artifact_hash: str
    training_hash: str
    split_hash: str
    feature_schema: str
    context: AssayProfile
    role: Literal["objective", "required_screen", "advisory"]
    reference: tuple[float, ...] = ()
    maximum: float | None = None
    screening_evidence: str = ""
    direction: Literal["higher_is_better", "lower_is_better"] = "higher_is_better"
    provenance: str = "fixture"

    def __post_init__(self) -> None:
        if not all(
            (
                self.endpoint_id,
                self.model_id,
                self.units,
                self.chemistry,
                self.artifact_hash,
                self.training_hash,
                self.split_hash,
                self.feature_schema,
                self.provenance,
            )
        ):
            raise ValueError("Complete endpoint manifest required")
        if self.role not in ("objective", "required_screen", "advisory"):
            raise ValueError("Invalid endpoint role")
        if self.direction not in ("higher_is_better", "lower_is_better"):
            raise ValueError("Invalid endpoint direction")
        if any(not isfinite(v) for v in self.reference):
            raise ValueError("Nonfinite reference")
        if self.role == "objective" and not self.reference:
            raise ValueError("Objective requires frozen training-only reference")
        if self.role == "required_screen" and (
            self.maximum is None or not self.screening_evidence
        ):
            raise ValueError("Required screen needs cutoff and validation evidence")
        if self.maximum is not None and not isfinite(self.maximum):
            raise ValueError("Nonfinite screening threshold")


@dataclass(frozen=True)
class EndpointResult:
    endpoint_id: str
    model_id: str
    units: str
    role: str
    direction: str
    status: str
    raw: float | None
    normalized: float | None
    reason: str
    manifest_identity: str
    uncertainty: float | None = None
    uncertainty_type: str = "none"
    diagnostics: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class CandidateScore:
    candidate_id: str
    scorer_identity: str
    status: str
    reward: float | None
    endpoints: tuple[EndpointResult, ...]
    reasons: tuple[str, ...]
    provenance: str
    assay_identity: str
    assessment_id: str
    disagreement: tuple[tuple[str, float], ...]

    def terminal_reward(self) -> float:
        return self.reward if self.reward is not None else -1.0


@dataclass(frozen=True)
class EndpointPrediction:
    value: float
    uncertainty: float | None = None
    uncertainty_type: str = "none"
    diagnostics: tuple[tuple[str, float], ...] = ()


Inference = Callable[
    [Candidate, TargetContext, AssayProfile], float | EndpointPrediction
]


@dataclass(frozen=True)
class FrozenEndpoint:
    manifest: EndpointManifest
    predict: Inference | None


class FrozenScorer:
    """Pure frozen scoring; private inference callables receive no label capability."""

    def __init__(self, endpoints: tuple[FrozenEndpoint, ...], provenance: str) -> None:
        manifests = tuple(e.manifest for e in endpoints)
        if sum(m.role == "objective" for m in manifests) != 1:
            raise ValueError("Exactly one potency objective required")
        if len({m.model_id for m in manifests}) != len(manifests):
            raise ValueError("Model identities must be unique")
        if any(e.predict is None and e.manifest.role != "advisory" for e in endpoints):
            raise ScorerError("Required model bundle is unavailable")
        if not provenance:
            raise ValueError("Scorer provenance required")
        self._endpoints = endpoints
        self.provenance = provenance
        self.identity = identity(
            {
                "schema": "aso-scorer-v1",
                "heads": [asdict(m) for m in manifests],
                "available": [e.predict is not None for e in endpoints],
                "provenance": provenance,
                "percentile": "midrank-v1",
            }
        )

    @property
    def manifests(self) -> tuple[EndpointManifest, ...]:
        return tuple(e.manifest for e in self._endpoints)

    def score(
        self,
        candidate: Candidate,
        target: TargetContext,
        assay: AssayProfile,
        on_call: Callable[[str], None] = lambda model_id: None,
    ) -> CandidateScore:
        key = candidate_identity(candidate, target)
        if candidate.helm != build_helm(candidate.sequence, candidate.chemistry):
            raise ValueError("Candidate HELM/template mismatch")
        scores, reasons = [], []
        utility = None
        for endpoint in self._endpoints:
            m = endpoint.manifest
            raw, normalized = None, None
            uncertainty, uncertainty_type = None, "none"
            diagnostics = ()
            status, reason = "scored", ""
            if candidate.chemistry != m.chemistry:
                status, reason = "not_applicable", "CHEMISTRY_UNSUPPORTED"
            elif assay != m.context:
                status, reason = "out_of_domain", "ASSAY_UNSUPPORTED"
            elif endpoint.predict is None:
                status, reason = "unavailable", "MODEL_UNAVAILABLE"
            else:
                on_call(m.model_id)
                try:
                    prediction = endpoint.predict(candidate, target, assay)
                    if isinstance(prediction, EndpointPrediction):
                        raw = float(prediction.value)
                        uncertainty = prediction.uncertainty
                        uncertainty_type = prediction.uncertainty_type
                        diagnostics = prediction.diagnostics
                    else:
                        raw = float(prediction)
                except Exception as exc:
                    raise ScorerError(f"Inference failed: {m.model_id}") from exc
                if not isfinite(raw):
                    raise ScorerError(f"Nonfinite prediction: {m.model_id}")
                if (
                    (uncertainty is None) != (uncertainty_type == "none")
                    or uncertainty is not None
                    and (not isfinite(uncertainty) or uncertainty < 0)
                ):
                    raise ScorerError(f"Invalid uncertainty: {m.model_id}")
                if any(not isfinite(v) for _, v in diagnostics):
                    raise ScorerError(f"Invalid diagnostics: {m.model_id}")
                if m.role == "objective":
                    normalized = (
                        sum(v < raw for v in m.reference)
                        + 0.5 * sum(v == raw for v in m.reference)
                    ) / len(m.reference)
                    if m.direction == "lower_is_better":
                        normalized = 1.0 - normalized
                    utility = normalized
                if m.role == "required_screen" and raw > m.maximum:
                    status, reason = "rejected", "SCREEN_MAXIMUM"
            if status != "scored" and m.role != "advisory":
                reasons.append(f"{m.model_id}:{reason}")
            scores.append(
                EndpointResult(
                    m.endpoint_id,
                    m.model_id,
                    m.units,
                    m.role,
                    m.direction,
                    status,
                    raw,
                    normalized,
                    reason,
                    identity(asdict(m)),
                    uncertainty,
                    uncertainty_type,
                    diagnostics,
                )
            )
        groups: dict[str, list[float]] = {}
        for score, endpoint in zip(scores, self._endpoints, strict=True):
            if score.raw is not None:
                m = endpoint.manifest
                group = identity(
                    {
                        "endpoint": m.endpoint_id,
                        "units": m.units,
                        "direction": m.direction,
                        "context": asdict(m.context),
                        "chemistry": m.chemistry,
                    }
                )
                groups.setdefault(group, []).append(score.raw)
        disagreement = tuple(
            (group, max(values) - min(values))
            for group, values in sorted(groups.items())
            if len(values) > 1
        )
        return CandidateScore(
            key,
            self.identity,
            "rejected" if reasons else "accepted",
            None if reasons else utility,
            tuple(scores),
            tuple(reasons),
            self.provenance,
            identity(asdict(assay)),
            identity({"candidate": key, "assay": asdict(assay)}),
            disagreement,
        )


class ScoringSession:
    """One optimizer's cache, exact cost ledger and archive; never share across arms."""

    def __init__(
        self,
        scorer: FrozenScorer,
        target: TargetContext,
        assay: AssayProfile,
        max_unique: int,
        max_requests: int,
    ) -> None:
        if any(type(v) is not int or v < 1 for v in (max_unique, max_requests)):
            raise ValueError("Positive integer scoring budgets required")
        if any(
            m.role != "advisory"
            and (m.context != assay or m.chemistry != "ps_moe_gapmer")
            for m in scorer.manifests
        ):
            raise ScorerError(
                "Required model does not support the fixed assay/chemistry"
            )
        self.scorer, self.target, self.assay = scorer, target, assay
        self.max_unique, self.max_requests = max_unique, max_requests
        self._identity = scorer.identity
        self._cache: dict[str, CandidateScore] = {}
        self.archive: dict[str, dict] = {}
        self.requests = self.misses = self.head_calls = 0
        self.per_head_calls: dict[str, int] = {}
        self.elapsed_seconds = 0.0
        self.trace: list[dict] = []

    def _call(self, model_id: str) -> None:
        self.head_calls += 1
        self.per_head_calls[model_id] = self.per_head_calls.get(model_id, 0) + 1

    def score(self, candidate: Candidate) -> CandidateScore:
        if self.scorer.identity != self._identity:
            raise ScorerError("Scorer changed during an optimization run")
        key = identity(
            {
                "candidate": candidate_identity(candidate, self.target),
                "assay": asdict(self.assay),
                "scorer": self._identity,
            }
        )
        if self.requests >= self.max_requests or (
            key not in self._cache and self.misses >= self.max_unique
        ):
            raise BudgetExhausted("Scoring budget exhausted before inference")
        self.requests += 1
        hit = key in self._cache
        if not hit:
            self.misses += 1  # Failed inference is still charged; exception aborts run.
            started = perf_counter()
            try:
                self._cache[key] = self.scorer.score(
                    candidate, self.target, self.assay, self._call
                )
            finally:
                self.elapsed_seconds += perf_counter() - started
            self.archive[key] = {
                "candidate": asdict(candidate),
                "score": asdict(self._cache[key]),
            }
        result = self._cache[key]
        self.trace.append(
            {
                "request": self.requests,
                "unique": self.misses,
                "cache_hit": hit,
                "candidate_id": result.candidate_id,
                "reward": result.reward,
            }
        )
        return result

    def receipt(self) -> dict:
        return {
            "scorer_identity": self._identity,
            "target_identity": self.target.identity,
            "assay": asdict(self.assay),
            "requests": self.requests,
            "unique_scores": self.misses,
            "head_calls": self.head_calls,
            "per_head_calls": dict(self.per_head_calls),
            "inference_batches": self.head_calls,
            "batch_size": 1,
            "model_manifests": [asdict(m) for m in self.scorer.manifests],
            "provenance": self.scorer.provenance,
            "max_unique": self.max_unique,
            "max_requests": self.max_requests,
            "elapsed_seconds": self.elapsed_seconds,
            "archive": list(self.archive.values()),
            "trace": self.trace,
        }
