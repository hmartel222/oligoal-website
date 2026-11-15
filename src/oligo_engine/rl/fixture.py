"""Synthetic engineering fixture, never a measured ASO predictor."""

from oligo_engine.design.aso_space import ASODesignSpace, TargetContext
from oligo_engine.reward.scoring import (
    AssayProfile,
    EndpointManifest,
    FrozenEndpoint,
    FrozenScorer,
    ScoringSession,
)

ASSAY = AssayProfile("fixture", "fixture", "fixture", 10.0, 24.0)


def fixture_space(gene: str = "FIXTURE", start: int = 1) -> ASODesignSpace:
    target = TargetContext(
        gene, f"synthetic-{gene}.1", "ACGT" * 7, "synthetic", "synthetic-fixture-v1"
    )
    return ASODesignSpace(target, start, radius=8, horizon=8)


def fixture_scorer() -> FrozenScorer:
    """Known increasing site utility: rightmost reachable window is optimal."""

    def manifest(endpoint: str, role: str, chemistry: str = "ps_moe_gapmer"):
        return EndpointManifest(
            endpoint,
            f"synthetic-{endpoint}-v1",
            "fixture units",
            chemistry,
            "synthetic-artifact",
            "synthetic-training",
            "synthetic-split",
            "synthetic-site-v1",
            ASSAY,
            role,
            reference=tuple(float(i) for i in range(10)),
        )

    return FrozenScorer(
        (
            FrozenEndpoint(
                manifest("potency", "objective"), lambda c, t, a: float(c.start)
            ),
            FrozenEndpoint(manifest("neurotox", "advisory"), lambda c, t, a: 2.0),
            FrozenEndpoint(manifest("tlr7", "advisory", "ps_2ome_gapmer"), None),
            FrozenEndpoint(manifest("hepatic", "advisory"), None),
        ),
        "synthetic_fixture_no_measured_labels",
    )


def fixture_session(
    space: ASODesignSpace,
    max_unique: int = 10,
    max_requests: int = 100,
) -> ScoringSession:
    return ScoringSession(
        fixture_scorer(), space.target, ASSAY, max_unique, max_requests
    )
