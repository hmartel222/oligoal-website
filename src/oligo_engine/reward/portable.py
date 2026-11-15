"""Inference for the frozen 20-mer Atlas efficacy model."""

import hashlib
import json
from math import isfinite
from pathlib import Path

from oligo_engine.design.candidates import Candidate, build_helm
from oligo_engine.reward.scoring import AssayProfile

CHEMISTRY = "ps_moe_20_gapmer"
MODEL_SHA256 = "44f975ae47c1a073ec52bde1e3403baa21de8a1b0b733e4bcac1a9b236252a84"


class PortableEfficacy:
    """Run the Atlas supported-aso-v1 Ridge checkpoint in its fixed assay."""

    def __init__(self, path: Path) -> None:
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != MODEL_SHA256:
            raise ValueError("Atlas efficacy bundle does not match the verified checkpoint.")
        bundle = json.loads(raw)
        if bundle["schema"] != "supported-aso-ridge-v1" or bundle["chemistry"] != CHEMISTRY:
            raise ValueError("The Atlas efficacy bundle has an unsupported chemistry.")
        self.assay = AssayProfile(**bundle["assay"])
        self.coefficients = tuple(bundle["coefficients"])
        self.intercept = bundle["intercept"]
        if len(self.coefficients) != 80 or not all(
            isfinite(value) for value in (*self.coefficients, self.intercept)
        ):
            raise ValueError("Invalid fitted Atlas efficacy model parameters.")

    def predict(self, candidate: Candidate) -> float:
        """Score the exact full-PS 5-10-5 MOE profile in the bundle's assay."""
        sequence = candidate.sequence
        if (
            candidate.chemistry != CHEMISTRY
            or len(sequence) != 20
            or candidate.helm != build_helm(sequence, CHEMISTRY)
        ):
            raise ValueError("Candidate is outside the Atlas efficacy model chemistry.")
        # Position-major ACGT one-hot coefficients, in the original model's order.
        weights = (
            self.coefficients[4 * position + "ACGT".index(base)]
            for position, base in enumerate(sequence)
        )
        return self.intercept + sum(weights)
