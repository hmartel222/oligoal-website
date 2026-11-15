"""Stateless, label-free adapters over the pinned OligoGym featurizers.

OligoGym is imported lazily and verified loudly: a missing or mismatched upstream
environment raises ``UpstreamEnvironmentError`` rather than degrading silently.
Nothing here is fitted, so no featurizer state can depend on revealed, unrevealed or
held-out labels, or on which rows a predictor was fitted with.
"""

import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from oligo_engine.audit_upstream import PINNED_REVISION
from oligo_engine.contracts import Observation

REQUIRED_SKLEARN = "1.4.0"
# Canonical nucleobase alphabet: a data-independent constant, so the encoding does not
# depend on which rows are visible when a model is fitted.
BASE_ALPHABET = ("A", "C", "G", "T", "U")
# MOE neurotox ASOs carry 5-methylcytosine rather than plain cytosine. The alphabet
# stays a declared constant: plain C is kept so the symbol set does not depend on the
# rows in any particular dataset or split.
MODIFIED_BASE_ALPHABET = ("A", "C", "G", "T", "U", "m5C")
# Sugar and phosphate symbols spell out the gapmer design (2'-MOE wings, DNA core)
# and the backbone (sp = phosphorothioate, p = phosphodiester). "EMPTY" is the
# terminal position that carries no phosphate; it is an explicit symbol, never an
# unknown silently encoded as zeros.
SUGAR_ALPHABET = ("d", "r", "moe")
PHOSPHATE_ALPHABET = ("p", "sp", "EMPTY")
# Upstream's OneHotEncoder reorders components; mirror that order in the config.
UPSTREAM_COMPONENT_ORDER = ("phosphate", "sugar", "base")
# Huesken siRNAs are two 21-monomer strands (RNA1 + RNA2).
HUESKEN_MAX_SEQUENCE_LENGTH = 42
# MOE neurotox ASOs are a single strand of 15-19 monomers.
MOE_NEUROTOX_MAX_SEQUENCE_LENGTH = 19


class UpstreamEnvironmentError(RuntimeError):
    """The pinned OligoGym environment is absent or does not match the audit."""


def verify_upstream_environment() -> dict:
    """Return identity of the upstream environment or raise; never skip silently."""
    try:
        import oligogym
    except ImportError as error:
        raise UpstreamEnvironmentError(
            "OligoGym is not importable. Install the pinned upstream environment "
            "(docs/upstream-verification.md, docs/gate1-resolution.md)."
        ) from error
    try:
        sklearn_version = version("scikit-learn")
    except PackageNotFoundError as error:
        raise UpstreamEnvironmentError("scikit-learn is not installed") from error
    if sklearn_version != REQUIRED_SKLEARN:
        raise UpstreamEnvironmentError(
            f"scikit-learn {REQUIRED_SKLEARN} is required, found {sklearn_version}"
        )
    package_dir = Path(oligogym.__file__).resolve().parent
    result = subprocess.run(
        ["git", "-C", str(package_dir), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    head = result.stdout.strip() if result.returncode == 0 else None
    if head != PINNED_REVISION:
        raise UpstreamEnvironmentError(
            f"OligoGym must be a git checkout at {PINNED_REVISION}; found {head}"
        )
    return {
        "oligogym_revision": head,
        "scikit_learn": sklearn_version,
        "oligogym_path": str(package_dir),
    }


class OneHotBaseFeaturizer:
    """OligoGym ``OneHotEncoder`` over nucleobases with a fixed map and padding."""

    def __init__(
        self,
        max_sequence_length: int = HUESKEN_MAX_SEQUENCE_LENGTH,
        alphabet: tuple[str, ...] = BASE_ALPHABET,
        component_alphabets: dict[str, tuple[str, ...]] | None = None,
    ):
        """Encode ``base`` over ``alphabet``, or the components named explicitly.

        ``component_alphabets`` maps each encoded component ("base", "sugar",
        "phosphate") to its declared symbol set and supersedes ``alphabet``. Every
        alphabet is a module constant, never inferred from the rows being encoded.
        """
        verify_upstream_environment()
        from oligogym.features import OneHotEncoder

        if component_alphabets is None:
            component_alphabets = {"base": tuple(alphabet)}
        unknown_components = set(component_alphabets) - set(UPSTREAM_COMPONENT_ORDER)
        if unknown_components or not component_alphabets:
            raise ValueError(
                f"Encoded components must be a non-empty subset of "
                f"{list(UPSTREAM_COMPONENT_ORDER)}; got {sorted(component_alphabets)}"
            )
        self.component_alphabets = {
            component: tuple(component_alphabets[component])
            for component in UPSTREAM_COMPONENT_ORDER
            if component in component_alphabets
        }
        if max_sequence_length < 1 or any(
            len(set(symbols)) != len(symbols) or not symbols
            for symbols in self.component_alphabets.values()
        ):
            raise ValueError("Positive length and unique, non-empty alphabets required")
        self.max_sequence_length = max_sequence_length
        self.alphabet = self.component_alphabets.get("base", ())
        maps = {
            component: {
                symbol: [1 if i == j else 0 for j in range(len(symbols))]
                for i, symbol in enumerate(symbols)
            }
            for component, symbols in self.component_alphabets.items()
        }
        self._encoder = OneHotEncoder(
            encode_components=list(self.component_alphabets),
            encoding_maps=maps,
            max_sequence_length=max_sequence_length,
        )

    @property
    def encode_components(self) -> tuple[str, ...]:
        return tuple(self.component_alphabets)

    @property
    def n_features(self) -> int:
        width = sum(len(symbols) for symbols in self.component_alphabets.values())
        return self.max_sequence_length * width

    @property
    def config(self) -> dict:
        return {
            "featurizer": "oligogym.OneHotEncoder",
            "encode_components": list(self.encode_components),
            "alphabet": list(self.alphabet),
            "component_alphabets": {
                component: list(symbols)
                for component, symbols in self.component_alphabets.items()
            },
            "max_sequence_length": self.max_sequence_length,
            "n_features": self.n_features,
            "fitted": False,
        }

    def transform(self, observations: tuple[Observation, ...]):
        """Return an (n, n_features) float array; reject anything not representable."""
        import numpy as np
        from oligogym.features import _extract_monomers

        if not observations:
            raise ValueError("At least one observation is required")
        for row in observations:
            monomers = _extract_monomers(row.sequence)
            if len(monomers["base"]) > self.max_sequence_length:
                raise ValueError(
                    f"{row.sample_id}: {len(monomers['base'])} monomers exceed the "
                    f"fixed length {self.max_sequence_length}"
                )
            for component, symbols in self.component_alphabets.items():
                # Undeclared symbols (including "NA") would be encoded as all-zeros
                # by upstream; refuse them instead of silently losing the monomer.
                unknown = set(monomers[component]) - set(symbols)
                if unknown:
                    raise ValueError(
                        f"{row.sample_id}: unsupported {component} {sorted(unknown)}"
                    )
        features = self._encoder.transform([row.sequence for row in observations])
        features = np.asarray(features, dtype=np.float64)
        return features.reshape(len(observations), -1)
