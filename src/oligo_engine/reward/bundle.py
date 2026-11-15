"""Hash-checked trusted artifact boundary, without pickle or implicit model fitting."""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from oligo_engine.reward.scoring import EndpointManifest, FrozenEndpoint, Inference


@dataclass(frozen=True)
class TrustedArtifact:
    path: Path
    sha256: str
    source_revision: str
    reconstruction: str

    def verify(self) -> bytes:
        if not self.source_revision or not self.reconstruction:
            raise ValueError("Artifact source and reconstruction receipt required")
        raw = self.path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != self.sha256:
            raise ValueError("Artifact hash mismatch")
        return raw


def load_endpoint(
    manifest: EndpointManifest,
    artifact: TrustedArtifact,
    reconstruct: Callable[[bytes], Inference],
) -> FrozenEndpoint:
    """Use caller-reviewed reconstruction, without dynamic imports or unpickling."""
    if artifact.sha256 != manifest.artifact_hash:
        raise ValueError("Manifest and artifact identities disagree")
    predict = reconstruct(artifact.verify())
    if not callable(predict):
        raise ValueError("Reconstruction did not return an inference callable")
    return FrozenEndpoint(manifest, predict)
