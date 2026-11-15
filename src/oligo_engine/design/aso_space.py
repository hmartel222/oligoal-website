"""Label-free, offline fixed-template ASO target-site search."""

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import IntEnum
from functools import cached_property

from oligo_engine.design.candidates import (
    Candidate,
    build_helm,
    reverse_complement,
    tile,
)


def identity(value: object) -> str:
    """Hash JSON-compatible content without depending on Python hash seeds."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class Action(IntEnum):
    STOP = 0
    LEFT = 1
    RIGHT = 2


@dataclass(frozen=True)
class TargetContext:
    gene: str
    transcript_id: str
    sequence: str
    species: str
    source: str
    transcript_type: str = "cdna"
    coordinates: str = "zero_based_half_open"

    def __post_init__(self) -> None:
        if not all((self.gene, self.transcript_id, self.species, self.source)):
            raise ValueError("Versioned transcript, gene, species and source required")
        if not self.sequence or set(self.sequence) - set("ACGTN"):
            raise ValueError("Pinned transcript must be uppercase A/C/G/T/N")
        if self.coordinates != "zero_based_half_open" or not self.transcript_type:
            raise ValueError("Unsupported coordinate convention or transcript type")

    @property
    def identity(self) -> str:
        return identity(asdict(self))


@dataclass(frozen=True)
class ASODesignSpace:
    target: TargetContext
    start: int
    radius: int = 8
    horizon: int = 8

    def __post_init__(self) -> None:
        if any(type(v) is not int for v in (self.start, self.radius, self.horizon)):
            raise ValueError("Integer space bounds required")
        if self.radius < 0 or self.horizon < 1:
            raise ValueError("Nonnegative radius and positive horizon required")
        if self.start < 0 or self.start not in self.candidates:
            raise ValueError("Starting window is unconstructable or out of bounds")

    @cached_property
    def candidates(self) -> dict[int, Candidate]:
        # Candidate construction is deterministic; no transcript network calls.
        reach = min(self.radius, self.horizon)
        return {
            c.start: c
            for c in tile(
                self.target.gene,
                self.target.transcript_id,
                self.target.sequence,
                start_at=max(0, self.start - reach),
                stop_at=self.start + reach + 1,
            )
        }

    @property
    def identity(self) -> str:
        return identity({"schema": "aso-space-v1", **asdict(self)})

    def reachable(self) -> tuple[int, ...]:
        """Connected component reachable from the start, including N barriers."""
        sites = self.candidates
        seen, pending = {self.start}, [self.start]
        while pending:
            position = pending.pop()
            for neighbor in (position - 1, position + 1):
                if neighbor in sites and neighbor not in seen:
                    seen.add(neighbor)
                    pending.append(neighbor)
        return tuple(sorted(seen))

    def candidate(self, position: int) -> Candidate:
        if position not in self.reachable():
            raise ValueError("Site outside reachable design region")
        return self.candidates[position]

    def mask(self, position: int, moves: int) -> tuple[bool, bool, bool]:
        self.candidate(position)
        if not 0 <= moves <= self.horizon:
            raise ValueError("Invalid move count")
        sites = self.candidates
        return (
            True,
            moves < self.horizon and position - 1 in sites,
            moves < self.horizon and position + 1 in sites,
        )

    def transition(self, position: int, moves: int, action: int) -> int:
        if action not in tuple(Action) or not self.mask(position, moves)[action]:
            raise ValueError("Illegal design action")
        return position + {Action.STOP: 0, Action.LEFT: -1, Action.RIGHT: 1}[action]


def candidate_identity(candidate: Candidate, target: TargetContext) -> str:
    """Chemical/site identity is separate from the existing display identifier."""
    window = target.sequence[candidate.start : candidate.end]
    if (
        candidate.gene != target.gene
        or candidate.transcript_id != target.transcript_id
        or candidate.start < 0
        or candidate.end > len(target.sequence)
        or candidate.end - candidate.start != 19
        or len(window) != 19
        or set(window) - set("ACGT")
        or candidate.target_window != window
        or candidate.sequence != reverse_complement(window)
        or candidate.chemistry != "ps_moe_gapmer"
        or candidate.helm != build_helm(candidate.sequence, candidate.chemistry)
    ):
        raise ValueError("Candidate does not match pinned transcript/template")
    return identity(
        {"candidate": asdict(candidate), "target_identity": target.identity}
    )
