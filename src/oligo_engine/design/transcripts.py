"""Fetch a target transcript by gene symbol from Ensembl, with an on-disk cache.

The predictors are sequence models: they need the transcript the ASO is designed
against, which OligoGym does not ship (it carries a target identifier only). This is
the "new surface area" the v5 plan flagged, kept deliberately small: one lookup, one
sequence fetch, cached to disk so a demo runs offline after the first call.

Only a gene symbol and species leave this machine. No user identity, no API key and
no request body is sent, and the User-Agent is a generic client string.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

ENSEMBL = "https://rest.ensembl.org"
USER_AGENT = "oligo-design-engine/0.1 (research prototype)"
CACHE = Path(__file__).resolve().parents[3] / "artifacts" / "transcripts"
SYMBOL = re.compile(r"^[A-Za-z0-9._-]{1,32}$")
TIMEOUT = 20


class TranscriptError(RuntimeError):
    """The transcript could not be resolved or fetched."""


@dataclass(frozen=True)
class Transcript:
    gene: str
    transcript_id: str
    sequence: str
    species: str
    source: str
    description: str = ""

    def as_dict(self) -> dict:
        return {
            "gene": self.gene,
            "transcript_id": self.transcript_id,
            "length": len(self.sequence),
            "species": self.species,
            "source": self.source,
            "description": self.description,
        }


def _get(path: str) -> dict:
    import requests

    response = requests.get(
        f"{ENSEMBL}{path}",
        headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
        timeout=TIMEOUT,
    )
    if response.status_code != 200:
        raise TranscriptError(
            f"Ensembl returned {response.status_code} for {path}: {response.text[:200]}"
        )
    return response.json()


def cache_path(gene: str, species: str) -> Path:
    return CACHE / f"{species}.{gene.upper()}.json"


def fetch(gene: str, species: str = "homo_sapiens", refresh: bool = False):
    """Return the canonical cDNA for ``gene``, from cache when available."""
    if not SYMBOL.match(gene):
        raise TranscriptError(f"Invalid gene symbol {gene!r}")
    path = cache_path(gene, species)
    if path.exists() and not refresh:
        stored = json.loads(path.read_text())
        return Transcript(**stored)

    lookup = _get(f"/lookup/symbol/{species}/{gene.upper()}?expand=1")
    transcripts = lookup.get("Transcript") or []
    if not transcripts:
        raise TranscriptError(f"No transcripts found for {gene!r} in {species}")
    # Ensembl flags exactly one canonical transcript; fall back to the longest.
    canonical = next((t for t in transcripts if t.get("is_canonical")), None)
    if canonical is None:
        canonical = max(transcripts, key=lambda t: t.get("end", 0) - t.get("start", 0))
    sequence = _get(f"/sequence/id/{canonical['id']}?type=cdna")["seq"].upper()
    if not sequence or set(sequence) - set("ACGTN"):
        raise TranscriptError(f"Unexpected sequence content for {canonical['id']}")
    transcript = Transcript(
        gene=lookup.get("display_name", gene.upper()),
        transcript_id=canonical["id"],
        sequence=sequence,
        species=species,
        source=f"Ensembl REST {ENSEMBL}",
        description=(lookup.get("description") or "").split("[")[0].strip(),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(transcript.__dict__, indent=2) + "\n")
    return transcript


def from_sequence(sequence: str, gene: str = "PASTED", transcript_id: str = "pasted"):
    """Wrap a pasted sequence so the surface works with no network at all."""
    cleaned = "".join(sequence.split()).upper()
    if not cleaned.startswith(">") and (set(cleaned) - set("ACGTUN")):
        raise TranscriptError("Pasted sequence must contain only A, C, G, T, U or N")
    if cleaned.startswith(">"):
        cleaned = "".join(
            line for line in sequence.splitlines() if not line.startswith(">")
        )
        cleaned = "".join(cleaned.split()).upper()
    if len(cleaned) < 40:
        raise TranscriptError("Pasted sequence is too short to tile")
    return Transcript(
        gene=gene,
        transcript_id=transcript_id,
        sequence=cleaned.replace("U", "T"),
        species="supplied",
        source="pasted by the user",
    )
