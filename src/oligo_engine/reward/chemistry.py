"""Strict single-strand parsing for the explicitly supported Atlas vocabulary.

Selectively ported from codex/atlas-lora-scaling at
1b3fc1e119146739746dc94e339e31356cdec9e4; no model or study code imported.

Source adapter inspected at bb90c946013f183379da04678340e193088e5f99.
Unlike its permissive parser, every token must be consumed. Raw HELM is retained.
Absent internal linkage token means PO under the Atlas release convention; the
last nucleotide has END, never an invented inter-nucleotide linkage.
"""

import hashlib
import json
import re
from dataclasses import dataclass
from functools import lru_cache

SUGARS = {"d": "DNA", "moe": "MOE", "cet": "cEt", "r": "RNA", "m": "OMe", "": "RNA"}
BASES = {"A": "A", "C": "C", "G": "G", "T": "T", "U": "U", "5meC": "C"}
TOKEN = re.compile(
    r"(?P<sugar>\[[A-Za-z]+\]|[drm])?\((?P<base>\[5meC\]|[ACGTU])\)(?P<link>\[sp\]|p)?"
)


@dataclass(frozen=True)
class ASOChemistry:
    raw: str
    bases: tuple[str, ...]
    sugars: tuple[str, ...]
    links: tuple[str, ...]
    terminal: str

    @property
    def natural_rna(self) -> str:
        return "".join(BASES[b] for b in self.bases).replace("T", "U")

    @property
    def identity(self) -> str:
        value = [self.bases, self.sugars, self.links, self.terminal]
        return hashlib.sha256(json.dumps(value).encode()).hexdigest()


@lru_cache(maxsize=180000)
def parse_helm(helm: str) -> ASOChemistry:
    if not isinstance(helm, str):
        raise ValueError("HELM must be a string")
    if helm.startswith("RNA1{{") and helm.endswith("}}$$$$"):
        body = helm[6:-6]
    elif helm.startswith("RNA1{") and helm.endswith("}$$$$"):
        body = helm[5:-5]
    else:
        raise ValueError("Expected exactly one RNA1 polymer without connections")
    tokens = body.split(".")
    if not 1 <= len(tokens) <= 64:
        raise ValueError("Supported ASO length is 1–64; no truncation")
    bases, sugars, links = [], [], []
    for token in tokens:
        match = TOKEN.fullmatch(token)
        if match is None:
            raise ValueError(f"Unsupported nucleotide token: {token}")
        sugar = (match["sugar"] or "").strip("[]").lower()
        if sugar not in SUGARS:
            raise ValueError(f"Unsupported sugar: {sugar}")
        bases.append(match["base"].strip("[]"))
        sugars.append(SUGARS[sugar])
        links.append("PS" if match["link"] == "[sp]" else "PO")
    terminal = (
        "PS"
        if tokens[-1].endswith("[sp]")
        else ("PO" if tokens[-1].endswith("p") else "none")
    )
    return ASOChemistry(
        helm, tuple(bases), tuple(sugars), tuple(links[:-1]) + ("END",), terminal
    )
