"""Candidate ASO generation: tile a transcript, build HELM, score sequence heuristics.

Everything here is label-free and deterministic. Nothing in this module consults a
measured label, a model, or a held-out partition: it turns a target transcript into
candidate oligonucleotides and computes sequence-derived properties of them.

Two chemistries are supported, each chosen to match a training set exactly rather
than to be generally expressive:

* ``ps_moe_gapmer`` - 5-10-4 2'-MOE wings over a DNA core, full phosphorothioate,
  5-methylcytosine throughout. This is the chemistry of ``MOE_Neurotox_1`` and the
  chemistry Dr. Sun identified as what the field actually runs.
* ``ps_2ome_gapmer`` - 5-10-5 2'-O-methyl wings over a DNA core, full
  phosphorothioate. This is the chemistry of the Alharbi 2020 TLR screens.

A predictor must only be applied to candidates in the chemistry it was fitted on;
``engine.py`` enforces that and labels anything else as out-of-chemistry rather than
silently extrapolating.

The heuristic terms are **literature-motivated flags, not fitted model outputs**, and
they are reported separately from predictions for that reason.
"""

from dataclasses import dataclass

from oligo_engine.contracts import Observation

COMPLEMENT = str.maketrans("ACGTU", "TGCAA")

# Each chemistry's design is pinned to the dominant design in the dataset the
# matching predictor was fitted on, because the fitted featurizer has a fixed width
# and a candidate outside it cannot be encoded at all.
#
# * ps_moe_gapmer: MOE_Neurotox_1's most common molecule is a 19-mer with a 5-monomer
#   5' wing, a 10-monomer DNA core and a 4-monomer 3' wing (2,006 of 2,437 rows).
# * ps_2ome_gapmer: every one of Alharbi 2020's 192 ASOs is a 20-mer 5-10-5.
DESIGNS = {
    "ps_moe_gapmer": {"length": 19, "wing5": 5, "wing3": 4},
    "ps_2ome_gapmer": {"length": 20, "wing5": 5, "wing3": 5},
}
CHEMISTRIES = tuple(DESIGNS)


@dataclass(frozen=True)
class Candidate:
    """One designed ASO, its target window, and its label-free properties."""

    candidate_id: str
    gene: str
    transcript_id: str
    start: int  # 0-based inclusive index into the supplied transcript
    end: int  # exclusive
    target_window: str  # transcript sense strand, 5'->3'
    sequence: str  # the ASO itself, 5'->3', DNA letters
    chemistry: str
    helm: str
    gc_fraction: float
    cpg_count: int
    guanine_distance_from_3prime: int | None
    longest_g_run: int
    hairpin_dg: float | None = None
    target_accessibility: float | None = None

    def observation(self) -> Observation:
        return Observation(
            sample_id=self.candidate_id,
            sequence=self.helm,
            target_id=self.gene,
            modality="ASO",
            chemistry=self.chemistry,
            assay="designed_candidate_unmeasured",
            context=(
                ("transcript_id", self.transcript_id),
                ("start", str(self.start)),
                ("end", str(self.end)),
            ),
        )


def reverse_complement(sequence: str) -> str:
    """Reverse complement in DNA letters; U in the input is read as T."""
    cleaned = sequence.upper()
    if set(cleaned) - set("ACGTUN"):
        raise ValueError("Transcript may only contain A, C, G, T, U or N")
    return cleaned.translate(COMPLEMENT)[::-1]


def build_helm(
    sequence: str,
    chemistry: str,
    wing5: int | None = None,
    wing3: int | None = None,
) -> str:
    """Build the HELM string for one ASO in a supported gapmer chemistry."""
    if chemistry not in CHEMISTRIES:
        raise ValueError(f"Unsupported chemistry {chemistry!r}; use {CHEMISTRIES}")
    if set(sequence) - set("ACGT"):
        raise ValueError("ASO sequence must be A, C, G or T")
    design = DESIGNS[chemistry]
    wing5 = design["wing5"] if wing5 is None else wing5
    wing3 = design["wing3"] if wing3 is None else wing3
    if len(sequence) <= wing5 + wing3:
        raise ValueError("Sequence must be longer than its two wings combined")
    units = []
    last = len(sequence) - 1
    for index, base in enumerate(sequence):
        in_wing = index < wing5 or index > last - wing3
        if chemistry == "ps_moe_gapmer":
            # Every cytosine in MOE_Neurotox_1 is 5-methylcytosine, in core and wing.
            symbol = "[m5C]" if base == "C" else base
            sugar = "[moe]" if in_wing else "d"
        else:
            # Alharbi wings are 2'OMe RNA, so thymine is written as uracil there.
            symbol = "U" if (in_wing and base == "T") else base
            sugar = "m" if in_wing else "d"
        linkage = "[sp]" if index < last else ""
        units.append(f"{sugar}({symbol}){linkage}")
    return "RNA1{" + ".".join(units) + "}$$$$"


def gc_fraction(sequence: str) -> float:
    return (sequence.count("G") + sequence.count("C")) / len(sequence)


def cpg_count(sequence: str) -> int:
    """CpG dinucleotides: the motif class associated with immunostimulation."""
    return sum(1 for i in range(len(sequence) - 1) if sequence[i : i + 2] == "CG")


def guanine_distance_from_3prime(sequence: str) -> int | None:
    """Monomers between the 3' end and the nearest guanine; None if there is none.

    A shorter distance correlates with acute neurotoxicity in the literature. This
    is a flag taken from published association, not something fitted here.
    """
    for offset, base in enumerate(reversed(sequence)):
        if base == "G":
            return offset
    return None


def longest_g_run(sequence: str) -> int:
    best = run = 0
    for base in sequence:
        run = run + 1 if base == "G" else 0
        best = max(best, run)
    return best


def hairpin_dg(sequence: str) -> float | None:
    """Minimum free energy of the oligo folding on itself, via ViennaRNA.

    ViennaRNA carries RNA parameters, so for a DNA-core gapmer this is an
    approximation used as a relative flag between candidates, not a thermodynamic
    measurement of the real molecule. Returns None if ViennaRNA is unavailable.
    """
    try:
        import RNA
    except ImportError:
        return None
    return float(RNA.fold(sequence.replace("T", "U"))[1])


def tile(
    gene: str,
    transcript_id: str,
    transcript: str,
    chemistry: str = "ps_moe_gapmer",
    length: int | None = None,
    step: int = 1,
    *,
    start_at: int = 0,
    stop_at: int | None = None,
) -> list[Candidate]:
    """Every ASO of ``length`` complementary to a window of the transcript.

    Windows containing N are skipped: an ambiguous base cannot be encoded, and
    guessing one would invent a molecule the transcript does not specify.
    """
    if chemistry not in CHEMISTRIES:
        raise ValueError(f"Unsupported chemistry {chemistry!r}; use {CHEMISTRIES}")
    design = DESIGNS[chemistry]
    length = design["length"] if length is None else length
    sequence = transcript.upper()
    if length <= design["wing5"] + design["wing3"] or step < 1:
        raise ValueError("Length must exceed both wings and step must be positive")
    if start_at < 0 or (stop_at is not None and stop_at < start_at):
        raise ValueError("Invalid tiling bounds")
    max_start = len(sequence) - length + 1
    end = max_start if stop_at is None else min(max_start, stop_at)
    candidates = []
    for start in range(start_at, end, step):
        window = sequence[start : start + length]
        if "N" in window:
            continue
        aso = reverse_complement(window)
        candidates.append(
            Candidate(
                candidate_id=f"{gene}:{transcript_id}:{start}:{chemistry}",
                gene=gene,
                transcript_id=transcript_id,
                start=start,
                end=start + length,
                target_window=window,
                sequence=aso,
                chemistry=chemistry,
                helm=build_helm(aso, chemistry),
                gc_fraction=gc_fraction(aso),
                cpg_count=cpg_count(aso),
                guanine_distance_from_3prime=guanine_distance_from_3prime(aso),
                longest_g_run=longest_g_run(aso),
            )
        )
    return candidates


def target_accessibility(
    transcript: str, start: int, end: int, flank: int = 70
) -> float | None:
    """Mean unpaired probability of the target window, folded in local context.

    The v5 plan listed target-site accessibility as a stretch item because OligoGym
    supplies a target identifier rather than a transcript. With the transcript in
    hand it is a local partition-function fold: higher means the site is more often
    single-stranded and so more reachable. Returns None without ViennaRNA.

    Only ``flank`` bases either side are folded, so the number is a local estimate,
    not the equilibrium structure of a full-length mRNA.
    """
    try:
        import RNA
    except ImportError:
        return None
    sequence = transcript.upper().replace("T", "U")
    left = max(0, start - flank)
    local = sequence[left : min(len(sequence), end + flank)]
    if "N" in local or len(local) < (end - start):
        return None
    fold = RNA.fold_compound(local)
    fold.pf()
    pairs = fold.bpp()  # 1-indexed upper-triangular probabilities
    size = len(local)
    paired = [0.0] * (size + 1)
    for i in range(1, size + 1):
        for j in range(i + 1, size + 1):
            probability = pairs[i][j]
            if probability:
                paired[i] += probability
                paired[j] += probability
    offset = start - left
    window = range(offset + 1, offset + 1 + (end - start))
    return float(sum(1.0 - min(1.0, paired[i]) for i in window) / (end - start))


def matches_chemistry(
    sequence_helm: str,
    sugars: tuple[str, ...],
    phosphates: tuple[str, ...],
    bases: tuple[str, ...],
    max_length: int,
) -> bool:
    """True when a HELM string sits inside a declared chemistry and length budget.

    Used both to select the training rows a deployment predictor may be fitted on
    and to check candidates before scoring, so the two definitions cannot drift.
    """
    from oligogym.features import _extract_monomers

    monomers = _extract_monomers(sequence_helm)
    return (
        set(monomers["sugar"]) <= set(sugars)
        and set(monomers["phosphate"]) <= set(phosphates)
        and set(monomers["base"]) <= set(bases)
        and len(monomers["base"]) <= max_length
    )


# MOE_Neurotox_1 annotates every cytosine as 5-methylcytosine; ASOptimizer writes
# plain C for the same PS/2'-MOE chemistry. The molecules are the same and the
# conventions are not, so a candidate is rewritten into whichever convention the
# receiving model was fitted on. This reconciles annotation, never chemistry: sugar,
# backbone and length are untouched and are still checked separately.
BASE_ANNOTATIONS = ("m5C", "plain")


def annotate_bases(helm: str, annotation: str) -> str:
    """Rewrite cytosine annotation in a HELM string without altering chemistry."""
    if annotation not in BASE_ANNOTATIONS:
        raise ValueError(f"Unknown annotation {annotation!r}; use {BASE_ANNOTATIONS}")
    if annotation == "plain":
        return helm.replace("([m5C])", "(C)")
    return helm.replace("(C)", "([m5C])")
