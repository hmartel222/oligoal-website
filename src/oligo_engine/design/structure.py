"""An idealized ASO:RNA duplex, written as PDB for a viewer.

**This is a schematic, not a predicted structure.** No structure prediction is run
anywhere in this project. Coordinates come from textbook A-form helical parameters
applied to the designed sequence, so the picture shows length, register, gapmer
architecture and where flagged bases sit - and nothing about the real conformation
of any particular molecule. The PDB carries that warning in its REMARK records so
it travels with the file into PyMOL or anywhere else.

Per residue three pseudo-atoms are written: the backbone phosphate ``P``, the sugar
``C1'`` and a base centroid ``N1``. That is enough for a viewer to draw a recognisable
double helix and to colour by chemistry, and too little to be mistaken for an
all-atom model.
"""

from math import cos, pi, radians, sin

from oligo_engine.design.candidates import MOE_CHEMISTRIES

# Idealized A-form parameters (Arnott): the hybrid duplex an RNase H gapmer forms is
# intermediate between A and B, so these are indicative only.
RISE = 2.81
TWIST = radians(32.7)
PHOSPHATE_RADIUS = 8.8
SUGAR_RADIUS = 6.4
BASE_RADIUS = 3.2
# Angular offset of the complementary strand about the helix axis.
STRAND_OFFSET = radians(154.0)
PAIR = {"A": "U", "T": "A", "G": "C", "C": "G", "U": "A"}


def _atom(serial, name, residue, chain, index, x, y, z, element):
    return (
        # PDB column layout: 13-16 atom, 17 altLoc, 18-20 residue, 22 chain,
        # 23-26 sequence, 27 iCode, 31-38/39-46/47-54 coordinates, 77-78 element.
        f"ATOM  {serial:>5d} {name:<4s} {residue:>3s} {chain}{index:>4d}    "
        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00          {element:>2s}"
    )


def _residue_name(base: str, strand: str, in_wing: bool, chemistry: str) -> str:
    if strand == "B":
        return {"A": "  A", "C": "  C", "G": "  G", "U": "  U"}.get(base, "  N")
    if in_wing:
        return "MOE" if chemistry in MOE_CHEMISTRIES else "OMe"
    return {"A": " DA", "C": " DC", "G": " DG", "T": " DT"}.get(base, " DN")


def duplex_pdb(
    sequence: str,
    chemistry: str,
    wing5: int,
    wing3: int,
    title: str = "designed ASO",
    *,
    target_window: str | None = None,
    flank5: str = "",
    flank3: str = "",
) -> str:
    """Return PDB text for the ASO (chain A) paired with its target RNA (chain B).

    ``flank5`` and ``flank3`` are the transcript bases immediately 5' and 3' of the
    binding window. They are written as part of chain B so the picture shows a short
    ASO bound to a longer mRNA rather than a blunt duplex. Those bases are unpaired
    in reality; the helical path is continued through them purely to place them, and
    a REMARK records that.
    """
    if not sequence or set(sequence) - PAIR.keys():
        raise ValueError("A nucleotide sequence is required")
    expected = "".join(PAIR[base] for base in reversed(sequence))
    target = expected if target_window is None else target_window.replace("T", "U")
    if target != expected:
        raise ValueError("The ASO does not complement the supplied target window.")
    flank5 = flank5.upper().replace("T", "U")
    flank3 = flank3.upper().replace("T", "U")
    for flank in (flank5, flank3):
        if set(flank) - set("ACGUN"):
            raise ValueError("Transcript flanks must be RNA bases.")
    length = len(sequence)
    title = " ".join(title.split()).encode("ascii", "replace").decode()
    lines = [
        f"TITLE     IDEALIZED SCHEMATIC OF {title.upper()[:50]}",
        "REMARK   1 NOT A PREDICTED STRUCTURE. Coordinates are textbook A-form",
        "REMARK   1 helical parameters applied to the designed sequence. No",
        "REMARK   1 structure prediction, docking or energy minimisation was run.",
        "REMARK   1 Three pseudo-atoms per residue: P, C1', base centroid as N1.",
        f"REMARK   2 CHEMISTRY {chemistry}; wings {wing5} (5') and {wing3} (3');",
        f"REMARK   2 DNA core {length - wing5 - wing3} monomers; chain A is the ASO,",
        "REMARK   2 chain B the complementary target RNA, antiparallel.",
    ]
    if flank5 or flank3:
        lines += [
            f"REMARK   3 CHAIN B CARRIES {len(flank5)} TRANSCRIPT BASES 5' AND",
            f"REMARK   3 {len(flank3)} BASES 3' OF THE BINDING WINDOW. Those flanks are",
            "REMARK   3 single stranded in reality. The A-form path is continued",
            "REMARK   3 through them only to position them; their conformation here",
            "REMARK   3 carries no meaning at all.",
        ]
    serial = 1
    for index, base in enumerate(sequence):
        angle = index * TWIST
        height = index * RISE
        in_wing = index < wing5 or index > length - 1 - wing3
        residue = _residue_name(base, "A", in_wing, chemistry)
        for name, radius, element in (
            ("P", PHOSPHATE_RADIUS, "P"),
            ("C1'", SUGAR_RADIUS, "C"),
            ("N1", BASE_RADIUS, "N"),
        ):
            lines.append(
                _atom(
                    serial,
                    name,
                    residue,
                    "A",
                    index + 1,
                    radius * cos(angle),
                    radius * sin(angle),
                    height,
                    element,
                )
            )
            serial += 1
    # Chain B runs antiparallel: residue i of B pairs with residue (length-1-i) of A.
    rna = flank5 + target + flank3
    for index, base in enumerate(rna):
        # Pairing position on the chain A axis; negative and over-length values
        # place the unpaired flanks beyond either end of the duplex.
        partner = length - 1 + len(flank5) - index
        angle = partner * TWIST + STRAND_OFFSET
        height = partner * RISE
        residue = _residue_name(base, "B", False, chemistry)
        for name, radius, element in (
            ("P", PHOSPHATE_RADIUS, "P"),
            ("C1'", SUGAR_RADIUS, "C"),
            ("N1", BASE_RADIUS, "N"),
        ):
            lines.append(
                _atom(
                    serial,
                    name,
                    residue,
                    "B",
                    index + 1,
                    radius * cos(angle),
                    radius * sin(angle),
                    height,
                    element,
                )
            )
            serial += 1
    for start, residues in ((1, length), (1 + 3 * length, len(rna))):
        for index in range(residues):
            phosphate = start + 3 * index
            lines.append(f"CONECT{phosphate:>5}{phosphate + 1:>5}")
            lines.append(f"CONECT{phosphate + 1:>5}{phosphate + 2:>5}")
            if index + 1 < residues:
                lines.append(f"CONECT{phosphate:>5}{phosphate + 3:>5}")
    lines.append("END")
    return "\n".join(lines) + "\n"


def pymol_script(pdb: str, length: int, wing5: int, wing3: int) -> str:
    """A self-contained PyMOL script for the same schematic shown in the browser."""
    return "\n".join(
        (
            "# Idealized ASO and RNA duplex made from schematic pseudo-atoms.",
            "python",
            "cmd.delete('oligo_duplex')",
            f"cmd.read_pdbstr({pdb!r}, 'oligo_duplex')",
            "python end",
            "hide everything, oligo_duplex",
            "set_color oligo_wing, [0.43, 0.85, 0.70]",
            "set_color oligo_core, [0.38, 0.66, 0.98]",
            "set_color oligo_rna, [0.97, 0.67, 0.39]",
            "color oligo_rna, oligo_duplex and chain B",
            "color oligo_wing, oligo_duplex and chain A",
            "color oligo_core, oligo_duplex and chain A and resi "
            f"{wing5 + 1}-{length - wing3}",
            "show sticks, oligo_duplex",
            "show spheres, oligo_duplex and name P",
            "set stick_radius, 0.24, oligo_duplex",
            "set sphere_scale, 0.32, oligo_duplex",
            "orient oligo_duplex",
            "zoom oligo_duplex, 4",
            "",
        )
    )


def helix_turns(length: int) -> float:
    return length * TWIST / (2 * pi)
