"""Build an idealized heavy-atom ASO–RNA duplex for structural inspection.

The helical step and residue placements are derived from the crystal structure
1FIX and the wwPDB chemical component dictionary. Wing and phosphorothioate
placements are illustrative. This builder does no folding, docking, or dynamics.
"""

import json
from math import cos, dist, pi, sin, sqrt
from pathlib import Path

from oligo_engine.design.candidates import MOE_CHEMISTRIES

DATA = json.loads(Path(__file__).with_name("duplex_templates.json").read_text())
TWIST = DATA["twist"]
RISE = DATA["rise"]
PAIR = str.maketrans("ACGT", "UGCA")
HYDROGEN_BONDS = {
    "A": (("N6", "O4"), ("N1", "N3")),
    "T": (("O4", "N6"), ("N3", "N1")),
    "G": (("O6", "N4"), ("N1", "N3"), ("N2", "O2")),
    "C": (("N4", "O6"), ("N3", "N1"), ("O2", "N2")),
}


def vector(left, right):
    return tuple(a - b for a, b in zip(left, right))


def add(origin, direction, length):
    return tuple(a + length * b for a, b in zip(origin, direction))


def unit(direction):
    length = sqrt(sum(a * a for a in direction))
    return tuple(a / length for a in direction)


def cross(a, b):
    return a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]


def coordinates(point, step):
    x, y, z = point
    angle = step * TWIST
    return x*cos(angle) - y*sin(angle), x*sin(angle) + y*cos(angle), z + step*RISE


def residue_atoms(base, chain, wing, moe, first):
    code = base if chain == "B" else "D" + base
    source, edges = DATA["residues"][code]
    atoms = dict(source)
    bonds = list(edges)
    if chain == "A":
        if not first:
            _, *oxygen = atoms.pop("OP2")
            _, *phosphate = atoms["P"]
            atoms["SP2"] = ("S", *add(phosphate, unit(vector(oxygen, phosphate)), 1.93))
            bonds = [[("SP2" if name == "OP2" else name) for name in edge] for edge in bonds]
        if moe and base == "C":
            _, *c5 = atoms["C5"]
            outward = unit(add(vector(c5, atoms["C4"][1:]), vector(c5, atoms["C6"][1:]), 1))
            atoms["C5M"] = ("C", *add(c5, outward, 1.50))
            bonds.append(("C5", "C5M"))
        if wing:
            if not moe and base == "T":
                atoms.pop("C7", None)
                bonds = [edge for edge in bonds if "C7" not in edge]
            oxygen = DATA["wing_oxygen"]
            outward = unit(vector(oxygen, atoms["C2'"][1:]))
            tangent = unit(cross(outward, (0, 0, 1)))
            sidechain = [("O2'", "O", oxygen)]
            last = oxygen
            for index, (name, element, length) in enumerate(
                [("CM2", "C", 1.43), ("CE2", "C", 1.52), ("OE2", "O", 1.43), ("CT2", "C", 1.43)]
            ):
                if index and not moe:
                    break
                direction = unit(add(outward, tangent, 0.60 * (-1)**index))
                last = add(last, direction, length)
                sidechain.append((name, element, last))
            previous = "C2'"
            for name, element, xyz in sidechain:
                atoms[name] = (element, *xyz)
                bonds.append((previous, name))
                previous = name
    if first:
        atoms = {key: point for key, point in atoms.items() if key not in {"P", "OP1", "OP2"}}
        bonds = [edge for edge in bonds if all(name in atoms for name in edge)]
    return atoms, bonds


def pdb_atom(serial, name, residue, chain, position, xyz, element):
    x, y, z = xyz
    return (
        f"ATOM  {serial:5d}  {name:<3s} {residue:>3s} {chain}{position:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}"
    )


def duplex_pdb(sequence, chemistry, wing5, wing3, title="ASO-RNA duplex", *, target_window=None):
    """Return chain A (ASO) and chain B (RNA target window) in 5′→3′ order."""
    if not sequence or set(sequence) - set("ACGT"):
        raise ValueError("A nucleotide sequence is required.")
    expected = sequence.translate(PAIR)[::-1]
    target = expected if target_window is None else target_window.replace("T", "U")
    if target != expected:
        raise ValueError("The ASO does not complement the supplied target window.")
    length = len(sequence)
    moe = chemistry in MOE_CHEMISTRIES
    title = " ".join(title.split()).encode("ascii", "replace").decode()
    lines = [
        f"TITLE     TEMPLATE-BUILT HEAVY-ATOM MODEL OF {title[:35].upper()}",
        "REMARK   1 ILLUSTRATIVE COORDINATES DERIVED FROM PDB 1FIX AND WWPDB CCD.",
        "REMARK   1 NO FOLDING, DOCKING, ENERGY MINIMIZATION OR SIMULATION.",
        "REMARK   1 PS AND WING GEOMETRY IS APPROXIMATE. PS STEREOCHEMISTRY UNKNOWN.",
        f"REMARK   2 {chemistry}; CHAIN A IS ASO; CHAIN B IS TARGET RNA WINDOW.",
    ]
    serial = 1
    links = []
    residues = {}
    for chain, strand in (("A", sequence), ("B", target)):
        previous_oxygen = None
        for index, base in enumerate(strand):
            wing = chain == "A" and (index < wing5 or index >= length - wing3)
            name = base if chain == "B" else "MOE" if wing and moe else "OME" if wing else "D" + base
            atoms, bonds = residue_atoms(base, chain, wing, moe, not index)
            step = (length - 1 - index if chain == "A" else index) - (length - 1) / 2
            index_by_atom = {}
            for atom, (element, *position) in atoms.items():
                index_by_atom[atom] = serial
                lines.append(pdb_atom(serial, atom, name, chain, index + 1, coordinates(position, step), element))
                serial += 1
            if previous_oxygen:
                links.append((previous_oxygen, index_by_atom["P"]))
            links.extend((index_by_atom[a], index_by_atom[b]) for a, b in bonds)
            residues[(chain, index)] = index_by_atom
            previous_oxygen = index_by_atom["O3'"]
        lines.append("TER")
    for i, base in enumerate(sequence):
        a, b = residues["A", i], residues["B", length - 1 - i]
        for donor, acceptor in HYDROGEN_BONDS[base]:
            lines.append(f"REMARK   3 PAIR {a[donor]} {b[acceptor]}")
    lines.extend(f"CONECT{a:5d}{b:5d}" for a, b in links)
    lines.append("END")
    return "\n".join(lines) + "\n"


def pymol_script(pdb, length, wing5, wing3):
    """Load the downloadable model with the same strand colors in PyMOL."""
    return "\n".join((
        "# Template-built model of the ASO and its RNA binding window.",
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
        f"color oligo_core, oligo_duplex and chain A and resi {wing5 + 1}-{length - wing3}",
        "color red, oligo_duplex and elem O",
        "color blue, oligo_duplex and elem N",
        "color yellow, oligo_duplex and elem S",
        "show sticks, oligo_duplex",
        "set stick_radius, 0.17, oligo_duplex",
        "orient oligo_duplex",
        "zoom oligo_duplex, 4",
        "",
    ))


def helix_turns(length: int) -> float:
    return length * TWIST / (2 * pi)
