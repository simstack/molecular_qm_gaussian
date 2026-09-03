"""Gaussian input writer and output parser.

Replaces ``pymatgen.io.gaussian.GaussianInput`` / ``GaussianOutput`` for the
subset of the API used by the Gaussian node.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional

from molecular_qm_models.molecule import Molecule

_ATOMIC_SYMBOLS = (
    "X",
    "H", "He",
    "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt",
    "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf",
    "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
)

_CHARGE_MULT_RE = re.compile(
    r"Charge\s*=\s*([+-]?\d+)\s+Multiplicity\s*=\s*(\d+)",
    re.IGNORECASE,
)
_SCF_DONE_RE = re.compile(r"SCF Done:\s+\S+\s+=\s+([+-]?\d+\.\d+)")
_ORIENT_ATOM_RE = re.compile(
    r"^\s+\d+\s+(\d+)\s+\d+\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)"
)


def _symbol_from_atomic_number(z: int) -> str:
    if 0 < z < len(_ATOMIC_SYMBOLS):
        return _ATOMIC_SYMBOLS[z]
    return f"X{z}"


def _route_dict_to_string(route_parameters: Optional[Mapping[str, Any]]) -> str:
    if not route_parameters:
        return ""
    parts = []
    for key, value in route_parameters.items():
        if value is None or value == "":
            parts.append(str(key))
        else:
            parts.append(f"{key}={value}")
    return " " + " ".join(parts)


class GaussianInput:
    """Minimal Gaussian ``.com`` writer used by the Gaussian node."""

    def __init__(
        self,
        mol: Molecule,
        charge: Optional[int] = None,
        spin_multiplicity: Optional[int] = None,
        title: str = "Gaussian Calculation",
        functional: Optional[str] = None,
        basis_set: Optional[str] = None,
        route_parameters: Optional[Mapping[str, Any]] = None,
        link0_parameters: Optional[Mapping[str, Any]] = None,
        dieze_tag: str = "P",
    ):
        self.mol = mol
        self.charge = mol.charge if charge is None else charge
        self.spin_multiplicity = (
            mol.spin_multiplicity if spin_multiplicity is None else spin_multiplicity
        )
        self.title = title
        self.functional = functional
        self.basis_set = basis_set
        self.route_parameters = dict(route_parameters or {})
        self.link0_parameters = dict(link0_parameters or {})
        self.dieze_tag = dieze_tag

    def to_str(self, cart_coords: bool = True) -> str:
        lines = []
        for key, value in self.link0_parameters.items():
            lines.append(f"{key}={value}")

        route = f"#{self.dieze_tag}"
        if self.functional and self.basis_set:
            route += f" {self.functional}/{self.basis_set}"
        elif self.functional:
            route += f" {self.functional}"
        route += _route_dict_to_string(self.route_parameters)
        lines.append(route)
        lines.append("")
        lines.append(self.title)
        lines.append("")
        lines.append(f"{int(self.charge)} {int(self.spin_multiplicity)}")
        for atom in self.mol.atoms:
            if cart_coords:
                lines.append(
                    f"{atom.element} {atom.x:.10f} {atom.y:.10f} {atom.z:.10f}"
                )
            else:
                lines.append(str(atom.element))
        lines.append("")
        return "\n".join(lines) + "\n"

    def write_file(self, filename: str, cart_coords: bool = True) -> None:
        with open(filename, "w", encoding="utf-8") as handle:
            handle.write(self.to_str(cart_coords=cart_coords))


class GaussianOutput:
    """Minimal Gaussian log parser used by the Gaussian node."""

    def __init__(self, filename: str):
        self.filename = filename
        with open(filename, "r", encoding="utf-8", errors="replace") as handle:
            self._text = handle.read()
        self.charge = 0
        self.spin_multiplicity = 1
        self.energies: list[float] = []
        self.properly_terminated = "Normal termination of Gaussian" in self._text
        self.final_structure: Optional[Molecule] = None
        self._parse()

    @property
    def final_energy(self) -> Optional[float]:
        if self.energies:
            return self.energies[-1]
        return None

    def _parse(self) -> None:
        charge_match = _CHARGE_MULT_RE.search(self._text)
        if charge_match:
            self.charge = int(charge_match.group(1))
            self.spin_multiplicity = int(charge_match.group(2))

        self.energies = [float(v) for v in _SCF_DONE_RE.findall(self._text)]

        last_block = None
        for label in ("Standard orientation:", "Input orientation:"):
            idx = self._text.rfind(label)
            if idx != -1:
                last_block = self._text[idx:]
                break
        if last_block is None:
            return

        species = []
        coords = []
        in_table = False
        dashed = 0
        for line in last_block.splitlines():
            stripped = line.strip()
            if stripped and set(stripped) == {"-"}:
                dashed += 1
                if dashed == 2:
                    in_table = True
                elif dashed >= 3:
                    break
                continue
            if not in_table:
                continue
            match = _ORIENT_ATOM_RE.match(line)
            if match:
                species.append(_symbol_from_atomic_number(int(match.group(1))))
                coords.append(
                    [float(match.group(2)), float(match.group(3)), float(match.group(4))]
                )

        if species:
            self.final_structure = Molecule.from_sites(species, coords)
            self.final_structure.properties["charge"] = self.charge
            self.final_structure.properties["spin_multiplicity"] = self.spin_multiplicity
