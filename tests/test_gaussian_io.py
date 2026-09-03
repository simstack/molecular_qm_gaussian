from pathlib import Path

from molecular_qm_models import Molecule

from molecular_qm_gaussian.lib.gaussian_excited_states_parser import (
    parse_gaussian_excited_states_file,
)
from molecular_qm_gaussian.lib.gaussian_io import GaussianInput, GaussianOutput

DATA = Path(__file__).parent / "data"


def _water() -> Molecule:
    return Molecule.from_sites(
        ["O", "H", "H"],
        [
            [0.0, 0.0, 0.1173],
            [0.0, 0.7572, -0.4692],
            [0.0, -0.7572, -0.4692],
        ],
    )


def test_gaussian_input_writes_route_and_geometry(tmp_path):
    mol = _water()
    gin = GaussianInput(
        mol=mol,
        charge=0,
        spin_multiplicity=1,
        functional="b3lyp",
        basis_set="6-31g",
        route_parameters={"opt": "tight", "nosymm": ""},
        link0_parameters={"%mem": "2GB", "%nprocshared": "4", "%chk": "gaussian.chk"},
    )
    text = gin.to_str()
    assert "%mem=2GB" in text
    assert "#P b3lyp/6-31g opt=tight nosymm" in text
    assert "0 1" in text
    assert "O " in text
    path = tmp_path / "gaussian.com"
    gin.write_file(str(path))
    assert path.read_text(encoding="utf-8") == text


def test_gaussian_output_parses_energy_and_geometry():
    gout = GaussianOutput(str(DATA / "sample.log"))
    assert gout.properly_terminated
    assert gout.final_energy == -76.123456
    assert gout.charge == 0
    assert gout.spin_multiplicity == 1
    assert gout.final_structure is not None
    assert len(gout.final_structure.atoms) == 3
    assert gout.final_structure.atoms[0].element == "O"


def test_excited_states_parser():
    states, transitions = parse_gaussian_excited_states_file(str(DATA / "sample.log"))
    assert len(states.row) == 2
    assert states.row[0]["number"] == 1
    assert abs(states.row[0]["energy_ev"] - 5.1234) < 1e-6
    assert len(transitions.row) == 2
    assert transitions.row[0]["orb1"] == 5
    assert transitions.row[0]["orb2"] == 6
