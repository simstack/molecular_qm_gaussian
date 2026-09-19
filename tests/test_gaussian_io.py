from pathlib import Path
from types import SimpleNamespace
import inspect

import pytest
from molecular_qm_models import Molecule

from molecular_qm_gaussian.lib.gaussian_excited_states_parser import (
    parse_gaussian_excited_states_file,
)
from molecular_qm_gaussian.lib.gaussian_io import GaussianInput, GaussianOutput
from molecular_qm_gaussian.nodes.gaussian import (
    GAUSSIAN_RESULT_FILES,
    _basis_set_name,
    _functional_name,
    _link0_parameters,
    gaussian as gaussian_node,
)

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


def test_link0_converts_slurm_mem_g_to_gb():
    params = SimpleNamespace(
        slurm_parameters=SimpleNamespace(mem="10G", tasks_per_node=8)
    )
    link0 = _link0_parameters(params)
    assert link0["%mem"] == "10GB"
    assert link0["%nprocshared"] == "8"


def test_link0_keeps_gaussian_mem_units():
    params = SimpleNamespace(
        slurm_parameters=SimpleNamespace(mem="2GB", tasks_per_node=None)
    )
    assert _link0_parameters(params)["%mem"] == "2GB"


def test_link0_rejects_empty_and_unknown_mem():
    with pytest.raises(ValueError, match="slurm_parameters.mem is empty"):
        _link0_parameters(SimpleNamespace(slurm_parameters=SimpleNamespace(mem="  ")))
    with pytest.raises(ValueError, match="slurm_parameters.mem"):
        _link0_parameters(SimpleNamespace(slurm_parameters=SimpleNamespace(mem="lots")))


def test_functional_and_basis_map_to_gaussian_keywords():
    assert _functional_name(SimpleNamespace(functional=SimpleNamespace(functional="PBE"))) == "PBEPBE"
    assert _functional_name(SimpleNamespace(functional=SimpleNamespace(functional="PBE0"))) == "PBE1PBE"
    assert _basis_set_name(SimpleNamespace(basis_set=SimpleNamespace(basis_set="def2-SVP"))) == "Def2SVP"
    gin = GaussianInput(
        mol=_water(),
        charge=0,
        spin_multiplicity=1,
        functional="PBEPBE",
        basis_set="Def2SVP",
        route_parameters={"opt": "tight", "freq": "", "pop": "full"},
    )
    assert "#P PBEPBE/Def2SVP opt=tight freq pop=full" in gin.to_str()
    with pytest.raises(ValueError, match="Functional"):
        _functional_name(SimpleNamespace(functional=SimpleNamespace(functional="GFN2-xTB")))
    with pytest.raises(ValueError, match="Basis set"):
        _basis_set_name(SimpleNamespace(basis_set=SimpleNamespace(basis_set="def2-mSVP")))


def test_scratch_inputs_omit_missing_checkpoint():
    source = inspect.getsource(gaussian_node)
    assert 'if Path("gaussian.chk").exists():' in source
    assert 'input_files.append("gaussian.chk")' in source
    assert GAUSSIAN_RESULT_FILES == ["gaussian.log", "gaussian.chk", "gaussian.fchk"]
    assert "GAUSSIAN_INPUT_FILES" not in source


def test_checkpoint_files_are_passed_into_qm_result():
    source = inspect.getsource(gaussian_node)
    assert "file_list = FileList()" in source
    assert "files=file_list" in source
    assert "Gaussian did not write gaussian.chk" in source
    assert "Gaussian did not write gaussian.fchk" in source
    assert 'node_runner.execute("formchk")' in source
    formchk_at = source.index('node_runner.execute("formchk")')
    retrieve_at = source.index("node_runner.retrieve(output_files=GAUSSIAN_RESULT_FILES)")
    assert formchk_at < retrieve_at
    assert "node_runner.result.files.append" not in source
