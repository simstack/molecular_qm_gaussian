import glob
import logging
from pathlib import Path

from molecular_qm_models import (
    GridType,
    Molecule,
    MoleculeList,
    QMInput,
    QMResult,
    SCFAccuracy,
)
from simstack.core.context import context
from simstack.core.definitions import TaskStatus
from simstack.core.node import node
from simstack.core.node_runner import NodeRunner
from simstack.core.simstack_result import SimstackResult
from simstack.models.file_list import FileList
from simstack.models.files import FileStack

from molecular_qm_gaussian.lib.gaussian_excited_states_parser import (
    parse_gaussian_excited_states_file,
)
from molecular_qm_gaussian.lib.gaussian_io import GaussianInput, GaussianOutput

logger = logging.getLogger("GaussianNode")

GAUSSIAN_RESULT_FILES = ["gaussian.log", "gaussian.chk"]

_DISPERSION_ROUTE = {
    "D2": "GD2",
    "D3": "GD3",
    "D3BJ": "GD3BJ",
}

# Model/ORCA names -> Gaussian 16 route keywords. Bare "PBE" is ambiguous (QPErr).
_GAUSSIAN_FUNCTIONAL = {
    "PBE": "PBEPBE",
    "BLYP": "BLYP",
    "BP86": "BP86",
    "PW91": "PW91PW91",
    "OLYP": "OLYP",
    "B97D": "B97D",
    "B3LYP": "B3LYP",
    "PBE0": "PBE1PBE",
    "HSE06": "HSEH1PBE",
    "O3LYP": "O3LYP",
    "X3LYP": "X3LYP",
    "BHANDHLYP": "BHandHLYP",
    "TPSS": "TPSSTPSS",
    "M06-L": "M06L",
    "TPSSh": "TPSSh",
    "M06": "M06",
    "M06-2X": "M062X",
    "B2PLYP": "B2PLYP",
    "CAM-B3LYP": "CAM-B3LYP",
    "wB97X-D": "WB97XD",
    "wB97X": "WB97X",
    "wB97": "WB97",
    "LC-BLYP": "LC-BLYP",
    "MN15": "MN15",
}

_GAUSSIAN_BASIS = {
    "STO3G": "STO-3G",
    "STO6G": "STO-6G",
    "6-31G": "6-31G",
    "6-31G*": "6-31G*",
    "6-31G**": "6-31G**",
    "cc-pVDZ": "cc-pVDZ",
    "cc-pVTZ": "cc-pVTZ",
    "cc-pVQZ": "cc-pVQZ",
    "cc-pV5Z": "cc-pV5Z",
    "def2-SVP": "Def2SVP",
    "def2-SVPD": "Def2SVPD",
    "def2-TZVP": "Def2TZVP",
    "def2-TZVPD": "Def2TZVPD",
    "def2-TZVPP": "Def2TZVPP",
    "def2-TZVPPD": "Def2TZVPPD",
    "def2-QZVP": "Def2QZVP",
    "def2-QZVPD": "Def2QZVPD",
    "def2-QZVPP": "Def2QZVPP",
    "def2-QZVPPD": "Def2QZVPPD",
    "aug-cc-pVDZ": "aug-cc-pVDZ",
    "aug-cc-pVTZ": "aug-cc-pVTZ",
}


def _enum_value(value):
    if value is None:
        return None
    return value.value if hasattr(value, "value") else value


def _dispersion_type(qm_input: QMInput):
    correction = getattr(qm_input.functional, "dispersion_correction", None)
    raw = _enum_value(getattr(correction, "value", None))
    if raw is None:
        return None
    text = str(raw).upper()
    if text == "NONE":
        return None
    return text


def _route_parameters(qm_input: QMInput) -> dict:
    if qm_input.states == 0:
        route = {"opt": "tight", "freq": "", "pop": "full", "nosymm": ""}
    else:
        route = {
            "td": f"(nstates={qm_input.states},root={qm_input.focus_state})",
            "nosymm": "",
            "force": "",
        }

    if qm_input.scf_accuracy in (SCFAccuracy.Sloppy, "Sloppy"):
        route["scf"] = "Sleazy"
    elif qm_input.scf_accuracy in (
        SCFAccuracy.Tight,
        SCFAccuracy.VeryTight,
        SCFAccuracy.Extreme,
        "Tight",
        "VeryTight",
        "Extreme",
    ):
        route["scf"] = "Tight"

    if qm_input.grid_type in (GridType.Grid5, "Grid5"):
        route["integral"] = "UltraFine"
    elif qm_input.grid_type in (GridType.Grid4, "Grid4"):
        route["integral"] = "FineGrid"

    dispersion_type = _dispersion_type(qm_input)
    if dispersion_type:
        mapped = _DISPERSION_ROUTE.get(dispersion_type)
        if mapped is None:
            raise ValueError(f"Unsupported dispersion correction type: {dispersion_type}")
        route["EmpiricalDispersion"] = mapped

    solvent = (qm_input.solvent or "none").lower()
    if solvent != "none":
        route["SCRF"] = f"(PCM,Solvent={qm_input.solvent})"
    return route


def _link0_parameters(parent_parameters) -> dict:
    link0 = {"%mem": "10GB", "%nprocshared": "12", "%chk": "gaussian.chk"}
    slurm = getattr(parent_parameters, "slurm_parameters", None)
    if slurm is None:
        return link0
    mem = getattr(slurm, "mem", None)
    if mem is not None:
        text = str(mem).strip().upper()
        if not text:
            raise ValueError("slurm_parameters.mem is empty")
        number = None
        unit = None
        for suffix in ("KW", "MW", "GW", "TW", "KB", "MB", "GB", "TB"):
            if text.endswith(suffix):
                number, unit = text[: -len(suffix)], suffix
                break
        if unit is None and text[-1] in "KMGT":
            number, unit = text[:-1], f"{text[-1]}B"
        if number is None or not number.isdigit():
            raise ValueError(
                f"slurm_parameters.mem={mem!r} is not a Gaussian %mem value; "
                "use a Slurm size such as 10G or a Gaussian size such as 10GB"
            )
        link0["%mem"] = f"{number}{unit}"
    tasks = getattr(slurm, "tasks_per_node", None)
    if tasks:
        link0["%nprocshared"] = str(tasks)
    return link0


def _functional_name(qm_input: QMInput) -> str:
    raw = str(_enum_value(qm_input.functional.functional))
    mapped = _GAUSSIAN_FUNCTIONAL.get(raw)
    if mapped is None:
        raise ValueError(
            f"Functional {raw!r} is not a Gaussian 16 route keyword"
        )
    return mapped


def _basis_set_name(qm_input: QMInput) -> str:
    raw = str(_enum_value(qm_input.basis_set.basis_set))
    mapped = _GAUSSIAN_BASIS.get(raw)
    if mapped is None:
        raise ValueError(
            f"Basis set {raw!r} is not a Gaussian 16 route keyword"
        )
    return mapped


@node
async def gaussian(qm_input: QMInput, **kwargs) -> SimstackResult:
    """Run a Gaussian calculation from ``QMInput`` and parse the log.

    The Gaussian binary is launched from ``config.toml``
    (``[<resource>.program.gaussian]``), not from hardcoded Justus scripts.

    Parameters:
        qm_input (QMInput): Molecular geometry, charge, multiplicity, functional,
            basis set, and excited-state options.

    Returns:
        SimstackResult: Parsed Gaussian result.

    SimstackResult:
        result (QMResult): Energies, final structure, optional excited states, and
            checkpoint files.
    """
    task_id = kwargs.get("task_id", "NA")
    node_runner: NodeRunner | None = kwargs.get("node_runner", None)
    if node_runner is None:
        raise ValueError(f"Gaussian: task_id: {task_id} node_runner not provided")

    node_runner.info("Starting Gaussian calculation")
    try:
        if getattr(qm_input, "name", None):
            node_runner.custom_name = qm_input.name

        mol = Molecule.from_molecule(qm_input.molecule)
        mol.properties["charge"] = qm_input.charge
        mol.properties["spin_multiplicity"] = qm_input.multiplicity

        parent_parameters = kwargs.get("parent_parameters", None)
        if parent_parameters:
            node_runner.info(f"parent_parameters: {parent_parameters}")

        route_params = _route_parameters(qm_input)
        node_runner.info(f"Handling PCM solvent model {qm_input.solvent}")

        link0_dict = _link0_parameters(parent_parameters)
        node_runner.info(f"link0_dict: {link0_dict}")

        gin = GaussianInput(
            mol=mol,
            charge=qm_input.charge,
            spin_multiplicity=qm_input.multiplicity,
            title="Gaussian Calculation",
            functional=_functional_name(qm_input),
            basis_set=_basis_set_name(qm_input),
            route_parameters=route_params,
            link0_parameters=link0_dict,
        )
        gin.write_file("gaussian.com", cart_coords=True)
    except Exception as exc:
        return node_runner.fail(f"error creating Gaussian input file: {str(exc)} ")

    try:
        input_files = ["gaussian.com"]
        if Path("gaussian.chk").exists():
            input_files.append("gaussian.chk")
        node_runner.stage(input_files=input_files)
        if not node_runner.execute("gaussian"):
            raise RuntimeError("execution of Gaussian failed")
        node_runner.retrieve(output_files=GAUSSIAN_RESULT_FILES)

        gout = GaussianOutput("gaussian.log")
        node_runner.info("output file parsed")

        molecule = gout.final_structure
        if molecule is None:
            raise RuntimeError(f"task_id: {task_id} Gaussian output has no final structure")

        chk_paths = sorted(glob.glob("*.chk"))
        if not any(Path(path).name == "gaussian.chk" for path in chk_paths):
            raise RuntimeError(
                f"task_id: {task_id} Gaussian did not write gaussian.chk"
            )

        file_list = FileList()
        for out_file in chk_paths:
            file_stack = FileStack.from_local_file(
                out_file,
                in_memory=False,
                is_hashable=True,
                secure_source=True,
                task_id=task_id,
            )
            await context.db.save(file_stack)
            file_list.append(file_stack)
            node_runner.info(f"attached {out_file} to QMResult.files")

        node_runner.result = QMResult(
            scf_converged=gout.properly_terminated,
            final_energy=gout.final_energy,
            energies=gout.energies,
            final_structure=molecule,
            structures=MoleculeList(),
            task_status=TaskStatus.COMPLETED,
            files=file_list,
        )
        node_runner.result.excited_states, node_runner.result.excited_state_transitions = (
            parse_gaussian_excited_states_file("gaussian.log")
        )

        return node_runner.succeed()
    except Exception as exc:
        raise RuntimeError(f"task_id: {task_id} Gaussian Failed {str(exc)}") from exc
    finally:
        await node_runner.make_info_files("*.com")
