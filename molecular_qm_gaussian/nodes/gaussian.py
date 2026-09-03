import glob
import logging
import subprocess

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
from simstack.models.files import FileStack

from molecular_qm_gaussian.lib.gaussian_excited_states_parser import (
    parse_gaussian_excited_states_file,
)
from molecular_qm_gaussian.lib.gaussian_io import GaussianInput, GaussianOutput

logger = logging.getLogger("GaussianNode")

GAUSSIAN_INPUT_FILES = ["gaussian.com", "gaussian.chk"]
GAUSSIAN_RESULT_FILES = ["gaussian.log", "gaussian.chk"]

_DISPERSION_ROUTE = {
    "D2": "GD2",
    "D3": "GD3",
    "D3BJ": "GD3BJ",
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
    if getattr(slurm, "mem", None):
        link0["%mem"] = slurm.mem
    tasks = getattr(slurm, "tasks_per_node", None)
    if tasks:
        link0["%nprocshared"] = str(tasks)
    return link0


def _functional_name(qm_input: QMInput) -> str:
    return str(_enum_value(qm_input.functional.functional)).lower()


def _basis_set_name(qm_input: QMInput) -> str:
    return str(_enum_value(qm_input.basis_set.basis_set)).lower()


def gaussian_run_command() -> int:
    try:
        context.resource_config.run("gaussian", GAUSSIAN_INPUT_FILES, GAUSSIAN_RESULT_FILES)
        return 0
    except subprocess.CalledProcessError as exc:
        return exc.returncode


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
        returncode = gaussian_run_command()
        if returncode != 0:
            raise RuntimeError(f"execution of Gaussian failed with return code {returncode}")

        gout = GaussianOutput("gaussian.log")
        node_runner.info("output file parsed")

        molecule = gout.final_structure
        if molecule is None:
            raise RuntimeError(f"task_id: {task_id} Gaussian output has no final structure")

        node_runner.result = QMResult(
            scf_converged=gout.properly_terminated,
            final_energy=gout.final_energy,
            energies=gout.energies,
            final_structure=molecule,
            structures=MoleculeList(),
            task_status=TaskStatus.COMPLETED,
        )
        node_runner.result.excited_states, node_runner.result.excited_state_transitions = (
            parse_gaussian_excited_states_file("gaussian.log")
        )

        for out_file in glob.glob("*.chk"):
            file_stack = FileStack.from_local_file(
                out_file,
                in_memory=True,
                is_hashable=True,
                secure_source=True,
                task_id=task_id,
            )
            await context.db.save(file_stack)
            node_runner.result.files.append(file_stack)

        return node_runner.succeed()
    except Exception as exc:
        raise RuntimeError(f"task_id: {task_id} Gaussian Failed {str(exc)}") from exc
    finally:
        await node_runner.make_info_files("*.com")
