import asyncio

from molecular_qm_models import (
    BasisSet,
    DispersionCorrection,
    DispersionCorrectionEnum,
    Functional,
    Molecule,
    QMInput,
)
from molecular_qm_models.molecule import Atom
from simstack.core.context import context
from simstack.models import Parameters

from molecular_qm_gaussian import gaussian


def make_water() -> Molecule:
    coords = [
        [0.0, 0.0, 0.1173],
        [0.0, 0.7572, -0.4692],
        [0.0, -0.7572, -0.4692],
    ]
    molecule = Molecule()
    for element, xyz in zip(["O", "H", "H"], coords):
        molecule.add_atom(Atom.from_coords(element=element, coords=xyz))
    molecule.formula = "H2O"
    return molecule


async def main():
    await context.initialize()
    water = make_water()
    water.atoms[0].z += 0.1
    qm_input = QMInput(
        molecule=water,
        states=3,
        excited_states=True,
        charge=0,
        multiplicity=1,
        functional=Functional(
            functional="B3LYP",
            dispersion_correction=DispersionCorrection(value=DispersionCorrectionEnum.D3),
        ),
        basis_set=BasisSet(basis_set="6-31G"),
        optimization=True,
        gradients=False,
        solvent="cyclohexane",
    )
    result = await gaussian(
        qm_input,
        parameters=Parameters(resource="justus", queue="slurm-queue", force_rerun=True),
    )
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
