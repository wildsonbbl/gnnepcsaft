"""Core helpers to build FEOS PC-SAFT equations of state."""

from typing import List, Optional

from feos import BinaryRecord  # pyright: ignore[reportAttributeAccessIssue]
from feos import EquationOfState  # pyright: ignore[reportAttributeAccessIssue]
from feos import Identifier  # pyright: ignore[reportAttributeAccessIssue]
from feos import IdentifierOption  # pyright: ignore[reportAttributeAccessIssue]
from feos import Parameters  # pyright: ignore[reportAttributeAccessIssue]
from feos import PureRecord  # pyright: ignore[reportAttributeAccessIssue]


def pc_saft(parameters: List[float]) -> EquationOfState.pcsaft:
    """
    Returns a PCSAFT equation of state.

    Args:
        parameters: A list with
         `[m, sigma, epsilon/kB, kappa_ab, epsilon_ab/kB, dipole moment, na, nb, mw]`

    Returns:
        out (EquationOfState.pcsaft): Configured PC-SAFT equation of state for a pure component.

    """

    return pc_saft_mixture([parameters])


def pc_saft_mixture(
    mixture_parameters: List[List[float]],
    kij_matrix: Optional[List[List[float]]] = None,
    epsilon_ab: Optional[List[List[float]]] = None,
) -> EquationOfState.pcsaft:
    """
    Returns a PCSAFT equation of state.

    Args:
        mixture_parameters: A list of
         `[m, sigma, epsilon/kB, kappa_ab, epsilon_ab/kB, dipole moment, na, nb, MW]`
         for each component of the mixture
        kij_matrix: A matrix of binary interaction parameters
        epsilon_ab: A matrix of cross association energy parameters

    Returns:
        out (EquationOfState.pcsaft): Configured PC-SAFT equation of state for a mixture.
    """
    records = get_records(mixture_parameters)

    def _create_binary_record(
        i: int, j: int, kij: Optional[float] = None, eps_ab: Optional[float] = None
    ) -> BinaryRecord:
        """Helper function to create a binary record."""
        kwargs = {}
        if kij is not None:
            kwargs["k_ij"] = kij
        if eps_ab is not None:
            kwargs["epsilon_k_ab"] = eps_ab

        return BinaryRecord(
            id1=Identifier(name=f"comp_{i}"),
            id2=Identifier(name=f"comp_{j}"),
            **kwargs,
        )

    if kij_matrix or epsilon_ab:
        binary_records = [
            _create_binary_record(
                i,
                j,
                kij=kij_matrix[i][j] if kij_matrix else None,
                eps_ab=epsilon_ab[i][j] if epsilon_ab else None,
            )
            for i in range(len(records))
            for j in range(len(records))
            if i != j
        ]
    else:
        binary_records = []
    pcsaftparameters = Parameters.from_records(
        records, binary_records=binary_records, identifier_option=IdentifierOption.Name
    )
    eos = EquationOfState.pcsaft(pcsaftparameters)
    return eos


def get_records(mixture_parameters: List[List[float]]) -> list[PureRecord]:
    """
    Returns a list of `feos.pcsaft.PureRecord`.

    Args:
        mixture_parameters: A list of
         `[m, sigma, epsilon/kB, kappa_ab, epsilon_ab/kB, dipole moment, na, nb, MW]`
         for each component of the mixture

    Returns:
        out (list[PureRecord]): FEOS pure-component records for all mixture components.
    """
    records = []
    for idx, mol_parameters in enumerate(mixture_parameters):
        records.append(
            PureRecord(
                identifier=Identifier(name=f"comp_{idx}"),
                molarweight=mol_parameters[8],  # g/mol
                m=mol_parameters[0],  # units
                sigma=mol_parameters[1],  # Å
                epsilon_k=mol_parameters[2],  # K
                mu=mol_parameters[5],  # Debye
                association_sites=[
                    {
                        "kappa_ab": mol_parameters[3],
                        "epsilon_k_ab": mol_parameters[4],  # K
                        "na": mol_parameters[6],
                        "nb": mol_parameters[7],
                    }
                ],
            )
        )

    return records
