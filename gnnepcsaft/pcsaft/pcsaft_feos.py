"Module to calculate properties with PCSAFT using FEOS."

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import si_units as si
from feos import BinaryRecord  # pyright: ignore[reportAttributeAccessIssue]
from feos import Contributions  # pyright: ignore[reportAttributeAccessIssue]
from feos import EquationOfState  # pyright: ignore[reportAttributeAccessIssue]
from feos import GcParameters  # pyright: ignore[reportAttributeAccessIssue]
from feos import (
    HelmholtzEnergyFunctional,  # pyright: ignore[reportAttributeAccessIssue]
)
from feos import Identifier  # pyright: ignore[reportAttributeAccessIssue]
from feos import IdentifierOption  # pyright: ignore[reportAttributeAccessIssue]
from feos import Parameters  # pyright: ignore[reportAttributeAccessIssue]
from feos import PhaseDiagram  # pyright: ignore[reportAttributeAccessIssue]
from feos import PhaseEquilibrium  # pyright: ignore[reportAttributeAccessIssue]
from feos import PureRecord  # pyright: ignore[reportAttributeAccessIssue]
from feos import SegmentRecord  # pyright: ignore[reportAttributeAccessIssue]
from feos import SmartsRecord  # pyright: ignore[reportAttributeAccessIssue]
from feos import State  # pyright: ignore[reportAttributeAccessIssue]
from feos import SurfaceTensionDiagram  # pyright: ignore[reportAttributeAccessIssue]


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


def mix_gibbs_energy(
    parameters: List[List[float]],
    state: List[float],
    kij_matrix: Optional[List[List[float]]] = None,
    epsilon_ab: Optional[List[List[float]]] = None,
):
    """
    Calculates mixture `Molar Gibbs Energy/RT` with PCSAFT.

    Args:
        parameters: A list of
         `[m, sigma, epsilon/kB, kappa_ab, epsilon_ab/kB, dipole moment, na, nb, MW]`
         for each component of the mixture
        state: A list with
         `[Temperature (K), Pressure (Pa), mole_fractions_1, mole_fractions_2, ...]`
        kij_matrix: A matrix of binary interaction parameters
        epsilon_ab: A matrix of cross association energy parameters
    """
    x = np.asarray(state[2:], dtype=np.float64)  # mole fractions

    excess_g = mix_e_gibbs_energy(parameters, state, kij_matrix, epsilon_ab)

    return excess_g + np.sum(x * np.log(x))


def mix_ln_fugacity_coefficient_pure(
    parameters: List[List[float]],
    state: List[float],
    kij_matrix: Optional[List[List[float]]] = None,
    epsilon_ab: Optional[List[List[float]]] = None,
):
    """
    Calculates mixture `ln(fugacity coefficient)` with PCSAFT for each pure component.

    Args:
        parameters: A list of
         `[m, sigma, epsilon/kB, kappa_ab, epsilon_ab/kB, dipole moment, na, nb, MW]`
         for each component of the mixture
        state: A list with
         `[Temperature (K), Pressure (Pa), mole_fractions_1, mole_fractions_2, ...]`
        kij_matrix: A matrix of binary interaction parameters
        epsilon_ab: A matrix of cross association energy parameters
    """

    t = state[0]  # Temperature, K
    p = state[1]  # Pa
    x = np.asarray(state[2:], dtype=np.float64)  # mole fractions

    eos = pc_saft_mixture(parameters, kij_matrix, epsilon_ab)

    statenpt = State(
        eos,
        temperature=t * si.KELVIN,
        pressure=p * si.PASCAL,
        molefracs=x,
        density_initialization="liquid",
    )

    return statenpt.ln_phi_pure_liquid()


def mix_ln_activity_coefficient(
    parameters: List[List[float]],
    state: List[float],
    kij_matrix: Optional[List[List[float]]] = None,
    epsilon_ab: Optional[List[List[float]]] = None,
) -> List[float]:
    """
    Calculates mixture `ln(activity coefficient)` with PCSAFT for each component.

    Args:
        parameters: A list of
         `[m, sigma, epsilon/kB, kappa_ab, epsilon_ab/kB, dipole moment, na, nb, MW]`
         for each component of the mixture
        state: A list with
         `[Temperature (K), Pressure (Pa), mole_fractions_1, mole_fractions_2, ...]`
        kij_matrix: A matrix of binary interaction parameters
        epsilon_ab: A matrix of cross association energy parameters

    Returns:
        out (List[float]): Natural logarithm of activity coefficients for each component.
    """

    t = state[0]  # Temperature, K
    p = state[1]  # Pa
    x = np.asarray(state[2:], dtype=np.float64)  # mole fractions

    eos = pc_saft_mixture(parameters, kij_matrix, epsilon_ab)

    statenpt = State(
        eos,
        temperature=t * si.KELVIN,
        pressure=p * si.PASCAL,
        molefracs=x,
        density_initialization="liquid",
    )

    return statenpt.ln_symmetric_activity_coefficient()


def mix_e_gibbs_energy(
    parameters: List[List[float]],
    state: List[float],
    kij_matrix: Optional[List[List[float]]] = None,
    epsilon_ab: Optional[List[List[float]]] = None,
) -> float:
    """
    Calculates mixture `Molar Excess Gibbs Energy/RT` with PCSAFT.

    Args:
        parameters: A list of
         `[m, sigma, epsilon/kB, kappa_ab, epsilon_ab/kB, dipole moment, na, nb, MW]`
         for each component of the mixture
        state: A list with
         `[Temperature (K), Pressure (Pa), mole_fractions_1, mole_fractions_2, ...]`
        kij_matrix: A matrix of binary interaction parameters
        epsilon_ab: A matrix of cross association energy parameters

    Returns:
        out (float): Mixture excess Gibbs energy divided by RT.
    """

    x = np.asarray(state[2:], dtype=np.float64)  # mole fractions

    return np.sum(
        mix_ln_activity_coefficient(parameters, state, kij_matrix, epsilon_ab) * x
    )


def mix_ln_fugacity_coefficient(
    parameters: List[List[float]],
    state: List[float],
    kij_matrix: Optional[List[List[float]]] = None,
    epsilon_ab: Optional[List[List[float]]] = None,
) -> List[float]:
    """
    Calculates mixture `ln(fugacity coefficient)` with PCSAFT for each component.

    Args:
        parameters: A list of
         `[m, sigma, epsilon/kB, kappa_ab, epsilon_ab/kB, dipole moment, na, nb, MW]`
         for each component of the mixture
        state: A list with
         `[Temperature (K), Pressure (Pa), mole_fractions_1, mole_fractions_2, ...]`
        kij_matrix: A matrix of binary interaction parameters
        epsilon_ab: A matrix of cross association energy parameters

    Returns:
        out (List[float]): Natural logarithm of fugacity coefficients for each component.
    """
    t = state[0]  # Temperature, K
    p = state[1]  # Pa
    x = np.asarray(state[2:], dtype=np.float64)  # mole fractions

    eos = pc_saft_mixture(parameters, kij_matrix, epsilon_ab)

    statenpt = State(
        eos,
        temperature=t * si.KELVIN,
        pressure=p * si.PASCAL,
        molefracs=x,
        density_initialization="liquid",
    )

    return statenpt.ln_phi()


def mix_r_gibbs_energy(
    parameters: List[List[float]],
    state: List[float],
    kij_matrix: Optional[List[List[float]]] = None,
    epsilon_ab: Optional[List[List[float]]] = None,
) -> float:
    """
    Calculates mixture `Molar Residual Gibbs Energy/RT` with PCSAFT.

    Args:
        parameters: A list of
         `[m, sigma, epsilon/kB, kappa_ab, epsilon_ab/kB, dipole moment, na, nb, MW]`
         for each component of the mixture
        state: A list with
         `[Temperature (K), Pressure (Pa), mole_fractions_1, mole_fractions_2, ...]`
        kij_matrix: A matrix of binary interaction parameters
        epsilon_ab: A matrix of cross association energy parameters

    Returns:
        out (float): Mixture residual Gibbs energy divided by RT.
    """
    x = np.asarray(state[2:], dtype=np.float64)  # mole fractions
    return np.sum(
        mix_ln_fugacity_coefficient(parameters, state, kij_matrix, epsilon_ab) * x
    )


def mix_den_feos(
    parameters: List[List[float]],
    state: List[float],
    kij_matrix: Optional[List[List[float]]] = None,
    epsilon_ab: Optional[List[List[float]]] = None,
) -> float:
    """
    Calculates mixture liquid density (mol/m³) with PCSAFT.

    Args:
        parameters: A list of
         `[m, sigma, epsilon/kB, kappa_ab, epsilon_ab/kB, dipole moment, na, nb, MW]`
         for each component of the mixture
        state: A list with
         `[Temperature (K), Pressure (Pa), mole_fractions_1, mole_fractions_2, ...]`
        kij_matrix: A matrix of binary interaction parameters
        epsilon_ab: A matrix of cross association energy parameters

    Returns:
        out (float): Mixture liquid density in mol/m^3.
    """

    t = state[0]  # Temperature, K
    p = state[1]  # Pa
    x = np.asarray(state[2:], dtype=np.float64)  # mole fractions

    eos = pc_saft_mixture(parameters, kij_matrix, epsilon_ab)

    statenpt = State(
        eos,
        temperature=t * si.KELVIN,
        pressure=p * si.PASCAL,
        molefracs=x,
        density_initialization="liquid",
    )

    den = statenpt.density * (si.METER**3) / si.MOL

    return den


def pure_den_feos(parameters: List[float], state: List[float]) -> float:
    """
    Calculates pure component liquid density (mol/m³) with PCSAFT.

    Args:
        parameters: A list with
         `[m, sigma, epsilon/kB, kappa_ab, epsilon_ab/kB, dipole moment, na, nb, mw]`
        state: A list with `[Temperature (K), Pressure (Pa)]`

    Returns:
        out (float): Pure-component liquid density in mol/m^3.
    """

    t = state[0]  # Temperature, K
    p = state[1]  # Pa

    eos = pc_saft(parameters)
    statenpt = State(
        eos,
        temperature=t * si.KELVIN,
        pressure=p * si.PASCAL,
        density_initialization="liquid",
    )

    den = statenpt.density * (si.METER**3) / si.MOL

    return den


def mix_vp_feos(
    parameters: List[List[float]],
    state: List[float],
    kij_matrix: Optional[List[List[float]]] = None,
    epsilon_ab: Optional[List[List[float]]] = None,
) -> Tuple[float, float]:
    """
    Calculates mixture `(Bubble point (Pa), Dew point (Pa))` with PCSAFT.

    Args:
        parameters: A list of
         `[m, sigma, epsilon/kB, kappa_ab, epsilon_ab/kB, dipole moment, na, nb, MW]`
         for each component of the mixture
        state: A list with
         `[Temperature (K), Pressure (Pa), mole_fractions_1, molefractions_2, ...]`
        kij_matrix: A matrix of binary interaction parameters
        epsilon_ab: A matrix of cross association energy parameters

    Returns:
        out (Tuple[float, float]): Bubble-point and dew-point pressures in Pascal.
    """

    t = state[0]  # Temperature, K
    x = np.asarray(state[2:], dtype=np.float64)  # mole fractions

    eos = pc_saft_mixture(parameters, kij_matrix, epsilon_ab)

    vle_bubble_point = PhaseEquilibrium.bubble_point(
        eos, temperature_or_pressure=t * si.KELVIN, liquid_molefracs=x
    )

    vle_dew_point = PhaseEquilibrium.dew_point(
        eos, temperature_or_pressure=t * si.KELVIN, vapor_molefracs=x
    )

    assert (
        t == vle_bubble_point.liquid.temperature / si.KELVIN
    ), "Temperature mismatch for bubble point"
    assert (
        t == vle_dew_point.vapor.temperature / si.KELVIN
    ), "Temperature mismatch for dew point"
    return (
        vle_bubble_point.liquid.pressure() / si.PASCAL,
        vle_dew_point.vapor.pressure() / si.PASCAL,
    )


def pure_vp_feos(parameters: List[float], state: List[float]) -> float:
    """
    Calculates pure component vapor pressure (Pa) with PCSAFT.

    Args:
        parameters: A list with
         `[m, sigma, epsilon/kB, kappa_ab, epsilon_ab/kB, dipole moment, na, nb, mw]`
        state: A list with `[Temperature (K)]`

    Returns:
        out (float): Pure-component vapor pressure in Pascal.
    """

    t = state[0]  # Temperature, K

    eos = pc_saft(parameters)
    vle = PhaseEquilibrium.pure(eos, temperature_or_pressure=t * si.KELVIN)

    assert t == vle.liquid.temperature / si.KELVIN

    return vle.liquid.pressure() / si.PASCAL


def pure_h_lv_feos(parameters: List[float], state: List[float]) -> float:
    """
    Calculates pure component enthalpy of vaporization (kJ/mol) with PCSAFT.

    Args:
        parameters: A list with
         `[m, sigma, epsilon/kB, kappa_ab, epsilon_ab/kB, dipole moment, na, nb, mw]`
        state: A list with `[Temperature (K)]`

    Returns:
        out (float): Residual enthalpy of vaporization in kJ/mol.
    """

    t = state[0]  # Temperature, K

    eos = pc_saft(parameters)
    vle = PhaseEquilibrium.pure(eos, temperature_or_pressure=t * si.KELVIN)

    liquid_state = vle.liquid
    vapor_state = vle.vapor

    assert t == liquid_state.temperature / si.KELVIN

    return (
        vapor_state.molar_enthalpy(Contributions.Residual)
        - liquid_state.molar_enthalpy(Contributions.Residual)
    ) * (si.MOL / si.KILO / si.JOULE)


def pure_s_lv_feos(parameters: List[float], state: List[float]) -> float:
    """
    Calcules pure component entropy of vaporization (J/mol*K) with PCSAFT.

    Args:
        parameters: A list with
         `[m, sigma, epsilon/kB, kappa_ab, epsilon_ab/kB, dipole moment, na, nb, mw]`
        state: A list with `[Temperature (K)]`

    Returns:
        out (float): Residual entropy of vaporization in J/(mol*K).
    """
    t = state[0]  # Temperature, K
    eos = pc_saft(parameters)
    vle = PhaseEquilibrium.pure(eos, temperature_or_pressure=t * si.KELVIN)
    liquid_state = vle.liquid
    vapor_state = vle.vapor
    assert t == liquid_state.temperature / si.KELVIN
    return (
        vapor_state.molar_entropy(Contributions.Residual)
        - liquid_state.molar_entropy(Contributions.Residual)
    ) * (si.MOL * si.KELVIN / si.JOULE)


def critical_points_feos(parameters: List[float]) -> List[float]:
    """
    Calculates critical points `[Tc (K), Pc (Pa), Dc (mol/m³)]` with PCSAFT.

    Args:
        parameters: A list with
         `[m, sigma, epsilon/kB, kappa_ab, epsilon_ab/kB, dipole moment, na, nb, mw]`

    Returns:
        out (List[float]): Critical temperature (K), pressure (Pa), and density (mol/m^3).
    """
    eos = pc_saft(parameters)
    critical_point = State.critical_point(eos)
    return [
        critical_point.temperature / si.KELVIN,
        critical_point.pressure() / si.PASCAL,
        critical_point.density * (si.METER**3) / si.MOL,
    ]


def pure_viscosity_feos(parameters: List[float], state: List[float]) -> float:
    """
    Calcules pure component viscosity (kPa*s) with PCSAFT.

    Args:
        parameters: A list with
         `[m, sigma, epsilon/kB, kappa_ab, epsilon_ab/kB, dipole moment, na, nb, mw]`
        state: A list with `[Temperature (K), Pressure (Pa)]`

    Returns:
        out (float): Dynamic viscosity from FEOS.
    """
    t = state[0]  # Temperature, K
    p = state[1]  # Pa

    eos = pc_saft(parameters)
    statenpt = State(
        eos,
        temperature=t * si.KELVIN,
        pressure=p * si.PASCAL,
        density_initialization="liquid",
    )

    return statenpt.viscosity()  # / (KILO * PASCAL * SECOND)


def phase_diagram_feos(
    parameters: List[float], state: List[float]
) -> Dict[str, List[float]]:
    """
    Calculates pure component phase diagram from
    state temperature up to the critical temperature with PCSAFT.

    Args:
        parameters: A list with
         `[m, sigma, epsilon/kB, kappa_ab, epsilon_ab/kB, dipole moment, na, nb, mw]`
        state: A list with `[Temperature (K)]`


    Returns:
        out (Dict[str, List[float]]):
          - temperature: K
          - pressure: Pa
          - density [liquid/vapor]: mol / m³
          - mass density [liquid/vapor]: kg / m³
          - residual molar enthalpy [liquid/vapor]: kJ / mol
          - residual molar entropy [liquid/vapor]: kJ / mol / K
          - residual specific enthalpy [liquid/vapor]: kJ / kg
          - residual specific entropy [liquid/vapor]: kJ / kg / K
    """
    t = state[0]  # Temperature, K
    eos = pc_saft(parameters)
    phase_diagram = PhaseDiagram.pure(eos, min_temperature=t * si.KELVIN, npoints=200)

    return phase_diagram.to_dict(Contributions.Residual)


def is_stable_feos(
    parameters: List[List[float]],
    state: List[float],
    kij_matrix: Optional[List[List[float]]] = None,
    epsilon_ab: Optional[List[List[float]]] = None,
    density_initialization: Optional[str] = None,
) -> bool:
    """
    Calculates stability of the mixture.

    Args:
        parameters: A list of
         `[m, sigma, epsilon/kB, kappa_ab, epsilon_ab/kB, dipole moment, na, nb, mw]`
         for each component of the mixture
        state:
         A list with `[Temperature (K), Pressure (Pa), mole_fractions_1, mole_fractions_2, ...]`
        kij_matrix: A matrix of binary interaction parameters
        epsilon_ab: A matrix of cross association energy parameters
        density_initialization: Initialization method for density ("liquid", "vapor", None)

    Returns:
        out (bool): True if the state is stable, otherwise False.
    """
    t = state[0]  # Temperature, K
    p = state[1]  # Pressure, Pa
    x = np.asarray(state[2:], dtype=np.float64)  # mole fractions
    eos = pc_saft_mixture(parameters, kij_matrix=kij_matrix, epsilon_ab=epsilon_ab)
    statenpt = State(
        eos,
        temperature=t * si.KELVIN,
        pressure=p * si.PASCAL,
        molefracs=x,
        density_initialization=density_initialization,
    )
    return statenpt.is_stable()


def mix_tp_flash_feos(
    parameters: List[List[float]],
    state: List[float],
    kij_matrix: Optional[List[List[float]]] = None,
    epsilon_ab: Optional[List[List[float]]] = None,
) -> PhaseEquilibrium:
    """
    Calculates mixture phase equilibrium at
    state temperature and pressure with PCSAFT.

    Args:
        parameters: A list of
         `[m, sigma, epsilon/kB, kappa_ab, epsilon_ab/kB, dipole moment, na, nb, mw]`
         for each component of the mixture
        state:
         A list with `[Temperature (K), Pressure (Pa), mole_fractions_1, mole_fractions_2, ...]`
        kij_matrix: A matrix of binary interaction parameters
        epsilon_ab: A matrix of cross association energy parameters

    Returns:
        out (PhaseEquilibrium): TP flash result with coexisting phases.
    """
    t = state[0]  # Temperature, K
    p = state[1]  # Pressure, Pa
    x = np.asarray(state[2:], dtype=np.float64)  # mole fractions
    eos = pc_saft_mixture(parameters, kij_matrix=kij_matrix, epsilon_ab=epsilon_ab)
    tp_flash = PhaseEquilibrium.tp_flash(
        eos,
        temperature=t * si.KELVIN,
        pressure=p * si.PASCAL,
        feed=x * si.MOL,
        max_iter=1_000,
    )

    return tp_flash


def henry_constant_feos(
    parameters: List[List[float]],
    state: List[float],
    kij_matrix: Optional[List[List[float]]] = None,
    epsilon_ab: Optional[List[List[float]]] = None,
    density_initialization: Optional[str] = None,
) -> np.ndarray:
    """
    Calculates Henry's constant (Pa) of every solute at
    state temperature and pressure with PCSAFT.
    Solute at x_i = 0.0 and solvents at x_i > 0.0.

    Args:
        parameters: A list of
         `[m, sigma, epsilon/kB, kappa_ab, epsilon_ab/kB, dipole moment, na, nb, mw]`
         for each component of the mixture
        state:
         A list with `[Temperature (K), Pressure (Pa), mole_fractions_1, mole_fractions_2, ...]`
        kij_matrix: A matrix of binary interaction parameters
        epsilon_ab: A matrix of cross association energy parameters
        density_initialization: Initialization method for density ("liquid", "vapor", None)

    Returns:
        out (np.ndarray): Henry constants for each component in Pascal.
    """
    t = state[0]  # Temperature, K
    p = state[1]  # Pressure, Pa
    x = np.asarray(state[2:], dtype=np.float64)  # mole fractions
    eos = pc_saft_mixture(parameters, kij_matrix=kij_matrix, epsilon_ab=epsilon_ab)
    statenpt = State(
        eos,
        temperature=t * si.KELVIN,
        pressure=p * si.PASCAL,
        molefracs=x,
        density_initialization=density_initialization,
    )

    return statenpt.henrys_law_constant(eos, t * si.KELVIN, x) / si.PASCAL


def mix_lle_diagram_feos(
    parameters: List[List[float]],
    state: List[float],
    kij_matrix: Optional[List[List[float]]] = None,
    epsilon_ab: Optional[List[List[float]]] = None,
) -> Dict[str, List[float]]:
    """
    Calculates mixture LLE phase diagram at
    state constant pressure and variable temperature with PCSAFT.

    Args:
        parameters: A list of
         `[m, sigma, epsilon/kB, kappa_ab, epsilon_ab/kB, dipole moment, na, nb, mw]`
         for each component of the mixture
        state:
         A list with `[Temperature (K), Pressure (Pa), mole_fractions_1, mole_fractions_2, ...]`
        kij_matrix: A matrix of binary interaction parameters
        epsilon_ab: A matrix of cross association energy parameters

    Returns:
        out (Dict[str, List[float]]):
          - For LLE, vapor identifies the liquid phase 2.
          - temperature: K
          - pressure: Pa
          - density [liquid/vapor]: mol / m³
          - mass density [liquid/vapor]: kg / m³
          - residual molar enthalpy [liquid/vapor]: kJ / mol
          - residual molar entropy [liquid/vapor]: kJ / mol / K
          - residual specific enthalpy [liquid/vapor]: kJ / kg
          - residual specific entropy [liquid/vapor]: kJ / kg / K
          - xi: phase 1 liquid molefraction of component i
          - yi: phase 2 liquid molefraction of component i
    """
    t = state[0]  # Temperature, K
    p = state[1]  # Pressure, Pa
    x = np.asarray(state[2:], dtype=np.float64)  # mole fractions
    eos = pc_saft_mixture(parameters, kij_matrix=kij_matrix, epsilon_ab=epsilon_ab)
    dia_t = PhaseDiagram.lle(
        eos,
        temperature_or_pressure=p * si.PASCAL,
        feed=x * si.MOL,
        min_tp=t * si.KELVIN,
        max_tp=(t + 50) * si.KELVIN,
        npoints=200,
    )

    if len(dia_t.states) == 0:
        raise ValueError("No LLE found at the given conditions.")

    return dia_t.to_dict(Contributions.Residual)


def mix_lle_feos(
    parameters: List[List[float]],
    state: List[float],
    kij_matrix: Optional[List[List[float]]] = None,
    epsilon_ab: Optional[List[List[float]]] = None,
) -> Dict[str, List[float]]:
    """
    Calculates mixture LLE at state pressure and temperature with PCSAFT.

    Args:
        parameters: A list of
         `[m, sigma, epsilon/kB, kappa_ab, epsilon_ab/kB, dipole moment, na, nb, mw]`
         for each component of the mixture
        state:
         A list with `[Temperature (K), Pressure (Pa), mole_fractions_1, mole_fractions_2, ...]`
        kij_matrix: A matrix of binary interaction parameters
        epsilon_ab: A matrix of cross association energy parameters

    Returns:
        out (Dict[str, List[float]]):
          - For LLE, vapor identifies the liquid phase 2.
          - temperature: K
          - pressure: Pa
          - density [liquid/vapor]: mol / m³
          - mass density [liquid/vapor]: kg / m³
          - residual molar enthalpy [liquid/vapor]: kJ / mol
          - residual molar entropy [liquid/vapor]: kJ / mol / K
          - residual specific enthalpy [liquid/vapor]: kJ / kg
          - residual specific entropy [liquid/vapor]: kJ / kg / K
          - xi: phase 1 liquid molefraction of component i
          - yi: phase 2 liquid molefraction of component i
    """
    t = state[0]  # Temperature, K
    p = state[1]  # Pressure, Pa
    x = np.asarray(state[2:], dtype=np.float64)  # mole fractions
    eos = pc_saft_mixture(parameters, kij_matrix=kij_matrix, epsilon_ab=epsilon_ab)
    dia_t = PhaseDiagram.lle(
        eos,
        temperature_or_pressure=p * si.PASCAL,
        feed=x * si.MOL,
        min_tp=t * si.KELVIN,
        max_tp=t * si.KELVIN,
        npoints=1,
    )

    if len(dia_t.states) == 0:
        raise ValueError("No LLE found at the given conditions.")

    return dia_t.to_dict(Contributions.Residual)


def mix_vle_diagram_feos(
    parameters: List[List[float]],
    state: List[float],
    kij_matrix: Optional[List[List[float]]] = None,
    epsilon_ab: Optional[List[List[float]]] = None,
) -> Dict[str, List[float]]:
    """
    Calculates binary mixture VLE phase diagram at
    state constant pressure and variable temperature with PCSAFT.

    Args:
        parameters: A list of
         `[m, sigma, epsilon/kB, kappa_ab, epsilon_ab/kB, dipole moment, na, nb, mw]`
         for each component of the mixture
        state:
         A list with `[Pressure (Pa)]`
        kij_matrix: A matrix of binary interaction parameters
        epsilon_ab: A matrix of cross association energy parameters

    Returns:
        out (Dict[str, List[float]]):
          - temperature: K
          - pressure: Pa
          - density [liquid/vapor]: mol / m³
          - mass density [liquid/vapor]: kg / m³
          - residual molar enthalpy [liquid/vapor]: kJ / mol
          - residual molar entropy [liquid/vapor]: kJ / mol / K
          - residual specific enthalpy [liquid/vapor]: kJ / kg
          - residual specific entropy [liquid/vapor]: kJ / kg / K
          - xi: liquid molefraction of component i
          - yi: vapor molefraction of component i
    """
    p = state[0]  # Pressure, Pa
    eos = pc_saft_mixture(parameters, kij_matrix=kij_matrix, epsilon_ab=epsilon_ab)
    dia_t = PhaseDiagram.binary_vle(
        eos,
        temperature_or_pressure=p * si.PASCAL,
    )

    if len(dia_t.states) == 0:
        raise ValueError("No VLE found at the given conditions.")

    return dia_t.to_dict(Contributions.Residual)


def mix_vle_pxy_diagram_feos(
    parameters: List[List[float]],
    temperature: float,
    kij_matrix: Optional[List[List[float]]] = None,
    epsilon_ab: Optional[List[List[float]]] = None,
) -> Dict[str, List[float]]:
    """
    Calculates binary mixture VLE phase diagram at
    state constant temperature and variable pressure with PCSAFT.

    Args:
        parameters: A list of
         `[m, sigma, epsilon/kB, kappa_ab, epsilon_ab/kB, dipole moment, na, nb, mw]`
         for each component of the mixture
        temperature: Temperature (K)
        kij_matrix: A matrix of binary interaction parameters
        epsilon_ab: A matrix of cross association energy parameters

    Returns:
        out (Dict[str, List[float]]):
          - temperature: K
          - pressure: Pa
          - density [liquid/vapor]: mol / m³
          - mass density [liquid/vapor]: kg / m³
          - residual molar enthalpy [liquid/vapor]: kJ / mol
          - residual molar entropy [liquid/vapor]: kJ / mol / K
          - residual specific enthalpy [liquid/vapor]: kJ / kg
          - residual specific entropy [liquid/vapor]: kJ / kg / K
          - xi: liquid molefraction of component i
          - yi: vapor molefraction of component i
    """

    eos = pc_saft_mixture(parameters, kij_matrix=kij_matrix, epsilon_ab=epsilon_ab)
    dia_p = PhaseDiagram.binary_vle(
        eos,
        temperature_or_pressure=temperature * si.KELVIN,
    )

    if len(dia_p.states) == 0:
        raise ValueError("No VLE found at the given conditions.")

    return dia_p.to_dict(Contributions.Residual)


def mix_vlle_diagram_feos(
    parameters: List[List[float]],
    state: List[float],
    kij_matrix: Optional[List[List[float]]] = None,
    epsilon_ab: Optional[List[List[float]]] = None,
) -> Dict[str, List[float]]:
    """
    Calculates binary mixture VLLE phase diagram at
    state constant pressure and variable temperature with PCSAFT.

    Args:
        parameters: A list of
         `[m, sigma, epsilon/kB, kappa_ab, epsilon_ab/kB, dipole moment, na, nb, mw]`
         for each component of the mixture
        state:
         A list with `[Temperature (K), Pressure (Pa), mole_fractions_1]`
        kij_matrix: A matrix of binary interaction parameters
        epsilon_ab: A matrix of cross association energy parameters

    Returns:
        out (Dict[str, List[float]]): VLLE diagram data returned by FEOS.
    """
    t = state[0]  # Temperature, K
    p = state[1]  # Pressure, Pa
    x = np.asarray(state[2:], dtype=np.float64)  # mole fractions
    eos = pc_saft_mixture(parameters, kij_matrix=kij_matrix, epsilon_ab=epsilon_ab)
    dia_t = PhaseDiagram.binary_vlle(
        eos,
        temperature_or_pressure=p * si.PASCAL,
        x_lle=x,
        tp_lim_lle=t * si.KELVIN,
        tp_init_vlle=t * si.KELVIN,
    )
    if len(dia_t.states) == 0:
        raise ValueError("No VLLE found at the given conditions.")

    return dia_t.to_dict(Contributions.Residual)


def mix_r_isobaric_heat_capacity_feos(
    parameters: List[List[float]],
    state: List[float],
    kij_matrix: Optional[List[List[float]]] = None,
    epsilon_ab: Optional[List[List[float]]] = None,
) -> float:
    """
    Calculates mixture residual molar isobaric heat capacity (J / (mol*K)) with PCSAFT

    Args:
        parameters: A list of
          `[m, sigma, epsilon/kB, kappa_ab, epsilon_ab/kB, dipole moment, na, nb, mw]`
          for each component of the mixture
        state:
          A list with `[Temperature (K), Pressure (Pa), mole_fractions_1, mole_fractions_2, ...]`
        kij_matrix: A matrix of binary interaction parameters
        epsilon_ab: A matrix of cross association energy parameters

    Returns:
        out (float): Residual molar isobaric heat capacity in J/(mol*K).
    """
    t = state[0]  # Temperature, K
    p = state[1]  # Pressure, Pa
    x = np.asarray(state[2:], dtype=np.float64)  # mole fractions
    eos = pc_saft_mixture(parameters, kij_matrix=kij_matrix, epsilon_ab=epsilon_ab)

    statenpt = State(
        eos,
        temperature=t * si.KELVIN,
        pressure=p * si.PASCAL,
        molefracs=x,
    )

    return statenpt.molar_isobaric_heat_capacity(Contributions.Residual) / (
        si.JOULE / si.MOL / si.KELVIN
    )


def pure_surface_tension_feos(
    parameters: List[float], state: List[float]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates pure component `([Surface Tension (mN/m)], [Temperature (K)])` with PCSAFT
    from state temperature up to the critical temperature with PCSAFT.

    Args:
        parameters: A list with
         `[m, sigma, epsilon/kB, kappa_ab, epsilon_ab/kB, dipole moment, na, nb, MW]`
        state: A list with `[Temperature (K)]`

    Returns:
        out (Tuple[np.ndarray, np.ndarray]): Surface tension (mN/m) and
          corresponding temperatures (K).
    """
    t = state[0]  # Temperature, K
    records = get_records([parameters])

    pcsaftparameters = Parameters.from_records(records)
    functional = HelmholtzEnergyFunctional.pcsaft(pcsaftparameters)
    phase_diagram = PhaseDiagram.pure(functional, t * si.KELVIN, 100)
    st_diagram = SurfaceTensionDiagram(phase_diagram.states, n_grid=1024)

    st = st_diagram.surface_tension / (si.MILLI * si.NEWTON / si.METER)
    temp = st_diagram.liquid.temperature / si.KELVIN
    return st, temp


def parameters_gc_pcsaft(smiles: str) -> List[float]:
    """
    Calculates PC-SAFT parameters with Group Contribution method.

    Args:
        smiles (str): SMILES of the compound

    Returns:
        out (List[float]): Estimated PC-SAFT parameters in the order
            [m, sigma, epsilon_k, kappa_ab, epsilon_k_ab, mu, na, nb].
    """

    parameters = GcParameters.from_smiles(
        [smiles],
        SmartsRecord.from_json(
            str(
                Path(__file__).resolve().parent.parent
                / "data/gc_pcsaft/sauer2014_smarts.json"
            )
        ),
        SegmentRecord.from_json(
            str(
                Path(__file__).resolve().parent.parent
                / "data/gc_pcsaft/rehner2023_hetero.json"
            )
        ),
    )
    eos = EquationOfState.pcsaft(parameters)
    gc_parameters = eos.parameters
    assert isinstance(gc_parameters, Dict)

    m = gc_parameters.get("m")
    sigma = gc_parameters.get("sigma")
    e = gc_parameters.get("epsilon_k")
    mu = gc_parameters.get("mu")
    kab = gc_parameters.get("kappa_ab")
    eab = gc_parameters.get("epsilon_k_ab")
    na = gc_parameters.get("na")
    nb = gc_parameters.get("nb")

    return [
        m.item() if m else 0.0,
        sigma.item() if sigma else 0.0,
        e.item() if e else 0.0,
        kab.item() if kab else 0.0,
        eab.item() if eab else 0.0,
        mu.item() if mu else 0.0,
        na.item() if na else 0.0,
        nb.item() if nb else 0.0,
    ]
