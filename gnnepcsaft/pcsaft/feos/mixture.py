"""Mixture thermodynamic properties using FEOS PC-SAFT."""

from typing import List, Optional

import numpy as np
import si_units as si
from feos import Contributions  # pyright: ignore[reportAttributeAccessIssue]
from feos import State  # pyright: ignore[reportAttributeAccessIssue]

from .core import pc_saft_mixture


def state_npt_feos(
    parameters: List[List[float]],
    state: List[float],
    kij_matrix: Optional[List[List[float]]] = None,
    epsilon_ab: Optional[List[List[float]]] = None,
    density_initialization: Optional[str] = None,
) -> State:
    """
    Set up a thermodynamic state at NPT.

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
        out (feos.State): State at NPT
    """
    t = state[0]  # Temperature, K
    p = state[1]  # Pressure, Pa
    x = np.asarray(state[2:], dtype=np.float64)  # mole fractions
    eos = pc_saft_mixture(parameters, kij_matrix=kij_matrix, epsilon_ab=epsilon_ab)
    statenpt = State(
        eos,
        temperature=t * si.KELVIN,
        pressure=p * si.PASCAL,
        composition=x * si.MOL,
        density_initialization=density_initialization,
    )
    return statenpt


def mix_gibbs_energy(
    parameters: List[List[float]],
    state: List[float],
    kij_matrix: Optional[List[List[float]]] = None,
    epsilon_ab: Optional[List[List[float]]] = None,
    density_initialization: Optional[str] = None,
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
        density_initialization: Initialization method for density ("liquid", "vapor", None)
    """
    x = np.asarray(state[2:], dtype=np.float64)  # mole fractions

    excess_g = mix_e_gibbs_energy(
        parameters, state, kij_matrix, epsilon_ab, density_initialization
    )

    return excess_g + np.sum(x * np.log(x))


def mix_ln_fugacity_coefficient_pure(
    parameters: List[List[float]],
    state: List[float],
    kij_matrix: Optional[List[List[float]]] = None,
    epsilon_ab: Optional[List[List[float]]] = None,
    density_initialization: Optional[str] = None,
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
        density_initialization: Initialization method for density ("liquid", "vapor", None)
    """
    statenpt = state_npt_feos(
        parameters=parameters,
        state=state,
        kij_matrix=kij_matrix,
        epsilon_ab=epsilon_ab,
        density_initialization=density_initialization,
    )

    return statenpt.ln_phi_pure_liquid()


def mix_ln_activity_coefficient(
    parameters: List[List[float]],
    state: List[float],
    kij_matrix: Optional[List[List[float]]] = None,
    epsilon_ab: Optional[List[List[float]]] = None,
    density_initialization: Optional[str] = None,
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
        density_initialization: Initialization method for density ("liquid", "vapor", None)

    Returns:
        out (List[float]): Natural logarithm of activity coefficients for each component.
    """

    statenpt = state_npt_feos(
        parameters=parameters,
        state=state,
        kij_matrix=kij_matrix,
        epsilon_ab=epsilon_ab,
        density_initialization=density_initialization,
    )

    return statenpt.ln_symmetric_activity_coefficient()


def mix_e_gibbs_energy(
    parameters: List[List[float]],
    state: List[float],
    kij_matrix: Optional[List[List[float]]] = None,
    epsilon_ab: Optional[List[List[float]]] = None,
    density_initialization: Optional[str] = None,
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
        density_initialization: Initialization method for density ("liquid", "vapor", None)

    Returns:
        out (float): Mixture excess Gibbs energy divided by RT.
    """

    x = np.asarray(state[2:], dtype=np.float64)  # mole fractions

    return np.sum(
        mix_ln_activity_coefficient(
            parameters, state, kij_matrix, epsilon_ab, density_initialization
        )
        * x
    )


def mix_ln_fugacity_coefficient(
    parameters: List[List[float]],
    state: List[float],
    kij_matrix: Optional[List[List[float]]] = None,
    epsilon_ab: Optional[List[List[float]]] = None,
    density_initialization: Optional[str] = None,
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
        density_initialization: Initialization method for density ("liquid", "vapor", None)

    Returns:
        out (List[float]): Natural logarithm of fugacity coefficients for each component.
    """
    statenpt = state_npt_feos(
        parameters=parameters,
        state=state,
        kij_matrix=kij_matrix,
        epsilon_ab=epsilon_ab,
        density_initialization=density_initialization,
    )

    return statenpt.ln_phi()


def mix_r_gibbs_energy(
    parameters: List[List[float]],
    state: List[float],
    kij_matrix: Optional[List[List[float]]] = None,
    epsilon_ab: Optional[List[List[float]]] = None,
    density_initialization: Optional[str] = None,
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
        density_initialization: Initialization method for density ("liquid", "vapor", None)

    Returns:
        out (float): Mixture residual Gibbs energy divided by RT.
    """
    x = np.asarray(state[2:], dtype=np.float64)  # mole fractions
    return np.sum(
        mix_ln_fugacity_coefficient(
            parameters,
            state,
            kij_matrix,
            epsilon_ab,
            density_initialization,
        )
        * x
    )


def mix_den_feos(
    parameters: List[List[float]],
    state: List[float],
    kij_matrix: Optional[List[List[float]]] = None,
    epsilon_ab: Optional[List[List[float]]] = None,
    density_initialization: Optional[str] = None,
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
        density_initialization: Initialization method for density ("liquid", "vapor", None)

    Returns:
        out (float): Mixture liquid density in mol/m^3.
    """

    statenpt = state_npt_feos(
        parameters=parameters,
        state=state,
        kij_matrix=kij_matrix,
        epsilon_ab=epsilon_ab,
        density_initialization=density_initialization,
    )

    return statenpt.density * (si.METER**3) / si.MOL


def mix_r_isobaric_heat_capacity_feos(
    parameters: List[List[float]],
    state: List[float],
    kij_matrix: Optional[List[List[float]]] = None,
    epsilon_ab: Optional[List[List[float]]] = None,
    density_initialization: Optional[str] = None,
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
        density_initialization: Initialization method for density ("liquid", "vapor", None)

    Returns:
        out (float): Residual molar isobaric heat capacity in J/(mol*K).
    """
    statenpt = state_npt_feos(
        parameters=parameters,
        state=state,
        kij_matrix=kij_matrix,
        epsilon_ab=epsilon_ab,
        density_initialization=density_initialization,
    )

    return statenpt.molar_isobaric_heat_capacity(Contributions.Residual) / (
        si.JOULE / si.MOL / si.KELVIN
    )


def mix_r_entropy_feos(
    parameters: List[List[float]],
    state: List[float],
    kij_matrix: Optional[List[List[float]]] = None,
    epsilon_ab: Optional[List[List[float]]] = None,
    density_initialization: Optional[str] = None,
) -> float:
    """
    Calculates residual entropy (J / K / mol) at NPT.

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
        out (float): Entropy (J / K / mol)
    """

    state_npt = state_npt_feos(
        parameters=parameters,
        state=state,
        kij_matrix=kij_matrix,
        epsilon_ab=epsilon_ab,
        density_initialization=density_initialization,
    )

    return state_npt.entropy(Contributions.Residual) / (si.JOULE / si.KELVIN)


def mix_r_enthalpy_feos(
    parameters: List[List[float]],
    state: List[float],
    kij_matrix: Optional[List[List[float]]] = None,
    epsilon_ab: Optional[List[List[float]]] = None,
    density_initialization: Optional[str] = None,
) -> float:
    """
    Calculates residual enthalpy (J / mol) at NPT.

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
        out (float): Enthalpy (J / mol)
    """

    state_npt = state_npt_feos(
        parameters=parameters,
        state=state,
        kij_matrix=kij_matrix,
        epsilon_ab=epsilon_ab,
        density_initialization=density_initialization,
    )

    return state_npt.enthalpy(Contributions.Residual) / (si.JOULE)
