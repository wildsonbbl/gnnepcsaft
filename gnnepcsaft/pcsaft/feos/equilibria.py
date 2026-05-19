"""Phase-equilibrium and stability routines using FEOS PC-SAFT."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import si_units as si
from feos import Contributions  # pyright: ignore[reportAttributeAccessIssue]
from feos import PhaseDiagram  # pyright: ignore[reportAttributeAccessIssue]
from feos import PhaseEquilibrium  # pyright: ignore[reportAttributeAccessIssue]
from feos import State  # pyright: ignore[reportAttributeAccessIssue]

from .core import pc_saft_mixture


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
    npoints: int = 500,
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
        npoints: Number of data points in the LLE diagram (default: 500)

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
          - xi: phase 1 molefraction of component i
          - yi: phase 2 molefraction of component i
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
        npoints=npoints,
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
          - xi: phase 1 molefraction of component i
          - yi: phase 2 molefraction of component i
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
    npoints: int = 500,
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
        npoints: Number of data points in the VLE diagram (default: 500)

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
          - xi: phase 1 molefraction of component i
          - yi: phase 2 molefraction of component i
    """
    p = state[0]  # Pressure, Pa
    eos = pc_saft_mixture(parameters, kij_matrix=kij_matrix, epsilon_ab=epsilon_ab)
    dia_t = PhaseDiagram.binary_vle(
        eos,
        temperature_or_pressure=p * si.PASCAL,
        npoints=npoints,
    )

    if len(dia_t.states) == 0:
        raise ValueError("No VLE found at the given conditions.")

    return dia_t.to_dict(Contributions.Residual)


def mix_vle_pxy_diagram_feos(
    parameters: List[List[float]],
    temperature: float,
    kij_matrix: Optional[List[List[float]]] = None,
    epsilon_ab: Optional[List[List[float]]] = None,
    npoints: int = 500,
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
        npoints: Number of data points in the VLE diagram (default: 500)

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
          - xi: phase 1 molefraction of component i
          - yi: phase 2 molefraction of component i
    """

    eos = pc_saft_mixture(parameters, kij_matrix=kij_matrix, epsilon_ab=epsilon_ab)
    dia_p = PhaseDiagram.binary_vle(
        eos,
        temperature_or_pressure=temperature * si.KELVIN,
        npoints=npoints,
    )

    if len(dia_p.states) == 0:
        raise ValueError("No VLE found at the given conditions.")

    return dia_p.to_dict(Contributions.Residual)


def mix_vlle_diagram_feos(
    parameters: List[List[float]],
    state: List[float],
    kij_matrix: Optional[List[List[float]]] = None,
    epsilon_ab: Optional[List[List[float]]] = None,
    npoints: int = 500,
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
        npoints: Number of data points in the VLLE diagram (default: 500)

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
        npoints=npoints,
    )
    if len(dia_t.states) == 0:
        raise ValueError("No VLLE found at the given conditions.")

    return dia_t.to_dict(Contributions.Residual)
