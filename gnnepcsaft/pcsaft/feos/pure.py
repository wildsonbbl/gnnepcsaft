"""Pure-component thermodynamic properties using FEOS PC-SAFT."""

from typing import Dict, List, Tuple

import numpy as np
import si_units as si
from feos import Contributions  # pyright: ignore[reportAttributeAccessIssue]
from feos import (
    HelmholtzEnergyFunctional,  # pyright: ignore[reportAttributeAccessIssue]
)
from feos import Parameters  # pyright: ignore[reportAttributeAccessIssue]
from feos import PhaseDiagram  # pyright: ignore[reportAttributeAccessIssue]
from feos import PhaseEquilibrium  # pyright: ignore[reportAttributeAccessIssue]
from feos import State  # pyright: ignore[reportAttributeAccessIssue]
from feos import SurfaceTensionDiagram  # pyright: ignore[reportAttributeAccessIssue]

from .core import get_records, pc_saft


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
