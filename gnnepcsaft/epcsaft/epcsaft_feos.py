"Module to calculate properties with ePC-SAFT using FEOS."

# pylint: disable = E0401,E0611
import numpy as np
from feos.eos import (
    Contributions,
    EquationOfState,
    PhaseDiagram,
    PhaseEquilibrium,
    State,
)
from feos.pcsaft import Identifier, PcSaftParameters, PcSaftRecord, PureRecord

# pylint: enable = E0401,E0611
from si_units import JOULE, KELVIN, KILO, METER, MOL, PASCAL


def pc_saft(parameters: list) -> EquationOfState.pcsaft:
    """Returns a ePC-SAFT equation of state for a pure component."""
    parameters.append("unknown")
    parameters.append("unknown")
    parameters.append(1)
    return pc_saft_mixture([parameters])


def pc_saft_mixture(
    mixture_parameters: list, kij_matrix: list = None
) -> EquationOfState.pcsaft:
    """Returns a ePC-SAFT equation of state."""
    records = get_records(mixture_parameters)

    if kij_matrix:
        binary_records = np.asarray(kij_matrix)
    else:
        binary_records = np.zeros((len(records), len(records)))
    pcsaftparameters = PcSaftParameters.from_records(
        records, binary_records=binary_records
    )
    eos = EquationOfState.pcsaft(pcsaftparameters)
    return eos


def get_records(mixture_parameters):
    """Returns a list of PureRecord."""
    records = []
    for mol_parameters in mixture_parameters:
        records.append(
            PureRecord(
                identifier=Identifier(
                    smiles=mol_parameters[8],
                    inchi=mol_parameters[9],
                ),
                molarweight=mol_parameters[10],
                model_record=PcSaftRecord(
                    m=mol_parameters[0],  # units
                    sigma=mol_parameters[1],  # Å
                    epsilon_k=mol_parameters[2],  # K
                    kappa_ab=mol_parameters[3],
                    epsilon_k_ab=mol_parameters[4],  # K
                    mu=mol_parameters[5],  # Debye
                    na=mol_parameters[6],
                    nb=mol_parameters[7],
                ),
            )
        )

    return records


def mix_den_feos(parameters: list, state: list, kij_matrix: list = None) -> float:
    """Calcules mixture density with ePC-SAFT."""

    t = state[0]  # Temperature, K
    p = state[1]  # Pa
    x = np.asarray(state[2:])  # mole fractions

    eos = pc_saft_mixture(parameters, kij_matrix)

    statenpt = State(
        eos,
        temperature=t * KELVIN,
        pressure=p * PASCAL,
        molefracs=x,
        density_initialization="liquid",
    )

    den = statenpt.density * (METER**3) / MOL

    return den


def pure_den_feos(parameters: list, state: list) -> float:
    """Calcules pure component density with ePC-SAFT."""

    t = state[0]  # Temperature, K
    p = state[1]  # Pa

    eos = pc_saft(parameters)
    statenpt = State(
        eos,
        temperature=t * KELVIN,
        pressure=p * PASCAL,
        density_initialization="liquid",
    )

    den = statenpt.density * (METER**3) / MOL

    return den


def mix_vp_feos(parameters: list, state: list, kij_matrix: list = None) -> float:
    """Calcules mixture vapor pressure with ePC-SAFT."""

    t = state[0]  # Temperature, K
    x = np.asarray(state[2:])  # mole fractions

    eos = pc_saft_mixture(parameters, kij_matrix=kij_matrix)

    vle = PhaseEquilibrium.bubble_point(
        eos, temperature_or_pressure=t * KELVIN, liquid_molefracs=x
    )

    assert t == vle.liquid.temperature / KELVIN

    return vle.liquid.pressure() / PASCAL


def pure_vp_feos(parameters: list, state: list) -> float:
    """Calcules pure component vapor pressure with ePC-SAFT."""

    t = state[0]  # Temperature, K

    eos = pc_saft(parameters)
    vle = PhaseEquilibrium.pure(eos, temperature_or_pressure=t * KELVIN)

    assert t == vle.liquid.temperature / KELVIN

    return vle.liquid.pressure() / PASCAL


def pure_h_lv_feos(parameters: list, state: list) -> float:
    """Calcules pure component enthalpy of vaporization with ePC-SAFT."""

    t = state[0]  # Temperature, K

    eos = pc_saft(parameters)
    vle = PhaseEquilibrium.pure(eos, temperature_or_pressure=t * KELVIN)

    liquid_state = vle.liquid
    vapor_state = vle.vapor

    assert t == liquid_state.temperature / KELVIN

    return (
        vapor_state.molar_enthalpy(Contributions.Residual)
        - liquid_state.molar_enthalpy(Contributions.Residual)
    ) * (MOL / KILO / JOULE)


def pure_s_lv_feos(parameters: list, state: list) -> float:
    """Calcules pure component entropy of vaporization with ePC-SAFT."""
    t = state[0]  # Temperature, K
    eos = pc_saft(parameters)
    vle = PhaseEquilibrium.pure(eos, temperature_or_pressure=t * KELVIN)
    liquid_state = vle.liquid
    vapor_state = vle.vapor
    assert t == liquid_state.temperature / KELVIN
    return (
        vapor_state.molar_entropy(Contributions.Residual)
        - liquid_state.molar_entropy(Contributions.Residual)
    ) * (MOL * KELVIN / JOULE)


def critical_points_feos(parameters: list) -> list:
    """Calculates critical points with ePC-SAFT."""
    eos = pc_saft(parameters)
    critical_point = State.critical_point(eos)
    return [
        critical_point.temperature / KELVIN,
        critical_point.pressure() / PASCAL,
        critical_point.density * (METER**3) / MOL,
    ]


def pure_viscosity_feos(parameters: list, state: list) -> float:
    """Calcules pure component viscosity with ePC-SAFT."""
    t = state[0]  # Temperature, K
    p = state[1]  # Pa

    eos = pc_saft(parameters)
    statenpt = State(
        eos,
        temperature=t * KELVIN,
        pressure=p * PASCAL,
        density_initialization="liquid",
    )

    return statenpt.viscosity()  # / (KILO * PASCAL * SECOND)


def phase_diagram_feos(parameters: list, state: list) -> dict:
    """Calculates phase diagram with ePC-SAFT."""
    t = state[0]  # Temperature, K
    eos = pc_saft(parameters)
    phase_diagram = PhaseDiagram.pure(eos, min_temperature=t * KELVIN, npoints=200)

    return phase_diagram.to_dict(Contributions.Residual)


def parameters_gc_pcsaft(smiles: str) -> tuple:
    "Calculates PC-SAFT parameters with Group Contribution method."
    pure_record = (
        PcSaftParameters.from_json_smiles(
            [smiles],
            "./gnnepcsaft/data/gc_pcsaft/sauer2014_smarts.json",
            "./gnnepcsaft/data/gc_pcsaft/rehner2023_hetero.json",
        )
        .pure_records[0]
        .model_record
    )

    m = pure_record.m
    sigma = pure_record.sigma
    e = pure_record.epsilon_k
    mu = pure_record.mu if pure_record.mu else 0
    kab = pure_record.kappa_ab if pure_record.kappa_ab else 0
    eab = pure_record.epsilon_k_ab if pure_record.epsilon_k_ab else 0
    na = pure_record.na if pure_record.na else 0
    nb = pure_record.nb if pure_record.nb else 0

    return (m, sigma, e, kab, eab, mu, na, nb)
