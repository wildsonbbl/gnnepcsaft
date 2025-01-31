"Module to calculate properties with ePC-SAFT using FEOS."

import numpy as np

# pylint: disable = E0401,E0611
from feos.eos import EquationOfState, PhaseEquilibrium, State
from feos.pcsaft import PcSaftParameters, PcSaftRecord

# pylint: enable = E0401,E0611
from si_units import KELVIN, METER, MOL, PASCAL


# pylint: disable=R0914
def pure_den_feos(parameters: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Calcules pure component density with ePC-SAFT."""

    t = state[0]  # Temperature, K

    m = max(parameters[0], 1.0)  # units
    s = parameters[1]  # Å
    e = parameters[2]  # K
    kappa_ab = parameters[3]
    epsilon_k_ab = parameters[4]  # K
    mu = parameters[5]  # Debye
    na = parameters[6]
    nb = parameters[7]

    p = state[1]  # Pa

    record = PcSaftRecord(
        m=m,
        sigma=s,
        epsilon_k=e,
        kappa_ab=kappa_ab,
        epsilon_k_ab=epsilon_k_ab,
        na=na,
        nb=nb,
        mu=mu,
    )
    para = PcSaftParameters.from_model_records([record])
    eos = EquationOfState.pcsaft(para)
    statenpt = State(
        eos,
        temperature=t * KELVIN,
        pressure=p * PASCAL,
        density_initialization="liquid",
    )

    den = statenpt.density * (METER**3) / MOL

    return den


def pure_vp_feos(parameters: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Calcules pure component density with ePC-SAFT."""

    t = state[0]  # Temperature, K

    m = max(parameters[0], 1.0)  # units
    s = parameters[1]  # Å
    e = parameters[2]  # K
    kappa_ab = parameters[3]
    epsilon_k_ab = parameters[4]  # K
    mu = parameters[5]  # Debye
    na = parameters[6]
    nb = parameters[7]

    record = PcSaftRecord(
        m=m,
        sigma=s,
        epsilon_k=e,
        kappa_ab=kappa_ab,
        epsilon_k_ab=epsilon_k_ab,
        na=na,
        nb=nb,
        mu=mu,
    )
    para = PcSaftParameters.from_model_records([record])
    eos = EquationOfState.pcsaft(para)
    vle = PhaseEquilibrium.pure(eos, temperature_or_pressure=t * KELVIN)

    assert t == vle.liquid.temperature / KELVIN

    return vle.liquid.pressure() / PASCAL


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
