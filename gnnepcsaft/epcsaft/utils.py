"""Module for ePC-SAFT calculations. """

import numpy as np
import PCSAFTsuperanc
import teqp
import torch

# pylint: disable = E0401,E0611
from feos.eos import EquationOfState, PhaseEquilibrium, State
from feos.pcsaft import PcSaftParameters, PcSaftRecord

# pylint: enable = E0401,E0611
from si_units import KELVIN, METER, MOL, PASCAL

N_A = PCSAFTsuperanc.N_A * (1e-10) ** 3  # adjusted to angstron unit


def pure_den_teqp(parameters: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Calcules pure component density with ePC-SAFT."""

    t = state[0]  # Temperature, K

    # pylint: disable=E1101,I1101
    c = teqp.SAFTCoeffs()
    c.m = max(parameters[0], 1.0)  # units
    c.sigma_Angstrom = parameters[1]  # Å
    c.epsilon_over_k = parameters[2]  # K
    model = teqp.PCSAFTEOS([c])

    # pylint: disable=I1101
    # T tilde = T / (e / kB) https://teqp.readthedocs.io/en/latest/models/PCSAFT.html
    [ttilde_crit, ttilde_min] = PCSAFTsuperanc.get_Ttilde_crit_min(m=c.m)
    ttilde = t / c.epsilon_over_k
    ttilde = min(max(ttilde, ttilde_min), ttilde_crit)
    # Rho tilde = RhoN * sigma ** 3 https://teqp.readthedocs.io/en/latest/models/PCSAFT.html
    [tilderhol, tilderhov] = PCSAFTsuperanc.PCSAFTsuperanc_rhoLV(Ttilde=ttilde, m=c.m)
    rhol_guess, rhov_guess = [
        tilderho / (N_A * c.sigma_Angstrom**3) for tilderho in [tilderhol, tilderhov]
    ]

    rhol, rhov = model.pure_VLE_T(t, rhol_guess * 0.98, rhov_guess * 1.02, 10)
    den = rhol if state[2] == 1 else rhov

    return den


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


def pure_vp_teqp(parameters: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Calculates pure component vapor pressure with ePC-SAFT."""

    t = state[0]  # Temperature, K
    x = np.array([1.0])  # mole fraction

    m = max(parameters[0], 1.0)  # units
    s = parameters[1]  # Å
    e = parameters[2]  # K

    # pylint: disable=E1101,I1101
    c = teqp.SAFTCoeffs()
    c.m = m
    c.sigma_Angstrom = s
    c.epsilon_over_k = e
    coeffs = [c]
    model = teqp.PCSAFTEOS(coeffs)

    rho = pure_den_teqp(parameters, state)

    # P = rho * R * T * (1 + Ar01) https://teqp.readthedocs.io/en/latest/derivs/derivs.html
    p = rho * model.get_R(x) * t * (1 + model.get_Ar01(t, rho, x))

    return p


# pylint: disable = abstract-method
class DenFromTensor(torch.autograd.Function):
    """Custom `torch` function to calculate pure component density with ePC-SAFT."""

    # pylint: disable = arguments-differ
    @staticmethod
    def forward(ctx, para: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        parameters = para.cpu().numpy()
        state = state.cpu().numpy()

        ctx.parameters = parameters
        ctx.state = state

        result = np.zeros(state.shape[0])

        for i, row in enumerate(state):
            try:
                den = pure_den_feos(parameters, row)
            except RuntimeError:
                den = np.nan
            result[i] = den
        return torch.tensor(result)

    @staticmethod
    def backward(ctx, dg1: torch.Tensor):
        grad_result = dg1
        return grad_result, None


class VpFromTensor(torch.autograd.Function):
    """Custom `torch` function to calculate pure component vapor pressure with ePC-SAFT."""

    # pylint: disable = arguments-differ
    @staticmethod
    def forward(ctx, para: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        parameters = para.cpu().numpy()
        state = state.cpu().numpy()

        ctx.parameters = parameters
        ctx.state = state

        result = np.zeros(state.shape[0])

        for i, row in enumerate(state):
            try:
                vp = pure_vp_feos(parameters, row)
            except RuntimeError:
                vp = np.nan
            result[i] = vp
        return torch.tensor(result)

    @staticmethod
    def backward(ctx, dg1: torch.Tensor):
        grad_result = dg1
        return grad_result, None


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
