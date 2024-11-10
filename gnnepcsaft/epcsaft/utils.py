"""Module for ePC-SAFT calculations. """

# pylint: disable=I1101,E0611
import numpy as np
import PCSAFTsuperanc
import teqp
import torch
from pcsaft import flashTQ, pcsaft_den

N_A = PCSAFTsuperanc.N_A * (1e-10) ** 3  # adjusted to angstron unit


def pure_den_teqp(parameters: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Calcules pure component density with ePC-SAFT."""

    t = state[0]  # Temperature, K

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

    # pylint: disable=I1101
    # T tilde = T / (e / kB) https://teqp.readthedocs.io/en/latest/models/PCSAFT.html
    [ttilde_crit, ttilde_min] = PCSAFTsuperanc.get_Ttilde_crit_min(m=m)
    ttilde = t / e
    ttilde = min(max(ttilde, ttilde_min), ttilde_crit)
    # Rho tilde = RhoN * sigma ** 3 https://teqp.readthedocs.io/en/latest/models/PCSAFT.html
    [tilderhol, tilderhov] = PCSAFTsuperanc.PCSAFTsuperanc_rhoLV(Ttilde=ttilde, m=m)
    rhol_guess, rhov_guess = [
        tilderho / (N_A * s**3) for tilderho in [tilderhol, tilderhov]
    ]

    rhol, rhov = model.pure_VLE_T(t, rhol_guess * 0.98, rhov_guess * 1.02, 10)
    den = rhol if state[2] == 1 else rhov

    return den


def pure_den_pcsaft(parameters: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Calcules pure component density with ePC-SAFT."""

    x = np.asarray([1.0])
    phase = ["liq" if state[2] == 1 else "vap"][0]
    t = state[0]  # Temperature, K

    m = np.asarray(max(parameters[0], 1.0))  # units
    s = parameters[1]  # Å
    e = parameters[2]  # K
    p = state[1]

    params = {"m": m, "s": s, "e": e}

    den = pcsaft_den(t, p, x, params, phase=phase)

    return den


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

    rho = pure_den_pcsaft(parameters, state)

    # P = rho * R * T * (1 + Ar01) https://teqp.readthedocs.io/en/latest/derivs/derivs.html
    p = rho * model.get_R(x) * t * (1 + model.get_Ar01(t, rho, x))

    return p


def pure_vp_pcsaft(parameters: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Calculates pure component vapor pressure with ePC-SAFT."""
    x = np.asarray([1.0])
    t = state[0]  # Temperature, K

    m = np.asarray(max(parameters[0], 1.0))  # units
    s = parameters[1]  # Å
    e = parameters[2]  # K

    params = {"m": m, "s": s, "e": e}
    p, _, _ = flashTQ(t, 0, x, params)

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
            den = pure_den_pcsaft(parameters, row)
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
            vp = pure_vp_teqp(parameters, row)
            result[i] = vp
        return torch.tensor(result)

    @staticmethod
    def backward(ctx, dg1: torch.Tensor):
        grad_result = dg1
        return grad_result, None
