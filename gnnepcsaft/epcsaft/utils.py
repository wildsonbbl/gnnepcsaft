"""Module for ePC-SAFT calculations. """

import numpy as np
import PCSAFTsuperanc
import teqp
import torch

# pylint: disable=I1101
N_A = PCSAFTsuperanc.N_A * (1e-10) ** 3


def pure_den(parameters: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Calcules pure component density with ePC-SAFT."""

    t = state[0]  # Temperatue

    m = max(parameters[0], 1.0)
    s = parameters[1]
    e = parameters[2]
    # pylint: disable=E1101,I1101,C0103
    [Ttilde_crit, Ttilde_min] = PCSAFTsuperanc.get_Ttilde_crit_min(m=m)
    Ttilde = t / e
    Ttilde = min(max(Ttilde, Ttilde_min), Ttilde_crit)
    [tilderhoL, tilderhoV] = PCSAFTsuperanc.PCSAFTsuperanc_rhoLV(Ttilde=Ttilde, m=m)

    rhoL, rhoV = [tilderho / (N_A * s**3) for tilderho in [tilderhoL, tilderhoV]]

    den = rhoL if state[2] == 1 else rhoV

    return den


def pure_vp(parameters: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Calculates pure component vapor pressure with ePC-SAFT."""

    t = state[0]
    z = np.array([1.0])

    m = max(parameters[0], 1.0)
    s = parameters[1]
    e = parameters[2]

    # pylint: disable=E1101,I1101
    c = teqp.SAFTCoeffs()

    c.m = m
    c.sigma_Angstrom = s
    c.epsilon_over_k = e
    coeffs = [c]
    model = teqp.PCSAFTEOS(coeffs)

    rho = pure_den(parameters, state)

    p = rho * model.get_R(z) * t * (1 + model.get_Ar01(t, rho, z))

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
            den = pure_den(parameters, row)
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
            vp = pure_vp(parameters, row)
            result[i] = vp
        return torch.tensor(result)

    @staticmethod
    def backward(ctx, dg1: torch.Tensor):
        grad_result = dg1
        return grad_result, None
