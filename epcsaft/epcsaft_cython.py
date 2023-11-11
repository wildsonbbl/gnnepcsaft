"""Module for ePC-SAFT calculations. """
import numpy as np
import torch

# pylint: disable = no-name-in-module
from pcsaft import flashTQ, pcsaft_den, pcsaft_fugcoef


def gamma(x, t, p, params):
    """Calculates infinity activity coefficients"""

    x1 = (x < 0.5) * 1.0

    rho = pcsaft_den(t, p, x, params, phase="liq")

    fungcoef = pcsaft_fugcoef(t, rho, x, params).T @ x1

    rho = pcsaft_den(t, p, x1, params, phase="liq")

    fungcoefpure = pcsaft_fugcoef(t, rho, x1, params).T @ x1

    gamma1 = fungcoef / fungcoefpure

    return gamma1.squeeze()


def pure_den(parameters: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Calcules pure component density with ePC-SAFT."""
    x = np.asarray([1.0])
    t = state[0]
    p = state[1]
    phase = ["liq" if state[2] == 1 else "vap"][0]

    m = parameters[0]
    s = parameters[1]
    e = parameters[2]

    params = {"m": m, "s": s, "e": e}

    den = pcsaft_den(t, p, x, params, phase=phase)

    return den


def pure_vp(parameters: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Calculates pure component vapor pressure with ePC-SAFT."""
    x = np.asarray([1.0])
    t = state[0]

    m = parameters[0]
    s = parameters[1]
    e = parameters[2]

    params = {"m": m, "s": s, "e": e}
    vp, _, _ = flashTQ(t, 0, x, params)

    return vp


# pylint: disable = abstract-method
class DenFromTensor(torch.autograd.Function):
    """Custom `torch` function to calculate pure component density with ePC-SAFT."""

    # pylint: disable = arguments-differ
    @staticmethod
    def forward(ctx, para: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        parameters = para.numpy()
        state = state.numpy()

        ctx.parameters = parameters
        ctx.state = state

        result = np.zeros(state.shape[0])

        for i, row in enumerate(state):
            try:
                den = pure_den(parameters, row)
            # pylint: disable = broad-exception-caught
            except Exception:
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
        parameters = para.numpy()
        state = state.numpy()

        ctx.parameters = parameters
        ctx.state = state

        result = np.zeros(state.shape[0])

        for i, row in enumerate(state):
            try:
                vp = pure_vp(parameters, row)
            # pylint: disable = broad-exception-caught
            except Exception:
                vp = np.nan
            result[i] = vp
        return torch.tensor(result)

    @staticmethod
    def backward(ctx, dg1: torch.Tensor):
        grad_result = dg1
        return grad_result, None
