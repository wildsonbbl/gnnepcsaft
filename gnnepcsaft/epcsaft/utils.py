"""Module for ePC-SAFT calculations. """

import numpy as np
import torch

from .epcsaft_feos import *  # pylint: disable=wildcard-import,unused-wildcard-import
from .epcsaft_teqp import *  # pylint: disable=wildcard-import,unused-wildcard-import


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
