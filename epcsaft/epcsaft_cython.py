import numpy as np
import torch
from pcsaft import (dielc_water, flashPQ, flashTQ, pcsaft_ares, pcsaft_cp,
                    pcsaft_dadt, pcsaft_den, pcsaft_fugcoef, pcsaft_gres,
                    pcsaft_hres, pcsaft_Hvap, pcsaft_osmoticC, pcsaft_p,
                    pcsaft_sres)


def gamma(
    x, m, s, e, t, p, k_ij, l_ij, khb_ij, e_assoc, vol_a, dipm, dip_num, z, dielc, phase
):
    x1 = (x < 0.5) * 1.0

    rho = pcsaft_den(
        x,
        m,
        s,
        e,
        t,
        p,
        k_ij,
        l_ij,
        khb_ij,
        e_assoc,
        vol_a,
        dipm,
        dip_num,
        z,
        dielc,
        1.0,
    )

    fungcoef = (
        pcsaft_fugcoef(
            x,
            m,
            s,
            e,
            t,
            rho,
            k_ij,
            l_ij,
            khb_ij,
            e_assoc,
            vol_a,
            dipm,
            dip_num,
            z,
            dielc,
        ).T
        @ x1
    )

    rho = pcsaft_den(
        x1,
        m,
        s,
        e,
        t,
        p,
        k_ij,
        l_ij,
        khb_ij,
        e_assoc,
        vol_a,
        dipm,
        dip_num,
        z,
        dielc,
        1.0,
    )

    fungcoefpure = (
        pcsaft_fugcoef(
            x1,
            m,
            s,
            e,
            t,
            rho,
            k_ij,
            l_ij,
            khb_ij,
            e_assoc,
            vol_a,
            dipm,
            dip_num,
            z,
            dielc,
        ).T
        @ x1
    )

    gamma1 = fungcoef / fungcoefpure

    return gamma1.squeeze()


def epcsaft_pure_den(parameters: np.ndarray, state: np.ndarray) -> np.ndarray:
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


def epcsaft_pure_VP(parameters: np.ndarray, state: np.ndarray) -> np.ndarray:
    x = np.asarray([1.0])
    t = state[0]
    p = state[1]
    phase = ["liq" if state[2] == 1 else "vap"][0]

    m = parameters[0]
    s = parameters[1]
    e = parameters[2]

    params = {"m": m, "s": s, "e": e}
    vp, xl, xv = flashTQ(t, 0, x, params)

    return vp


class PCSAFT_den(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        parameters = input.numpy()
        state = state.numpy()

        ctx.parameters = parameters
        ctx.state = state

        result = np.zeros(state.shape[0])

        for i, row in enumerate(state):
            try:
                den = epcsaft_pure_den(parameters, row)
            except:
                den = np.nan
            result[i] = den
        return torch.tensor(result)

    @staticmethod
    def backward(ctx, dg1: torch.Tensor):
        grad_result = dg1
        return grad_result, None


class PCSAFT_vp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        parameters = input.numpy()
        state = state.numpy()

        ctx.parameters = parameters
        ctx.state = state

        result = np.zeros(state.shape[0])

        for i, row in enumerate(state):
            try:
                vp = epcsaft_pure_VP(parameters, row)
            except:
                vp = np.nan
            result[i] = vp
        return torch.tensor(result)

    @staticmethod
    def backward(ctx, dg1: torch.Tensor):
        grad_result = dg1
        return grad_result, None
