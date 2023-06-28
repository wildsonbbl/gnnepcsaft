import jax.numpy as jnp
import jax
from jax import dlpack as jdlpack
from torch.utils import dlpack as tdlpack
from epcsaft_complete import pcsaft_den, pcsaft_VP, pcsaft_fugcoef
import torch


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


def epcsaft_pure_den(parameters: jax.Array, state: jax.Array) -> jax.Array:
    x = jnp.asarray([[1.0]])
    t = state[0]
    p = state[1]
    phase = state[2]
    fntype = state[3]

    m = parameters[0].reshape(1, 1)
    s = parameters[1].reshape(1, 1)
    e = parameters[2].reshape(1, 1)
    vol_a = parameters[3].reshape(1, 1)
    e_assoc = parameters[4].reshape(1, 1)
    dipm = parameters[5].reshape(1, 1)
    dip_num = parameters[6].reshape(1, 1)
    z = parameters[7].reshape(1, 1)
    dielc = parameters[8].reshape(1, 1)

    k_ij = jnp.zeros_like(m)
    l_ij = jnp.zeros_like(m)
    khb_ij = jnp.zeros_like(m)

    result = pcsaft_den(
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
        phase,
    )

    return result.squeeze()


vmap_den = jax.vmap(epcsaft_pure_den, (None, 0))
grad_den = jax.vmap(jax.jacfwd(epcsaft_pure_den), (None, 0))


def epcsaft_pure_VP(parameters: jax.Array, state: jax.Array) -> jax.Array:
    x = jnp.asarray([[1.0]])
    t = state[0]
    p = state[1]
    phase = state[2]
    fntype = state[3]

    m = parameters[0].reshape(1, 1)
    s = parameters[1].reshape(1, 1)
    e = parameters[2].reshape(1, 1)
    vol_a = parameters[3].reshape(1, 1)
    e_assoc = parameters[4].reshape(1, 1)
    dipm = parameters[5].reshape(1, 1)
    dip_num = parameters[6].reshape(1, 1)
    z = parameters[7].reshape(1, 1)
    dielc = parameters[8].reshape(1, 1)

    k_ij = jnp.zeros_like(m)
    l_ij = jnp.zeros_like(m)
    khb_ij = jnp.zeros_like(m)

    result = pcsaft_VP(
        x, m, s, e, t, p, k_ij, l_ij, khb_ij, e_assoc, vol_a, dipm, dip_num, z, dielc
    ).squeeze()

    return result


vmap_VP = jax.vmap(epcsaft_pure_VP, (None, 0))


class PCSAFT_den(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        parameters = tdlpack.to_dlpack(input)
        parameters = jdlpack.from_dlpack(parameters)

        state = tdlpack.to_dlpack(state)
        state = jdlpack.from_dlpack(state)

        ctx.parameters = parameters
        ctx.state = state
        result = vmap_den(parameters, state)
        result = jdlpack.to_dlpack(result)
        result = tdlpack.from_dlpack(result)
        return result

    @staticmethod
    def backward(ctx, dg1: torch.Tensor):
        grad_result = grad_den(ctx.parameters, ctx.state)
        grad_result = jdlpack.to_dlpack(grad_result)
        grad_result = tdlpack.from_dlpack(grad_result)
        grad_result = dg1[..., None] * grad_result
        return grad_result, None


class PCSAFT_vp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        parameters = tdlpack.to_dlpack(input)
        parameters = jdlpack.from_dlpack(parameters)

        state = tdlpack.to_dlpack(state)
        state = jdlpack.from_dlpack(state)

        result = vmap_VP(parameters, state)
        result = jdlpack.to_dlpack(result)
        result = tdlpack.from_dlpack(result)
        return result

    @staticmethod
    def backward(ctx, dg1: torch.Tensor):
        grad_result = dg1.sum()
        return grad_result, None
