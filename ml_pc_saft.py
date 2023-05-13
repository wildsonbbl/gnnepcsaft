import jax.numpy as jnp
import jax
from jax import dlpack as jdlpack
from torch.utils import dlpack as tdlpack
import epcsaft
import torch

from jax.config import config

config.update("jax_enable_x64", True)


@jax.jit
def epcsaft_layer(parameters: jax.Array, state: jax.Array) -> jax.Array:
    x = jnp.asarray([[state[0]], [state[1]]])
    t = state[2]
    p = state[3]
    phase = state[4]
    fntype = state[5]

    m = parameters[:, 0][..., jnp.newaxis]
    s = parameters[:, 1][..., jnp.newaxis]
    e = parameters[:, 2][..., jnp.newaxis]
    vol_a = parameters[:, 3][..., jnp.newaxis]
    e_assoc = parameters[:, 4][..., jnp.newaxis]
    dipm = parameters[:, 5][..., jnp.newaxis]
    dip_num = parameters[:, 6][..., jnp.newaxis]
    z = 0
    dielc = 0

    k_ij = jnp.asarray([[0.0, parameters[0, 7]], [parameters[1, 7], 0.0]])
    l_ij = jnp.asarray([[0.0, parameters[0, 8]], [parameters[1, 8], 0.0]])
    khb_ij = jnp.asarray([[0.0, parameters[0, 9]], [parameters[1, 9], 0.0]])

    result = jax.lax.cond(
        fntype == 1,
        epcsaft.pcsaft_den,
        gamma,
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

    failed = jnp.zeros_like(result)
    result = jax.lax.cond(result>0, lambda result: result , lambda result: failed, result)

    return result


epcsaft_layer_batch = jax.jit(jax.vmap(epcsaft_layer))


@jax.jit
def loss(parameters: jax.Array, state: jax.Array) -> jax.Array:
    y = state[:, 6]
    results = epcsaft_layer_batch(parameters, state)
    return jnp.abs(1.0 - results / y).sum()


loss_grad = jax.jit(jax.jacfwd(loss))


def gamma(
    x, m, s, e, t, p, k_ij, l_ij, khb_ij, e_assoc, vol_a, dipm, dip_num, z, dielc, phase
):
    x1 = (x < 0.5) * 1.0

    rho = epcsaft.pcsaft_den(
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
        epcsaft.pcsaft_fugcoef(
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

    rho = epcsaft.pcsaft_den(
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
        epcsaft.pcsaft_fugcoef(
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
    
    return gamma1[0,0]


def VP(
    x, m, s, e, t, p, k_ij, l_ij, khb_ij, e_assoc, vol_a, dipm, dip_num, z, dielc, phase
):
    return epcsaft.pcsaft_VP(
        x, m, s, e, t, k_ij, l_ij, khb_ij, e_assoc, vol_a, dipm, dip_num, z, dielc
    )


class PCSAFTLOSS(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, state):
        parameters = tdlpack.to_dlpack(input)
        parameters = jdlpack.from_dlpack(parameters)

        state = tdlpack.to_dlpack(state)
        state = jdlpack.from_dlpack(state)

        ctx.parameters = parameters
        ctx.state = state
        result = loss(parameters, state)
        result = jdlpack.to_dlpack(result)
        result = tdlpack.from_dlpack(result)
        return result

    @staticmethod
    def backward(ctx, dg1):
        grad_result = loss_grad(ctx.parameters, ctx.state)
        grad_result = jdlpack.to_dlpack(grad_result)
        grad_result = dg1 * tdlpack.from_dlpack(grad_result)
        return grad_result, None


@jax.jit
def epcsaft_layer_test(parameters: jax.Array, state: jax.Array) -> jax.Array:
    x = jnp.asarray([[state[0]], [state[1]]])
    t = state[2]
    p = state[3]
    phase = state[4]
    fntype = state[5]

    m = parameters[:, 0][..., jnp.newaxis]
    s = parameters[:, 1][..., jnp.newaxis]
    e = parameters[:, 2][..., jnp.newaxis]
    vol_a = parameters[:, 3][..., jnp.newaxis]
    e_assoc = parameters[:, 4][..., jnp.newaxis]
    dipm = parameters[:, 5][..., jnp.newaxis]
    dip_num = parameters[:, 6][..., jnp.newaxis]
    z = 0
    dielc = 0

    k_ij = jnp.asarray([[0.0, parameters[0, 7]], [parameters[1, 7], 0.0]])
    l_ij = jnp.asarray([[0.0, parameters[0, 8]], [parameters[1, 8], 0.0]])
    khb_ij = jnp.asarray([[0.0, parameters[0, 9]], [parameters[1, 9], 0.0]])

    result = epcsaft.pcsaft_VP(
        x, m, s, e, t, k_ij, l_ij, khb_ij, e_assoc, vol_a, dipm, dip_num, z, dielc
    )

    return result


epcsaft_layer_test_batch = jax.jit(jax.vmap(epcsaft_layer_test))


@jax.jit
def loss_test(parameters: jax.Array, state: jax.Array) -> jax.Array:
    y = state[:, 6]
    results = epcsaft_layer_test_batch(parameters, state)
    return jnp.abs(1.0 - results / y).sum()


class PCSAFTLOSS_test(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, state):
        parameters = tdlpack.to_dlpack(input)
        parameters = jdlpack.from_dlpack(parameters)

        state = tdlpack.to_dlpack(state)
        state = jdlpack.from_dlpack(state)

        ctx.parameters = parameters
        ctx.state = state
        result = loss_test(parameters, state)
        result = jdlpack.to_dlpack(result)
        result = tdlpack.from_dlpack(result)
        return result

    @staticmethod
    def backward(ctx, dg1):
        grad_result = dg1
        return grad_result, None
