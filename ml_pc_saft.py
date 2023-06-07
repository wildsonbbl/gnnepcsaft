import jax.numpy as jnp
import jax
import epcsaft


from jax.config import config

config.update("jax_enable_x64", True)


def epcsaft_layer(parameters: jax.Array, state: jax.Array) -> jax.Array:
    x = jnp.asarray([[state[0]], [state[1]]])
    t = state[2]
    p = state[3]
    phase = state[4]
    fntype = state[5]

    kij = parameters[-3:]
    parameters = parameters[:-3].reshape(2, 7)

    m = parameters[:, 0][..., jnp.newaxis]
    s = parameters[:, 1][..., jnp.newaxis]
    e = parameters[:, 2][..., jnp.newaxis]
    vol_a = parameters[:, 3][..., jnp.newaxis]
    e_assoc = parameters[:, 4][..., jnp.newaxis]
    dipm = parameters[:, 5][..., jnp.newaxis]
    dip_num = parameters[:, 6][..., jnp.newaxis]
    z = jnp.zeros_like(m)
    dielc = jnp.zeros_like(m)

    k_ij = jnp.asarray([[0.0, kij[0]], [kij[0], 0.0]])
    l_ij = jnp.asarray([[0.0, kij[1]], [kij[1], 0.0]])
    khb_ij = jnp.asarray([[0.0, kij[2]], [kij[2], 0.0]])

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

    return result.squeeze()


batch_pcsaft_layer = jax.jit(jax.vmap(epcsaft_layer, (0, 0)))

grad_pcsaft_layer = jax.jit(jax.vmap(jax.jacfwd(epcsaft_layer), (0, 0)))


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

    return gamma1.squeeze()


def VP(
    x, m, s, e, t, p, k_ij, l_ij, khb_ij, e_assoc, vol_a, dipm, dip_num, z, dielc, phase
):
    return epcsaft.pcsaft_VP(
        x, m, s, e, t, p, k_ij, l_ij, khb_ij, e_assoc, vol_a, dipm, dip_num, z, dielc
    ).squeeze()


def epcsaft_layer_test(parameters: jax.Array, state: jax.Array) -> jax.Array:
    x = jnp.asarray([[state[0]], [state[1]]])
    t = state[2]
    p = state[3]
    phase = state[4]
    fntype = state[5]

    kij = parameters[-3:]
    parameters = parameters[:-3].reshape(2, 7)

    m = parameters[:, 0][..., jnp.newaxis]
    s = parameters[:, 1][..., jnp.newaxis]
    e = parameters[:, 2][..., jnp.newaxis]
    vol_a = parameters[:, 3][..., jnp.newaxis]
    e_assoc = parameters[:, 4][..., jnp.newaxis]
    dipm = parameters[:, 5][..., jnp.newaxis]
    dip_num = parameters[:, 6][..., jnp.newaxis]
    z = jnp.zeros_like(m)
    dielc = jnp.zeros_like(m)

    k_ij = jnp.asarray([[0.0, kij[0]], [kij[0], 0.0]])
    l_ij = jnp.asarray([[0.0, kij[1]], [kij[1], 0.0]])
    khb_ij = jnp.asarray([[0.0, kij[2]], [kij[2], 0.0]])

    result = epcsaft.pcsaft_VP(
        x, m, s, e, t, k_ij, l_ij, khb_ij, e_assoc, vol_a, dipm, dip_num, z, dielc
    )

    return result.squeeze()


batch_epcsaft_layer_test = jax.jit(jax.vmap(epcsaft_layer_test, (0, 0)))


def epcsaft_pure(parameters: jax.Array, state: jax.Array) -> jax.Array:
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
    z = jnp.zeros_like(m)
    dielc = jnp.zeros_like(m)

    k_ij = jnp.zeros_like(m)
    l_ij = jnp.zeros_like(m)
    khb_ij = jnp.zeros_like(m)

    result = jax.lax.cond(
        fntype == 1,
        epcsaft.pcsaft_den,
        VP,
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


batch_epcsaft_pure = jax.jit(jax.vmap(epcsaft_pure, (None, 0)))
