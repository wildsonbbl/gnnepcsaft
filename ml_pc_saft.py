

import epcsaft
import jax
import jax.numpy as jnp


def rho(parameters: jax.Array, state: jax.Array) -> jax.Array:
    x = jnp.asarray(
        [[state[0]],
         [state[1]]]
    )
    t = state[2]
    p = state[3]
    m = parameters[:, 0][..., jnp.newaxis]
    s = parameters[:, 1][..., jnp.newaxis]
    e = parameters[:, 2][..., jnp.newaxis]
    e_assoc = parameters[:, 4][..., jnp.newaxis]
    vol_a = parameters[:, 3][..., jnp.newaxis]
    dipm = parameters[:, 5][..., jnp.newaxis]
    dip_num = parameters[:, 6][..., jnp.newaxis]
    z = parameters[:, 7][..., jnp.newaxis]
    dielc = parameters[:, 8][..., jnp.newaxis]
    phase = state[4]
    k_ij = jnp.asarray(
        [[0.0, parameters[0, 9]],
         [parameters[1, 9], 0.0]]
    )
    l_ij = jnp.asarray(
        [[0.0, parameters[0, 10]],
         [parameters[1, 10], 0.0]]
    )
    khb_ij = jnp.asarray(
        [[0.0, parameters[0, 11]],
         [parameters[1, 11], 0.0]]
    )

    rho = epcsaft.pcsaft_den(x, m, s, e, t, p, k_ij, l_ij,
                             khb_ij, e_assoc, vol_a, dipm,
                             dip_num, z, dielc, phase)

    return rho


def ActivityCoefficient(parameters: jax.Array, state: jax.Array) -> jax.Array:

    x = jnp.asarray(
        [[state[0]],
         [state[1]]]
    )
    t = state[2]
    p = state[3]
    m = parameters[:, 0][..., jnp.newaxis]
    s = parameters[:, 1][..., jnp.newaxis]
    e = parameters[:, 2][..., jnp.newaxis]
    e_assoc = parameters[:, 4][..., jnp.newaxis]
    vol_a = parameters[:, 3][..., jnp.newaxis]
    dipm = parameters[:, 5][..., jnp.newaxis]
    dip_num = parameters[:, 6][..., jnp.newaxis]
    z = parameters[:, 7][..., jnp.newaxis]
    dielc = parameters[:, 8][..., jnp.newaxis]
    phase = state[4]
    k_ij = jnp.asarray(
        [[0.0, parameters[0, 9]],
         [parameters[1, 9], 0.0]]
    )
    l_ij = jnp.asarray(
        [[0.0, parameters[0, 10]],
         [parameters[1, 10], 0.0]]
    )
    khb_ij = jnp.asarray(
        [[0.0, parameters[0, 11]],
         [parameters[1, 11], 0.0]]
    )

    x1 = (x < 0.5)*1.0

    rho = epcsaft.pcsaft_den(x, m, s, e, t, p, k_ij, l_ij,
                             khb_ij, e_assoc, vol_a, dipm,
                             dip_num, z, dielc, phase)
    
    fungcoef = epcsaft.pcsaft_fugcoef(x, m, s, e, t, rho, k_ij, l_ij,
                                        khb_ij, e_assoc, vol_a, dipm,
                                        dip_num, z, dielc).T @ x1
    

    rho = epcsaft.pcsaft_den(x1, m, s, e, t, p, k_ij, l_ij,
                             khb_ij, e_assoc, vol_a, dipm,
                             dip_num, z, dielc, phase)
    
    fungcoef01 = epcsaft.pcsaft_fugcoef(x1, m, s, e, t, rho, k_ij, l_ij,
                                        khb_ij, e_assoc, vol_a, dipm,
                                        dip_num, z, dielc).T @ x1
    

    gamma1 = (fungcoef/fungcoef01)
    return gamma1[0,0]
