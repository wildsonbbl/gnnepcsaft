"""
ePC-SAFT implementation with jax
---------------
Code reference: `https://github.com/zmeri/PC-SAFT`
"""

# @author: Wildson Lima

import jax
import jax.numpy as np

KB = 1.380648465952442093e-23  # Boltzmann constant, J K^-1
N_AV = 6.022140857e23  # Avogadro's number
E_CHRG = 1.6021766208  # elementary charge, units of coulomb / 1e-19
PERM_VAC = 8.854187817  # permittivity in vacuum, C V^-1 Angstrom^-1 / 1e-22
E_CHRG_P10 = 1e-19
PERM_VAC_P10 = 1e-22


# pylint: disable=R0914
@jax.jit
def pcsaft_ares(x, t, rho, params):
    """
    Calculates the residual Helmholtz energy.

    Parameters
    ----------
    x : ndarray, shape (n,1)
        Mole fractions of each component. It has a length of n, where n is
        the number of components in the system.
    m : ndarray, shape (n,1)
        Segment number for each component.
    s : ndarray, shape (n,1)
        Segment diameter for each component. For ions this is the diameter of
        the hydrated ion. Units of Angstrom.
    e : ndarray, shape (n,1)
        Dispersion energy of each component. For ions this is the dispersion
        energy of the hydrated ion. Units of K.
    t : float
        Temperature (K)
    rho : float
        Molar density (mol / m^3)
    k_ij : ndarray, shape (n,n)
        Binary interaction parameters between components in the mixture for dispersion energy.
        (dimensions: ncomp x ncomp)
    l_ij : ndarray, shape (n,n)
        Binary interaction parameters between components in the mixture for segment diameter.
        (dimensions: ncomp x ncomp)
    khb_ij : ndarray, shape (n,n)
        Binary interaction parameters between components in the mixture for association energy.
        (dimensions: ncomp x ncomp)
    e_assoc : ndarray, shape (n,1)
        Association energy of the associating components. For non associating
        compounds this is set to 0. Units of K.
    vol_a : ndarray, shape (n,1)
        Effective association volume of the associating components. For non
        associating compounds this is set to 0.
    dipm : ndarray, shape (n,1)
        Dipole moment of the polar components. For components where the dipole
        term is not used this is set to 0. Units of Debye.
    dip_num : ndarray, shape (n,1)
        The effective number of dipole functional groups on each component
        molecule.
    z : ndarray, shape (n,1)
        Charge number of the ions.
    dielc : ndarray, shape (n,1)
        relative permittivity of each component of the medium to be used for electrolyte
        calculations.

    Returns
    -------
    ares : float
        Residual Helmholtz energy (J mol^{-1})
    """
    m = params["m"]
    s = params["s"]
    e = params["e"]
    k_ij = params["k_ij"]
    khb_ij = params["khb_ij"]
    l_ij = params["l_ij"]
    e_assoc = params["e_assoc"]
    vol_a = params["vol_a"]
    dipm = params["dipm"]
    dip_num = params["dip_num"]
    z = params["z"]
    dielc = params["dielc"]
    ncomp = x.shape[0]  # number of components

    d = s * (1.0 - 0.12 * np.exp(-3 * e / t))

    den = rho * N_AV / 1.0e30

    a0 = np.asarray(
        [
            0.910563145,
            0.636128145,
            2.686134789,
            -26.54736249,
            97.75920878,
            -159.5915409,
            91.29777408,
        ]
    )
    a1 = np.asarray(
        [
            -0.308401692,
            0.186053116,
            -2.503004726,
            21.41979363,
            -65.25588533,
            83.31868048,
            -33.74692293,
        ]
    )
    a2 = np.asarray(
        [
            -0.090614835,
            0.452784281,
            0.596270073,
            -1.724182913,
            -4.130211253,
            13.77663187,
            -8.672847037,
        ]
    )
    b0 = np.asarray(
        [
            0.724094694,
            2.238279186,
            -4.002584949,
            -21.00357682,
            26.85564136,
            206.5513384,
            -355.6023561,
        ]
    )
    b1 = np.asarray(
        [
            -0.575549808,
            0.699509552,
            3.892567339,
            -17.21547165,
            192.6722645,
            -161.8264617,
            -165.2076935,
        ]
    )
    b2 = np.asarray(
        [
            0.097688312,
            -0.255757498,
            -9.155856153,
            20.64207597,
            -38.80443005,
            93.62677408,
            -29.66690559,
        ]
    )

    n = np.arange(4)[..., np.newaxis]
    zeta = np.pi / 6.0 * den * (d.T**n) @ (x * m)

    eta = zeta[3]
    m_avg = x.T @ m

    s_ij = (s + s.T) / 2.0 * (1 - l_ij)
    e_ij = np.sqrt(e @ e.T) * (1 - k_ij)
    m2es3 = np.sum((x @ x.T) * (m @ m.T) * (e_ij / t) * s_ij**3)
    m2e2s3 = np.sum((x @ x.T) * (m @ m.T) * (e_ij / t) ** 2 * s_ij**3)
    ghs = (
        1.0 / (1.0 - zeta[3])
        + (d @ d.T) / (d + d.T) * 3.0 * zeta[2] / (1.0 - zeta[3]) ** 2
        + ((d @ d.T) / (d + d.T)) ** 2 * 2.0 * zeta[2] ** 2 / (1.0 - zeta[3]) ** 3
    )

    ares_hs = (
        1
        / zeta[0]
        * (
            3 * zeta[1] * zeta[2] / (1 - zeta[3])
            + zeta[2] ** 3 / (zeta[3] * (1 - zeta[3]) ** 2)
            + (zeta[2] ** 3 / zeta[3] ** 2 - zeta[0]) * np.log(1 - zeta[3])
        )
    )

    a = (
        a0
        + (m_avg - 1.0) / m_avg * a1
        + (m_avg - 1.0) / m_avg * (m_avg - 2.0) / m_avg * a2
    )
    b = (
        b0
        + (m_avg - 1) / m_avg * b1
        + (m_avg - 1.0) / m_avg * (m_avg - 2.0) / m_avg * b2
    )

    idx = np.arange(7)
    i1 = np.sum(a * eta**idx)
    i2 = np.sum(b * eta**idx)
    c1 = 1.0 / (
        1.0
        + m_avg * (8 * eta - 2 * eta**2) / (1 - eta) ** 4
        + (1 - m_avg)
        * (20 * eta - 27 * eta**2 + 12 * eta**3 - 2 * eta**4)
        / ((1 - eta) * (2 - eta)) ** 2
    )

    ghs_diag = np.diagonal(ghs)[..., np.newaxis]

    summ = np.sum((x * (m - 1.0)) * (np.log(ghs_diag)))

    ares_hc = m_avg * ares_hs - summ
    ares_disp = -2 * np.pi * den * i1 * m2es3 - np.pi * den * m_avg * c1 * i2 * m2e2s3

    s_ij_diag = np.diagonal(s_ij)[..., np.newaxis]

    polar_params = (m, s, e, dipm, dip_num, e_ij, s_ij, eta)
    assoc_params = (e_assoc, vol_a, khb_ij, s_ij_diag, s_ij, ghs, ncomp)
    ion_params = (s, z, dielc)

    ares = (
        ares_hc
        + ares_disp
        + ares_polar(x, t, den, polar_params)
        + ares_assoc(x, t, den, assoc_params)
        + ares_ion(x, t, den, ion_params)
    )

    return ares.squeeze()


@jax.jit
def xa_find(xa_guess, delta_ij, den, x):
    """Iterate over this function in order to solve for XA"""

    return 1.0 / (1.0 + den * np.sum(x * xa_guess * delta_ij[..., np.newaxis], axis=1))


def ares_polar(x, t, den, params):
    "Polar term for ePC-SAFT."

    # Dipole term (Gross and Vrabec term) --------------------------------------
    # Gross, Joachim, e Jadran Vrabec. “An Equation-of-State Contribution for
    # Polar Components: Dipolar Molecules”. AIChE Journal 52, nº 3 (2006): 1194–1204.
    #  https://doi.org/10.1002/aic.10683.
    # Held, Christoph, Thomas Reschke, Sultan Mohammad, Armando Luza, e Gabriele Sadowski.
    #  “EPC-SAFT Revised”. Chemical Engineering Research and Design,
    #  Advances in Thermodynamics for Chemical Process and Product Design, 92,
    #  nº 12 (1º de dezembro de 2014): 2884–97. https://doi.org/10.1016/j.cherd.2014.05.017.

    m, s, e, dipm, dip_num, e_ij, s_ij, eta = params

    a0dip = np.asarray([0.3043504, -0.1358588, 1.4493329, 0.3556977, -2.0653308])[
        ..., np.newaxis, np.newaxis
    ]
    a1dip = np.asarray([0.9534641, -1.8396383, 2.0131180, -7.3724958, 8.2374135])[
        ..., np.newaxis, np.newaxis
    ]
    a2dip = np.asarray([-1.1610080, 4.5258607, 0.9751222, -12.281038, 5.9397575])[
        ..., np.newaxis, np.newaxis
    ]
    b0dip = np.asarray([0.2187939, -1.1896431, 1.1626889, 0, 0.0])[
        ..., np.newaxis, np.newaxis
    ]
    b1dip = np.asarray([-0.5873164, 1.2489132, -0.5085280, 0, 0])[
        ..., np.newaxis, np.newaxis
    ]
    b2dip = np.asarray([3.4869576, -14.915974, 15.372022, 0, 0])[
        ..., np.newaxis, np.newaxis
    ]

    c0dip = np.asarray([-0.0646774, 0.1975882, -0.8087562, 0.6902849, 0])[
        ..., np.newaxis, np.newaxis, np.newaxis
    ]
    c1dip = np.asarray([-0.9520876, 2.9924258, -2.3802636, -0.2701261, 0])[
        ..., np.newaxis, np.newaxis, np.newaxis
    ]
    c2dip = np.asarray([-0.6260979, 1.2924686, 1.6542783, -3.4396744, 0])[
        ..., np.newaxis, np.newaxis, np.newaxis
    ]

    idxd = np.arange(5)

    # conversion factor, see the note below Table 2 in Gross and Vrabec 2006
    conv = 7242.702976750923

    dipm_sq = dipm**2 / (m * e * s**3) * conv

    m_ij = np.sqrt(m * m.T)

    m_ij = np.minimum(m_ij, 2)

    adip = (
        a0dip
        + (m_ij - 1) / m_ij * a1dip
        + (m_ij - 1) / m_ij * (m_ij - 2) / m_ij * a2dip
    )
    bdip = (
        b0dip
        + (m_ij - 1) / m_ij * b1dip
        + (m_ij - 1) / m_ij * (m_ij - 2) / m_ij * b2dip
    )

    e_ij_diag = np.diagonal(e_ij)[..., np.newaxis]

    s_ij_diag = np.diagonal(s_ij)[..., np.newaxis]

    etan = (eta**idxd)[..., np.newaxis, np.newaxis]

    j2 = np.sum((adip + bdip * e_ij_diag / t) * etan, axis=0)  # e_ij or e_ij_diag ?

    a2 = np.sum(
        (x * x.T)
        * (e_ij_diag / t * e_ij_diag.T / t)
        * (s_ij_diag**3 * s_ij_diag.T**3)
        / s_ij**3
        * (dip_num * dip_num.T)
        * (dipm_sq * dipm_sq.T)
        * j2
    )

    m_ijk = ((m * m.T)[..., np.newaxis] * m.T) ** (1 / 3.0)

    m_ijk = np.minimum(m_ijk, 2)

    cdip = (
        c0dip
        + (m_ijk - 1) / m_ijk * c1dip
        + (m_ijk - 1) / m_ijk * (m_ijk - 2) / m_ijk * c2dip
    )

    etan = etan[..., np.newaxis]
    j3 = np.sum(cdip * etan, axis=0)

    a3 = np.sum(
        ((x * x.T)[..., np.newaxis] * x.T)
        * ((e_ij_diag / t * e_ij_diag.T / t)[..., np.newaxis] * e_ij_diag.T / t)
        * ((s_ij_diag**3 * s_ij_diag.T**3)[..., np.newaxis] * s_ij_diag.T**3)
        / ((s_ij * s_ij.T)[..., np.newaxis] * s_ij.T)
        * ((dip_num * dip_num.T)[..., np.newaxis] * dip_num.T)
        * ((dipm_sq * dipm_sq.T)[..., np.newaxis] * dipm_sq.T)
        * j3
    )

    a2 = -np.pi * den * a2
    a3 = -4 / 3.0 * np.pi**2 * den**2 * a3

    ares_polar_term = a2 / (1.0 - a3 / a2)

    return jax.lax.cond(
        np.any(jax.lax.is_finite(ares_polar_term)),
        lambda ares_polar_term: ares_polar_term,
        np.zeros_like(ares_polar_term),
        ares_polar_term,
    )


# pylint: disable=invalid-name
def ares_assoc(x, t, den, params):
    "Association term for ePC-SAFT."

    # Association term -------------------------------------------------------
    # 2B association type

    e_assoc, vol_a, khb_ij, s_ij_diag, s_ij, ghs, ncomp = params

    eABij = (e_assoc + e_assoc.T) / 2.0 * (1 - khb_ij)

    volABij = (
        np.sqrt(vol_a @ vol_a.T)
        * (np.sqrt(s_ij_diag @ s_ij_diag.T) / (1 / 2.0 * (s_ij_diag + s_ij_diag.T)))
        ** 3.0
    )

    delta_ij = ghs * volABij * s_ij**3 * (np.exp(eABij / t) - 1.0)

    delta_ij_diag = np.diagonal(delta_ij)[..., np.newaxis] + 1e-30

    XA = (
        np.ones((ncomp, 2))
        * (-1 + np.sqrt(1 + 8 * den * delta_ij_diag))
        / (4 * den * delta_ij_diag)
    )

    XA = jax.lax.fori_loop(
        0, 50, lambda i, XA: (XA + xa_find(XA, delta_ij, den, x)) / 2.0, XA
    )

    ares_assoc_term = np.sum(x * (np.log(XA) - XA / 2.0 + 1 / 2.0))

    return jax.lax.cond(
        np.any(jax.lax.is_finite(ares_assoc_term)),
        lambda ares_assoc_term: ares_assoc_term,
        np.zeros_like(ares_assoc_term),
        ares_assoc_term,
    )


def ares_ion(x, t, den, params):
    "Ion term for ePC-SAFT."

    # Ion term ---------------------------------------------------------------
    s, z, dielc = params
    kb_P10 = 1e-23
    kb = KB / kb_P10
    P10 = E_CHRG_P10**2 / kb_P10 / PERM_VAC_P10
    q = z * E_CHRG
    dielc = x.T @ dielc
    # the inverse Debye screening length. Equation 4 in Held et al. 2008.
    kappa = np.sqrt(den * E_CHRG**2 / kb / t / (dielc * PERM_VAC) * (x.T @ z**2) * P10)
    chi = (
        3.0
        / (kappa * s) ** 3
        * (
            1.5
            + np.log(1 + kappa * s)
            - 2.0 * (1.0 + kappa * s)
            + 0.5 * (1.0 + kappa * s) ** 2
        )
    )
    ares_ion_term = (
        -1
        / 12.0
        / np.pi
        / kb
        / t
        / (dielc * PERM_VAC)
        * np.sum(x * q**2 * chi)
        * kappa
        * P10
    )

    return jax.lax.cond(
        np.any(jax.lax.is_finite(ares_ion_term)),
        lambda ares_ion_term: ares_ion_term,
        np.zeros_like(ares_ion_term),
        ares_ion_term,
    )
