# Reference: https://github.com/zmeri/PC-SAFT

# @author: Wildson Lima


import jax.numpy as np
import jax


from jax.config import config


@jax.jit
def pcsaft_ares(
    x,
    m,
    s,
    e,
    t,
    rho,
    k_ij,
    l_ij,
):
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

    Returns
    -------
    ares : float
        Residual Helmholtz energy (J mol^{-1})
    """

    ncomp = x.shape[0]  # number of components
    kb = 1.380648465952442093e-23  # Boltzmann constant, J K^-1
    N_AV = 6.022140857e23  # Avogadro's number

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
    I1 = np.sum(a * eta**idx)
    I2 = np.sum(b * eta**idx)
    C1 = 1.0 / (
        1.0
        + m_avg * (8 * eta - 2 * eta**2) / (1 - eta) ** 4
        + (1 - m_avg)
        * (20 * eta - 27 * eta**2 + 12 * eta**3 - 2 * eta**4)
        / ((1 - eta) * (2 - eta)) ** 2
    )

    ghs_diag = np.diagonal(ghs)[..., np.newaxis]

    summ = np.sum((x * (m - 1.0)) * (np.log(ghs_diag)))

    ares_hc = m_avg * ares_hs - summ
    ares_disp = -2 * np.pi * den * I1 * m2es3 - np.pi * den * m_avg * C1 * I2 * m2e2s3

    ares = ares_hc + ares_disp

    return ares[0, 0]


@jax.jit
def XA_find(XA_guess, delta_ij, den, x):
    """Iterate over this function in order to solve for XA"""
    return 1.0 / (1.0 + den * np.sum(x * XA_guess * delta_ij[..., np.newaxis], axis=1))


dares_drho = jax.jit(jax.jacfwd(pcsaft_ares, 5))


@jax.jit
def pcsaft_Z(
    x,
    m,
    s,
    e,
    t,
    rho,
    k_ij,
    l_ij,
):
    """
    Calculate the compressibility factor.

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

    Returns
    -------
    Z : float
        Compressibility factor
    """

    return 1 + rho * dares_drho(
        x,
        m,
        s,
        e,
        t,
        rho,
        k_ij,
        l_ij,
    )


dares_dx = jax.jit(jax.jacfwd(pcsaft_ares, 0))


@jax.jit
def pcsaft_fugcoef(
    x,
    m,
    s,
    e,
    t,
    rho,
    k_ij,
    l_ij,
):
    """
    Calculate the fugacity coefficients for one phase of the system.

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

    Returns
    -------
    fugcoef : ndarray, shape (n,)
        Fugacity coefficients of each component.
    """

    Z = pcsaft_Z(
        x,
        m,
        s,
        e,
        t,
        rho,
        k_ij,
        l_ij,
    )
    lnZ = np.log(Z)
    ares = pcsaft_ares(
        x,
        m,
        s,
        e,
        t,
        rho,
        k_ij,
        l_ij,
    )

    grad = dares_dx(
        x,
        m,
        s,
        e,
        t,
        rho,
        k_ij,
        l_ij,
    )

    return np.exp(ares + (Z - 1) + grad - (x.T @ grad) - lnZ)


@jax.jit
def pcsaft_p(
    x,
    m,
    s,
    e,
    t,
    rho,
    k_ij,
    l_ij,
):
    """
    Calculate pressure.

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

    Returns
    -------
    P : float
        Pressure (Pa)
    """

    kb = 1.380648465952442093e-23  # Boltzmann constant, J K^-1
    N_AV = 6.022140857e23  # Avogadro's number
    den = rho * N_AV  # number density, units of Angstrom^-3

    Z = pcsaft_Z(
        x,
        m,
        s,
        e,
        t,
        rho,
        k_ij,
        l_ij,
    )
    P = Z * kb * t * den  # Pa
    return P


@jax.jit
def density_from_nu(nu, t, x, m, s, e):
    """
    density calculation from reduced density nu

    """
    N_AV = 6.022140857e23  # Avogadro's number

    d = s * (1.0 - 0.12 * np.exp(-3.0 * e / t))
    summ = np.sum(x * m * d**3.0)
    return 6.0 / np.pi * nu / summ * 1.0e30 / N_AV


@jax.jit
def nu_from_density(rho, t, x, m, s, e):
    """
    reduced density calculation from density

    """
    N_AV = 6.022140857e23  # Avogadro's number

    d = s * (1.0 - 0.12 * np.exp(-3.0 * e / t))
    summ = np.sum(x * m * d**3.0)
    return np.pi / 6 * rho * N_AV / 1.0e30 * summ


@jax.jit
def den_err(
    nu,
    x,
    m,
    s,
    e,
    t,
    p,
    k_ij,
    l_ij,
):
    """Find root of this function to calculate the reduced density or pressure."""

    rho_guess = density_from_nu(nu, t, x, m, s, e)

    P_fit = pcsaft_p(
        x,
        m,
        s,
        e,
        t,
        rho_guess,
        k_ij,
        l_ij,
    )

    return (P_fit - p) / p


@jax.jit
def den_errSQ(nu, x, m, s, e, t, p, k_ij, l_ij):
    """Find root of this function to calculate the reduced density or pressure."""

    return den_err(nu, x, m, s, e, t, p, k_ij, l_ij) ** 2


dden_errSQ_dnu = jax.jit(jax.jacfwd(den_errSQ))


@jax.jit
def pcsaft_den(x, m, s, e, t, p, k_ij, l_ij, phase):
    """
    Molar density at temperature and pressure given.

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
    P : float
        Pressure (Pa)
    k_ij : ndarray, shape (n,n)
        Binary interaction parameters between components in the mixture for dispersion energy.
        (dimensions: ncomp x ncomp)
    l_ij : ndarray, shape (n,n)
        Binary interaction parameters between components in the mixture for segment diameter.
        (dimensions: ncomp x ncomp)
    phase : int
        The phase for which the calculation is performed. Options: 1 (liquid),
        0 (vapor).

    Returns
    -------
    rho : float
        Molar density (mol / m^3)
    """

    nulow = 10 ** -np.arange(13, 4, -1, dtype=np.float64)[..., np.newaxis]
    nuhigh = np.arange(1.0e-4, 0.7405, 0.0001, dtype=np.float64)[..., np.newaxis]
    nu = np.concatenate([nulow, nuhigh], 0)
    err = vden_err(nu, x, m, s, e, t, p, k_ij, l_ij)

    nul = np.zeros_like(nu).repeat(3, 1)

    nul = jax.lax.fori_loop(
        0,
        nul.shape[0] - 1,
        lambda i, nul: jax.lax.cond(
            err[i + 1, 0] * err[i, 0] < 0,
            lambda i, nul: nul.at[i, :].set((nu[i, 0], nu[i + 1, 0], 1)),
            lambda i, nul: nul,
            i,
            nul,
        ),
        nul,
    )

    nul = np.sort(nul, 0)

    roots = np.sum(nul[:, 2]).astype(np.int_)

    nu_max = np.argmax(nul, 0)[0]

    a, b = jax.lax.cond(
        phase == 1,
        lambda nul: nul[nu_max, 0:2],
        lambda nul: nul[nu_max - roots + 1, 0:2],
        nul,
    )

    nu = np.asarray([(b + a) / 2.0, 0.0])

    def cond_fun(nu):
        f = den_errSQ(nu[0], x, m, s, e, t, p, k_ij, l_ij)
        test1 = f > 1.0e-5
        test2 = nu[1] < 10
        return (test1) & (test2)

    def updater(nu):
        f = den_errSQ(nu[0], x, m, s, e, t, p, k_ij, l_ij)

        gradf = dden_errSQ_dnu(nu[0], x, m, s, e, t, p, k_ij, l_ij)

        nu = nu.at[0].set(nu[0] - f / gradf)
        nu = nu.at[1].set(nu[1] + 1)
        return nu

    nu = jax.lax.while_loop(cond_fun, updater, nu)

    rho = density_from_nu(nu[0], t, x, m, s, e)

    return rho


vden_err = jax.jit(
    jax.vmap(
        den_err,
        in_axes=(
            0,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
    )
)


def dielc_water(t):
    """
    Return the permittivity of water at 1 bar and the given temperature.

    t : float
        Temperature (K)

    This equation was fit to values given in the reference over the temperature
    range of 263.15 to 368.15 K.

    Reference:
    D. G. Archer and P. Wang, “The permittivity of Water and Debye‐Hückel
    Limiting Law Slopes,” J. Phys. Chem. Ref. Data, vol. 19, no. 2, pp. 371–411,
    Mar. 1990.
    """
    dielc = 7.6555618295e-04 * t**2 - 8.1783881423e-01 * t + 2.5419616803e02
    return dielc


dares_dt = jax.jit(jax.jacfwd(pcsaft_ares, 4))


@jax.jit
def pcsaft_hres(
    x,
    m,
    s,
    e,
    t,
    rho,
    k_ij,
    l_ij,
):
    """
    Calculate the residual enthalpy for one phase of the system.

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

    Returns
    -------
    hres : float
        Residual enthalpy (J mol^-1)
    """

    kb = 1.380648465952442093e-23  # Boltzmann constant, J K^-1
    N_AV = 6.022140857e23  # Avogadro's number

    grad = dares_dt(x, m, s, e, t, rho, k_ij, l_ij)

    Z = pcsaft_Z(x, m, s, e, t, rho, k_ij, l_ij)

    return (-t * grad + (Z - 1)) * kb * N_AV * t


@jax.jit
def pcsaft_gres(
    x,
    m,
    s,
    e,
    t,
    rho,
    k_ij,
    l_ij,
):
    """
    Calculate the residual Gibbs energy for one phase of the system.

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

    Returns
    -------
    hres : float
        Residual Gibbs energy (J mol^-1)
    """

    kb = 1.380648465952442093e-23  # Boltzmann constant, J K^-1
    N_AV = 6.022140857e23  # Avogadro's number

    ares = pcsaft_ares(x, m, s, e, t, rho, k_ij, l_ij)

    Z = pcsaft_Z(x, m, s, e, t, rho, k_ij, l_ij)

    return (ares + (Z - 1) - np.log(Z)) * kb * N_AV * t


@jax.jit
def pcsaft_sres(
    x,
    m,
    s,
    e,
    t,
    rho,
    k_ij,
    l_ij,
):
    """
    Calculate the residual entropy (constant volume) for one phase of the system.

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

    Returns
    -------
    hres : float
        Residual entropy (J mol^-1)
    """

    hres = pcsaft_hres(x, m, s, e, t, rho, k_ij, l_ij)

    gres = pcsaft_gres(x, m, s, e, t, rho, k_ij, l_ij)

    return (hres - gres) / t


den_phase = jax.jit(
    jax.vmap(
        pcsaft_den,
        in_axes=(None, None, None, None, None, None, None, None, 0),
    )
)

fungcoef_phase = jax.jit(
    jax.vmap(
        pcsaft_fugcoef,
        (None, None, None, None, None, 0, None, None),
    )
)


@jax.jit
def k_i(p_guess, x, m, s, e, t, k_ij, l_ij):
    """Minimize this function to calculate the vapor pressure."""
    phases = np.asarray([1.0, 0.0])

    rho = den_phase(
        x,
        m,
        s,
        e,
        t,
        p_guess,
        k_ij,
        l_ij,
        phases,
    )

    fugcoef_l, fugcoef_v = fungcoef_phase(x, m, s, e, t, rho, k_ij, l_ij)
    return fugcoef_l / fugcoef_v


@jax.jit
def pcsaft_VP(
    x,
    m,
    s,
    e,
    t,
    p_guess,
    k_ij,
    l_ij,
):
    """
    Vapor pressure calculation

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
    p: float
        Guess for vapor pressure (Pa)
    k_ij : ndarray, shape (n,n)
        Binary interaction parameters between components in the mixture for dispersion energy.
        (dimensions: ncomp x ncomp)
    l_ij : ndarray, shape (n,n)
        Binary interaction parameters between components in the mixture for segment diameter.
        (dimensions: ncomp x ncomp)

    Returns
    -------
    VP : float
        Vapor Pressure (Pa)
    """

    pref = p_guess - 0.01 * p_guess

    pprime = jax.lax.cond(
        p_guess > 1.0e6,
        lambda p_guess: p_guess - 0.005 * p_guess,
        lambda p_guess: p_guess + 0.01 * p_guess,
        p_guess,
    )

    k = k_i(p_guess, x, m, s, e, t, k_ij, l_ij)
    kprime = k_i(pprime, x, m, s, e, t, k_ij, l_ij)

    dlnk_dt = (kprime - k) / (pprime - p_guess)
    t_weight = x * dlnk_dt
    t_sum = np.sum(t_weight)
    wi = t_weight / t_sum

    kb = np.sum(wi * np.log(k))
    kb = np.exp(kb)

    kbprime = np.sum(wi * np.log(kprime))
    kbprime = np.exp(kbprime)

    B = np.log(kbprime / kb) / (1 / pprime - 1 / p_guess)
    A = np.log(kb) - B * (1 / p_guess - 1 / pref)

    p = 1.0 / (1.0 / pref + (np.log(1.0) - A) / B)

    return p.squeeze()
