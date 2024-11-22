"""ePC-SAFT implementation with jax
---------------
Code reference: `https://github.com/zmeri/PC-SAFT`
"""

# @author: Wildson Lima


import jax
import jax.numpy as np

from .epcsaft_jax import pcsaft_ares

# pylint: disable=C0103,E1102
dares_drho = jax.jit(jax.jacfwd(pcsaft_ares, 5))


# pylint: disable = invalid-name
@jax.jit
def pcsaft_Z(
    x,
    t,
    rho,
    params,
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
        Charge number of the ions
    dielc : ndarray, shape (n,1)
        permittivity of each component of the medium to be used for electrolyte
        calculations.

    Returns
    -------
    Z : float
        Compressibility factor
    """

    return (1 + rho * dares_drho(x, t, rho, params)).squeeze()


dares_dx = jax.jit(jax.jacfwd(pcsaft_ares, 0))


@jax.jit
def pcsaft_fugcoef(x, t, rho, params):
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
        Charge number of the ions
    dielc : ndarray, shape (n,1)
        permittivity of each component of the medium to be used for electrolyte
        calculations.

    Returns
    -------
    fugcoef : ndarray, shape (n,)
        Fugacity coefficients of each component.
    """

    Z = pcsaft_Z(x, t, rho, params)
    lnZ = np.log(Z)
    ares = pcsaft_ares(x, t, rho, params)

    grad = dares_dx(x, t, rho, params)

    return np.exp(ares + (Z - 1) + grad - (x.T @ grad) - lnZ).squeeze()


@jax.jit
def pcsaft_p(x, t, rho, params):
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
        Charge number of the ions
    dielc : ndarray, shape (n,1)
        permittivity of each component of the medium to be used for electrolyte
        calculations.

    Returns
    -------
    P : float
        Pressure (Pa)
    """

    kb = 1.380648465952442093e-23  # Boltzmann constant, J K^-1
    nav = 6.022140857e23  # Avogadro's number
    den = rho * nav  # number density, units of Angstrom^-3

    Z = pcsaft_Z(x, t, rho, params)
    P = Z * kb * t * den  # Pa
    return P.squeeze()


@jax.jit
def density_from_nu(nu, t, x, params):
    """
    density calculation from reduced density nu

    """
    m = params["m"]
    s = params["s"]
    e = params["e"]
    nav = 6.022140857e23  # Avogadro's number

    d = s * (1.0 - 0.12 * np.exp(-3.0 * e / t))
    summ = np.sum(x * m * d**3.0)
    return 6.0 / np.pi * nu / summ * 1.0e30 / nav


@jax.jit
def nu_from_density(rho, t, x, params):
    """
    reduced density calculation from density

    """
    m = params["m"]
    s = params["s"]
    e = params["e"]
    nav = 6.022140857e23  # Avogadro's number

    d = s * (1.0 - 0.12 * np.exp(-3.0 * e / t))
    summ = np.sum(x * m * d**3.0)
    return np.pi / 6 * rho * nav / 1.0e30 * summ


@jax.jit
def den_err(nu, x, t, p, params):
    """Find root of this function to calculate the reduced density or pressure."""

    rho_guess = density_from_nu(nu, t, x, params)

    P_fit = pcsaft_p(x, t, rho_guess, params)

    return (P_fit - p) / p


@jax.jit
def den_errSQ(nu, x, t, p, params):
    """Find root of this function to calculate the reduced density or pressure."""

    return den_err(nu, x, t, p, params) ** 2


dden_errSQ_dnu = jax.jit(jax.jacfwd(den_errSQ))


@jax.jit
def pcsaft_den(x, t, p, phase, params):
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
        Charge number of the ions
    dielc : ndarray, shape (n,1)
        permittivity of each component of the medium to be used for electrolyte
        calculations.
    phase : int
        The phase for which the calculation is performed. Options: 1 (liquid),
        0 (vapor).

    Returns
    -------
    rho : float
        Molar density (mol / m^3)
    """

    nulow = (
        10
        ** -np.arange(
            13.0,
            4.0,
            -1,
        )[..., np.newaxis]
    )
    nuhigh = np.arange(
        1.0e-4,
        0.7405,
        0.0001,
    )[..., np.newaxis]
    nu = np.concatenate([nulow, nuhigh], 0)

    err = vden_err(nu, x, t, p, params)

    nul = np.zeros_like(nu).repeat(3, 1) * np.nan

    nul = jax.lax.fori_loop(
        0,
        nul.shape[0] - 1,
        lambda i, nul: jax.lax.cond(
            err[i + 1] * err[i] < 0,
            lambda i, nul: nul.at[i, :].set((nu[i, 0], nu[i + 1, 0], 1)),
            lambda i, nul: nul.at[i, :].set((np.nan, np.nan, 0)),
            i,
            nul,
        ),
        nul,
    )

    nu_max = np.nanargmax(nul, 0)[0]
    nu_min = np.nanargmin(nul, 0)[0]

    nu_max, nu_min = jax.lax.cond(
        phase == 1,
        lambda nul: nul[nu_max, 0:2],
        lambda nul: nul[nu_min, 0:2],
        nul,
    )

    nu = (nu_max + nu_min) / 2.0

    # pylint: disable = unused-argument
    def updater(i, nu):
        f = den_errSQ(nu, x, t, p, params)

        gradf = jax.lax.cond(
            f < 1.0e-5,
            lambda nu: np.inf,
            lambda nu: dden_errSQ_dnu(nu, x, t, p, params),
            nu,
        )

        tmp = nu - f / gradf
        nu = jax.lax.cond(
            np.any((tmp > 0) & (jax.lax.is_finite(tmp))),
            lambda nu: tmp,
            lambda nu: nu,
            nu,
        )

        return nu

    nu = jax.lax.fori_loop(0, 20, updater, nu)

    rho = density_from_nu(nu, t, x, params)

    return rho.squeeze()


vden_err = jax.jit(
    jax.vmap(
        den_err,
        in_axes=(
            0,
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


dares_dt = jax.jit(jax.jacfwd(pcsaft_ares, 1))


@jax.jit
def pcsaft_hres(x, t, rho, params):
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
        Charge number of the ions
    dielc : ndarray, shape (n,1)
        permittivity of each component of the medium to be used for electrolyte
        calculations.

    Returns
    -------
    hres : float
        Residual enthalpy (J mol^-1)
    """

    kb = 1.380648465952442093e-23  # Boltzmann constant, J K^-1
    nav = 6.022140857e23  # Avogadro's number

    grad = dares_dt(x, t, rho, params)

    Z = pcsaft_Z(x, t, rho, params)

    return ((-t * grad + (Z - 1)) * kb * nav * t).squeeze()


@jax.jit
def pcsaft_gres(x, t, rho, params):
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
        Charge number of the ions
    dielc : ndarray, shape (n,1)
        permittivity of each component of the medium to be used for electrolyte
        calculations.

    Returns
    -------
    hres : float
        Residual Gibbs energy (J mol^-1)
    """

    kb = 1.380648465952442093e-23  # Boltzmann constant, J K^-1
    nav = 6.022140857e23  # Avogadro's number

    ares = pcsaft_ares(x, t, rho, params)

    Z = pcsaft_Z(x, t, rho, params)

    return ((ares + (Z - 1) - np.log(Z)) * kb * nav * t).squeeze()


@jax.jit
def pcsaft_sres(x, t, rho, params):
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
        Charge number of the ions
    dielc : ndarray, shape (n,1)
        permittivity of each component of the medium to be used for electrolyte
        calculations.

    Returns
    -------
    hres : float
        Residual entropy (J mol^-1)
    """

    hres = pcsaft_hres(x, t, rho, params)

    gres = pcsaft_gres(x, t, rho, params)

    return ((hres - gres) / t).squeeze()


den_phase = jax.jit(
    jax.vmap(
        pcsaft_den,
        in_axes=(
            None,
            None,
            None,
            0,
            None,
        ),
    )
)

fungcoef_phase = jax.jit(
    jax.vmap(
        pcsaft_fugcoef,
        (
            None,
            None,
            0,
            None,
        ),
    )
)


@jax.jit
def k_i(p_guess, x, t, params):
    """Minimize this function to calculate the vapor pressure."""
    phases = np.asarray([1.0, 0.0])

    rho = den_phase(x, t, p_guess, phases, params)

    fugcoef_l, fugcoef_v = fungcoef_phase(
        x,
        t,
        rho,
    )
    return fugcoef_l / fugcoef_v


@jax.jit
def pcsaft_VP(x, t, p_guess, params):
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
        Charge number of the ions
    dielc : ndarray, shape (n,1)
        permittivity of each component of the medium to be used for electrolyte
        calculations.

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

    k = k_i(p_guess, x, t, params)
    kprime = k_i(pprime, x, t, params)

    dlnk_dt = (kprime - k) / (pprime - p_guess)
    wi = x * dlnk_dt / np.sum(x * dlnk_dt)

    kb = np.exp(np.sum(wi * np.log(k)))

    kbprime = np.exp(np.sum(wi * np.log(kprime)))

    B = np.log(kbprime / kb) / (1 / pprime - 1 / p_guess)
    A = np.log(kb) - B * (1 / p_guess - 1 / pref)

    p = 1.0 / (1.0 / pref + (np.log(1.0) - A) / B)

    return p.squeeze()
