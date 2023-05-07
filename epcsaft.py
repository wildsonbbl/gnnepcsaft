# Inspired by https://github.com/zmeri/PC-SAFT

# @author: Wildson Lima


import jax.numpy as np
import jax


from jax.config import config
config.update("jax_enable_x64", True)


@jax.jit
def pcsaft_ares(x, m, s, e, t, rho, k_ij, l_ij,
                khb_ij, e_assoc, vol_a, dipm,
                dip_num, z, dielc):
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

    x = np.minimum(np.abs(x) + 1.0e-10, 1.)
    m = np.abs(m)
    s = np.abs(s)
    e = np.abs(e)
    t = np.abs(t)
    rho = np.abs(rho)
    e_assoc = np.abs(e_assoc)
    vol_a = np.abs(vol_a)
    dipm = np.abs(dipm)
    dip_num = np.abs(dip_num)
    charged_ij = np.int8((z * z.T) <= 0)
    z = np.minimum(np.abs(z) + 1.0e-10, 3)
    dielc = np.abs(dielc)

    charged = np.int8(z < 1)

    ncomp = x.shape[0]  # number of components
    kb = 1.380648465952442093e-23  # Boltzmann constant, J K^-1
    N_AV = 6.022140857e23  # Avogadro's number

    d = s*(1. - 0.12*np.exp(-3*e*charged/t))

    den = rho*N_AV/1.0e30

    a0 = np.asarray([0.910563145, 0.636128145, 2.686134789,
                    -26.54736249, 97.75920878, -159.5915409,
                     91.29777408])
    a1 = np.asarray([-0.308401692, 0.186053116, -2.503004726,
                    21.41979363, -65.25588533, 83.31868048,
                    -33.74692293])
    a2 = np.asarray([-0.090614835, 0.452784281, 0.596270073,
                    -1.724182913, -4.130211253, 13.77663187,
                    -8.672847037])
    b0 = np.asarray([0.724094694, 2.238279186, -4.002584949,
                    -21.00357682, 26.85564136, 206.5513384,
                    -355.6023561])
    b1 = np.asarray([-0.575549808, 0.699509552, 3.892567339,
                    -17.21547165, 192.6722645, -161.8264617,
                    -165.2076935])
    b2 = np.asarray([0.097688312, -0.255757498, -9.155856153,
                    20.64207597, -38.80443005, 93.62677408,
                    -29.66690559])

    n = np.arange(4)[..., np.newaxis]
    zeta = np.pi/6.*den*(d.T**n) @ (x*m)

    eta = zeta[3]
    m_avg = x.T @ m

    s_ij = (s + s.T) / 2. * (1 - l_ij)
    e_ij = np.sqrt(e @ e.T) * (1-k_ij) * charged_ij
    m2es3 = np.sum((x@x.T) * (m@m.T)*(e_ij/t)*s_ij**3)
    m2e2s3 = np.sum((x@x.T) * (m@m.T)*(e_ij/t)**2*s_ij**3)
    ghs = 1./(1.-zeta[3]) + (d @ d.T)/(d + d.T) *\
        3.*zeta[2]/(1.-zeta[3])**2 + ((d @ d.T)/(d + d.T))**2 *\
        2.*zeta[2]**2/(1.-zeta[3])**3

    ares_hs = 1/zeta[0]*(
        3*zeta[1]*zeta[2]/(1-zeta[3]) +
        zeta[2]**3/(zeta[3]*(1-zeta[3])**2) +
        (zeta[2]**3/zeta[3]**2 - zeta[0]) *
        np.log(1-zeta[3])
    )

    a = a0 + (m_avg-1.)/m_avg*a1 +\
        (m_avg-1.)/m_avg*(m_avg-2.)/m_avg*a2
    b = b0 + (m_avg-1)/m_avg*b1 +\
        (m_avg-1.)/m_avg*(m_avg-2.)/m_avg*b2

    idx = np.arange(7)
    I1 = np.sum(a*eta**idx)
    I2 = np.sum(b*eta**idx)
    C1 = 1./(
        1. + m_avg*(8*eta-2*eta**2)/(1-eta)**4 +
        (1-m_avg) * (20*eta-27*eta**2 + 12*eta**3-2*eta**4) /
        ((1-eta)*(2-eta))**2
    )

    ghs_diag = np.diagonal(ghs)[..., np.newaxis]

    summ = np.sum(
        (x * (m-1.))*(np.log(ghs_diag))
    )

    ares_hc = m_avg*ares_hs - summ
    ares_disp = -2*np.pi*den*I1*m2es3 -\
        np.pi*den*m_avg*C1*I2*m2e2s3

    # Dipole term (Gross and Vrabec term) --------------------------------------
    # Gross, Joachim, e Jadran Vrabec. “An Equation-of-State Contribution for Polar Components: Dipolar Molecules”. AIChE Journal 52, nº 3 (2006): 1194–1204. https://doi.org/10.1002/aic.10683.
    # Held, Christoph, Thomas Reschke, Sultan Mohammad, Armando Luza, e Gabriele Sadowski. “EPC-SAFT Revised”. Chemical Engineering Research and Design, Advances in Thermodynamics for Chemical Process and Product Design, 92, nº 12 (1º de dezembro de 2014): 2884–97. https://doi.org/10.1016/j.cherd.2014.05.017.

    a0dip = np.asarray(
        [0.3043504, -0.1358588, 1.4493329, 0.3556977, -2.0653308]
    )[..., np.newaxis, np.newaxis]
    a1dip = np.asarray(
        [0.9534641, -1.8396383, 2.0131180, -7.3724958, 8.2374135]
    )[..., np.newaxis, np.newaxis]
    a2dip = np.asarray(
        [-1.1610080, 4.5258607, 0.9751222, -12.281038, 5.9397575]
    )[..., np.newaxis, np.newaxis]
    b0dip = np.asarray(
        [0.2187939, -1.1896431, 1.1626889, 0, 0.0]
    )[..., np.newaxis, np.newaxis]
    b1dip = np.asarray(
        [-0.5873164, 1.2489132, -0.5085280, 0, 0]
    )[..., np.newaxis, np.newaxis]
    b2dip = np.asarray(
        [3.4869576, -14.915974, 15.372022, 0, 0]
    )[..., np.newaxis, np.newaxis]

    c0dip = np.asarray(
        [-0.0646774, 0.1975882, -0.8087562, 0.6902849, 0]
    )[..., np.newaxis, np.newaxis, np.newaxis]
    c1dip = np.asarray(
        [-0.9520876, 2.9924258, -2.3802636, -0.2701261, 0]
    )[..., np.newaxis, np.newaxis, np.newaxis]
    c2dip = np.asarray(
        [-0.6260979, 1.2924686, 1.6542783, -3.4396744, 0]
    )[..., np.newaxis, np.newaxis, np.newaxis]

    idxd = np.arange(5)

    # conversion factor, see the note below Table 2 in Gross and Vrabec 2006
    conv = 7242.702976750923

    dipmSQ = dipm**2/(m*e*s**3)*conv

    m_ij = np.sqrt(m * m.T)

    m_ij = np.minimum(m_ij, 2)

    adip = a0dip + (m_ij-1)/m_ij*a1dip +\
        (m_ij-1)/m_ij*(m_ij-2)/m_ij*a2dip
    bdip = b0dip + (m_ij-1)/m_ij*b1dip +\
        (m_ij-1)/m_ij*(m_ij-2)/m_ij*b2dip

    e_ij_diag = np.diagonal(e_ij)[..., np.newaxis]

    s_ij_diag = np.diagonal(s_ij)[..., np.newaxis]

    etan = (eta**idxd)[..., np.newaxis, np.newaxis]

    J2 = np.sum((adip + bdip*e_ij_diag/t)*etan, axis=0)  # e_ij or e_ij_diag ?

    A2 = np.sum((x*x.T) * (e_ij_diag/t * e_ij_diag.T/t) *
                (s_ij_diag**3 * s_ij_diag.T**3)/s_ij**3 *
                (dip_num * dip_num.T) * (dipmSQ * dipmSQ.T) * J2) + 1e-10

    m_ijk = ((m * m.T)[..., np.newaxis] * m.T)**(1/3.)

    m_ijk = np.minimum(m_ijk, 2)

    cdip = c0dip + (m_ijk-1)/m_ijk*c1dip +\
        (m_ijk-1)/m_ijk*(m_ijk-2)/m_ijk*c2dip

    etan = etan[..., np.newaxis]
    J3 = np.sum(cdip*etan, axis=0)

    A3 = np.sum(
        ((x * x.T)[..., np.newaxis] * x.T) *
        ((e_ij_diag/t * e_ij_diag.T/t)[..., np.newaxis] * e_ij_diag.T/t) *
        ((s_ij_diag**3 * s_ij_diag.T**3)[..., np.newaxis] * s_ij_diag.T**3) /
        ((s_ij * s_ij.T)[..., np.newaxis] * s_ij.T) *
        ((dip_num * dip_num.T)[..., np.newaxis] * dip_num.T) *
        ((dipmSQ * dipmSQ.T)[..., np.newaxis] * dipmSQ.T) * J3
    ) + 1e-20

    A2 = -np.pi*den*A2
    A3 = -4/3.*np.pi**2*den**2*A3

    ares_polar = A2/(1.-A3/A2)

    # Association term -------------------------------------------------------
    # 2B association type

    eABij = (e_assoc+e_assoc.T)/2.*(1-khb_ij)

    volABij = np.sqrt(vol_a @ vol_a.T) *\
        (np.sqrt(s_ij_diag @ s_ij_diag.T) /
         (1/2.*(s_ij_diag + s_ij_diag.T)))**3.

    delta_ij = ghs * volABij * s_ij**3 * (np.exp(eABij/t)-1.)

    delta_ij_diag = np.diagonal(delta_ij)[..., np.newaxis] + 1e-30

    XA = np.ones((ncomp, 2)) *\
        (-1+np.sqrt(1+8*den * delta_ij_diag)) / \
        (4*den*delta_ij_diag)

    ctr = 0

    XA = jax.lax.fori_loop(
        0,
        50,
        lambda i, XA: (XA + XA_find(XA, delta_ij, den, x))/2.0,
        XA
    )

    ares_assoc = np.sum(x * (np.log(XA) - XA/2. + 1/2.))

    # Ion term ---------------------------------------------------------------

    E_CHRG = 1.6021766208  # elementary charge, units of coulomb / 1e-19
    perm_vac = 8.854187817  # permittivity in vacuum, C V^-1 Angstrom^-1 / 1e-22

    E_CHRG_P10 = 1e-19
    perm_vac_P10 = 1e-22
    kb_P10 = 1e-23
    kb = kb/kb_P10

    P10 = E_CHRG_P10**2/kb_P10/perm_vac_P10

    q = z*E_CHRG
    dielc = x.T @ dielc
    # the inverse Debye screening length. Equation 4 in Held et al. 2008.
    kappa = np.sqrt(
        den*E_CHRG**2/kb/t/(dielc*perm_vac) *
        (x.T @ z**2) * P10
    ) + 1e-10

    chi = 3./(kappa*s)**3 *\
        (1.5 + np.log(1+kappa*s) - 2.*(1. + kappa*s) +
         0.5*(1. + kappa*s)**2)

    ares_ion = -1/12./np.pi/kb/t/(dielc*perm_vac) * \
        np.sum(x*q**2*chi) * kappa * P10

    ares = ares_hc + ares_disp + \
        ares_polar + ares_assoc + ares_ion

    # print(ares_hc, ares_disp, ares_polar,
    # ares_assoc,ares_ion, sep = '\n new data: \n')

    return ares[0, 0]


@jax.jit
def XA_find(XA_guess, delta_ij, den, x):
    """Iterate over this function in order to solve for XA"""

    # print("inside: \n",XA_guess, delta_ij, den, x,sep = " \n ")

    return (1./(1. + den * np.sum(x * XA_guess * delta_ij[..., np.newaxis], axis=1)))


dares_drho = jax.jit(jax.jacfwd(pcsaft_ares, 5))


@jax.jit
def pcsaft_Z(x, m, s, e, t, rho, k_ij, l_ij,
             khb_ij, e_assoc, vol_a, dipm,
             dip_num, z, dielc):
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

    return 1 + rho*dares_drho(x, m, s, e, t, rho, k_ij, l_ij,
                              khb_ij, e_assoc, vol_a, dipm,
                              dip_num, z, dielc)


@jax.jit
def ares_eta(x, m, s, e, t, eta, k_ij, l_ij,
             khb_ij, e_assoc, vol_a, dipm,
             dip_num, z, dielc):
    """
    Calculates residual Helmholtz energy with reduced density"""

    rho = density_from_nu(eta, t, x, m, s, e)

    return pcsaft_ares(x, m, s, e, t, rho, k_ij, l_ij,
                       khb_ij, e_assoc, vol_a, dipm,
                       dip_num, z, dielc)


dares_dx = jax.jit(jax.jacfwd(pcsaft_ares, 0))


@jax.jit
def pcsaft_fugcoef(x, m, s, e, t, rho, k_ij, l_ij,
                   khb_ij, e_assoc, vol_a, dipm,
                   dip_num, z, dielc):
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

    Z = pcsaft_Z(x, m, s, e, t, rho, k_ij, l_ij,
                 khb_ij, e_assoc, vol_a, dipm,
                 dip_num, z, dielc)
    lnZ = np.log(Z)
    ares = pcsaft_ares(x, m, s, e, t, rho, k_ij, l_ij,
                       khb_ij, e_assoc, vol_a, dipm,
                       dip_num, z, dielc)

    grad = dares_dx(x, m, s, e, t, rho, k_ij, l_ij,
                    khb_ij, e_assoc, vol_a, dipm,
                    dip_num, z, dielc)

    return np.exp(
        ares + (Z - 1) + grad - (x.T @ grad) - lnZ
    )


@jax.jit
def pcsaft_gamma(x, m, s, e, t, rho, k_ij, l_ij,
                 khb_ij, e_assoc, vol_a, dipm,
                 dip_num, z, dielc):
    """
    Activity coefficient calculation.


    """
    kb = 1.380648465952442093e-23  # Boltzmann constant, J K^-1
    N_AV = 6.022140857e23  # Avogadro's number

    ares = pcsaft_ares(x, m, s, e, t, rho, k_ij, l_ij,
                       khb_ij, e_assoc, vol_a, dipm,
                       dip_num, z, dielc)

    return np.exp(-ares/t/(kb*N_AV))


@jax.jit
def pcsaft_p(x, m, s, e, t, rho, k_ij, l_ij,
             khb_ij, e_assoc, vol_a, dipm,
             dip_num, z, dielc):
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
    N_AV = 6.022140857e23  # Avogadro's number
    den = rho*N_AV  # number density, units of Angstrom^-3

    Z = pcsaft_Z(x, m, s, e, t, rho, k_ij, l_ij,
                 khb_ij, e_assoc, vol_a, dipm,
                 dip_num, z, dielc)
    P = Z*kb*t*den  # Pa
    return P


@jax.jit
def density_from_nu(nu, t, x, m, s, e):
    """
    density calculation from reduced density nu

    """
    N_AV = 6.022140857e23  # Avogadro's number

    d = s*(1.-0.12*np.exp(-3.*e/t))
    summ = np.sum(x * m * d**3.)
    return 6./np.pi*nu/summ*1.0e30/N_AV


@jax.jit
def nu_from_density(rho, t, x, m, s, e):
    """
    reduced density calculation from density

    """
    N_AV = 6.022140857e23  # Avogadro's number

    d = s*(1.-0.12*np.exp(-3.*e/t))
    summ = np.sum(x * m * d**3.)
    return np.pi/6*rho*N_AV/1.0e30*summ


@jax.jit
def den_err(nu, x, m, s, e, t, p, k_ij, l_ij,
            khb_ij, e_assoc, vol_a, dipm,
            dip_num, z, dielc):
    """Find root of this function to calculate the reduced density or pressure."""

    rho_guess = density_from_nu(nu, t, x, m, s, e)

    P_fit = pcsaft_p(x, m, s, e, t, rho_guess, k_ij, l_ij,
                     khb_ij, e_assoc, vol_a, dipm,
                     dip_num, z, dielc)

    return ((P_fit-p)/p)


@jax.jit
def den_errSQ(nu, x, m, s, e, t, p, k_ij, l_ij,
              khb_ij, e_assoc, vol_a, dipm,
              dip_num, z, dielc):
    """Find root of this function to calculate the reduced density or pressure."""

    return den_err(nu, x, m, s, e, t, p, k_ij, l_ij,
                   khb_ij, e_assoc, vol_a, dipm,
                   dip_num, z, dielc)**2


dden_errSQ_dnu = jax.jit(jax.jacfwd(den_errSQ))


@jax.jit
def pcsaft_den(x, m, s, e, t, p, k_ij, l_ij,
               khb_ij, e_assoc, vol_a, dipm,
               dip_num, z, dielc, phase):
    """
    Solve for the molar density when temperature and pressure are given.

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

    b, a = bainterval_den(x, m, s, e, t, p, k_ij, l_ij,
                          khb_ij, e_assoc, vol_a, dipm,
                          dip_num, z, dielc, phase)

    nu = (b+a)/2

    def updater(i, nu):
        f = den_errSQ(nu, x, m, s, e, t, p, k_ij, l_ij,
                      khb_ij, e_assoc, vol_a, dipm,
                      dip_num, z, dielc)
        f = f * (f >= 1.0e-6)
        gradf = dden_errSQ_dnu(nu, x, m, s, e, t, p, k_ij, l_ij,
                               khb_ij, e_assoc, vol_a, dipm,
                               dip_num, z, dielc)
        nu = nu - f/gradf
        return nu

    nu = jax.lax.fori_loop(
        0, 40,
        updater,
        nu
    )

    rho = density_from_nu(nu, t, x, m, s, e)

    return rho


root_den = jax.jit(jax.vmap(
    den_err,
    in_axes=(0, None, None, None, None, None,
             None, None, None, None, None, None,
             None, None, None, None)
))


@jax.jit
def bainterval_den(x, m, s, e, t, p, k_ij, l_ij,
                   khb_ij, e_assoc, vol_a, dipm,
                   dip_num, z, dielc, phase):

    n = np.arange(0.1, 0.7405, 0.1, dtype=np.float64)
    nu = n*10**np.arange(-12.0, 1.0)[..., np.newaxis]
    nu = nu.flatten()[..., np.newaxis]

    err = root_den(nu, x, m, s, e, t, p, k_ij, l_ij,
                   khb_ij, e_assoc, vol_a, dipm,
                   dip_num, z, dielc)

    nul = np.zeros((91, 2))

    nul = jax.lax.fori_loop(
        0, 90,
        lambda i, nul: jax.lax.cond(
            err[i+1, 0]*err[i, 0] < 0,
            lambda i, nul: nul.at[(i, i+1), 0].set(
                (nu[i, 0], nu[i+1, 0])
            ).at[i, 1].set(1.0),
            lambda i, nul: nul,
            i, nul
        ),
        nul
    )

    nul = jax.lax.sort(nul, 0)
    nul = nul[-1::-1]

    roots = nul[:, 1].sum()

    nnu = (nul[:, 0] != 0).sum()

    b, a = jax.lax.cond(
        roots == 1,
        lambda nul: [nul[0, 0], nul[1, 0]],
        lambda nul: jax.lax.cond(
            phase == 1,
            lambda nul: [nul[0, 0], nul[1, 0]],
            lambda nul: [nul[nnu-2, 0], nul[nnu-1, 0]],
            nul),
        nul
    )

    return b, a


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
    dielc = 7.6555618295E-04*t**2 - 8.1783881423E-01*t + 2.5419616803E+02
    return dielc


dares_dt = jax.jit(jax.jacfwd(pcsaft_ares, 4))


@jax.jit
def pcsaft_hres(x, m, s, e, t, rho, k_ij, l_ij,
                khb_ij, e_assoc, vol_a, dipm,
                dip_num, z, dielc):
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
    N_AV = 6.022140857e23  # Avogadro's number

    grad = dares_dt(x, m, s, e, t, rho, k_ij, l_ij,
                    khb_ij, e_assoc, vol_a, dipm,
                    dip_num, z, dielc)

    Z = pcsaft_Z(x, m, s, e, t, rho, k_ij, l_ij,
                 khb_ij, e_assoc, vol_a, dipm,
                 dip_num, z, dielc)

    return (-t*grad + (Z-1))*kb*N_AV*t


@jax.jit
def pcsaft_gres(x, m, s, e, t, rho, k_ij, l_ij,
                khb_ij, e_assoc, vol_a, dipm,
                dip_num, z, dielc):
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
    N_AV = 6.022140857e23  # Avogadro's number

    ares = pcsaft_ares(x, m, s, e, t, rho, k_ij, l_ij,
                       khb_ij, e_assoc, vol_a, dipm,
                       dip_num, z, dielc)

    Z = pcsaft_Z(x, m, s, e, t, rho, k_ij, l_ij,
                 khb_ij, e_assoc, vol_a, dipm,
                 dip_num, z, dielc)

    return (ares + (Z - 1) - np.log(Z))*kb*N_AV*t


@jax.jit
def pcsaft_sres(x, m, s, e, t, rho, k_ij, l_ij,
                khb_ij, e_assoc, vol_a, dipm,
                dip_num, z, dielc):
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

    hres = pcsaft_hres(x, m, s, e, t, rho, k_ij, l_ij,
                       khb_ij, e_assoc, vol_a, dipm,
                       dip_num, z, dielc)

    gres = pcsaft_gres(x, m, s, e, t, rho, k_ij, l_ij,
                       khb_ij, e_assoc, vol_a, dipm,
                       dip_num, z, dielc)

    return (hres - gres)/t


den_phase = jax.jit(jax.vmap(
    pcsaft_den,
    in_axes=(None, None, None, None,
             None, None, None, None,
             None, None, None, None,
             None, None, None, 0)
))

fungcoef_phase = jax.jit(jax.vmap(
    pcsaft_fugcoef,
    (None, None, None, None,
     None, 0, None, None,
     None, None, None, None,
     None, None, None)
))


@jax.jit
def VP_err(p_guess, x, m, s, e, t, k_ij, l_ij,
              khb_ij, e_assoc, vol_a, dipm,
              dip_num, z, dielc):
    """Minimize this function to calculate the vapor pressure."""
    phases = np.asarray([1.0, 0.0])

    rho = den_phase(x, m, s, e, t, p_guess, k_ij, l_ij,
                    khb_ij, e_assoc, vol_a, dipm,
                    dip_num, z, dielc, phases)

    fugcoef_l, fugcoef_v = fungcoef_phase(x, m, s, e, t, rho, k_ij, l_ij,
                                          khb_ij, e_assoc, vol_a, dipm,
                                          dip_num, z, dielc)
    
    err = (fugcoef_l[0, 0] - fugcoef_v[0, 0])**2

    err = jax.lax.cond(err > 1.0e-7,
                       lambda err: err,
                       lambda err: np.float64('inf'),
                       err)
    return err


@jax.jit
def pcsaft_VP(x, m, s, e, t, p, k_ij, l_ij,
              khb_ij, e_assoc, vol_a, dipm,
              dip_num, z, dielc):
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
    p : float
        A guess for Vapor Pressure (Pa)
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

    p = (10**np.arange(-5.0, 7)[..., np.newaxis]
         * np.arange(1, 10, 0.1)).flatten()

    err = vperr(p, x, m, s, e, t, k_ij, l_ij,
                     khb_ij, e_assoc, vol_a, dipm,
                     dip_num, z, dielc)

    a = p[err.argmin()]
    b = p[err.argmin()+2]
    n = np.arange(0, 1, 1/1080)
    newp = b * n + a * (1-n)

    newerr = vperr(newp, x, m, s, e, t, k_ij, l_ij,
                     khb_ij, e_assoc, vol_a, dipm,
                     dip_num, z, dielc)

    return newp[newerr.argmin()]


vperr = jax.jit(jax.vmap(
    VP_err,
    in_axes=(0, None, None, None, None,
             None, None, None, None, None,
             None, None, None, None, None)
))


den_phase2 = jax.jit(jax.vmap(
    pcsaft_den,
    in_axes=(0, None, None, None,
             None, None, None, None,
             None, None, None, None,
             None, None, None, 0)
))


fungcoef_phase2 = jax.jit(jax.vmap(
    pcsaft_fugcoef,
    (0, None, None, None,
     None, 0, None, None,
     None, None, None, None,
     None, None, None)
))


@jax.jit
def BP_err(p, x, m, s, e, t, k_ij, l_ij,
           khb_ij, e_assoc, vol_a, dipm,
           dip_num, z, dielc):
    """Minimize this function to calculate the Bubble pressure."""

    phases = np.asarray([1.0, 0.0])
    y = x

    def update(i, y):
        xyguess = np.concatenate([x[np.newaxis, ...], y[np.newaxis, ...]], 0)

        rho = den_phase2(xyguess, m, s, e, t, p, k_ij, l_ij,
                         khb_ij, e_assoc, vol_a, dipm,
                         dip_num, z, dielc, phases)

        fugcoef_l, fugcoef_v = fungcoef_phase2(xyguess, m, s, e, t, rho, k_ij, l_ij,
                                               khb_ij, e_assoc, vol_a, dipm,
                                               dip_num, z, dielc)

        y = x*fugcoef_l/fugcoef_v

        return y/y.sum()

    y = jax.lax.fori_loop(
        0, 2, update, y
    )

    xyguess = np.concatenate([x[np.newaxis, ...], y[np.newaxis, ...]], 0)

    rho = den_phase2(xyguess, m, s, e, t, p, k_ij, l_ij,
                     khb_ij, e_assoc, vol_a, dipm,
                     dip_num, z, dielc, phases)

    fugcoef_l, fugcoef_v = fungcoef_phase2(xyguess, m, s, e, t, rho, k_ij, l_ij,
                                           khb_ij, e_assoc, vol_a, dipm,
                                           dip_num, z, dielc)

    err = np.sum((x*fugcoef_l - y*fugcoef_v)**2)

    err = jax.lax.cond(err > 1.0e-7,
                       lambda err: err,
                       lambda err: np.float64('inf'),
                       err)

    return err, y


vbperr = jax.jit(jax.vmap(
    BP_err,
    in_axes=(0, None, None, None, None,
             None, None, None, None, None,
             None, None, None, None, None)
))


@jax.jit
def pcsaft_BP(x, m, s, e, t, k_ij, l_ij, khb_ij,
              e_assoc, vol_a, dipm, dip_num, z, dielc):
    """
    Bubble pressure calculation

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
    BP : float
        Bubble Pressure (Pa)
    """

    p = (10**np.arange(-5.0, 7)[..., np.newaxis]
         * np.arange(1, 10, 0.1)).flatten()

    err, _ = vbperr(p, x, m, s, e, t, k_ij, l_ij,
                 khb_ij, e_assoc, vol_a, dipm,
                 dip_num, z, dielc)

    a = p[err.argmin()]
    b = p[err.argmin()+2]
    n = np.arange(0, 1, 1/1080)
    newp = b * n + a * (1-n)

    newerr, y = vbperr(newp, x, m, s, e, t, k_ij, l_ij,
                    khb_ij, e_assoc, vol_a, dipm,
                    dip_num, z, dielc)

    return newp[newerr.argmin()], y[newerr.argmin()]