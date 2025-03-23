"Module to calculate properties with ePC-SAFT using TEQP."

import numpy as np
import PCSAFTsuperanc
import teqp

N_A = PCSAFTsuperanc.N_A * (1e-10) ** 3  # adjusted to angstron unit


def pure_den_teqp(parameters: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Calcules pure component density with ePC-SAFT."""

    t = state[0]  # Temperature, K

    # pylint: disable=E1101,I1101
    c = teqp.SAFTCoeffs()  # type: ignore
    c.m = max(parameters[0], 1.0)  # units
    c.sigma_Angstrom = parameters[1]  # Å
    c.epsilon_over_k = parameters[2]  # K
    model = teqp.PCSAFTEOS([c])

    # pylint: disable=I1101
    # T tilde = T / (e / kB) https://teqp.readthedocs.io/en/latest/models/PCSAFT.html
    [ttilde_crit, ttilde_min] = PCSAFTsuperanc.get_Ttilde_crit_min(m=c.m)
    ttilde = t / c.epsilon_over_k
    ttilde = min(max(ttilde, ttilde_min), ttilde_crit)
    # Rho tilde = RhoN * sigma ** 3 https://teqp.readthedocs.io/en/latest/models/PCSAFT.html
    [tilderhol, tilderhov] = PCSAFTsuperanc.PCSAFTsuperanc_rhoLV(Ttilde=ttilde, m=c.m)
    rhol_guess, rhov_guess = [
        tilderho / (N_A * c.sigma_Angstrom**3) for tilderho in [tilderhol, tilderhov]
    ]

    rhol, rhov = model.pure_VLE_T(t, rhol_guess * 0.98, rhov_guess * 1.02, 10)
    den = rhol if state[2] == 1 else rhov

    return den


def pure_vp_teqp(parameters: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Calculates pure component vapor pressure with ePC-SAFT."""

    t = state[0]  # Temperature, K
    x = np.array([1.0])  # mole fraction

    m = max(parameters[0], 1.0)  # units
    s = parameters[1]  # Å
    e = parameters[2]  # K

    # pylint: disable=E1101,I1101
    c = teqp.SAFTCoeffs()  # type: ignore
    c.m = m
    c.sigma_Angstrom = s
    c.epsilon_over_k = e
    coeffs = [c]
    model = teqp.PCSAFTEOS(coeffs)

    rho = pure_den_teqp(parameters, state)

    # P = rho * R * T * (1 + Ar01) https://teqp.readthedocs.io/en/latest/derivs/derivs.html
    p = rho * model.get_R(x) * t * (1 + model.get_Ar01(t, rho, x))

    return p
