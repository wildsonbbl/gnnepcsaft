"Module to calculate properties with ePC-SAFT using FEOS."

from typing import Dict, List, Optional, Tuple

import numpy as np
import si_units as si
from feos import dft  # type: ignore # pylint: disable = E0401
from feos.eos import (  # type: ignore # pylint: disable = E0401
    Contributions,
    EquationOfState,
    PhaseDiagram,
    PhaseEquilibrium,
    State,
)
from feos.pcsaft import (  # type: ignore # pylint: disable = E0401
    Identifier,
    PcSaftParameters,
    PcSaftRecord,
    PureRecord,
)


def pc_saft(parameters: List[float]) -> EquationOfState.pcsaft:
    """
    Returns a ePC-SAFT equation of state.

    Args:
        parameters: A list with
         `[m, sigma, epsilon/kB, kappa_ab, epsilon_ab/kB, dipole moment, na, nb]`

    """
    parameters.append(1.0)
    return pc_saft_mixture([parameters])


def pc_saft_mixture(
    mixture_parameters: List[List[float]],
    kij_matrix: Optional[List[List[float]]] = None,
) -> EquationOfState.pcsaft:
    """
    Returns a ePC-SAFT equation of state.

    Args:
        mixture_parameters: A list of
         `[m, sigma, epsilon/kB, kappa_ab, epsilon_ab/kB, dipole moment, na, nb, MW]`
         for each component of the mixture
        kij_matrix: A matrix of binary interaction parameters
    """
    records = get_records(mixture_parameters)

    if kij_matrix:
        binary_records = np.asarray(kij_matrix, dtype=np.float64)
    else:
        binary_records = np.zeros((len(records), len(records)), dtype=np.float64)
    pcsaftparameters = PcSaftParameters.from_records(
        records, binary_records=binary_records
    )
    eos = EquationOfState.pcsaft(pcsaftparameters)
    return eos


def get_records(mixture_parameters: List[List[float]]) -> list[PureRecord]:
    """
    Returns a list of `feos.pcsaft.PureRecord`.

    Args:
        mixture_parameters: A list of
         `[m, sigma, epsilon/kB, kappa_ab, epsilon_ab/kB, dipole moment, na, nb, MW]`
         for each component of the mixture
    """
    records = []
    for idx, mol_parameters in enumerate(mixture_parameters):
        records.append(
            PureRecord(
                identifier=Identifier(
                    smiles=f"SMILES_{idx}",
                    inchi=f"InChI_{idx}",
                ),
                molarweight=mol_parameters[-1],  # g/mol
                model_record=PcSaftRecord(
                    m=mol_parameters[0],  # units
                    sigma=mol_parameters[1],  # Å
                    epsilon_k=mol_parameters[2],  # K
                    kappa_ab=mol_parameters[3],
                    epsilon_k_ab=mol_parameters[4],  # K
                    mu=mol_parameters[5],  # Debye
                    na=mol_parameters[6],
                    nb=mol_parameters[7],
                ),
            )
        )

    return records


def mix_gibbs_energy(
    parameters: List[List[float]],
    state: List[float],
    kij_matrix: Optional[List[List[float]]] = None,
):
    """
    Calculates mixture `Molar Gibbs Energy/RT` with ePC-SAFT.

    Args:
        parameters: A list of
         `[m, sigma, epsilon/kB, kappa_ab, epsilon_ab/kB, dipole moment, na, nb, MW]`
         for each component of the mixture
        state: A list with
         `[Temperature (K), Pressure (Pa), mole_fractions_1, mole_fractions_2, ...]`
        kij_matrix: A matrix of binary interaction parameters
    """
    x = np.asarray(state[2:], dtype=np.float64)  # mole fractions

    excess_g = mix_e_gibbs_energy(parameters, state, kij_matrix)

    return excess_g + np.sum(x * np.log(x))


def mix_ln_fugacity_coefficient_pure(
    parameters: List[List[float]],
    state: List[float],
    kij_matrix: Optional[List[List[float]]] = None,
):
    """
    Calculates mixture `ln(fugacity coefficient)` with ePC-SAFT for each pure component.

    Args:
        parameters: A list of
         `[m, sigma, epsilon/kB, kappa_ab, epsilon_ab/kB, dipole moment, na, nb, MW]`
         for each component of the mixture
        state: A list with
         `[Temperature (K), Pressure (Pa), mole_fractions_1, mole_fractions_2, ...]`
        kij_matrix: A matrix of binary interaction parameters
    """

    t = state[0]  # Temperature, K
    p = state[1]  # Pa
    x = np.asarray(state[2:], dtype=np.float64)  # mole fractions

    eos = pc_saft_mixture(parameters, kij_matrix)

    statenpt = State(
        eos,
        temperature=t * si.KELVIN,
        pressure=p * si.PASCAL,
        molefracs=x,
        density_initialization="liquid",
    )

    return statenpt.ln_phi_pure_liquid()


def mix_ln_activity_coefficient(
    parameters: List[List[float]],
    state: List[float],
    kij_matrix: Optional[List[List[float]]] = None,
) -> List[float]:
    """
    Calculates mixture `ln(activity coefficient)` with ePC-SAFT for each component.

    Args:
        parameters: A list of
         `[m, sigma, epsilon/kB, kappa_ab, epsilon_ab/kB, dipole moment, na, nb, MW]`
         for each component of the mixture
        state: A list with
         `[Temperature (K), Pressure (Pa), mole_fractions_1, mole_fractions_2, ...]`
        kij_matrix: A matrix of binary interaction parameters
    """

    t = state[0]  # Temperature, K
    p = state[1]  # Pa
    x = np.asarray(state[2:], dtype=np.float64)  # mole fractions

    eos = pc_saft_mixture(parameters, kij_matrix)

    statenpt = State(
        eos,
        temperature=t * si.KELVIN,
        pressure=p * si.PASCAL,
        molefracs=x,
        density_initialization="liquid",
    )

    return statenpt.ln_symmetric_activity_coefficient()


def mix_e_gibbs_energy(
    parameters: List[List[float]],
    state: List[float],
    kij_matrix: Optional[List[List[float]]] = None,
) -> float:
    """
    Calculates mixture `Molar Excess Gibbs Energy/RT` with ePC-SAFT.

    Args:
        parameters: A list of
         `[m, sigma, epsilon/kB, kappa_ab, epsilon_ab/kB, dipole moment, na, nb, MW]`
         for each component of the mixture
        state: A list with
         `[Temperature (K), Pressure (Pa), mole_fractions_1, mole_fractions_2, ...]`
        kij_matrix: A matrix of binary interaction parameters
    """

    x = np.asarray(state[2:], dtype=np.float64)  # mole fractions

    return np.sum(mix_ln_activity_coefficient(parameters, state, kij_matrix) * x)


def mix_ln_fugacity_coefficient(
    parameters: List[List[float]],
    state: List[float],
    kij_matrix: Optional[List[List[float]]] = None,
) -> List[float]:
    """
    Calculates mixture `ln(fugacity coefficient)` with ePC-SAFT for each component.

    Args:
        parameters: A list of
         `[m, sigma, epsilon/kB, kappa_ab, epsilon_ab/kB, dipole moment, na, nb, MW]`
         for each component of the mixture
        state: A list with
         `[Temperature (K), Pressure (Pa), mole_fractions_1, mole_fractions_2, ...]`
        kij_matrix: A matrix of binary interaction parameters
    """
    t = state[0]  # Temperature, K
    p = state[1]  # Pa
    x = np.asarray(state[2:], dtype=np.float64)  # mole fractions

    eos = pc_saft_mixture(parameters, kij_matrix)

    statenpt = State(
        eos,
        temperature=t * si.KELVIN,
        pressure=p * si.PASCAL,
        molefracs=x,
        density_initialization="liquid",
    )

    return statenpt.ln_phi()


def mix_r_gibbs_energy(
    parameters: List[List[float]],
    state: List[float],
    kij_matrix: Optional[List[List[float]]] = None,
) -> float:
    """
    Calculates mixture `Molar Residual Gibbs Energy/RT` with ePC-SAFT.

    Args:
        parameters: A list of
         `[m, sigma, epsilon/kB, kappa_ab, epsilon_ab/kB, dipole moment, na, nb, MW]`
         for each component of the mixture
        state: A list with
         `[Temperature (K), Pressure (Pa), mole_fractions_1, mole_fractions_2, ...]`
        kij_matrix: A matrix of binary interaction parameters
    """
    x = np.asarray(state[2:], dtype=np.float64)  # mole fractions
    return np.sum(mix_ln_fugacity_coefficient(parameters, state, kij_matrix) * x)


def mix_den_feos(
    parameters: List[List[float]],
    state: List[float],
    kij_matrix: Optional[List[List[float]]] = None,
) -> float:
    """
    Calculates mixture liquid density (mol/m³) with ePC-SAFT.

    Args:
        parameters: A list of
         `[m, sigma, epsilon/kB, kappa_ab, epsilon_ab/kB, dipole moment, na, nb, MW]`
         for each component of the mixture
        state: A list with
         `[Temperature (K), Pressure (Pa), mole_fractions_1, mole_fractions_2, ...]`
        kij_matrix: A matrix of binary interaction parameters
    """

    t = state[0]  # Temperature, K
    p = state[1]  # Pa
    x = np.asarray(state[2:], dtype=np.float64)  # mole fractions

    eos = pc_saft_mixture(parameters, kij_matrix)

    statenpt = State(
        eos,
        temperature=t * si.KELVIN,
        pressure=p * si.PASCAL,
        molefracs=x,
        density_initialization="liquid",
    )

    den = statenpt.density * (si.METER**3) / si.MOL

    return den


def pure_den_feos(parameters: List[float], state: List[float]) -> float:
    """
    Calculates pure component liquid density (mol/m³) with ePC-SAFT.

    Args:
        parameters: A list with
         `[m, sigma, epsilon/kB, kappa_ab, epsilon_ab/kB, dipole moment, na, nb]`
        state: A list with `[Temperature (K), Pressure (Pa)]`
    """

    t = state[0]  # Temperature, K
    p = state[1]  # Pa

    eos = pc_saft(parameters)
    statenpt = State(
        eos,
        temperature=t * si.KELVIN,
        pressure=p * si.PASCAL,
        density_initialization="liquid",
    )

    den = statenpt.density * (si.METER**3) / si.MOL

    return den


def mix_vp_feos(
    parameters: List[List[float]],
    state: List[float],
    kij_matrix: Optional[List[List[float]]] = None,
) -> Tuple[float, float]:
    """
    Calculates mixture `(Bubble point (Pa), Dew point (Pa))` with ePC-SAFT.

    Args:
        parameters: A list of
         `[m, sigma, epsilon/kB, kappa_ab, epsilon_ab/kB, dipole moment, na, nb, MW]`
         for each component of the mixture
        state: A list with
         `[Temperature (K), Pressure (Pa), mole_fractions_1, molefractions_2, ...]`
        kij_matrix: A matrix of binary interaction parameters
    """

    t = state[0]  # Temperature, K
    x = np.asarray(state[2:], dtype=np.float64)  # mole fractions

    eos = pc_saft_mixture(parameters, kij_matrix=kij_matrix)

    vle_bubble_point = PhaseEquilibrium.bubble_point(
        eos, temperature_or_pressure=t * si.KELVIN, liquid_molefracs=x
    )

    vle_dew_point = PhaseEquilibrium.dew_point(
        eos, temperature_or_pressure=t * si.KELVIN, vapor_molefracs=x
    )

    assert (
        t == vle_bubble_point.liquid.temperature / si.KELVIN
    ), "Temperature mismatch for bubble point"
    assert (
        t == vle_dew_point.vapor.temperature / si.KELVIN
    ), "Temperature mismatch for dew point"
    return (
        vle_bubble_point.liquid.pressure() / si.PASCAL,
        vle_dew_point.vapor.pressure() / si.PASCAL,
    )


def pure_vp_feos(parameters: List[float], state: List[float]) -> float:
    """
    Calculates pure component vapor pressure (Pa) with ePC-SAFT.

    Args:
        parameters: A list with
         `[m, sigma, epsilon/kB, kappa_ab, epsilon_ab/kB, dipole moment, na, nb]`
        state: A list with `[Temperature (K)]`
    """

    t = state[0]  # Temperature, K

    eos = pc_saft(parameters)
    vle = PhaseEquilibrium.pure(eos, temperature_or_pressure=t * si.KELVIN)

    assert t == vle.liquid.temperature / si.KELVIN

    return vle.liquid.pressure() / si.PASCAL


def pure_h_lv_feos(parameters: List[float], state: List[float]) -> float:
    """
    Calculates pure component enthalpy of vaporization (kJ/mol) with ePC-SAFT.

    Args:
        parameters: A list with
         `[m, sigma, epsilon/kB, kappa_ab, epsilon_ab/kB, dipole moment, na, nb]`
        state: A list with `[Temperature (K)]`
    """

    t = state[0]  # Temperature, K

    eos = pc_saft(parameters)
    vle = PhaseEquilibrium.pure(eos, temperature_or_pressure=t * si.KELVIN)

    liquid_state = vle.liquid
    vapor_state = vle.vapor

    assert t == liquid_state.temperature / si.KELVIN

    return (
        vapor_state.molar_enthalpy(Contributions.Residual)
        - liquid_state.molar_enthalpy(Contributions.Residual)
    ) * (si.MOL / si.KILO / si.JOULE)


def pure_s_lv_feos(parameters: List[float], state: List[float]) -> float:
    """
    Calcules pure component entropy of vaporization (J/mol*K) with ePC-SAFT.

    Args:
        parameters: A list with
         `[m, sigma, epsilon/kB, kappa_ab, epsilon_ab/kB, dipole moment, na, nb]`
        state: A list with `[Temperature (K)]`
    """
    t = state[0]  # Temperature, K
    eos = pc_saft(parameters)
    vle = PhaseEquilibrium.pure(eos, temperature_or_pressure=t * si.KELVIN)
    liquid_state = vle.liquid
    vapor_state = vle.vapor
    assert t == liquid_state.temperature / si.KELVIN
    return (
        vapor_state.molar_entropy(Contributions.Residual)
        - liquid_state.molar_entropy(Contributions.Residual)
    ) * (si.MOL * si.KELVIN / si.JOULE)


def critical_points_feos(parameters: List[float]) -> List[float]:
    """
    Calculates critical points `[Tc (K), Pc (Pa), Dc (mol/m³)]` with ePC-SAFT.

    Args:
        parameters: A list with
         `[m, sigma, epsilon/kB, kappa_ab, epsilon_ab/kB, dipole moment, na, nb]`
    """
    eos = pc_saft(parameters)
    critical_point = State.critical_point(eos)
    return [
        critical_point.temperature / si.KELVIN,
        critical_point.pressure() / si.PASCAL,
        critical_point.density * (si.METER**3) / si.MOL,
    ]


def pure_viscosity_feos(parameters: List[float], state: List[float]) -> float:
    """
    Calcules pure component viscosity (kPa*s) with ePC-SAFT.

    Args:
        parameters: A list with
         `[m, sigma, epsilon/kB, kappa_ab, epsilon_ab/kB, dipole moment, na, nb]`
        state: A list with `[Temperature (K), Pressure (Pa)]`
    """
    t = state[0]  # Temperature, K
    p = state[1]  # Pa

    eos = pc_saft(parameters)
    statenpt = State(
        eos,
        temperature=t * si.KELVIN,
        pressure=p * si.PASCAL,
        density_initialization="liquid",
    )

    return statenpt.viscosity()  # / (KILO * PASCAL * SECOND)


def phase_diagram_feos(parameters: List[float], state: List[float]) -> Dict[str, float]:
    """
    Calculates phase diagram from
    state temperature up to the critical temperature with ePC-SAFT.

    Args:
        parameters: A list with
         `[m, sigma, epsilon/kB, kappa_ab, epsilon_ab/kB, dipole moment, na, nb]`
        state: A list with `[Temperature (K)]`
    """
    t = state[0]  # Temperature, K
    eos = pc_saft(parameters)
    phase_diagram = PhaseDiagram.pure(eos, min_temperature=t * si.KELVIN, npoints=200)

    return phase_diagram.to_dict(Contributions.Residual)


def pure_surface_tension_feos(
    parameters: List[float], state: List[float]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates pure component `[Surface Tension (mN/m), Temperature (K)]` with ePC-SAFT
    from state temperature up to the critical temperature with ePC-SAFT.

    Args:
        parameters: A list with
         `[m, sigma, epsilon/kB, kappa_ab, epsilon_ab/kB, dipole moment, na, nb, MW]`
        state: A list with `[Temperature (K)]`
    """
    t = state[0]  # Temperature, K
    records = get_records([parameters])

    pcsaftparameters = PcSaftParameters.from_records(records)
    functional = dft.HelmholtzEnergyFunctional.pcsaft(pcsaftparameters)
    phase_diagram = dft.PhaseDiagram.pure(functional, t * si.KELVIN, 100)
    st_diagram = dft.SurfaceTensionDiagram(phase_diagram.states, n_grid=1024)

    st = st_diagram.surface_tension / (si.MILLI * si.NEWTON / si.METER)
    temp = st_diagram.liquid.temperature / si.KELVIN
    return st, temp


def parameters_gc_pcsaft(smiles: str) -> List[float]:
    """
    Calculates PC-SAFT parameters with Group Contribution method.

    Args:
        smiles (str): SMILES of the compound
    """
    pure_record = (
        PcSaftParameters.from_json_smiles(
            [smiles],
            "./gnnepcsaft/data/gc_pcsaft/sauer2014_smarts.json",
            "./gnnepcsaft/data/gc_pcsaft/rehner2023_hetero.json",
        )
        .pure_records[0]
        .model_record
    )

    m = pure_record.m
    sigma = pure_record.sigma
    e = pure_record.epsilon_k
    mu = pure_record.mu if pure_record.mu else 0
    kab = pure_record.kappa_ab if pure_record.kappa_ab else 0
    eab = pure_record.epsilon_k_ab if pure_record.epsilon_k_ab else 0
    na = pure_record.na if pure_record.na else 0
    nb = pure_record.nb if pure_record.nb else 0

    return [m, sigma, e, kab, eab, mu, na, nb]


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    parameters_1 = [
        [2.36931, 2.13688, 229.08936, 0.35375, 2191.07976, 0.0, 1, 1],  # water
        [3.42422, 4.05778, 287.37372, 0.00236, 3004.83009, 0.0, 1, 1],  # octanol
    ]
    parameters_2 = [
        [1.23923, 3.31619, 89.78633, 0.00086, 1159.69411, 0.0, 2, 0],  # N2
        [1.0, 3.70632, 151.87309, 0.0, 0.0, 0.0, 0, 0],  # CH4
    ]

    parameters_3 = [
        [2.50956, 3.43242, 271.01627, 0.0, 0.0, 0.0, 0, 0],  # ClC(Cl)Cl
        [3.48832, 3.79358, 237.78816, 0.0, 0.0, 0.0, 0, 0],  # CCCCCCC
    ]

    parameters_4 = [
        [2.77575, 3.24163, 230.60544, 0.00018, 1448.962, 0.0, 1, 0],  # CC(=O)C
        [2.24243, 2.82464, 182.92549, 0.0872, 2447.75103, 0.0, 1, 1],  # CO
    ]
    parameters_5 = [
        [2.77575, 3.24163, 230.60544, 0.00018, 1448.962, 0.0, 1, 0],  # CC(=O)C
        [2.50956, 3.43242, 271.01627, 0.0, 0.0, 0.0, 0, 0],  # ClC(Cl)Cl
    ]

    parameters_6 = [
        [2.87326, 2.96227, 187.37704, 0.05594, 2460.62173, 0.0, 1, 1],  # CCO
        [3.48832, 3.79358, 237.78816, 0.0, 0.0, 0.0, 0, 0],  # CCCCCCC
    ]

    parameters_7 = [
        [2.87326, 2.96227, 187.37704, 0.05594, 2460.62173, 0.0, 1, 1],  # CCO
        [2.50956, 3.43242, 271.01627, 0.0, 0.0, 0.0, 0, 0],  # ClC(Cl)Cl
    ]

    parameters_8 = [
        [2.87326, 2.96227, 187.37704, 0.05594, 2460.62173, 0.0, 1, 1],  # CCO
        [2.36931, 2.13688, 229.08936, 0.35375, 2191.07976, 0.0, 1, 1],  # water
    ]

    parameters_9 = [
        [
            1.77033,
            4.11536,
            408.98859,
            0.01119,
            2028.69335,
            0.0,
            3,
            3,
        ],  # C(C(CO)O)O
        [
            1.87119,
            4.36345,
            421.9549,
            0.00102,
            2253.55334,
            0.0,
            1.0,
            1.0,
        ],  # C[N+](C)(C)CCO.[Cl-]
    ]

    parameters_10 = [
        [
            3.18316,
            2.91089,
            288.20007,
            0.0001,
            198.64934,
            0.0,
            1,
            1,
        ],  #  CC(=O)O
        [
            1.87119,
            4.36345,
            421.9549,
            0.00102,
            2253.55334,
            0.0,
            1.0,
            1.0,
        ],  # C[N+](C)(C)CCO.[Cl-]
    ]

    parameters_11 = [
        [
            2.50956,
            3.43242,
            271.01627,
            0.0,
            0.0,
            0.0,
            0,
            0,
        ],  # ClC(Cl)Cl
        [
            1.87119,
            4.36345,
            421.9549,
            0.00102,
            2253.55334,
            0.0,
            1.0,
            1.0,
        ],  # C[N+](C)(C)CCO.[Cl-]
    ]

    parameters_12 = [
        [
            2.77575,
            3.24163,
            230.60544,
            0.00018,
            1448.962,
            0.0,
            1,
            0,
        ],  # CC(=O)C
        [
            1.87119,
            4.36345,
            421.9549,
            0.00102,
            2253.55334,
            0.0,
            1.0,
            1.0,
        ],  # C[N+](C)(C)CCO.[Cl-]
    ]

    parameters_13 = [
        [
            2.77575,
            3.24163,
            230.60544,
            0.00018,
            1448.962,
            0.0,
            1,
            0,
        ],  # CC(=O)C
        [
            2.47904,
            3.18442,
            254.75797,
            0.0,
            0.0,
            0.0,
            0,
            0,
        ],  # ClCCl
    ]

    temperatures = np.linspace(300, 370, 10)
    fig1 = plt.figure(1)
    fig2 = plt.figure(2)
    fig3 = plt.figure(3)
    for tp in temperatures:
        ge = []
        molxs = []
        g = []
        for molx in np.linspace(1e-5, 0.9999, 500):
            g.append(mix_gibbs_energy(parameters_1, [tp, 101325, molx, 1 - molx], None))
            ge.append(
                mix_e_gibbs_energy(parameters_1, [tp, 101325, molx, 1 - molx], None)
            )
            molxs.append(molx)

        plt.figure(fig1.number)
        plt.plot(molxs, ge, label=f"T = {round(tp, 2)} K")
        plt.figure(fig2.number)
        plt.plot(molxs, g, label=f"T = {round(tp, 2)} K")

    plt.figure(fig1.number)
    plt.legend(loc=(1.01, 0.0))
    plt.xlabel("x-water")
    plt.ylabel(r"$G^{E}/RT$")

    plt.figure(fig2.number)
    plt.legend(loc=(1.01, 0.0))
    plt.xlabel("x-water")
    plt.ylabel(r"$G^{mix}/RT$")

    import math

    N = 13  # número de subplots
    NCOLS = 4
    nrows = math.ceil(N / NCOLS)
    fig, axes = plt.subplots(nrows, NCOLS, figsize=(16, 10), sharex=True, sharey=False)
    axes = axes.flatten()

    for i in range(1, N + 1):
        PARA_NAME = "parameters_" + str(i)
        para = eval(PARA_NAME)  # pylint: disable=eval-used
        ge = []
        molxs = []
        TEMP = 273.15 + 50
        for molx in np.linspace(1e-5, 0.9999, 500):
            ge.append(
                mix_e_gibbs_energy(para, [TEMP, 101325, molx, 1 - molx], None)
                * si.RGAS
                * TEMP
                * si.KELVIN
                / (si.JOULE / si.MOL)
            )
            molxs.append(molx)
        ax = axes[i - 1]
        ax.plot(molxs, ge)
        ax.set_title(PARA_NAME)
        ax.set_xlabel("x")
        ax.set_ylabel(r"$G^{E}$ (J $mol^{-1}$)")

    # Remove subplots vazios, se houver
    for j in range(N, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
