"""
Module to handle phase equilibria calculations.
"""

from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import si_units as si
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.optimize import root_scalar

from gnnepcsaft.data.rdkit_util import smilestoinchi
from gnnepcsaft.pcsaft.pcsaft_feos import (
    is_stable_feos,
    mix_tp_flash_feos,
    pure_vp_feos,
)

from .pcsaft_feos import mix_ln_activity_coefficient

LABEL_FS = 11
TICKS_FS = 10
TITLE_FS = 11
mpl.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": TITLE_FS,
        "axes.labelsize": LABEL_FS,
        "xtick.labelsize": TICKS_FS,
        "ytick.labelsize": TICKS_FS,
    }
)

sns.set_theme(style="ticks")


# pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
def co2_binary_px(
    inchis: List[str],
    data: pl.DataFrame,
    inchi_to_params: Dict[str, List[float]],
    k_12: Optional[float] = None,
    epsilon_a1b2: Optional[float] = None,
    n_fractions: int = 50,
    n_pressure: int = 50,
) -> Tuple[Figure, List[List[Axes]]]:
    """Plot CO2 solubility in solvent from ThermoML data and GNNPCSAFT predictions.

    Args:
        inchis (List[str]): List of two InChI strings.
        data (pl.DataFrame): Polars DataFrame containing ThermoML data.
        inchi_to_params (Dict[str, List[float]]): Dictionary mapping
         InChI strings to PC-SAFT parameters.
        k_12 (Optional[float]): Binary interaction parameter between CO2 and solvent.
        epsilon_a1b2 (Optional[float]): Association energy parameter between CO2 and solvent.
        n_fractions (int): Number of fractions to check for VLE.
        n_pressure (int): Number of pressure points to calculate.

    Returns:
      out (Tuple[Figure, List[List[Axes]]]): Matplotlib figure and List of axes containing the plot.
    """

    params = [inchi_to_params[inchi] for inchi in inchis]

    kij_matrix = (
        [
            [0.0, k_12],
            [k_12, 0.0],
        ]
        if k_12 is not None
        else None
    )

    epsilon_ab = (
        [
            [0.0, epsilon_a1b2],
            [epsilon_a1b2, 0.0],
        ]
        if epsilon_a1b2 is not None
        else None
    )

    vle = data.filter(
        pl.col("inchi1").is_in(inchis),
        pl.col("inchi2").is_in(inchis),
        pl.col("mole_fraction_c1p2").is_not_null(),
    )

    x1_name = (
        "mole_fraction_c1p2" if vle["inchi1"][0] == inchis[0] else "mole_fraction_c2p2"
    )

    isotherms = (
        vle.sort("T_K")
        .group_by("T_K")
        .agg(
            pl.col("P_kPa").min().alias("min_p_kpa"),
            pl.col("P_kPa").max().alias("max_p_kpa"),
            pl.col("P_kPa").count().alias("n"),
        )
        .filter(pl.col("n") > 1)
    )
    if len(isotherms) == 0:
        raise ValueError("No data available for the given InChIs.")
    fig, axs = plt.subplots(
        len(isotherms), 1, figsize=(6, 4 * len(isotherms)), squeeze=False
    )

    feed_x1s = np.linspace(1e-5, 0.99, n_fractions)
    for ax, isotherm in zip(axs, isotherms.iter_rows(named=True)):
        temperature = isotherm["T_K"]
        pressures_kpa = np.linspace(
            isotherm["min_p_kpa"], isotherm["max_p_kpa"], n_pressure, dtype=np.float64
        )
        pred_x = []

        exp_vle = vle.filter(
            pl.col("T_K") == temperature,
        ).sort("P_kPa")
        if len(exp_vle) < 2:
            continue
        exp_x = exp_vle[x1_name].to_list()
        exp_p = exp_vle["P_kPa"].to_list()

        vp = (
            (
                pure_vp_feos(
                    parameters=inchi_to_params["InChI=1S/CO2/c2-1-3"],
                    state=[temperature],
                )
                / 1e3
            )
            if temperature < 304.2
            else 7377.3
        )

        for pressure in pressures_kpa:
            pred_x1 = np.nan
            for feed_x1 in feed_x1s:
                try:
                    if not is_stable_feos(
                        parameters=params,
                        state=[temperature, pressure * 1e3, feed_x1, 1 - feed_x1],
                        kij_matrix=kij_matrix,
                        epsilon_ab=epsilon_ab,
                        density_initialization=None,
                    ):
                        flash = mix_tp_flash_feos(
                            params,
                            [temperature, pressure * 1e3, feed_x1, 1 - feed_x1],
                            kij_matrix=kij_matrix,
                            epsilon_ab=epsilon_ab,
                        )

                        pred_x1 = (
                            flash.liquid.molefracs[0].item()
                            if flash.liquid.density > flash.vapor.density
                            else flash.vapor.molefracs[0].item()
                        )
                        break
                except RuntimeError:
                    continue
            pred_x.append(pred_x1)
        ax[0].plot(exp_p, exp_x, "x", color="black", label="Exp")
        ax[0].plot(pressures_kpa, pred_x, "-", color="r", label="Pred")
        ax[0].axvline(vp, color="gray", linestyle="--", label="CO2 Vapor Pressure")
        ax[0].set_xlabel("Pressure (kPa)")
        ax[0].set_ylabel("Mole Fraction CO2 in Liquid Phase")
        ax[0].set_title(f"T = {temperature} K")
        ax[0].legend()
    fig.tight_layout()
    return fig, axs.tolist()


def co2_ternary_px(
    smiles: List[str],
    data: pl.DataFrame,
    inchi_to_params: Dict[str, List[float]],
    kij_matrix: Optional[List[List[float]]] = None,
    epsilon_ab: Optional[List[List[float]]] = None,
) -> Tuple[Figure, List[List[Axes]]]:
    """Plot CO2 solubility in solvent mixtures from ThermoML data and GNNPCSAFT predictions.

    Args:
        smiles (List[str]): List of three SMILES strings.
        data (pl.DataFrame): Polars DataFrame containing ThermoML VLE data.
        inchi_to_params (Dict[str, List[float]]): Dictionary mapping
         InChI strings to PC-SAFT parameters.
        kij_matrix (Optional[List[List[float]]]): Binary interaction parameter matrix.
        epsilon_ab (Optional[List[List[float]]]): Association energy parameter matrix.

    Returns:
      out (Tuple[Figure, List[List[Axes]]): Matplotlib figure and List of axes containing the plot.
    """

    params = [inchi_to_params[smilestoinchi(smi)] for smi in smiles]

    vle = data.filter(
        pl.col("inchi1").is_in([smilestoinchi(smi) for smi in smiles]),
        pl.col("inchi2").is_in([smilestoinchi(smi) for smi in smiles]),
        pl.col("inchi3").is_in([smilestoinchi(smi) for smi in smiles]),
    )

    x1_name, x2_name, x3_name = _get_mole_fraction_names(vle, smiles)

    temperatures = (
        vle.filter(
            pl.col(x1_name) > 1e-10,
            pl.col(x2_name) > 1e-10,
            pl.col(x3_name) > 1e-10,
        )
        .select("T_K")
        .sort("T_K")
        .unique("T_K")
        .to_series()
        .to_list()
    )
    fig, axs = plt.subplots(
        len(temperatures), 1, figsize=(6, 4 * len(temperatures)), squeeze=False
    )
    feed_x1s = np.linspace(1e-5, 0.99, 10)
    for ax, t in zip(axs, temperatures):
        exp_x = []
        pred_x = []
        pressures = []
        for row in (
            vle.filter(
                pl.col(x1_name).is_not_null(),
                pl.col("T_K") == t,
                pl.col(x1_name) > 1e-10,
                pl.col(x2_name) > 1e-10,
                pl.col(x3_name) > 1e-10,
            )
            .sort("P_kPa")
            .iter_rows(named=True)
        ):
            x2 = row[x2_name]
            x3 = row[x3_name]
            p_pa = row["P_kPa"] * 1e3
            pred_x1 = _get_x1_ternary(
                kij_matrix, epsilon_ab, params, feed_x1s, t, x2, x3, p_pa
            )
            exp_x.append(row[x1_name])
            pred_x.append(pred_x1)
            pressures.append(row["P_kPa"])
        ax[0].plot(pressures, exp_x, "x", color="black", label="Exp")
        ax[0].plot(pressures, pred_x, "o-", color="r", label="Pred")
        ax[0].set_xlabel("Pressure (kPa)")
        ax[0].set_ylabel("Mole Fraction CO2 in Liquid Phase")
        ax[0].set_title(f"T = {t} K")
        ax[0].legend()
        fig.tight_layout()
    return fig, axs.tolist()


def _get_mole_fraction_names(
    vle: pl.DataFrame, smiles: List[str]
) -> Tuple[str, str, str]:
    """Map SMILES strings to their corresponding mole fraction column names in VLE data.

    Args:
        vle (pl.DataFrame): Polars DataFrame containing VLE data with InChI columns.
        smiles (List[str]): List of three SMILES strings to match against InChI data.

    Returns:
        out (Tuple[str, str, str]): Column names for mole fractions of the three components.
    """
    x1_name = (
        "mole_fraction_c1p2"
        if vle["inchi1"][0] == smilestoinchi(smiles[0])
        else (
            "mole_fraction_c2p2"
            if vle["inchi2"][0] == smilestoinchi(smiles[0])
            else "mole_fraction_c3p2"
        )
    )
    x2_name = (
        "mole_fraction_c1p2"
        if vle["inchi1"][0] == smilestoinchi(smiles[1])
        else (
            "mole_fraction_c2p2"
            if vle["inchi2"][0] == smilestoinchi(smiles[1])
            else "mole_fraction_c3p2"
        )
    )
    x3_name = (
        "mole_fraction_c1p2"
        if vle["inchi1"][0] == smilestoinchi(smiles[2])
        else (
            "mole_fraction_c2p2"
            if vle["inchi2"][0] == smilestoinchi(smiles[2])
            else "mole_fraction_c3p2"
        )
    )

    return x1_name, x2_name, x3_name


def _get_x1_ternary(
    kij_matrix: Optional[List[List[float]]],
    epsilon_ab: Optional[List[List[float]]],
    params: List[List[float]],
    feed_x1s: np.ndarray,
    t: float,
    x2: float,
    x3: float,
    p_pa: float,
) -> float:
    """Calculate mole fraction of component 1 in ternary mixture at flash conditions.

    Args:
        kij_matrix (Optional[List[List[float]]]): Binary interaction parameter matrix.
        epsilon_ab (Optional[List[List[float]]]): Association energy parameter matrix.
        params (List[List[float]]): PC-SAFT parameters for each component.
        feed_x1s (np.ndarray): Array of feed mole fractions for component 1 to try.
        t (float): Temperature in Kelvin.
        x2 (float): Mole fraction of component 2.
        x3 (float): Mole fraction of component 3.
        p_pa (float): Pressure in Pascal.

    Returns:
        out (float): Mole fraction of component 1 in the liquid phase,
          or np.nan if convergence fails.
    """
    pred_x1 = np.nan
    for feed_x1 in feed_x1s:
        try:
            if not is_stable_feos(
                parameters=params,
                state=[
                    t,
                    p_pa,
                    feed_x1,
                    x2,
                    x3,
                ],
                kij_matrix=kij_matrix,
                epsilon_ab=epsilon_ab,
                density_initialization=None,
            ):
                flash = mix_tp_flash_feos(
                    params,
                    [
                        t,
                        p_pa,
                        feed_x1,
                        x2,
                        x3,
                    ],
                    kij_matrix=kij_matrix,
                    epsilon_ab=epsilon_ab,
                )

                pred_x1 = (
                    flash.liquid.molefracs[0].item()
                    if flash.liquid.density > flash.vapor.density
                    else flash.vapor.molefracs[0].item()
                )
                break
        except RuntimeError:
            continue
    return pred_x1


def get_kij_matrix_ternary(
    kij_df: pl.DataFrame, inchi1: str, inchi2: str, inchi3: str
) -> List[List[float]]:
    """Extract binary interaction parameters from dataframe and construct ternary matrix.

    Args:
        kij_df (pl.DataFrame): Polars DataFrame containing binary interaction parameters.
        inchi1 (str): InChI string for component 1.
        inchi2 (str): InChI string for component 2.
        inchi3 (str): InChI string for component 3.

    Returns:
        out (List[List[float]]): 3x3 symmetric matrix of binary interaction parameters.
    """
    k_12 = (
        kij_df.filter(
            (pl.col("inchi1").is_in([inchi1, inchi2])),
            (pl.col("inchi2").is_in([inchi1, inchi2])),
        )["k_12"].to_list()
        or [0.0]
    )[0]
    k_13 = (
        kij_df.filter(
            (pl.col("inchi1").is_in([inchi1, inchi3])),
            (pl.col("inchi2").is_in([inchi1, inchi3])),
        )["k_12"].to_list()
        or [0.0]
    )[0]
    k_23 = (
        kij_df.filter(
            (pl.col("inchi1").is_in([inchi2, inchi3])),
            (pl.col("inchi2").is_in([inchi2, inchi3])),
        )["k_12"].to_list()
        or [0.0]
    )[0]

    kij_matrix = [[0.0, k_12, k_13], [k_12, 0.0, k_23], [k_13, k_23, 0.0]]
    return kij_matrix


def mix_melting_point_ideal(
    mole_fraction_i: np.ndarray, tm_i: np.ndarray, delta_h_sl: np.ndarray
) -> np.ndarray:
    """Calculates mixture melting point using ideal mixing rule and
    pure component melting points and enthalpies of fusion.


    Args:
        mole_fraction_i (np.ndarray): Mole fraction of component i in the mixture.
        tm_i (np.ndarray): Melting point of pure component i in Kelvin
        delta_h_sl (np.ndarray): Enthalpy of fusion of pure component i in kJ/mol.
    """

    temperature = 1 / (
        -np.log(mole_fraction_i) / (delta_h_sl * si.KILO * si.JOULE / si.MOL) * si.RGAS
        + 1 / (tm_i * si.KELVIN)
    )

    return temperature / si.KELVIN


def fit_melting_point(
    temperature: float,
    pressure: float,
    mole_fraction_i: np.ndarray,
    parameters: List[List[float]],
    k_ij: float,
    tm_i: np.ndarray,
    delta_h_sl: np.ndarray,
    comp_idx: int,
) -> np.ndarray:
    """Calculates the difference between the left and right side of the
    melting point equation for a given temperature, pressure,
    mole fraction, and PC-SAFT parameters for a binary mixture.
    Used for fitting melting point parameters.

    Args:
        temperature (float): Temperature in Kelvin.
        pressure (float): Pressure in Pa.
        mole_fraction_i (np.ndarray): Mole fraction of component i in the mixture.
        parameters (List[List[float]]): PC-SAFT parameters for each component.
        k_ij (float): Interaction parameter between components i and j.
        tm_i (np.ndarray): Melting point of pure component i in Kelvin.
        delta_h_sl (np.ndarray): Enthalpy of fusion of pure component i in kJ/mol.
        comp_idx (int): Index of the component for which to calculate the difference.
    """

    gamma_i = np.exp(
        mix_ln_activity_coefficient(
            parameters=parameters,
            state=[temperature, pressure, *mole_fraction_i],
            kij_matrix=[[0.0, k_ij], [k_ij, 0.0]],
        )
    )

    left_side = np.log(mole_fraction_i * gamma_i)
    right_side = (
        (delta_h_sl * si.KILO * si.JOULE / si.MOL)
        / si.RGAS
        * (1 / (tm_i * si.KELVIN) - 1 / (temperature * si.KELVIN))
    )

    return left_side[comp_idx] - right_side[comp_idx]


def fit_kij_with_tm(
    kij: np.ndarray,
    data: np.ndarray,
    pressure: float,
    parameters: List[List[float]],
    tm_i: np.ndarray,
    delta_h_sl: np.ndarray,
    comp_idx: int,
) -> np.ndarray:
    """Calculates the difference between the left and right side of the
    melting point equation for a given kij, pressure, PC-SAFT parameters,
    pure component melting points, and enthalpies of fusion for a binary mixture.
    Used for fitting the binary interaction parameter kij.


    Args:
        kij (float): Interaction parameter between components i and j.
        data (np.ndarray): Array of shape (n, 2) containing
          mole fractions of component 2 and melting temperature data.
        pressure (float): Pressure in Pa.
        parameters (List[List[float]]): PC-SAFT parameters for each component.
        tm_i (np.ndarray): Melting point of pure component i in Kelvin.
        delta_h_sl (np.ndarray): Enthalpy of fusion of pure component i in kJ/mol.
        comp_idx (int): Index of the component for which to calculate the difference.
    """

    x_all = data[:, 0][..., np.newaxis]
    tm_all = data[:, 1][..., np.newaxis]
    mole_fractions_i = np.hstack([1 - x_all, x_all])

    gamma_i = np.array(
        [
            np.exp(
                mix_ln_activity_coefficient(
                    parameters=parameters,
                    state=[tm, pressure, 1 - x_i, x_i],
                    kij_matrix=[[0.0, kij.item()], [kij.item(), 0.0]],
                )
            )
            for x_i, tm in data
        ]
    )

    left_side = np.log(mole_fractions_i * gamma_i)
    right_side = (
        (delta_h_sl * si.KILO * si.JOULE / si.MOL)
        / si.RGAS
        * (1 / (tm_i * si.KELVIN) - 1 / (tm_all * si.KELVIN))
    )

    return left_side[:, comp_idx] - right_side[:, comp_idx]


def gamma_from_exp_data(
    tm_data: np.ndarray, tm_i: np.ndarray, delta_h_sl: np.ndarray, idx: int = -5
) -> np.ndarray:
    """Calculates activity coefficient from experimental data of
    melting points and pure component melting points and enthalpies of fusion
    for a binary mixture.

    Args:
        tm_data (np.ndarray): Array of shape (n, 2) containing
          mole fractions of component 2 and melting temperature data.
        tm_i (np.ndarray): Melting point of pure component i in Kelvin.
        delta_h_sl (np.ndarray): Enthalpy of fusion of pure component i in kJ/mol.
        idx (int): Index of the data point where the activity coefficient
         changes from component 1 to component 2. This is used to determine which
         component's activity coefficient to calculate.

    """

    x_all = tm_data[:, 0][..., np.newaxis]
    tm_all = tm_data[:, 1][..., np.newaxis]
    mole_fractions_i = np.hstack([1 - x_all, x_all])

    gammas = (
        np.exp(
            (delta_h_sl * si.KILO * si.JOULE / si.MOL)
            / si.RGAS
            * (1 / (tm_i * si.KELVIN) - 1 / (tm_all * si.KELVIN))
        )
        / mole_fractions_i
    )

    return np.concatenate([gammas[:idx, 0], gammas[idx:, 1]])


def fit_kij_with_gamma(
    kij: np.ndarray,
    data: np.ndarray,
    pressure: float,
    parameters: List[List[float]],
    comp_idx: int,
) -> np.ndarray:
    """Calculate residuals for fitting binary interaction parameter using activity coefficients.

    Calculates the difference between the activity coefficient calculated from
    experimental data and the activity coefficient calculated from PC-SAFT
    for a given kij, pressure, and PC-SAFT parameters for a binary mixture.
    Used for fitting the binary interaction parameter kij.

    Args:
        kij (np.ndarray): Interaction parameter between components i and j.
        data (np.ndarray): Array of shape (n, 3) containing
            mole fractions of component 2, melting temperature, and experimental
            activity coefficient data.
        pressure (float): Pressure in Pascal.
        parameters (List[List[float]]): PC-SAFT parameters for each component.
        comp_idx (int): Index of the component for which to calculate the difference.

    Returns:
        out (np.ndarray): Array of residuals (experimental minus predicted activity coefficients).
    """

    exp_gamma_i = data[:, 2]

    gamma_i = np.array(
        [
            np.exp(
                mix_ln_activity_coefficient(
                    parameters=parameters,
                    state=[tm, pressure, 1 - x_i, x_i],
                    kij_matrix=[[0.0, kij.item()], [kij.item(), 0.0]],
                )
            )
            for x_i, tm, _ in data
        ]
    )

    return exp_gamma_i - gamma_i[:, comp_idx]


def mape_tm(
    tm_data: np.ndarray,
    parameters: List[List[float]],
    k_ij: float,
    pressure: float,
    exp_tm_i: np.ndarray,
    exp_delta_h_sl: np.ndarray,
) -> np.floating:
    """Calculate mean absolute percentage error (MAPE) for melting point predictions.

    Calculates the mean absolute percentage error (MAPE) between the experimental
    melting points and the melting points calculated from PC-SAFT for a binary mixture.

    Args:
        tm_data (np.ndarray): Array of shape (n, 2) containing
            mole fractions of component 2 and melting temperature data.
        parameters (List[List[float]]): PC-SAFT parameters for each component.
        k_ij (float): Interaction parameter between components i and j.
        pressure (float): Pressure in Pascal.
        exp_tm_i (np.ndarray): Melting point of pure component i in Kelvin.
        exp_delta_h_sl (np.ndarray): Enthalpy of fusion of pure component i in kJ/mol.

    Returns:
        out (np.floating): Mean absolute percentage error between experimental and
          predicted melting points.
    """

    mape = []
    for x_i, tm_exp in tm_data:
        tm_0, tm_1 = _get_tm(parameters, k_ij, x_i, pressure, exp_tm_i, exp_delta_h_sl)
        tm = max(tm_0, tm_1)
        if tm == 0.0:
            continue
        mape.append(np.abs((tm - tm_exp) / tm_exp))
    return np.mean(mape)


def _get_tm(
    parameters: List[List[float]],
    k_ij: float,
    x_i: float,
    pressure: float,
    exp_tm_i: np.ndarray,
    exp_delta_h_sl: np.ndarray,
) -> Tuple[float, float]:
    """Solve for melting temperatures of both components in a binary mixture.

    Args:
        parameters (List[List[float]]): PC-SAFT parameters for each component.
        k_ij (float): Interaction parameter between components i and j.
        x_i (float): Mole fraction of component 2.
        pressure (float): Pressure in Pascal.
        exp_tm_i (np.ndarray): Melting point of pure component i in Kelvin.
        exp_delta_h_sl (np.ndarray): Enthalpy of fusion of pure component i in kJ/mol.

    Returns:
        out (Tuple[float, float]): Melting temperatures for components 1 and 2 respectively.
            Returns 0.0 for any component where convergence fails.
    """
    mole_fraction_i = np.array([1 - x_i, x_i], dtype=np.float64)
    try:
        res = root_scalar(
            f=fit_melting_point,
            bracket=exp_tm_i,
            x0=exp_tm_i.min(),
            args=(
                pressure,
                mole_fraction_i,
                parameters,
                k_ij,
                exp_tm_i,
                exp_delta_h_sl,
                0,
            ),
            method="newton",
            xtol=1e-8,
        )
        tm_0 = res.root
        if not res.converged:
            raise ValueError("not converged")

    except (RuntimeError, ValueError) as e:
        print(x_i, e)
        tm_0 = 0.0

    try:
        res = root_scalar(
            f=fit_melting_point,
            bracket=exp_tm_i,
            x0=(exp_tm_i).min(),
            args=(
                pressure,
                mole_fraction_i,
                parameters,
                k_ij,
                exp_tm_i,
                exp_delta_h_sl,
                1,
            ),
            method="newton",
            xtol=1e-8,
        )
        tm_1 = res.root
        if not res.converged:
            raise ValueError("not converged")

    except (RuntimeError, ValueError) as e:
        print(x_i, e)
        tm_1 = 0.0

    return tm_0, tm_1


def plot_tm(
    all_k_ij: List[float],
    all_parameters: List[List[List[float]]],
    all_exp_tm_i: List[np.ndarray],
    all_exp_delta_h_sl: List[np.ndarray],
    all_tm_data: List[pl.DataFrame],
    fig_name: str = "fig11.png",
    mole_fraction_step: float = 0.01,
    pressure: float = 101325.0,
    plot_tm0_tm1: bool = False,
) -> Tuple[Figure, List[List[Axes]]]:
    """
    Plots the melting temperatures and activity coefficients vs mole fractions
    for a list of binary mixtures. This function generates a multi-panel figure
    comparing PC-SAFT predictions with ideal mixing behavior and experimental
    data for melting point depression and activity coefficients across
    a range of mole fractions.

    Args:
      all_k_ij (List[float]):
        List of binary interaction parameters (k_ij) for each mixture.
      all_parameters (List[List[List[float]]]):
        List of PC-SAFT parameters for each component in each mixture.
      all_exp_tm_i (List[np.ndarray]):
        List of arrays containing experimental melting
        temperatures for pure components in each mixture.
      all_exp_delta_h_sl (List[np.ndarray]):
        List of arrays containing experimental enthalpy of
        fusion for pure components in each mixture.
      all_tm_data (List[pl.DataFrame]):
        List of Polars DataFrames containing experimental
        melting point data with columns 'x' (mole fraction)
        and 'tm' (melting temperature) for each mixture.
      fig_name (str, optional):
        Output filename for the saved figure. Default is "fig11.png".
      mole_fraction_step (float, optional):
        Step size for mole fraction calculations, ranging from 0.001 to 1.0. Default is 0.01.
      pressure (float): System pressure in Pascal. Default is 101325.0.
      plot_tm0_tm1 (bool): Whether to plot tm_0 and tm_1. Default is `False`.

    Saves the figure to "images/{fig_name}" and displays it.

    Returns:
      out (Tuple[Figure, List[List[Axes]]]):
        Matplotlib figure and List of axes containing the plot.

    Notes
    -----
    - Each row in the figure corresponds to one mixture.
    - Left column (ax0): Melting temperature vs mole fraction,
      comparing PC-SAFT, ideal mixing, and experimental data.
    - Right column (ax1): Activity coefficients vs mole fraction
      for both components, with experimental data.
    - The function uses root-finding (Newton's method) to solve
      for melting points and catches convergence failures.
    - Activity coefficients are calculated from PC-SAFT mixing
      rules at the computed melting point.
    """

    fig, axs = plt.subplots(
        len(all_parameters),
        2,
        figsize=(4.68, 2.22 * len(all_parameters)),
        squeeze=False,
    )

    for idx in np.arange(len(all_parameters)):
        k_ij = all_k_ij[idx]
        parameters = all_parameters[idx]
        exp_tm_i = all_exp_tm_i[idx]
        exp_delta_h_sl = all_exp_delta_h_sl[idx]
        tm_data = all_tm_data[idx]

        melting_points = []
        melting_points_ideal = []
        melting_points_0 = []
        melting_points_1 = []
        mole_fractions_i = []
        gammas = []
        gammas_exp = gamma_from_exp_data(tm_data.to_numpy(), exp_tm_i, exp_delta_h_sl)

        for x_i in np.arange(0.001, 1.0, mole_fraction_step, dtype=np.float64):
            mole_fraction_i = np.array([1 - x_i, x_i], dtype=np.float64)

            tm_0, tm_1 = _get_tm(
                parameters, k_ij, x_i, pressure, exp_tm_i, exp_delta_h_sl
            )
            tm = max(tm_0, tm_1)
            melting_points_0.append(tm_0)
            melting_points_1.append(tm_1)
            melting_points.append(tm)
            mole_fractions_i.append(x_i)
            melting_points_ideal.append(
                mix_melting_point_ideal(mole_fraction_i, exp_tm_i, exp_delta_h_sl)
            )
            gamma = np.exp(
                mix_ln_activity_coefficient(
                    parameters=parameters,
                    state=[tm, 101325.0, *mole_fraction_i],
                    kij_matrix=[[0.0, k_ij], [k_ij, 0.0]],
                )
            )
            gammas.append(gamma)

        ax0 = axs[idx, 0]
        ax0.plot(
            mole_fractions_i,
            melting_points,
            label="PC-SAFT",
            color="C0",
            linestyle="-",
        )
        ax0.plot(
            mole_fractions_i,
            melting_points_ideal,
            label="Ideal",
            linestyle="--",
            color="#d62728",
        )
        ax0.plot(
            tm_data["x"],
            tm_data["tm"],
            label="Experimental",
            linestyle="",
            marker="o",
            color="black",
            markerfacecolor="none",
        )
        if plot_tm0_tm1:
            tm0_plot = np.where(
                np.array(melting_points_0) == 0.0, np.nan, melting_points_0
            )
            tm1_plot = np.where(
                np.array(melting_points_1) == 0.0, np.nan, melting_points_1
            )
            ax0.plot(
                mole_fractions_i,
                tm0_plot,
                label=r"$T_{m,1}$",
                color="#9467bd",
            )
            ax0.plot(
                mole_fractions_i,
                tm1_plot,
                label=r"$T_{m,2}$",
                color="#8c564b",
            )
        ax0.set_xlabel(r"$x_1$")
        ax0.set_ylabel(r"T / K")

        ax1 = axs[idx, 1]
        gammas_array = np.array(gammas)
        ax1.plot(
            mole_fractions_i,
            gammas_array[:, 0],
            color="C0",
            label=r"$\gamma_1$",
        )
        ax1.plot(
            mole_fractions_i,
            gammas_array[:, 1],
            color="C0",
            label=r"$\gamma_2$",
        )
        ax1.plot(
            tm_data["x"],
            gammas_exp,
            label="Experimental",
            linestyle="",
            marker="o",
            color="black",
            markerfacecolor="none",
        )
        ax1.set_xlabel(r"$x_1$")
        ax1.set_ylabel(r"$\gamma_i$")
        ax1.axhline(y=1.0, linestyle="--", color="r")

        for ax in (ax0, ax1):
            # Adiciona ticks para dentro em todos os lados
            # ax.tick_params(direction="in", top=False, right=False)
            ax.set_xlim([0, 1])
            ax.grid(False)
    sns.despine(trim=True)
    plt.tight_layout()
    plt.savefig(
        "images/" + fig_name,
        dpi=600,
        format="png",
        bbox_inches="tight",
        transparent=False,
    )

    return fig, axs.tolist()
