"""CO2 binary/ternary VLE helpers for phase equilibria plots."""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.optimize import minimize_scalar, root_scalar

from gnnepcsaft.data.rdkit_util import smilestoinchi
from gnnepcsaft.pcsaft.feos.equilibria import mix_vp_feos
from gnnepcsaft.pcsaft.pcsaft_feos import (
    pure_vp_feos,
)

from ._phase_equilibria_style import (
    CO2_CRITICAL_P_KPA,
    CO2_CRITICAL_T_K,
    CO2_INCHI,
)

# pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals


def co2_binary_px(
    inchis: List[str],
    data: pl.DataFrame,
    inchi_to_params: Dict[str, List[float]],
    k_12: Optional[float] = None,
    epsilon_a1b2: Optional[float] = None,
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

    for ax, isotherm in zip(axs, isotherms.iter_rows(named=True)):
        temperature = isotherm["T_K"]
        pred_bubble_points = []

        exp_vle = vle.filter(
            pl.col("T_K") == temperature,
        ).sort("P_kPa")
        if len(exp_vle) < 2:
            continue
        exp_x = exp_vle[x1_name].to_list()
        exp_p = exp_vle["P_kPa"].to_list()

        vp = (
            (
                pure_vp_feos(parameters=inchi_to_params[CO2_INCHI], state=[temperature])
                / 1e3
            )
            if temperature < CO2_CRITICAL_T_K
            else CO2_CRITICAL_P_KPA
        )

        for x1 in exp_x:
            pred_bp, _ = mix_vp_feos(
                parameters=params,
                state=[temperature, np.nan, x1, 1 - x1],
                kij_matrix=kij_matrix,
                epsilon_ab=epsilon_ab,
            )
            pred_bubble_points.append(pred_bp / 1e3)
        ax[0].plot(exp_p, exp_x, "x", color="black", label="Exp")
        ax[0].plot(pred_bubble_points, exp_x, "-", color="r", label="Pred")
        ax[0].axvline(vp, color="gray", linestyle="--", label="CO2 Vapor Pressure")
        ax[0].set_xlabel("Pressure (kPa)")
        ax[0].set_ylabel("Mole Fraction CO2 in Liquid Phase")
        ax[0].set_title(f"T = {temperature} K")
        ax[0].legend()
    fig.tight_layout()
    return fig, axs.tolist()


def _solve_co2_binary_tx_temperature(
    x1: float,
    pressure_kpa: float,
    params: List[List[float]],
    kij_matrix: Optional[List[List[float]]],
    epsilon_ab: Optional[List[List[float]]],
    temp_min: float,
    temp_max: float,
) -> float:
    """Find the temperature where the predicted bubble pressure matches the target pressure."""

    def pressure_residual(temp: float) -> float:
        pred_bp, _ = mix_vp_feos(
            parameters=params,
            state=[temp, np.nan, x1, 1 - x1],
            kij_matrix=kij_matrix,
            epsilon_ab=epsilon_ab,
        )
        return pred_bp / 1e3 - pressure_kpa

    try:
        lower_residual = pressure_residual(temp_min)
        upper_residual = pressure_residual(temp_max)

        if np.isfinite(lower_residual) and lower_residual == 0.0:
            return temp_min
        if np.isfinite(upper_residual) and upper_residual == 0.0:
            return temp_max

        if (
            np.isfinite(lower_residual)
            and np.isfinite(upper_residual)
            and (lower_residual * upper_residual < 0.0)
        ):
            root = root_scalar(
                pressure_residual,
                bracket=[temp_min, temp_max],
                method="brentq",
                xtol=1e-8,
            )
            if root.converged:
                return float(root.root)
            return np.nan

        minimization = minimize_scalar(
            lambda temp: float(np.abs(pressure_residual(temp))),
            bounds=(temp_min, temp_max),
            method="bounded",
            options={"xatol": 1e-4},
        )
        return float(getattr(minimization, "x", np.nan))
    except (RuntimeError, ValueError):
        return np.nan


def co2_binary_tx(
    inchis: List[str],
    data: pl.DataFrame,
    inchi_to_params: Dict[str, List[float]],
    k_12: Optional[float] = None,
    epsilon_a1b2: Optional[float] = None,
) -> Tuple[Figure, List[List[Axes]]]:
    """Plot CO2 solubility in solvent from ThermoML data and GNNPCSAFT predictions (T-x).

    Args:
        inchis (List[str]): List of two InChI strings.
        data (pl.DataFrame): Polars DataFrame containing ThermoML data.
        inchi_to_params (Dict[str, List[float]]): Dictionary mapping
         InChI strings to PC-SAFT parameters.
        k_12 (Optional[float]): Binary interaction parameter between CO2 and solvent.
        epsilon_a1b2 (Optional[float]): Association energy parameter between CO2 and solvent.
        n_fractions (int): Number of fractions to check for VLE.
        n_temperatures (int): Number of temperature points to calculate.

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

    isobars = (
        vle.sort("P_kPa")
        .group_by("P_kPa")
        .agg(
            pl.col("T_K").min().alias("min_t_k"),
            pl.col("T_K").max().alias("max_t_k"),
            pl.col("T_K").count().alias("n"),
        )
        .filter(pl.col("n") > 1)
    )
    if len(isobars) == 0:
        raise ValueError("No data available for the given InChIs.")
    fig, axs = plt.subplots(
        len(isobars), 1, figsize=(6, 4 * len(isobars)), squeeze=False
    )

    for ax, isobar in zip(axs, isobars.iter_rows(named=True)):
        pressure = isobar["P_kPa"]
        pred_t = []

        exp_vle = vle.filter(
            pl.col("P_kPa") == pressure,
        ).sort("T_K")
        if len(exp_vle) < 2:
            continue
        exp_x = exp_vle[x1_name].to_list()
        exp_t = exp_vle["T_K"].to_list()

        temp_min = max(1.0, float(min(exp_t)) - 25.0)
        temp_max = float(max(exp_t)) + 25.0

        for x1 in exp_x:
            pred_t.append(
                _solve_co2_binary_tx_temperature(
                    x1=x1,
                    pressure_kpa=pressure,
                    params=params,
                    kij_matrix=kij_matrix,
                    epsilon_ab=epsilon_ab,
                    temp_min=temp_min,
                    temp_max=temp_max,
                )
            )

        ax[0].plot(exp_t, exp_x, "x", color="black", label="Exp")
        ax[0].plot(pred_t, exp_x, "-", color="r", label="Pred")
        ax[0].set_xlabel("Temperature (K)")
        ax[0].set_ylabel("Mole Fraction CO2 in Liquid Phase")
        ax[0].set_title(f"P = {pressure} kPa")
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

    vle_with_ratio = vle.with_columns(
        (pl.col(x2_name) / (pl.col(x2_name) + pl.col(x3_name))).alias("ratio").round(2)
    )

    temperature_ratio_groups = (
        vle_with_ratio.filter(
            pl.col(x1_name) > 1e-10,
            pl.col(x2_name) > 1e-10,
            pl.col(x3_name) > 1e-10,
        )
        .group_by(["T_K", "ratio"])
        .agg(
            pl.col("P_kPa").min().alias("min_p_kpa"),
            pl.col("P_kPa").max().alias("max_p_kpa"),
            pl.col("P_kPa").count().alias("n"),
        )
        .filter(pl.col("n") > 1)
        .sort(["ratio", "T_K"])
    )
    fig, axs = plt.subplots(
        len(temperature_ratio_groups),
        1,
        figsize=(6, 4 * len(temperature_ratio_groups)),
        squeeze=False,
    )

    for ax, group in zip(axs, temperature_ratio_groups.iter_rows(named=True)):
        t = group["T_K"]
        ratio = group["ratio"]
        exp_x = []
        pred_bubble_points = []
        pressures = []
        for row in (
            vle_with_ratio.filter(
                pl.col(x1_name).is_not_null(),
                pl.col("T_K") == t,
                pl.col("ratio").round(2) == ratio,
                pl.col(x1_name) > 1e-10,
                pl.col(x2_name) > 1e-10,
                pl.col(x3_name) > 1e-10,
            )
            .sort("P_kPa")
            .iter_rows(named=True)
        ):
            x1 = row[x1_name]
            x2 = row[x2_name]
            x3 = row[x3_name]
            pred_bubble_point, _ = mix_vp_feos(
                parameters=params,
                state=[t, np.nan, x1, x2, x3],
                kij_matrix=kij_matrix,
                epsilon_ab=epsilon_ab,
            )
            exp_x.append(row[x1_name])
            pred_bubble_points.append(pred_bubble_point / 1e3)
            pressures.append(row["P_kPa"])
        ax[0].plot(pressures, exp_x, "x", color="black", label="Exp")
        ax[0].plot(pred_bubble_points, exp_x, "o-", color="r", label="Pred")
        ax[0].set_xlabel("Pressure (kPa)")
        ax[0].set_ylabel("Mole Fraction CO2 in Liquid Phase")
        ax[0].set_title(f"T = {t} K, ratio = {ratio:.2f}")
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
