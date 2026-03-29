"""CO2 binary/ternary VLE helpers for phase equilibria plots."""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from gnnepcsaft.data.rdkit_util import smilestoinchi
from gnnepcsaft.pcsaft.pcsaft_feos import (
    is_stable_feos,
    mix_tp_flash_feos,
    pure_vp_feos,
)

from ._phase_equilibria_style import (
    CO2_CRITICAL_P_KPA,
    CO2_CRITICAL_T_K,
    CO2_INCHI,
    MOLE_FRACTION_SCAN_MAX,
    MOLE_FRACTION_SCAN_MIN,
)

# pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals


def _scan_pred_x1_from_feed(
    params: List[List[float]],
    feed_x1s: np.ndarray,
    state_template: List[float],
    feed_index: int,
    complement_index: Optional[int] = None,
    kij_matrix: Optional[List[List[float]]] = None,
    epsilon_ab: Optional[List[List[float]]] = None,
) -> float:
    """Scan feed compositions and return x1 from first unstable flash solution."""

    for feed_x1 in feed_x1s:
        state = list(state_template)
        state[feed_index] = float(feed_x1)
        if complement_index is not None:
            state[complement_index] = float(1.0 - feed_x1)
        try:
            if not is_stable_feos(
                parameters=params,
                state=state,
                kij_matrix=kij_matrix,
                epsilon_ab=epsilon_ab,
                density_initialization=None,
            ):
                flash = mix_tp_flash_feos(
                    params,
                    state,
                    kij_matrix=kij_matrix,
                    epsilon_ab=epsilon_ab,
                )

                return (
                    flash.liquid.molefracs[0].item()
                    if flash.liquid.density > flash.vapor.density
                    else flash.vapor.molefracs[0].item()
                )
        except RuntimeError:
            continue
    return np.nan


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

    feed_x1s = np.linspace(MOLE_FRACTION_SCAN_MIN, MOLE_FRACTION_SCAN_MAX, n_fractions)
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
                pure_vp_feos(parameters=inchi_to_params[CO2_INCHI], state=[temperature])
                / 1e3
            )
            if temperature < CO2_CRITICAL_T_K
            else CO2_CRITICAL_P_KPA
        )

        for pressure in pressures_kpa:
            pressure_pa = float(pressure * 1e3)
            pred_x1 = _scan_pred_x1_from_feed(
                params=params,
                feed_x1s=feed_x1s,
                state_template=[temperature, pressure_pa, np.nan, np.nan],
                feed_index=2,
                complement_index=3,
                kij_matrix=kij_matrix,
                epsilon_ab=epsilon_ab,
            )
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
    feed_x1s = np.linspace(MOLE_FRACTION_SCAN_MIN, MOLE_FRACTION_SCAN_MAX, 10)
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
    return _scan_pred_x1_from_feed(
        params=params,
        feed_x1s=feed_x1s,
        state_template=[t, p_pa, np.nan, x2, x3],
        feed_index=2,
        kij_matrix=kij_matrix,
        epsilon_ab=epsilon_ab,
    )


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
