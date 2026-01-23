"""
Module to handle phase equilibria calculations.
"""

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.axes import Axes

from ..data.rdkit_util import smilestoinchi
from .pcsaft_feos import is_stable_feos, mix_tp_flash_feos, pure_vp_feos


# pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
def co2_binary_px(
    inchis: List[str],
    data: pl.DataFrame,
    inchi_to_params: Dict[str, List[float]],
    k_12: Optional[float] = None,
    epsilon_a1b2: Optional[float] = None,
):
    """Plot CO2 solubility in solvent from ThermoML data and GNNPCSAFT predictions.

    Args:
        inchis (List[str]): List of two InChI strings.
        data (pl.DataFrame): Polars DataFrame containing ThermoML data.
        inchi_to_params (Dict[str, List[float]]): Dictionary mapping
         InChI strings to PC-SAFT parameters.
        k_12 (Optional[float]): Binary interaction parameter between CO2 and solvent.
        epsilon_a1b2 (Optional[float]): Association energy parameter between CO2 and solvent.
        feed_x1 (float): Feed mole fraction of CO2 in the liquid phase.
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

    temperatures = vle.select("T_K").sort("T_K").unique("T_K").to_series().to_list()
    if len(temperatures) == 0:
        raise ValueError("No data available for the given InChIs.")
    fig, axs = plt.subplots(len(temperatures), 1, figsize=(6, 4 * len(temperatures)))
    if isinstance(axs, Axes):
        axs = [axs]
    axs: List[Axes]
    feed_x1s = np.linspace(1e-5, 0.99, 10)
    for ax, t in zip(axs, temperatures):
        exp_x = []
        pred_x = []
        pressures = []
        vp = (
            (
                pure_vp_feos(
                    parameters=inchi_to_params["InChI=1S/CO2/c2-1-3"], state=[t]
                )
                / 1e3
            )
            if t < 304.2
            else 7377.3
        )
        for row in (
            vle.filter(
                pl.col("T_K") == t,
            )
            .sort("P_kPa")
            .iter_rows(named=True)
        ):
            pred_x1 = np.nan
            for feed_x1 in feed_x1s:
                try:
                    if not is_stable_feos(
                        parameters=params,
                        state=[t, row["P_kPa"] * 1e3, feed_x1, 1 - feed_x1],
                        kij_matrix=kij_matrix,
                        epsilon_ab=epsilon_ab,
                        density_initialization=None,
                    ):
                        flash = mix_tp_flash_feos(
                            params,
                            [t, row["P_kPa"] * 1e3, feed_x1, 1 - feed_x1],
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
            exp_x.append(row[x1_name])
            pred_x.append(pred_x1)
            pressures.append(row["P_kPa"])
        ax.plot(pressures, exp_x, "x", color="black", label="Exp")
        ax.plot(pressures, pred_x, "o-", color="r", label="Pred")
        ax.axvline(vp, color="gray", linestyle="--", label="CO2 Vapor Pressure")
        ax.set_xlabel("Pressure (kPa)")
        ax.set_ylabel("Mole Fraction CO2 in Liquid Phase")
        ax.set_title(f"T = {t} K")
        ax.legend()
    fig.tight_layout()
    return fig, axs


def co2_ternary_px(
    smiles: List[str],
    data: pl.DataFrame,
    inchi_to_params: Dict[str, List[float]],
    kij_matrix: Optional[List[List[float]]] = None,
    epsilon_ab: Optional[List[List[float]]] = None,
):
    """
    Plot CO2 solubility in solvent mixtures from ThermoML data and GNNPCSAFT predictions.

    Args:
        smiles (List[str]): List of three SMILES strings.
        data (pl.DataFrame): Polars DataFrame containing ThermoML VLE data.
        inchi_to_params (Dict[str, List[float]]): Dictionary mapping
         InChI strings to PC-SAFT parameters.
        kij_matrix (Optional[List[List[float]]]): Binary interaction parameter matrix.
        epsilon_ab (Optional[List[List[float]]]): Association energy parameter matrix.

    """

    params = [inchi_to_params[smilestoinchi(smi)] for smi in smiles]

    vle = data.filter(
        pl.col("inchi1").is_in([smilestoinchi(smi) for smi in smiles]),
        pl.col("inchi2").is_in([smilestoinchi(smi) for smi in smiles]),
        pl.col("inchi3").is_in([smilestoinchi(smi) for smi in smiles]),
    )

    def _get_mole_fraction_names(vle):
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

    x1_name, x2_name, x3_name = _get_mole_fraction_names(vle)

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
    fig, axs = plt.subplots(len(temperatures), 1, figsize=(6, 4 * len(temperatures)))
    if isinstance(axs, Axes):
        axs = [axs]
    axs: List[Axes]
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
            pred_x1 = np.nan
            for feed_x1 in feed_x1s:
                try:
                    if not is_stable_feos(
                        parameters=params,
                        state=[
                            t,
                            row["P_kPa"] * 1e3,
                            feed_x1,
                            row[x2_name],
                            row[x3_name],
                        ],
                        kij_matrix=kij_matrix,
                        epsilon_ab=epsilon_ab,
                        density_initialization=None,
                    ):
                        flash = mix_tp_flash_feos(
                            params,
                            [
                                t,
                                row["P_kPa"] * 1e3,
                                feed_x1,
                                row[x2_name],
                                row[x3_name],
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
            exp_x.append(row[x1_name])
            pred_x.append(pred_x1)
            pressures.append(row["P_kPa"])
        ax.plot(pressures, exp_x, "x", color="black", label="Exp")
        ax.plot(pressures, pred_x, "o-", color="r", label="Pred")
        ax.set_xlabel("Pressure (kPa)")
        ax.set_ylabel("Mole Fraction CO2 in Liquid Phase")
        ax.set_title(f"T = {t} K")
        ax.legend()
        fig.tight_layout()
    return fig, axs
