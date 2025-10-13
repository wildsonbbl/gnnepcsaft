"""
Module to handle phase equilibria calculations.
"""

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from gnnepcsaft_mcp_server.utils import batch_predict_epcsaft_parameters
from matplotlib.axes import Axes

from ..data.rdkit_util import mw, smilestoinchi
from .epcsaft_feos import mix_tp_flash_feos


# pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
def co2_binary_px(
    smiles: List[str],
    data: pl.DataFrame,
    inchi_to_mu: Dict[str, float],
    k_12: Optional[float] = None,
    epsilon_a1b2: Optional[float] = None,
    feed_x1: float = 0.5,
):
    """Plot CO2 solubility in solvent from ThermoML data and GNNePCSAFT predictions.

    Args:
        smiles (List[str]): List of two SMILES strings.
        data (pl.DataFrame): Polars DataFrame containing ThermoML data.
        inchi_to_mu (Dict[str, float]): Dictionary mapping InChI strings to dipole moments.
        k_12 (Optional[float]): Binary interaction parameter between CO2 and solvent.
        epsilon_a1b2 (Optional[float]): Association energy parameter between CO2 and solvent.
        feed_x1 (float): Feed mole fraction of CO2 in the liquid phase.
    """

    params = _retrieve_pcsaft_params(smiles, inchi_to_mu)

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
        pl.col("inchi1").is_in([smilestoinchi(smi) for smi in smiles]),
        pl.col("inchi2").is_in([smilestoinchi(smi) for smi in smiles]),
        pl.col("P_kPa") < 2000.0,
    )

    x1_name = (
        "mole_fraction_c1p2"
        if vle["inchi1"][0] == smilestoinchi(smiles[0])
        else "mole_fraction_c2p2"
    )

    temperatures = (
        vle.filter(
            pl.col(x1_name).is_not_null(),
        )
        .select("T_K")
        .sort("T_K")
        .unique("T_K")
        .to_series()
        .to_list()
    )
    if len(temperatures) == 0:
        raise ValueError("No data available for the given SMILES.")
    fig, axs = plt.subplots(len(temperatures), 1, figsize=(6, 4 * len(temperatures)))
    if isinstance(axs, Axes):
        axs = [axs]
    axs: List[Axes]
    for ax, t in zip(axs, temperatures):
        exp_x = []
        pred_x = []
        pressures = []
        for row in (
            vle.filter(
                pl.col(x1_name).is_not_null(),
                pl.col("T_K") == t,
            )
            .sort("P_kPa")
            .iter_rows(named=True)
        ):
            try:
                pred_x1 = (
                    mix_tp_flash_feos(
                        params,
                        [t, row["P_kPa"] * 1e3, feed_x1, 1 - feed_x1],
                        kij_matrix=kij_matrix,
                        epsilon_ab=epsilon_ab,
                    )
                    .liquid.molefracs[0]
                    .item()
                )
            except RuntimeError:
                pred_x1 = np.nan
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


def _retrieve_pcsaft_params(
    smiles: List[str], inchi_to_mu: Dict[str, float]
) -> List[List[float]]:
    params = batch_predict_epcsaft_parameters(smiles)
    for i, smi in enumerate(smiles):
        params[i].append(mw(smilestoinchi(smi)))
        if params[i][-1] > 0 and params[i][-2] > 0:
            continue
        params[i][5] = inchi_to_mu.get(smilestoinchi(smi), 0.0)
        params[i][4] /= 2
        print(f"{smi}: mu = {params[i][5]:.4f} D")
    return params


def co2_ternary_px(
    smiles: List[str],
    data: pl.DataFrame,
    inchi_to_mu: Dict[str, float],
    kij_matrix: Optional[List[List[float]]] = None,
    epsilon_ab: Optional[List[List[float]]] = None,
):
    """
    Plot CO2 solubility in solvent mixtures from ThermoML data and GNNePCSAFT predictions.

    Args:
        smiles (List[str]): List of three SMILES strings.
        data (pl.DataFrame): Polars DataFrame containing ThermoML VLE data.
        inchi_to_mu (Dict[str, float]): Dictionary mapping InChI strings to dipole moments.
        kij_matrix (Optional[List[List[float]]]): Binary interaction parameter matrix.
        epsilon_ab (Optional[List[List[float]]]): Association energy parameter matrix.

    """

    params = _retrieve_pcsaft_params(smiles, inchi_to_mu)

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
            pl.col(x1_name).is_not_null(),
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
    for ax, t in zip(axs, temperatures):
        exp_x = []
        pred_x = []
        pressures = []
        for row in (
            vle.filter(
                pl.col(x1_name).is_not_null(),
                pl.col("T_K") == t,
            )
            .sort("P_kPa")
            .iter_rows(named=True)
        ):
            try:
                pred_xi = (
                    mix_tp_flash_feos(
                        params,
                        [
                            t,
                            row["P_kPa"] * 1e3,
                            2.0,
                            row[x2_name],
                            row[x3_name],
                        ],
                        kij_matrix,
                        epsilon_ab,
                    )
                    .liquid.molefracs[0]
                    .item()
                )
            except RuntimeError:
                print(row[x3_name], row[x2_name], t, row["P_kPa"])
                pred_xi = np.nan
            exp_x.append(row[x1_name])
            pred_x.append(pred_xi)
            pressures.append(row["P_kPa"])
        ax.plot(pressures, exp_x, "x", color="black", label="Exp")
        ax.plot(pressures, pred_x, "o-", color="r", label="Pred")
        ax.set_xlabel("Pressure (kPa)")
        ax.set_ylabel("Mole Fraction CO2 in Liquid Phase")
        ax.set_title(f"T = {t} K")
        ax.legend()
        fig.tight_layout()
    return fig, axs
