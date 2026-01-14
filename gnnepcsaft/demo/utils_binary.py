"""Module for binary utilities."""

import os.path as osp
from typing import List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

from ..epcsaft.utils import mix_den_feos
from .utils import _customize_plot, _line_plot, _scatter_plot

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

# Configuration and global settings
sns.set_theme(style="ticks")

# Plot markers and styling
MARKERS = ("o", "v", "s", "<", ">", "*", "^", "p", "P", "D")
MARKERS_2 = ("o", "v", "x", "^", "<", ">", "*", "s", "p", "P", "D")

# Data loading and preprocessing
real_path = osp.dirname(__file__)


def binary_rho_plot(
    rho_data: pl.DataFrame, mw1: float, mw2: float, list_params: List[List[List[float]]]
):
    """Plot binary density"""

    rho_pred = []
    rho_tml = []
    temperatures = []
    _ = plt.figure(
        figsize=(3.3, 3.3),
        dpi=600,
    )

    for i, params in enumerate(list_params):
        rho_pred = []
        rho_tml = []
        temperatures = []
        for state in rho_data.sort("T_K").iter_rows(named=True):
            temperature = state["T_K"]
            pressure = state["P_kPa"]
            mole_fraction_c1 = state["mole_fraction_c1"]
            mole_fraction_c2 = state["mole_fraction_c2"]

            rho_tml.append(
                state["rho"] / (mole_fraction_c1 * mw1 + mole_fraction_c2 * mw2) * 1000
            )
            rho_pred.append(
                mix_den_feos(
                    parameters=params,
                    state=[
                        temperature,
                        pressure * 1000,
                        mole_fraction_c1,
                        mole_fraction_c2,
                    ],
                )
            )
            temperatures.append(temperature)

        _line_plot(temperatures, rho_pred, marker=MARKERS[i])

    _scatter_plot(temperatures, rho_tml)
    _customize_plot("linear", ylabel="Density (mol/m³)")


def mape_rho(
    rho_data: pl.DataFrame, mw1: float, mw2: float, list_params: List[List[List[float]]]
):
    """mean absolute percentage error for binary density"""

    for params in list_params:
        mape = []
        for state in rho_data.sort("T_K").iter_rows(named=True):
            temperature = state["T_K"]
            pressure = state["P_kPa"]
            mole_fraction_c1 = state["mole_fraction_c1"]
            mole_fraction_c2 = state["mole_fraction_c2"]

            rho_tml = (
                state["rho"] / (mole_fraction_c1 * mw1 + mole_fraction_c2 * mw2) * 1000
            )
            rho_pred = mix_den_feos(
                parameters=params,
                state=[
                    temperature,
                    pressure * 1000,
                    mole_fraction_c1,
                    mole_fraction_c2,
                ],
            )
            mape.append(abs((rho_pred - rho_tml) / rho_tml) * 100)
        print(f"MAPE: {np.mean(mape):.2f}%")
