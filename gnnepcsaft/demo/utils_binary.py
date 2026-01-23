"""Module for binary utilities."""

import os.path as osp
from typing import List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import torch

from ..data.graph import from_InChI
from ..pcsaft.pcsaft_feos import mix_den_feos
from ..train.models import GNNePCSAFT
from .utils import _customize_plot, _get_model_params, _line_plot, _scatter_plot

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


def binary_test(
    model: GNNePCSAFT, model_msigmae: Optional[GNNePCSAFT] = None, device="cuda"
) -> List[Tuple[Tuple[str, str], List[Tuple[float, float]]]]:
    """Test model performance on binary data."""
    binary_data = pl.read_parquet(
        osp.join(real_path, "../data/thermoml/raw/binary.parquet")
    )

    inchi_pairs = [
        (row["inchi1"], row["inchi2"])
        for row in binary_data.filter(pl.col("tp") == 1)
        .unique(("inchi1", "inchi2"))
        .to_dicts()
    ]

    with torch.no_grad():
        all_predictions = []
        for inchi1, inchi2 in inchi_pairs:
            mix_params = _get_mixture_params(
                model, model_msigmae, [inchi1, inchi2], device
            )

            rho_data = (
                binary_data.filter(
                    (pl.col("inchi1") == inchi1)
                    & (pl.col("inchi2") == inchi2)
                    & (pl.col("tp") == 1)
                )
                .select("m", "TK", "PPa", "mlc1", "mlc2")
                .to_numpy()
            )

            rho_predictions = []
            for state in rho_data:
                predicted_rho = mix_den_feos(mix_params, state[1:])
                reference_rho = (
                    state[0]
                    * 1000
                    / (mix_params[0][-1] * state[3] + mix_params[1][-1] * state[4])
                ).item()
                rho_predictions.append((predicted_rho, reference_rho))

            all_predictions.append(((inchi1, inchi2), rho_predictions))

    return all_predictions


def _get_mixture_params(
    model: GNNePCSAFT,
    model_msigmae: Optional[GNNePCSAFT],
    inchis: List[str],
    device="cuda",
) -> List[List[float]]:
    """Organize parameters for mixture calculations."""
    mix_params = []
    for inchi in inchis:
        gh = from_InChI(inchi).to(device)
        params = _get_model_params(model, model_msigmae, gh).tolist()
        mix_params.append(params)
    return mix_params
