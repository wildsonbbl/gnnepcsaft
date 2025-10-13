"""Module to optimize binary interaction parameters (kij) for mixtures using VLE data."""

from typing import Callable, Dict, List

import numpy as np
import polars as pl
from joblib import Parallel, delayed
from scipy.optimize import least_squares
from tqdm import tqdm

from .epcsaft_feos import mix_tp_flash_feos


def optimize_kij(
    binary_vle_tml: pl.DataFrame, inchi_to_params: Dict[str, List[float]]
) -> pl.DataFrame:
    """Optimize binary interaction parameters (kij) for mixtures using VLE data."""

    k_12_data = {"inchi1": [], "inchi2": [], "k_12": [], "loss": []}
    unique_inchis = binary_vle_tml.select("inchi1", "inchi2").unique()
    for row in tqdm(unique_inchis.iter_rows(named=True), total=unique_inchis.shape[0]):
        vle = binary_vle_tml.filter(
            pl.col("mole_fraction_c1p2").is_not_null(),
            pl.col("P_kPa") < 2000.0,
            pl.col("inchi1") == row["inchi1"],
            pl.col("inchi2") == row["inchi2"],
        )
        if vle.height == 0:
            continue
        params = [inchi_to_params[row["inchi1"]], inchi_to_params[row["inchi2"]]]
        x1 = vle["mole_fraction_c1p2"].to_numpy()
        temperature = vle["T_K"].to_numpy()
        pressure = vle["P_kPa"].to_numpy()

        def _pred_x1(t, p, k_12, params, feed_x1):
            if abs(k_12) > 1:
                return 5.0 * feed_x1
            try:
                return mix_tp_flash_feos(
                    parameters=params,
                    state=[t, p * 1e3, 0.5, 0.5],
                    kij_matrix=[[0.0, k_12], [k_12, 0.0]],
                    epsilon_ab=None,
                ).liquid.molefracs[0]
            except RuntimeError:
                return np.nan

        # pylint: disable=too-many-arguments, too-many-positional-arguments
        def _loss_fn(
            k_12: float,
            params: List[List[float]],
            x1: np.ndarray,
            temperature: np.ndarray,
            pressure: np.ndarray,
            _pred_x1: Callable[..., float],
        ) -> float:
            pred_x1 = np.asarray(
                Parallel(n_jobs=-1)(
                    delayed(_pred_x1)(T, P, k_12, params, feed_x1)
                    for T, P, feed_x1 in zip(temperature, pressure, x1)
                )
            )
            loss = np.log((pred_x1 + 1e-6) / (x1 + 1e-6))
            loss[np.isnan(loss)] = 1.0
            return loss.mean()

        res = least_squares(
            fun=_loss_fn,
            kwargs={
                "params": params,
                "x1": x1,
                "temperature": temperature,
                "pressure": pressure,
                "_pred_x1": _pred_x1,
            },
            x0=0.05,
            jac="3-point",
            method="lm",
            ftol=1e-8,
            xtol=1e-8,
            # diff_step=1e-1,
        )
        k_12 = res.x.item()
        loss = abs(res.fun.item())

        if loss >= 0.01:
            res = least_squares(
                fun=_loss_fn,
                kwargs={
                    "params": params,
                    "x1": x1,
                    "temperature": temperature,
                    "pressure": pressure,
                    "_pred_x1": _pred_x1,
                },
                x0=0.11,
                jac="3-point",
                method="lm",
                ftol=1e-8,
                xtol=1e-8,
                # diff_step=1e-1,
            )
            if np.abs(res.fun) < loss:
                k_12 = res.x.item()
                loss = abs(res.fun.item())

        if loss >= 0.01:
            res = least_squares(
                fun=_loss_fn,
                kwargs={
                    "params": params,
                    "x1": x1,
                    "temperature": temperature,
                    "pressure": pressure,
                    "_pred_x1": _pred_x1,
                },
                x0=0.22,
                jac="3-point",
                method="lm",
                ftol=1e-8,
                xtol=1e-8,
                # diff_step=1e-1,
            )
            if np.abs(res.fun) < loss:
                k_12 = res.x.item()
                loss = abs(res.fun.item())

        k_12_data["inchi1"].append(row["inchi1"])
        k_12_data["inchi2"].append(row["inchi2"])
        k_12_data["k_12"].append(k_12)
        k_12_data["loss"].append(loss)

    return pl.DataFrame(k_12_data)
