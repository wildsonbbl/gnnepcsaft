"""Module to optimize binary interaction parameters (kij) for mixtures using VLE data."""

from typing import Callable, Dict, List

import numpy as np
import polars as pl
from joblib import Parallel, delayed
from scipy.optimize import least_squares
from tqdm import tqdm

from .pcsaft_feos import is_stable_feos, mix_tp_flash_feos, pure_vp_feos


# pylint: disable=too-many-arguments, too-many-positional-arguments,too-many-locals
def optimize_kij(
    binary_vle_tml: pl.DataFrame, inchi_to_params: Dict[str, List[float]]
) -> pl.DataFrame:
    """Optimize binary interaction parameters (kij) for mixtures using VLE data."""

    k_12_data = {"inchi1": [], "inchi2": [], "k_12": [], "loss": [], "loss_nonan": []}
    unique_inchis = binary_vle_tml.select("inchi1", "inchi2").unique()
    for row in tqdm(unique_inchis.iter_rows(named=True), total=unique_inchis.shape[0]):
        vle = binary_vle_tml.filter(
            pl.col("mole_fraction_c1p2").is_not_null(),
            pl.col("inchi1") == row["inchi1"],
            pl.col("inchi2") == row["inchi2"],
        )
        if vle.height == 0:
            continue
        params = [inchi_to_params[row["inchi1"]], inchi_to_params[row["inchi2"]]]
        x1 = vle["mole_fraction_c1p2"].to_numpy()
        temperature = vle["T_K"].to_numpy()
        pressure = vle["P_kPa"].to_numpy()
        vp = (
            np.asarray(
                [
                    (
                        pure_vp_feos(
                            parameters=inchi_to_params["InChI=1S/CO2/c2-1-3"], state=[T]
                        )
                        if T < 304.2
                        else 7377.3e3
                    )
                    for T in temperature
                ]
            )
            / 1e3
        )
        check_gas_co2 = pressure / vp < 0.85
        if not np.any(check_gas_co2):
            continue
        x1 = x1[check_gas_co2]
        temperature = temperature[check_gas_co2]
        pressure = pressure[check_gas_co2]

        def _pred_x1(t, p, k_12, params):
            feed_x1s = np.linspace(1e-5, 0.99, 10)
            for feed_x1 in feed_x1s:
                try:
                    if not is_stable_feos(
                        parameters=params,
                        state=[t, p * 1e3, feed_x1, 1 - feed_x1],
                        kij_matrix=[[0.0, k_12], [k_12, 0.0]],
                        epsilon_ab=None,
                        density_initialization=None,
                    ):
                        flash = mix_tp_flash_feos(
                            parameters=params,
                            state=[t, p * 1e3, feed_x1, 1 - feed_x1],
                            kij_matrix=[[0.0, k_12], [k_12, 0.0]],
                            epsilon_ab=None,
                        )
                        return (
                            flash.liquid.molefracs[0]
                            if flash.liquid.density > flash.vapor.density
                            else flash.vapor.molefracs[0]
                        )
                except RuntimeError:
                    continue
            return np.nan

        def _loss_fn(
            k_12: np.ndarray,
            params: List[List[float]],
            x1: np.ndarray,
            temperature: np.ndarray,
            pressure: np.ndarray,
            _pred_x1: Callable[..., float],
        ) -> float:
            pred_x1 = np.asarray(
                Parallel(n_jobs=-1)(
                    delayed(_pred_x1)(T, P, k_12.item(), params)
                    for T, P in zip(temperature, pressure)
                )
            )
            loss = np.log((pred_x1 + 1e-6) / (x1 + 1e-6)) ** 2
            loss[np.isnan(loss)] = 10.0
            return loss.mean()

        def _loss_fn_nonan(
            k_12: float,
            params: List[List[float]],
            x1: np.ndarray,
            temperature: np.ndarray,
            pressure: np.ndarray,
            _pred_x1: Callable[..., float],
        ) -> float:
            pred_x1 = np.asarray(
                Parallel(n_jobs=-1)(
                    delayed(_pred_x1)(T, P, k_12, params)
                    for T, P in zip(temperature, pressure)
                )
            )
            loss = np.log((pred_x1 + 1e-6) / (x1 + 1e-6))
            loss = loss[~np.isnan(loss)]

            if loss.size == 0:
                return 1.0

            return np.abs(loss).mean()

        res = least_squares(
            fun=_loss_fn,
            kwargs={
                "params": params,
                "x1": x1,
                "temperature": temperature,
                "pressure": pressure,
                "_pred_x1": _pred_x1,
            },
            x0=0.20,
            jac="2-point",
            method="lm",
            ftol=1e-8,
            xtol=1e-8,
            # diff_step=1e-1,
        )
        k_12 = res.x.item()
        loss = abs(res.fun.item())

        loss_nonan = _loss_fn_nonan(k_12, params, x1, temperature, pressure, _pred_x1)

        k_12_data["inchi1"].append(row["inchi1"])
        k_12_data["inchi2"].append(row["inchi2"])
        k_12_data["k_12"].append(k_12)
        k_12_data["loss"].append(loss)
        k_12_data["loss_nonan"].append(loss_nonan)

    return pl.DataFrame(k_12_data)
