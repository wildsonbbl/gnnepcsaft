"""Module to optimize binary interaction parameters (kij) for mixtures using VLE data."""

from typing import Dict, List

import numpy as np
import polars as pl
from joblib import Parallel, delayed
from scipy.optimize import least_squares
from tqdm import tqdm

from gnnepcsaft.pcsaft.pcsaft_feos import (
    is_stable_feos,
    mix_tp_flash_feos,
    mix_vp_feos,
    pure_vp_feos,
)

# pylint: disable=too-many-arguments, too-many-positional-arguments,too-many-locals

EPS = 1e-6


def _pred_x1_worker(
    t: float, p: float, k_12: float, params: List[List[float]], feed_x1s: np.ndarray
) -> float:
    """Predict liquid-phase mole fraction of component 1 for one state point.

    Must be at module level for pickle compatibility on Windows.

    Args:
        t (float): Temperature in Kelvin.
        p (float): Pressure in kPa.
        k_12 (float): Binary interaction parameter.
        params (List[List[float]]): PC-SAFT parameters for the binary mixture.
        feed_x1s (np.ndarray): Candidate feed mole fractions for component 1.

    Returns:
        out (float): Predicted mole fraction of component 1 in the denser phase.
            Returns np.nan when no converged flash result is found.
    """
    for feed_x1 in feed_x1s:
        try:
            # Check stability.
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
                # Return the composition of the denser phase (usually liquid)
                if flash.liquid.density > flash.vapor.density:
                    return float(flash.liquid.molefracs[0])
                return float(flash.vapor.molefracs[0])
        except RuntimeError:
            continue
    return np.nan


def _loss_fn(
    k_12_arr: np.ndarray,
    params: List[List[float]],
    x1: np.ndarray,
    temperature: np.ndarray,
    pressure: np.ndarray,
    parallel: Parallel,
    feed_x1s: np.ndarray,
) -> np.ndarray:
    """Compute residual vector used in least-squares optimization.

    Args:
        k_12_arr (np.ndarray): Array containing a single optimization variable (k_12).
        params (List[List[float]]): PC-SAFT parameters for the binary mixture.
        x1 (np.ndarray): Experimental mole fractions of component 1.
        temperature (np.ndarray): Temperatures in Kelvin.
        pressure (np.ndarray): Pressures in kPa.
        parallel (Parallel): Active joblib parallel pool.
        feed_x1s (np.ndarray): Candidate feed mole fractions for flash calculations.

    Returns:
        out (np.ndarray): Residual vector defined as log(predicted/experimental).
            Failed flash evaluations are penalized with a large residual.
    """
    k_12 = k_12_arr[0]

    # Run predictions in parallel using the active pool
    pred_x1 = np.asarray(
        parallel(
            delayed(_pred_x1_worker)(T, P, k_12, params, feed_x1s)
            for T, P in zip(temperature, pressure)
        )
    )

    # Calculate residuals: log(pred) - log(exp) = log(pred/exp)
    residuals = np.log((pred_x1 + EPS) / (x1 + EPS))

    # Handle NaNs (failed flash) by assigning a large penalty
    nan_mask = np.isnan(residuals)
    residuals[nan_mask] = 10.0

    return residuals


def _pred_bp_worker(
    t: float, x1: float, k_12: float, params: List[List[float]]
) -> float:
    """Predict bubble point pressure for one state point.

    Must be at module level for pickle compatibility on Windows.

    Args:
        t (float): Temperature in Kelvin.
        x1 (float): Mole fraction of component 1.
        k_12 (float): Binary interaction parameter.
        params (List[List[float]]): PC-SAFT parameters for the binary mixture.

    Returns:
        out (float): Predicted bubble point pressure in kPa.
            Returns np.nan when bubble point calculation fails.
    """
    try:
        bp_pa, _ = mix_vp_feos(
            parameters=params,
            state=[t, 0.0, x1, 1 - x1],
            kij_matrix=[[0.0, k_12], [k_12, 0.0]],
            epsilon_ab=None,
        )
        return float(bp_pa / 1e3)  # Convert Pa to kPa
    except RuntimeError:
        return np.nan
    except BaseException as exc:  # pylint: disable=W0718
        exception_type = type(exc).__name__
        if exception_type == "PanicException":
            return np.nan
        raise


def _loss_fn_bubble_point(
    k_12_arr: np.ndarray,
    params: List[List[float]],
    x1: np.ndarray,
    temperature: np.ndarray,
    pressure: np.ndarray,
    parallel: Parallel,
) -> np.ndarray:
    """Compute residual vector for bubble point optimization.

    Args:
        k_12_arr (np.ndarray): Array containing a single optimization variable (k_12).
        params (List[List[float]]): PC-SAFT parameters for the binary mixture.
        x1 (np.ndarray): Mole fractions of component 1.
        temperature (np.ndarray): Temperatures in Kelvin.
        pressure (np.ndarray): Pressures in kPa.
        parallel (Parallel): Active joblib parallel pool.

    Returns:
        out (np.ndarray): Residual vector defined as log(predicted_P/experimental_P).
            Failed calculations are penalized with a large residual.
    """
    k_12 = k_12_arr[0]

    # Run predictions in parallel using the active pool
    pred_p = np.asarray(
        parallel(
            delayed(_pred_bp_worker)(T, X1, k_12, params)
            for T, X1 in zip(temperature, x1)
        )
    )

    # Calculate residuals: log(pred_P/exp_P)
    residuals = np.log((pred_p + EPS) / (pressure + EPS))

    # Handle NaNs (failed calculation) by assigning a large penalty
    nan_mask = np.isnan(residuals)
    residuals[nan_mask] = 10.0

    return residuals


def optimize_kij(
    binary_vle_tml: pl.DataFrame, inchi_to_params: Dict[str, List[float]], n: int = 50
) -> pl.DataFrame:
    """Optimize binary interaction parameters (kij) for mixtures using VLE data.

    Args:
        binary_vle_tml (pl.DataFrame): VLE dataset with columns including
            inchi1, inchi2, T_K, P_kPa, and mole_fraction_c1p2.
        inchi_to_params (Dict[str, List[float]]): Mapping from InChI to PC-SAFT parameters.
        n (int): Number of candidate feed mole fractions used in flash scans.

    Returns:
        out (pl.DataFrame): DataFrame with optimized k_12 and error metrics per binary pair.
            Columns: inchi1, inchi2, k_12, loss, loss_nonan, mape, n_nan.
    """

    k_12_data = {
        "inchi1": [],
        "inchi2": [],
        "k_12": [],
        "loss": [],
        "loss_nonan": [],
        "mape": [],
        "n_nan": [],
    }
    unique_inchis = binary_vle_tml.select("inchi1", "inchi2").unique()
    feed_x1s = np.linspace(1e-5, 0.99, n)

    # Create Parallel pool once and reuse it
    with Parallel(n_jobs=-1) as parallel:
        for row in tqdm(
            unique_inchis.iter_rows(named=True), total=unique_inchis.shape[0]
        ):
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

            # CO2 check
            vp = (
                np.asarray(
                    [
                        (
                            pure_vp_feos(
                                parameters=inchi_to_params["InChI=1S/CO2/c2-1-3"],
                                state=[T],
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

            try:
                # Optimize
                res = least_squares(
                    fun=_loss_fn,
                    x0=[0.20],
                    kwargs={
                        "params": params,
                        "x1": x1,
                        "temperature": temperature,
                        "pressure": pressure,
                        "parallel": parallel,
                        "feed_x1s": feed_x1s,
                    },
                    jac="2-point",
                    method="lm",
                    ftol=1e-8,
                    xtol=1e-8,
                )
                k_12 = res.x[0]

                # MSE Loss = 2 * Cost / N (since Cost = 0.5 * sum(residuals^2))
                loss = 2 * res.cost / len(res.fun)

                # Calculate MAE without NaNs
                pred_x1 = np.asarray(
                    parallel(
                        delayed(_pred_x1_worker)(T, P, k_12, params, feed_x1s)
                        for T, P in zip(temperature, pressure)
                    )
                )
                loss_vec = np.log((pred_x1 + EPS) / (x1 + EPS))
                n_nan = np.isnan(loss_vec).sum()
                loss_vec = loss_vec[~np.isnan(loss_vec)]
                loss_nonan = np.abs(loss_vec).mean() if loss_vec.size > 0 else 1.0
                mape_vec = (pred_x1 - x1) / (x1 + EPS)
                mape_vec = mape_vec[~np.isnan(mape_vec)]
                mape_nonan = np.abs(mape_vec).mean() if mape_vec.size > 0 else 1.0

                k_12_data["inchi1"].append(row["inchi1"])
                k_12_data["inchi2"].append(row["inchi2"])
                k_12_data["k_12"].append(k_12)
                k_12_data["loss"].append(loss)
                k_12_data["loss_nonan"].append(loss_nonan)
                k_12_data["mape"].append(mape_nonan)
                k_12_data["n_nan"].append(n_nan)

            except RuntimeError as e:
                print(e)
                continue

    return pl.DataFrame(k_12_data)


def optimize_kij_bubble_point(
    binary_vle_tml: pl.DataFrame, inchi_to_params: Dict[str, List[float]]
) -> pl.DataFrame:
    """Optimize binary interaction parameters (kij) for mixtures using bubble point data.

    This function uses fixed mole fractions and variable pressure calculations.
    Particularly useful when mole fractions are accurately known and you want to
    fit to bubble point pressure data.

    Args:
        binary_vle_tml (pl.DataFrame): VLE dataset with columns including
            inchi1, inchi2, T_K, P_kPa, and mole_fraction_c1p2.
        inchi_to_params (Dict[str, List[float]]): Mapping from InChI to PC-SAFT parameters.

    Returns:
        out (pl.DataFrame): DataFrame with optimized k_12 and error metrics per binary pair.
            Columns: inchi1, inchi2, k_12, loss, loss_nonan, mape, n_nan.
    """

    k_12_data = {
        "inchi1": [],
        "inchi2": [],
        "k_12": [],
        "loss": [],
        "loss_nonan": [],
        "mape": [],
        "n_nan": [],
    }
    unique_inchis = binary_vle_tml.select("inchi1", "inchi2").unique()

    # Create Parallel pool once and reuse it
    with Parallel(n_jobs=-1) as parallel:
        for row in tqdm(
            unique_inchis.iter_rows(named=True), total=unique_inchis.shape[0]
        ):
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

            try:
                # Optimize
                res = least_squares(
                    fun=_loss_fn_bubble_point,
                    x0=[0.20],
                    kwargs={
                        "params": params,
                        "x1": x1,
                        "temperature": temperature,
                        "pressure": pressure,
                        "parallel": parallel,
                    },
                    jac="2-point",
                    method="lm",
                    ftol=1e-8,
                    xtol=1e-8,
                )
                k_12 = res.x[0]

                # MSE Loss = 2 * Cost / N (since Cost = 0.5 * sum(residuals^2))
                loss = 2 * res.cost / len(res.fun)

                # Calculate error metrics without NaNs
                pred_p = np.asarray(
                    parallel(
                        delayed(_pred_bp_worker)(T, X1, k_12, params)
                        for T, X1 in zip(temperature, x1)
                    )
                )
                loss_vec = np.log((pred_p + EPS) / (pressure + EPS))
                n_nan = np.isnan(loss_vec).sum()
                loss_vec = loss_vec[~np.isnan(loss_vec)]
                loss_nonan = np.abs(loss_vec).mean() if loss_vec.size > 0 else 1.0
                mape_vec = (pred_p - pressure) / (pressure + EPS)
                mape_vec = mape_vec[~np.isnan(mape_vec)]
                mape_nonan = np.abs(mape_vec).mean() if mape_vec.size > 0 else 1.0

                k_12_data["inchi1"].append(row["inchi1"])
                k_12_data["inchi2"].append(row["inchi2"])
                k_12_data["k_12"].append(k_12)
                k_12_data["loss"].append(loss)
                k_12_data["loss_nonan"].append(loss_nonan)
                k_12_data["mape"].append(mape_nonan)
                k_12_data["n_nan"].append(n_nan)

            except RuntimeError as e:
                print(e)
                continue

    return pl.DataFrame(k_12_data)
