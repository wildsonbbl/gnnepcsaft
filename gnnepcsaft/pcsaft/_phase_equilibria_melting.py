"""Melting point and eutectic utilities for phase equilibria."""

from typing import Callable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import si_units as si
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.optimize import minimize_scalar, root_scalar

from ._phase_equilibria_style import (
    DEFAULT_ATM_PRESSURE_PA,
    MOLE_FRACTION_GRID_MAX_EXCLUSIVE,
    MOLE_FRACTION_GRID_MAX_INCLUSIVE,
    MOLE_FRACTION_GRID_MIN,
)
from .pcsaft_feos import mix_ln_activity_coefficient

# pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals


def mix_melting_point_ideal(
    mole_fraction_i: np.ndarray, tm_i: np.ndarray, delta_h_sl: np.ndarray
) -> np.ndarray:
    """Calculates mixture melting point using ideal mixing rule and
    pure component melting points and enthalpies of fusion.


    Args:
        mole_fraction_i (np.ndarray): Mole fraction of component i in the mixture.
        tm_i (np.ndarray): Melting point of pure component i in Kelvin
        delta_h_sl (np.ndarray): Enthalpy of fusion of pure component i in kJ/mol.

    Returns:
        out (np.ndarray): Estimated melting temperature in Kelvin.
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

    Returns:
        out (np.ndarray): Residual of the melting point equation for component ``comp_idx``.
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

    Returns:
        out (np.ndarray): Residual vector used to fit ``kij`` against melting temperature data.
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

    Returns:
        out (np.ndarray): Experimental activity coefficients stitched for both components.

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


def _get_eutectic_tm_values(
    tm_getter: Callable[[float], Tuple[float, float]],
    x_i: float,
) -> Tuple[float, float, float]:
    """Return (tm_0, tm_1, tm_0 - tm_1) for a given mole fraction."""

    tm_0, tm_1 = tm_getter(x_i)
    if tm_0 == 0.0 or tm_1 == 0.0:
        return np.nan, np.nan, np.nan
    return tm_0, tm_1, tm_0 - tm_1


def _sample_eutectic_points(
    x_grid: np.ndarray,
    tm_values_getter: Callable[[float], Tuple[float, float, float]],
    temperature_tolerance: float,
) -> Tuple[List[Tuple[float, float, float, float]], Optional[Tuple[float, float]]]:
    """Sample tm differences in x-grid and return points plus optional early solution."""
    sampled_points: List[Tuple[float, float, float, float]] = []
    for x_i in x_grid:
        tm_0, tm_1, delta_tm = tm_values_getter(float(x_i))
        if not np.isfinite(delta_tm):
            continue
        sampled_points.append((float(x_i), tm_0, tm_1, delta_tm))
        if np.abs(delta_tm) <= temperature_tolerance:
            return sampled_points, (float(0.5 * (tm_0 + tm_1)), float(x_i))
    return sampled_points, None


def _find_eutectic_candidates(
    sampled_points: List[Tuple[float, float, float, float]],
    tm_values_getter: Callable[[float], Tuple[float, float, float]],
    temperature_tolerance: float,
) -> List[Tuple[float, float]]:
    """Find candidate eutectic points where tm difference changes sign."""
    candidates: List[Tuple[float, float]] = []
    prev_x, _, _, prev_delta = sampled_points[0]
    for x_i, _, _, delta_tm in sampled_points[1:]:
        if prev_delta * delta_tm < 0.0:
            try:
                root = root_scalar(
                    lambda x: tm_values_getter(float(x))[2],
                    bracket=[prev_x, x_i],
                    method="brentq",
                    xtol=1e-10,
                )
                if root.converged:
                    tm_0, tm_1, delta_root = tm_values_getter(float(root.root))
                    if (
                        np.isfinite(delta_root)
                        and np.abs(delta_root) <= temperature_tolerance
                    ):
                        candidates.append(
                            (float(0.5 * (tm_0 + tm_1)), float(root.root))
                        )
            except (RuntimeError, ValueError):
                pass
        prev_x, prev_delta = x_i, delta_tm
    return candidates


def _find_eutectic_from_minimization(
    tm_values_getter: Callable[[float], Tuple[float, float, float]],
    mole_fraction_step: float,
    temperature_tolerance: float,
) -> Tuple[float, float]:
    """Fallback search for eutectic point by minimizing absolute tm difference."""

    def objective(x: float) -> float:
        delta_tm = tm_values_getter(float(x))[2]
        if not np.isfinite(delta_tm):
            return 1e9
        return float(np.abs(delta_tm))

    minimization = minimize_scalar(
        objective,
        bounds=(MOLE_FRACTION_GRID_MIN, MOLE_FRACTION_GRID_MAX_INCLUSIVE),
        method="bounded",
        options={"xatol": mole_fraction_step * 0.1},
    )
    if not bool(getattr(minimization, "success", False)):
        return np.nan, np.nan

    x_min = float(getattr(minimization, "x", np.nan))
    tm_0, tm_1, delta_tm = tm_values_getter(x_min)
    if np.isfinite(delta_tm) and np.abs(delta_tm) <= temperature_tolerance:
        return float(0.5 * (tm_0 + tm_1)), x_min
    return np.nan, np.nan


def get_eutectic_point(
    parameters: List[List[float]],
    k_ij: float,
    pressure: float,
    exp_tm_i: np.ndarray,
    exp_delta_h_sl: np.ndarray,
    mole_fraction_step: float = 0.001,
    temperature_tolerance: float = 1e-3,
) -> Tuple[float, float]:
    """Calculate eutectic melting point and mole fraction.


    Args:
        parameters (List[List[float]]): PC-SAFT parameters for each component.
        k_ij (float): Interaction parameter between components i and j.
        pressure (float): Pressure in Pascal.
        exp_tm_i (np.ndarray): Melting point of pure component i in Kelvin.
        exp_delta_h_sl (np.ndarray): Enthalpy of fusion of pure component i in kJ/mol.
        mole_fraction_step (float): Step size used to scan for sign changes in delta T.
        temperature_tolerance (float): Maximum allowed |tm_0 - tm_1| in Kelvin.

    Returns:
        out (Tuple[float, float]): Eutectic melting point in Kelvin and mole fraction
    """

    if mole_fraction_step <= 0.0:
        raise ValueError("mole_fraction_step must be > 0.")

    x_grid = np.arange(
        MOLE_FRACTION_GRID_MIN,
        MOLE_FRACTION_GRID_MAX_EXCLUSIVE,
        mole_fraction_step,
        dtype=np.float64,
    )

    def tm_getter(x_i: float) -> Tuple[float, float]:
        return _get_tm(
            parameters,
            k_ij,
            x_i,
            pressure,
            exp_tm_i,
            exp_delta_h_sl,
        )

    def tm_values_getter(x_i: float) -> Tuple[float, float, float]:
        return _get_eutectic_tm_values(tm_getter, x_i)

    sampled_points, early_solution = _sample_eutectic_points(
        x_grid,
        tm_values_getter,
        temperature_tolerance,
    )
    if early_solution is not None:
        return early_solution

    if not sampled_points:
        return np.nan, np.nan

    candidates = _find_eutectic_candidates(
        sampled_points,
        tm_values_getter,
        temperature_tolerance,
    )

    if candidates:
        return min(candidates, key=lambda item: item[0])

    return _find_eutectic_from_minimization(
        tm_values_getter,
        mole_fraction_step,
        temperature_tolerance,
    )


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
    pressure: float = DEFAULT_ATM_PRESSURE_PA,
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

        for x_i in np.arange(
            MOLE_FRACTION_GRID_MIN,
            MOLE_FRACTION_GRID_MAX_EXCLUSIVE,
            mole_fraction_step,
            dtype=np.float64,
        ):
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
                    state=[tm, DEFAULT_ATM_PRESSURE_PA, *mole_fraction_i],
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
