"""Compatibility facade for phase equilibria utilities.

This module keeps the original import path stable while implementation
is split into smaller logical modules.
"""

from ._phase_equilibria_co2 import (  # noqa: F401
    _get_mole_fraction_names,
    co2_binary_px,
    co2_binary_tx,
    co2_ternary_px,
    get_kij_matrix_ternary,
)
from ._phase_equilibria_melting import (  # noqa: F401
    _find_eutectic_candidates,
    _find_eutectic_from_minimization,
    _get_eutectic_tm_values,
    _get_tm,
    _sample_eutectic_points,
    fit_kij_with_gamma,
    fit_kij_with_tm,
    fit_melting_point,
    gamma_from_exp_data,
    get_eutectic_point,
    mape_tm,
    mix_melting_point_ideal,
    plot_tm,
)
from ._phase_equilibria_style import (  # noqa: F401
    CO2_CRITICAL_P_KPA,
    CO2_CRITICAL_T_K,
    CO2_INCHI,
    DEFAULT_ATM_PRESSURE_PA,
    LABEL_FS,
    MOLE_FRACTION_GRID_MAX_EXCLUSIVE,
    MOLE_FRACTION_GRID_MAX_INCLUSIVE,
    MOLE_FRACTION_GRID_MIN,
    MOLE_FRACTION_SCAN_MAX,
    MOLE_FRACTION_SCAN_MIN,
    TICKS_FS,
    TITLE_FS,
)

__all__ = [
    "LABEL_FS",
    "TICKS_FS",
    "TITLE_FS",
    "CO2_INCHI",
    "CO2_CRITICAL_T_K",
    "CO2_CRITICAL_P_KPA",
    "DEFAULT_ATM_PRESSURE_PA",
    "MOLE_FRACTION_SCAN_MIN",
    "MOLE_FRACTION_SCAN_MAX",
    "MOLE_FRACTION_GRID_MIN",
    "MOLE_FRACTION_GRID_MAX_EXCLUSIVE",
    "MOLE_FRACTION_GRID_MAX_INCLUSIVE",
    "co2_binary_px",
    "co2_binary_tx",
    "co2_ternary_px",
    "_get_mole_fraction_names",
    "get_kij_matrix_ternary",
    "mix_melting_point_ideal",
    "fit_melting_point",
    "fit_kij_with_tm",
    "gamma_from_exp_data",
    "fit_kij_with_gamma",
    "mape_tm",
    "_get_eutectic_tm_values",
    "_sample_eutectic_points",
    "_find_eutectic_candidates",
    "_find_eutectic_from_minimization",
    "get_eutectic_point",
    "_get_tm",
    "plot_tm",
]
