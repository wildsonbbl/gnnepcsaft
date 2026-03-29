"""Shared style and constants for phase equilibrium utilities."""

import matplotlib as mpl
import seaborn as sns

LABEL_FS = 11
TICKS_FS = 10
TITLE_FS = 11
CO2_INCHI = "InChI=1S/CO2/c2-1-3"
CO2_CRITICAL_T_K = 304.2
CO2_CRITICAL_P_KPA = 7377.3
DEFAULT_ATM_PRESSURE_PA = 101325.0
MOLE_FRACTION_SCAN_MIN = 1e-5
MOLE_FRACTION_SCAN_MAX = 0.99
MOLE_FRACTION_GRID_MIN = 0.001
MOLE_FRACTION_GRID_MAX_EXCLUSIVE = 1.0
MOLE_FRACTION_GRID_MAX_INCLUSIVE = 0.999

mpl.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": TITLE_FS,
        "axes.labelsize": LABEL_FS,
        "xtick.labelsize": TICKS_FS,
        "ytick.labelsize": TICKS_FS,
    }
)

sns.set_theme(style="ticks")
