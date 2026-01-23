"""Module for functions used in model demonstration."""

import os
import os.path as osp
from typing import List, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import plotly.graph_objects as go
import seaborn as sns
import torch
import xgboost as xgb
from joblib import Parallel, delayed
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from rdkit import Chem
from sklearn.ensemble import RandomForestRegressor
from torch.export.dynamic_shapes import Dim

from ..configs.default import get_config
from ..data.graph import Data, assoc_number, from_InChI, from_smiles
from ..data.graphdataset import Esper, Ramirez, ThermoMLDataset
from ..data.rdkit_util import smilestoinchi
from ..pcsaft.pcsaft_feos import (
    mix_gibbs_energy,
    mix_lle_diagram_feos,
    mix_lle_feos,
    mix_vle_diagram_feos,
)
from ..train.models import GNNePCSAFT, HabitchNN
from ..train.utils import rhovp_data

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
config = get_config()

# Plot markers and styling
MARKERS = ("o", "v", "s", "<", ">", "*", "^", "p", "P", "D")
MARKERS_2 = ("o", "v", "x", "^", "<", ">", "*", "s", "p", "P", "D")

# Data loading and preprocessing
real_path = osp.dirname(__file__)


def _load_ramirez_data() -> dict:
    """Load Ramirez data and return parameters dictionary."""
    ra_loader = Ramirez(osp.join(real_path, "../data/ramirez2022"))
    _ra_para = {}
    for graph in ra_loader:
        inchi, para = graph.InChI, graph.para.view(-1, 3).round(decimals=2)
        _ra_para[inchi] = para.tolist()[0]
    return _ra_para


def _load_thermoml_data() -> dict:
    """Load ThermoML data and return parameters dictionary."""
    tml_loader = ThermoMLDataset(osp.join(real_path, "../data/thermoml"))
    _tml_para = {}
    for graph in tml_loader:
        _tml_para[graph.InChI] = graph
    return _tml_para


def _load_esper_data() -> dict:
    """Load Esper data and return parameters dictionary."""
    es_loader = Esper(osp.join(real_path, "../data/esper2023"))
    _es_para = {}
    for graph in es_loader:
        _es_para[graph.InChI] = (
            graph.para,
            10 ** (graph.assoc * torch.tensor([-1.0, 1.0])),
            graph.munanb,
        )
    return _es_para


# Global data dictionaries
ra_para = _load_ramirez_data()
tml_para = _load_thermoml_data()
es_para = _load_esper_data()


# Main plotting functions
def plotdata(
    inchi: str,
    molecule_name: str,
    list_params: List[List[float]],
) -> None:
    """Plot ThermoML Archive experimental data and compare with model predictions."""

    _ensure_images_directory()

    if inchi not in tml_para:
        return

    gh = tml_para[inchi]

    pred_den_list, pred_vp_list = _predict_rho_vp(inchi, list_params, gh.rho, gh.vp)

    _plot_vapor_pressure(molecule_name, gh.vp, pred_vp_list)
    _plot_density(molecule_name, gh.rho, pred_den_list)
    _save_molecule_image(inchi, molecule_name)


def plotparams(
    smiles: List[List[str]],
    models: List[Union[GNNePCSAFT, HabitchNN]],
    list_xlabel: List[str],
    device: str = "cuda",
) -> Tuple[Figure, np.ndarray]:
    """Plot parameter behavior vs chain length."""
    _ensure_images_directory()

    list_array_params = [
        _predict_params_from_smiles(list_smiles, models, device=device)
        for list_smiles in smiles
    ]
    x = np.arange(2, len(smiles[0]) + 2)

    fig, axs = plt.subplots(len(smiles), 3, figsize=(4.68, 5.0 * len(smiles) / 3))
    if axs.ndim == 1:
        axs = np.array([axs])
    for i in range(len(smiles)):
        plt.sca(axs[i, 0])
        _plot_parameter_m(x, list_array_params[i], "")

        plt.sca(axs[i, 1])
        _plot_parameter_sigma(x, list_array_params[i], list_xlabel[i])

        plt.sca(axs[i, 2])
        _plot_parameter_epsilon(x, list_array_params[i], "")
    fig.tight_layout()
    _save_plot("parameters.png")

    return fig, axs


def plot_binary_gibbs_energy(
    params: List[List[float]],
    state: List[float],
    k_12: Optional[float] = None,
    epsilon_a1b2: Optional[float] = None,
) -> Figure:
    """
    Plot the binary Gibbs energy for a given set of PCSAFT parameters and state.

    Args:
        params: List of PCSAFT parameters
         `[m, sigma, epsilon/kB, kappa_ab, epsilon_ab/kB, dipole moment, na, nb]`
         for the two components.
        state: List containing `[T_min (K), T_max (K), P (Pa)]` for the plot.
        k_12: Binary interaction parameter.
        epsilon_a1b2: Cross-association energy parameter (in K).

    """
    x = np.linspace(1e-5, 0.999, 100)
    t = np.linspace(state[0], state[1], 20).round(2)
    p = state[2]

    def gibbs_at(tloc, xi):
        yi = 1 - xi
        return mix_gibbs_energy(
            params,
            [tloc, p, xi, yi],
            kij_matrix=(
                [
                    [0.0, k_12],
                    [k_12, 0.0],
                ]
                if k_12 is not None
                else None
            ),
            epsilon_ab=(
                [
                    [0.0, epsilon_a1b2],
                    [epsilon_a1b2, 0.0],
                ]
                if epsilon_a1b2 is not None
                else None
            ),
        )

    fig = plt.figure(figsize=(8, 6))
    for tloc in t:
        g = np.asarray(
            Parallel(n_jobs=-1, backend="loky")(delayed(gibbs_at)(tloc, xi) for xi in x)
        )
        plt.plot(x, g, "-")

    plt.xlim(0, 1)
    plt.xticks(np.arange(0, 1.04, 0.04), minor=True)
    plt.grid(which="minor", axis="both", color="gray", linestyle="--", linewidth=0.5)
    plt.grid(which="major", axis="both", color="black", linestyle="--", linewidth=1.0)
    plt.legend(t, title="T (K)", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xlabel("x1")
    plt.ylabel(r"$g_{mix}$")
    return fig


def plot_binary_lle_phase_diagram(
    params: List[List[float]],
    state: List[float],
    k_12: Optional[float] = None,
    epsilon_a1b2: Optional[float] = None,
) -> Tuple[Figure, List[Axes]]:
    """
    Plot the binary LLE phase diagram for a given set of PCSAFT parameters and state.


    Args:
        params: List of PCSAFT parameters
         `[m, sigma, epsilon/kB, kappa_ab, epsilon_ab/kB, dipole moment, na, nb]`
         for the two components.
        state:
         List containing initial state
         `[T (K), P (Pa), mole_fractions_1, mole_fractions_2]` for the plot.
        k_12: Binary interaction parameter.
        epsilon_a1b2: Cross-association energy parameter (in K).


    """

    dia_t = mix_lle_diagram_feos(
        params,
        state,
        kij_matrix=(
            [
                [0.0, k_12],
                [k_12, 0.0],
            ]
            if k_12 is not None
            else None
        ),
        epsilon_ab=(
            [
                [0.0, epsilon_a1b2],
                [epsilon_a1b2, 0.0],
            ]
            if epsilon_a1b2 is not None
            else None
        ),
    )

    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    axs: List[Axes]

    # --- Subplot 1: T vs x / y ---
    ax_t = axs[0]
    ax_t.plot(dia_t["x0"], dia_t["temperature"], color="b", label="Phase 1")
    ax_t.plot(dia_t["y0"], dia_t["temperature"], color="r", label="Phase 2")
    ax_t.set_xlabel(r"$x_1$")
    ax_t.set_ylabel("T (K)")
    ax_t.set_title("T–x–x")

    t_min, t_max = min(dia_t["temperature"]), max(dia_t["temperature"])
    ax_t.set_yticks(np.arange(t_min, t_max + 10, 10, dtype=np.int64))
    ax_t.set_yticks(np.arange(t_min, t_max + 10, 2, dtype=np.int64), minor=True)
    ax_t.legend(loc="best")

    # --- Subplot 2: y vs x ---
    ax_rho = axs[1]
    ax_rho.plot(dia_t["x0"], dia_t["y0"], color="b")
    ax_rho.set_xlabel(r"$x^{\alpha}_1$")
    ax_rho.set_ylabel(r"$x^{\beta}_1$")
    ax_rho.set_title(r"$x^{\alpha}$ vs $x^{\beta}$")

    for ax in axs:
        ax.set_xlim(0, 1)
        ax.grid(which="major", color="black", linestyle="--", alpha=1.0)
        ax.grid(which="minor", color="gray", linestyle="--", alpha=0.5)
        ax.set_xticks(np.linspace(0, 1, 11))

    fig.tight_layout()
    return fig, axs


def plot_binary_vle_phase_diagram(
    params: List[List[float]],
    state: List[float],
    k_12: Optional[float] = None,
    epsilon_a1b2: Optional[float] = None,
) -> Tuple[Figure, List[Axes]]:
    """Plot binary VLE diagrams T–x–y and y-x.

    Creates two side-by-side subplots:
      1. Temperature vs composition
      2. Vapor vs Liquid composition

    Args:
        params: List of PCSAFT parameters for the two components
            ``[m, sigma, epsilon/kB, kappa_ab, epsilon_ab/kB, dipole moment, na, nb, mw]``.
        state: state pressure `[P (Pa)]`.
        k_12: Binary interaction parameter (``k_ij``) between the two components.
        epsilon_a1b2: Cross-association energy parameter (in K).


    Returns:
        fig: Matplotlib ``Figure`` instance.
        axs: Numpy array of two ``Axes`` objects (index 0: T vs x/y, index 1: y vs x).
    """

    dia_t = mix_vle_diagram_feos(
        params,
        state,
        kij_matrix=(
            [
                [0, k_12],
                [k_12, 0],
            ]
            if k_12 is not None
            else None
        ),
        epsilon_ab=(
            [
                [0.0, epsilon_a1b2],
                [epsilon_a1b2, 0.0],
            ]
            if epsilon_a1b2 is not None
            else None
        ),
    )

    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    axs: List[Axes]

    # --- Subplot 1: T vs x / y ---
    ax_t = axs[0]
    ax_t.plot(dia_t["x0"], dia_t["temperature"], color="b", label="L (x)")
    ax_t.plot(dia_t["y0"], dia_t["temperature"], color="r", label="V (y)")
    ax_t.set_xlabel(r"$x_1$")
    ax_t.set_ylabel("T (K)")
    ax_t.set_title("T–x–y")

    t_min, t_max = min(dia_t["temperature"]), max(dia_t["temperature"])
    ax_t.set_yticks(np.arange(t_min, t_max + 10, 10, dtype=np.int64))
    ax_t.set_yticks(np.arange(t_min, t_max + 10, 2, dtype=np.int64), minor=True)
    ax_t.legend(loc="best")

    # --- Subplot 2: y vs x ---
    ax_rho = axs[1]
    ax_rho.plot(dia_t["x0"], dia_t["y0"], color="b")
    ax_rho.set_xlabel(r"$x_1$")
    ax_rho.set_ylabel(r"$y_1$")
    ax_rho.set_title("y vs x")

    # Ajustes gerais
    for ax in axs:
        ax.set_xlim(0, 1)
        ax.grid(which="major", color="black", linestyle="--", alpha=1.0)
        ax.grid(which="minor", color="gray", linestyle="--", alpha=0.5)
        ax.set_xticks(np.linspace(0, 1, 11))

    fig.tight_layout()
    return fig, axs


def plot_ternary_gibbs_surface(
    params: List[List[float]],
    state: List[float],
    kij_matrix: Optional[List[List[float]]] = None,
    epsilon_ab: Optional[List[List[float]]] = None,
) -> go.Figure:
    """
    Plot the ternary Gibbs energy surface for a given set of PCSAFT parameters and state.

    Args:
        params: List of PCSAFT parameters
         `[m, sigma, epsilon/kB, kappa_ab, epsilon_ab/kB, dipole moment, na, nb]`
         for the three components.
        state: List containing `[T (K), P (Pa)]` for the plot.
        kij_matrix: 3x3 matrix of binary interaction parameters.
        epsilon_ab: 3x3 matrix of cross-association energy parameters (in K).


    """

    # Malha de composições
    xi = np.linspace(1e-5, 0.999, 200, dtype=np.float64)
    x1, x2 = np.meshgrid(xi, xi, indexing="xy")
    x3 = 1.0 - x1 - x2

    # Máscara do simplesio (x3 >= 0)
    mask = x3 >= 0.0

    # Matriz de Gibbs com NaN fora do simplesio
    z = np.full_like(x1, np.nan)

    # Calcula somente nos pontos válidos
    valid_idx = np.argwhere(mask)
    for i, j in valid_idx:
        z[i, j] = mix_gibbs_energy(
            params,
            [state[0], state[1], x1[i, j].item(), x2[i, j].item(), x3[i, j].item()],
            kij_matrix=kij_matrix,
            epsilon_ab=epsilon_ab,
        )

    # Plot da superfície
    fig = go.Figure(
        data=[go.Surface(z=z, x=xi, y=xi, colorbar_title="g<sub>mix</sub>")]
    )
    fig.update_layout(
        width=800,
        height=800,
        scene={
            "xaxis_title": "x<sub>1</sub>",
            "yaxis_title": "x<sub>2</sub>",
            "zaxis": {"title": "g<sub>mix</sub>"},
        },
    )
    return fig


def plot_ternary_lle_diagram(
    params: List[List[float]],
    state: List[float],
    kij_matrix: Optional[List[List[float]]] = None,
    epsilon_ab: Optional[List[List[float]]] = None,
) -> go.Figure:
    """
    Plot the ternary LLE diagram for a given set of PCSAFT parameters and state.

    Args:
        params: List of PCSAFT parameters
         `[m, sigma, epsilon/kB, kappa_ab, epsilon_ab/kB, dipole moment, na, nb]`
         for the three components.
        state: List containing `[T (K), P (Pa)]` for the plot.
        kij_matrix: 3x3 matrix of binary interaction parameters.
        epsilon_ab: 3x3 matrix of cross-association energy parameters (in K).


    """

    def _grid(n_pts: int = 50):
        xi = np.linspace(1e-5, 0.999, n_pts, dtype=np.float64)
        x1_m, x2_m = np.meshgrid(xi, xi, indexing="xy")
        x3_m = 1.0 - x1_m - x2_m
        return x1_m, x2_m, x3_m, (x3_m >= 0.0)

    def _collect_tie_lines(x1_m, x2_m, x3_m, mask):
        valid_idx = np.argwhere(mask)
        lines = []
        for i, j in valid_idx:
            try:
                lle = mix_lle_feos(
                    params,
                    [
                        state[0],
                        state[1],
                        x1_m[i, j].item(),
                        x2_m[i, j].item(),
                        x3_m[i, j].item(),
                    ],
                    kij_matrix=kij_matrix,
                    epsilon_ab=epsilon_ab,
                )
            except (RuntimeError, ValueError):
                continue
            # For LLE, y is one phase and x is the other phase
            lines.append(
                [
                    lle["y0"] + lle["y1"] + lle["y2"],
                    lle["x0"] + lle["x1"] + lle["x2"],
                ]
            )
        return np.asarray(lines)

    x1, x2, x3, mask = _grid()
    tl = _collect_tie_lines(x1, x2, x3, mask)

    if tl.size == 0:
        # Nada convergiu; evita IndexError e informa usuário.
        fig = go.Figure()
        fig.update_layout(
            title=f"Nenhuma fase encontrada em T={state[0]} K, P={state[1]/1e5} bar"
        )

        return fig

    fig = go.Figure()
    # Fase 1
    fig.add_trace(
        go.Scatterternary(
            a=tl[:, 0, 0],
            b=tl[:, 0, 1],
            c=tl[:, 0, 2],
            mode="markers",
            marker={"symbol": "circle", "size": 5, "color": "blue"},
            name="Phase 1",
        )
    )
    # Fase 2
    fig.add_trace(
        go.Scatterternary(
            a=tl[:, 1, 0],
            b=tl[:, 1, 1],
            c=tl[:, 1, 2],
            mode="markers",
            marker={"symbol": "circle", "size": 5, "color": "red"},
            name="Phase 2",
        )
    )

    fig.update_layout(
        ternary={
            "sum": 1,
            "aaxis": {
                "title": "A",
                "min": 0.0,
                "linewidth": 2,
                "ticks": "outside",
            },
            "baxis": {
                "title": "B",
                "min": 0.0,
                "linewidth": 2,
                "ticks": "outside",
            },
            "caxis": {
                "title": "C",
                "min": 0.0,
                "linewidth": 2,
                "ticks": "outside",
            },
        },
        title=f"LLE Diagram at T={state[0]} K and P={state[1]/1e5} bar",
        width=800,
        height=800,
    )
    return fig


# Parameter prediction functions
def predict_params_from_inchi(
    inchi: str,
    model_assoc: Union[GNNePCSAFT, HabitchNN],
    model_msigmae: Union[GNNePCSAFT, HabitchNN],
    device: str = "cuda",
) -> List[float]:
    """Predict PCSAFT parameters from InChI."""
    with torch.no_grad():
        gh = from_InChI(inchi)
        graph = gh.to(device)

        params = _get_model_params(model_assoc, model_msigmae, graph)

    return params.tolist()


def _predict_params_from_smiles(
    smiles: List[str], models: List[Union[GNNePCSAFT, HabitchNN]], device="cuda"
) -> List[np.ndarray]:
    """Make a n models x m SMILES x 3 list of parameters for homologous series plot.
    Add Esper et al. (2023) reference parameters as last model if available.
    """
    list_array_params = []

    for model in models:
        model.eval()
        model_params = _predict_params_for_single_model(model, smiles, device=device)
        list_array_params.append(model_params)

    esper_params = _get_esper_reference_params(smiles)
    list_array_params.append(esper_params)

    return list_array_params


def _predict_params_for_single_model(
    model: Union[GNNePCSAFT, HabitchNN], smiles: List[str], device="cpu"
) -> np.ndarray:
    """Make a m SMILES x 3 array of parameters."""
    list_params = []

    with torch.no_grad():
        for smile in smiles:
            graphs = from_smiles(smile).to(device)
            parameters = model.pred_with_bounds(graphs)
            params = parameters.squeeze().to(torch.float64).cpu().numpy()
            list_params.append(params)

    return np.asarray(list_params)


def _get_esper_reference_params(smiles: List[str]) -> np.ndarray:
    """Get Esper et al. (2023) reference parameters."""
    list_params = []

    for smile in smiles:
        inchi = smilestoinchi(smile)
        if inchi in es_para:
            list_params.append(es_para[inchi][0].squeeze())
        else:
            list_params.append(np.zeros(3) * np.nan)

    return np.asarray(list_params)


def _extract_features(graphs: Data) -> np.ndarray:
    """Extract features from graph data for ML models."""
    return (
        torch.hstack(
            (
                graphs.ecfp,
                graphs.mw,
                graphs.atom_count,
                graphs.ring_count,
                graphs.rbond_count,
            )
        )
        .cpu()
        .numpy()
    )


def _predict_with_model(
    model: Union[GNNePCSAFT, HabitchNN, RandomForestRegressor, xgb.Booster],
    graphs: Data,
) -> np.ndarray:
    """Predict parameters using a single model."""
    if isinstance(model, (GNNePCSAFT, HabitchNN)):
        with torch.no_grad():
            return (
                model.pred_with_bounds(graphs).squeeze().to(torch.float64).cpu().numpy()
            )

    features = _extract_features(graphs)

    if isinstance(model, RandomForestRegressor):
        return model.predict(features).squeeze()
    if isinstance(model, xgb.Booster):
        x_matrix = xgb.DMatrix(features)
        return model.predict(x_matrix).squeeze()
    raise TypeError(f"Model type {type(model)} not supported.")


def _get_association_params(graphs: Data) -> Tuple[np.ndarray, np.ndarray]:
    """Get association parameters for the molecule."""
    if graphs.InChI in es_para:
        assoc = es_para[graphs.InChI][1][0].numpy()
        munanb = es_para[graphs.InChI][2][0].numpy()
    else:
        assoc = np.zeros(2)
        munanb = np.array((0,) + assoc_number(graphs.InChI), dtype=np.float64)

    return assoc, munanb


def _get_model_params(
    model: Union[GNNePCSAFT, HabitchNN, RandomForestRegressor, xgb.Booster],
    model_msigmae: Optional[
        Union[GNNePCSAFT, HabitchNN, RandomForestRegressor, xgb.Booster]
    ],
    graphs: Data,
) -> np.ndarray:
    """Extract parameters from models."""
    msigmae_or_log10assoc = _predict_with_model(model, graphs)
    assoc, munanb = _get_association_params(graphs)

    # Handle associating model case
    if msigmae_or_log10assoc.shape[0] == 2:
        if model_msigmae is None:
            raise ValueError("model_msigmae is required when using associating model.")

        msigmae = _predict_with_model(model_msigmae, graphs)

        return np.hstack(
            (
                msigmae,
                10 ** (msigmae_or_log10assoc * np.array([-1.0, 1.0])),
                munanb,
                graphs.mw[0].cpu().numpy(),
            )
        )

    return np.hstack((msigmae_or_log10assoc, assoc, munanb, graphs.mw[0].cpu().numpy()))


# Prediction and calculation functions
def _predict_rho_vp(
    inchi: str, list_params: List[List[float]], rho: np.ndarray, vp: np.ndarray
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Predict density and vapor pressure with PCSAFT."""
    pred_den_list, pred_vp_list = [], []
    print(f"#### {inchi} ####")

    for i, params in enumerate(list_params):
        print(f"#### Parameters for model {i+1} ####")
        print([round(para, 5) for para in params])
        pred_den, pred_vp = rhovp_data(params, rho, vp)
        pred_den_list.append(pred_den)
        pred_vp_list.append(pred_vp)

    return pred_den_list, pred_vp_list


# Plotting helper functions
def _plot_density(
    molecule_name: str,
    rho: np.ndarray,
    pred_den_list: List[np.ndarray],
) -> None:
    """Plot density data."""
    if np.all(rho == np.zeros_like(rho)):
        return

    idx_p = abs(rho[:, 1] - 101325) < 1_000
    rho_filtered = rho[idx_p]

    if rho_filtered.shape[0] == 0:
        return
    _ = plt.figure(
        figsize=(3.3, 3.3),
        dpi=600,
    )

    idx = np.argsort(rho_filtered[:, 0], 0)
    x = rho_filtered[idx, 0]
    y = rho_filtered[idx, -1]

    _scatter_plot(x, y)

    for i, pred_den in enumerate(pred_den_list):
        pred_den_filtered = pred_den[idx_p]
        y_pred = pred_den_filtered[idx]
        _line_plot(x, y_pred, MARKERS[i])

    _customize_plot("linear", "Density (mol / m³)")
    _save_plot(f"den_{molecule_name}.png")


def _plot_vapor_pressure(
    molecule_name: str,
    vp: np.ndarray,
    pred_vp_list: List[np.ndarray],
) -> None:
    """Plot vapor pressure data."""
    if np.all(vp == np.zeros_like(vp)):
        return
    _ = plt.figure(
        figsize=(3.3, 3.3),
        dpi=600,
    )

    idx = np.argsort(vp[:, 0], 0)
    x = vp[idx, 0]
    y = vp[idx, -1] / 100000

    _scatter_plot(x, y)

    for i, pred_vp in enumerate(pred_vp_list):
        y_pred = pred_vp[idx] / 100000
        _line_plot(x, y_pred, MARKERS[i])

    _customize_plot("log", "Vapor pressure (bar)")
    _save_plot(f"vp_{molecule_name}.png")


def _plot_parameter_m(
    x: np.ndarray, list_array_params: List[np.ndarray], xlabel: str
) -> None:
    """Plot parameter m vs chain length."""
    for i, array_params in enumerate(list_array_params):
        y = array_params[:, 0]
        _scatter_plot(x, y, MARKERS_2[i], None)

    _customize_plot_params(
        xlabel=xlabel,
        ylabel=r"$m$",
    )


def _plot_parameter_sigma(
    x: np.ndarray, list_array_params: List[np.ndarray], xlabel: str
) -> None:
    """Plot parameter sigma vs chain length."""
    for i, array_params in enumerate(list_array_params):
        y = array_params[:, 0] * array_params[:, 1] ** 3
        _scatter_plot(x, y, MARKERS_2[i], None)

    _customize_plot_params(xlabel=xlabel, ylabel=r"$m \cdot \sigma³ (Å³)$")


def _plot_parameter_epsilon(
    x: np.ndarray, list_array_params: List[np.ndarray], xlabel: str
) -> None:
    """Plot parameter epsilon vs chain length."""
    for i, array_params in enumerate(list_array_params):
        y = array_params[:, 0] * array_params[:, 2]
        _scatter_plot(x, y, MARKERS_2[i], None)

    _customize_plot_params(
        xlabel=xlabel,
        ylabel=r"$m \cdot \mu k_b^{-1} (K)$",
    )


# Basic plotting utilities
def _line_plot(
    x: Union[np.ndarray, List[float]],
    y: Union[np.ndarray, List[float]],
    marker: str = "x",
) -> None:
    """Create line plot."""
    plt.plot(x, y, marker=marker, linewidth=0.5, markersize=3)


def _scatter_plot(
    x: Union[np.ndarray, List[float]],
    y: Union[np.ndarray, List[float]],
    marker: str = "x",
    color: Optional[str] = "black",
) -> None:
    """Create scatter plot."""
    plt.scatter(x, y, marker=marker, c=color, zorder=10, s=9)


def plot_linear_fit(x: np.ndarray, y: np.ndarray, marker: str) -> None:
    """Plot linear fit."""
    plt.plot(
        x,
        np.poly1d(np.polyfit(x, y, 1, full=False))(x),
        color="red",
        marker=marker,
        linewidth=0.5,
        markersize=3,
    )


def _customize_plot(scale: str = "linear", ylabel: str = "") -> None:
    """Customize plot appearance for main plots."""
    plt.xlabel("T (K)")
    plt.ylabel(ylabel)
    plt.title("")
    plt.grid(False)
    plt.yscale(scale)
    sns.despine(trim=True)


def _customize_plot_params(
    scale: str = "linear", xlabel: str = "", ylabel: str = ""
) -> None:
    """Customize plot appearance for parameter plots."""
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("")
    plt.grid(False)
    plt.yscale(scale)
    sns.despine(trim=False)


# Utility functions
def _ensure_images_directory() -> None:
    """Ensure images directory exists."""
    if not osp.exists("images"):
        os.mkdir("images")


def _save_plot(filename: str) -> None:
    """Save current plot figure."""
    img_path = osp.join("images", filename)
    plt.savefig(img_path, dpi=600, format="png", bbox_inches="tight", transparent=False)


def _save_molecule_image(inchi: str, molecule_name: str) -> None:
    """Save molecule structure as image."""
    from rdkit.Chem import Draw  # pylint: disable=C0415; # type: ignore

    mol = Chem.MolFromInchi(inchi)
    img = Draw.MolToImage(mol, size=(600, 600))
    img_path = osp.join("images", f"mol_{molecule_name}.png")
    img.save(img_path, dpi=(300, 300), format="png", bitmap_format="png")


def save_exported_program(
    model: torch.nn.Module, example_input: tuple, path: str
) -> torch.export.ExportedProgram:
    """Save model as Exported Program."""
    model.eval()
    dynamic_shapes = {
        "x": (Dim.AUTO, 9),  # type: ignore
        "edge_index": (2, Dim.AUTO),  # type: ignore
        "edge_attr": (Dim.AUTO, 3),  # type: ignore
        "batch": None,
    }

    exported_program = torch.export.export(
        model, example_input, dynamic_shapes=dynamic_shapes
    )

    torch.onnx.export(
        exported_program,
        example_input,
        path,
        verbose=True,
        external_data=False,
        optimize=True,
        verify=True,
    )
    return exported_program


def test_onnx(
    loader: ThermoMLDataset,
    msigmae_path: str,
    assoc_path: str,
    pna_msigmae: GNNePCSAFT,
    pna_assoc: GNNePCSAFT,
):
    """test saved onnx"""

    msigmae_onnx = ort.InferenceSession(msigmae_path)
    assoc_onnx = ort.InferenceSession(assoc_path)
    pna_assoc.eval()
    pna_msigmae.eval()

    with torch.no_grad():
        for graph in loader:
            x, edge_index, edge_attr = (
                graph["x"],
                graph["edge_index"],
                graph["edge_attr"],
            )

            onnx_assoc = assoc_onnx.run(
                None,
                {
                    "x": x.cpu().numpy(),
                    "edge_index": edge_index.cpu().numpy(),
                    "edge_attr": edge_attr.cpu().numpy(),
                    "batch": None,
                },
            )

            assoc = pna_assoc(x, edge_index, edge_attr, None).cpu().numpy()

            if not np.allclose(onnx_assoc, assoc):  # type: ignore
                print(
                    f"missmatch onnx: {onnx_assoc}, pna: {assoc}, InChI: {graph.InChI}"
                )

            onnx_msigmae = msigmae_onnx.run(
                None,
                {
                    "x": x.cpu().numpy(),
                    "edge_index": edge_index.cpu().numpy(),
                    "edge_attr": edge_attr.cpu().numpy(),
                    "batch": None,
                },
            )

            msigmae = pna_msigmae(x, edge_index, edge_attr, None).cpu().numpy()

            if not np.allclose(onnx_msigmae, msigmae):  # type: ignore
                print(
                    f"missmatch onnx: {onnx_msigmae}, pna: {msigmae}, InChI: {graph.InChI}"
                )
