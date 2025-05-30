"""Module for functions used in model demonstration."""

import itertools
import os
import os.path as osp
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import torch
from rdkit import Chem
from torch.export.dynamic_shapes import Dim

from ..configs.default import get_config
from ..data.graph import Data, assoc_number, from_InChI, from_smiles
from ..data.graphdataset import Esper, Ramirez, ThermoMLDataset
from ..epcsaft.utils import mix_den_feos, parameters_gc_pcsaft
from ..train.models import GNNePCSAFT
from ..train.utils import rhovp_data

sns.set_theme(style="ticks")

markers = itertools.cycle(("o", "v", "^", "<", ">", "*", "s", "p", "P", "D"))

config = get_config()
# pylint: disable = invalid-name
model_dtype = torch.float64
real_path = osp.dirname(__file__)
ra_loader = Ramirez(osp.join(real_path, "../data/ramirez2022"))
ra_para = {}
for graph in ra_loader:
    InChI, para = graph.InChI, graph.para.view(-1, 3).round(decimals=2)
    ra_para[InChI] = para.tolist()[0]
tml_loader = ThermoMLDataset(osp.join(real_path, "../data/thermoml"))
tml_para = {}
for graph in tml_loader:
    tml_para[graph.InChI] = graph

es_loader = Esper(osp.join(real_path, "../data/esper2023"))
es_para = {}
for graph in es_loader:
    es_para[graph.InChI] = (
        graph.para,
        10 ** (graph.assoc * torch.tensor([-1.0, 1.0])),
        graph.munanb,
    )

device = "cpu"


def plotdata(
    inchi: str,
    molecule_name: str,
    models: List[GNNePCSAFT],
    model_msigmae: Optional[GNNePCSAFT],
):
    """Plots ThermoML Archive experimental density and/or vapor pressure
    and compares with predicted values by ePC-SAFT with model estimated
    parameters"""

    from rdkit.Chem import Draw  # pylint: disable=C0415; # type: ignore

    if not osp.exists("images"):
        os.mkdir("images")

    if inchi in tml_para:
        gh = tml_para[inchi]
    else:
        return
    list_params = predparams(inchi, models, model_msigmae)

    rho = gh.rho
    vp = gh.vp

    pred_den_list, pred_vp_list = pred_rhovp(inchi, list_params, rho, vp)

    plotvp(
        inchi,
        molecule_name,
        models,
        (
            vp,
            pred_vp_list,
        ),
    )

    plotden(
        inchi,
        molecule_name,
        models,
        (
            rho,
            pred_den_list,
        ),
    )

    mol = Chem.MolFromInchi(inchi)
    img = Draw.MolToImage(mol, size=(600, 600))
    img_path = osp.join("images", "mol_" + molecule_name + ".png")
    img.save(img_path, dpi=(300, 300), format="png", bitmap_format="png")


def pltline(x, y):
    "Line plot."
    return plt.plot(x, y, marker=next(iter(markers)), linewidth=0.5)


def pltscatter(x, y, marker="x"):
    "Scatter plot."
    return plt.scatter(x, y, marker=marker, s=10, c="black", zorder=10)


def plterr(x, y, m):
    "Add mean absolute percentage error to plot."
    tb = 0
    for i, maperror in enumerate(np.round(m, decimals=1)):
        ta = x[i]
        if (maperror > 1) & (ta - tb > 2):
            tb = ta
            plt.text(x[i], y[i], f"{maperror} %", ha="center", va="center", fontsize=8)


def pltcustom(inchi, scale="linear", ylabel="", n=2):
    """
    Add legend and lables for `plotdata`.
    """
    plt.xlabel("T (K)")
    plt.ylabel(ylabel)
    plt.title("")
    legend = ["ThermoML Archive"]
    for i in range(1, n + 1):
        legend += [f"GNNePCSAFT {i}"]
    if inchi in es_para:
        legend += ["Ref."]
    legend += ["GC PCSAFT"]
    plt.legend(legend, loc=(1.01, 0.75))
    plt.grid(False)
    plt.yscale(scale)
    sns.despine(trim=True)


def predparams(
    inchi: str, models: List[GNNePCSAFT], model_msigmae: Optional[GNNePCSAFT]
):
    "Use models to predict ePC-SAFT parameters from InChI."
    with torch.no_grad():
        gh = from_InChI(inchi)
        graphs = gh.to(device)
        list_params = []
        for model in models:
            model.eval()
            params = get_params(model, model_msigmae, graphs)
            list_params.append(params.tolist())
        if inchi in es_para:
            list_params.append(np.hstack(es_para[inchi]).squeeze().tolist())
        try:
            list_params.append(list(parameters_gc_pcsaft(gh.smiles)))
        # pylint: disable=W0702
        except:
            pass

    return list_params


def get_params(
    model: GNNePCSAFT, model_msigmae: Optional[GNNePCSAFT], graphs: Data
) -> torch.Tensor:
    "to get parameters from models"
    msigmae_or_log10assoc = model.pred_with_bounds(graphs).squeeze().to(torch.float64)
    if graphs.InChI in es_para:
        assoc = es_para[graphs.InChI][1][0]
        munanb = es_para[graphs.InChI][2][0]
    else:
        assoc = torch.zeros(2)
        munanb = torch.tensor((0,) + assoc_number(graphs.InChI), dtype=torch.float64)

    if msigmae_or_log10assoc.size(0) == 2:
        if model_msigmae:
            msigmae = model_msigmae.pred_with_bounds(graphs).squeeze().to(torch.float64)
            return torch.hstack(
                (
                    msigmae,
                    10 ** (msigmae_or_log10assoc * torch.tensor([-1.0, 1.0])),
                    munanb,
                )
            )
        raise ValueError("model_msigmae is None")
    return torch.hstack((msigmae_or_log10assoc, assoc, munanb))


def pred_rhovp(
    inchi: str, list_params: List[List[float]], rho: np.ndarray, vp: np.ndarray
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    "Predicted density and vapor pressure with ePC-SAFT."
    pred_den_list, pred_vp_list = [], []
    print(f"#### {inchi} ####")
    for i, params in enumerate(list_params):
        print(f"#### Parameters for model {i+1} ####")
        print([round(para, 5) for para in params])
        pred_den, pred_vp = rhovp_data(params, rho, vp)
        pred_den_list.append(pred_den)
        pred_vp_list.append(pred_vp)
    return pred_den_list, pred_vp_list


def plotden(
    inchi: str,
    molecule_name: str,
    models: List[GNNePCSAFT],
    data: Tuple[np.ndarray, List[np.ndarray]],
):
    "Plot density data."

    rho, pred_den_list = data
    if ~np.all(rho == np.zeros_like(rho)):
        idx_p = abs(rho[:, 1] - 101325) < 1_000
        rho = rho[idx_p]
        if rho.shape[0] != 0:
            idx = np.argsort(rho[:, 0], 0)
            x = rho[idx, 0]
            y = rho[idx, -1]
            pltscatter(x, y)

            for pred_den in pred_den_list:
                pred_den = pred_den[idx_p]
                y = pred_den[idx]
                pltline(x, y)
                # mden_model = 100 * np.abs(rho[idx, -1] - pred_den[idx]) / rho[idx, -1]
                # plterr(x, y, mden_model)

            # Customize the plot appearance
            pltcustom(inchi, "linear", "Density (mol / m³)", len(models))
            saveplot("den_" + molecule_name + ".png")


def plotvp(
    inchi: str,
    molecule_name: str,
    models: List[GNNePCSAFT],
    data: Tuple[np.ndarray, List[np.ndarray]],
):
    "Plot vapor pressure data."
    vp, pred_vp_list = data
    if ~np.all(vp == np.zeros_like(vp)):
        idx = np.argsort(vp[:, 0], 0)
        x = vp[idx, 0]
        y = vp[idx, -1] / 100000
        pltscatter(x, y)

        for pred_vp in pred_vp_list:
            y = pred_vp[idx] / 100000
            pltline(x, y)
            # mvp_model = 100 * np.abs(vp[idx, -1] - pred_vp[idx]) / vp[idx, -1]
            # plterr(x, y, mvp_model)

        # Customize the plot appearance
        pltcustom(inchi, "log", "Vapor pressure (bar)", len(models))

        # Save the plot as a high-quality image file
        saveplot("vp_" + molecule_name + ".png")


def pltcustom2(scale="linear", xlabel="", ylabel="", n=2):
    """Add legend and lable to `plotparams`"""
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("")
    plt.grid(False)
    plt.yscale(scale)
    legend = []
    for i in range(1, n + 1):
        legend += [f"GNNePCSAFT {i}"]
        legend += [f"Linear fit {i}"]
    plt.legend(legend, loc=(1.01, 0.75))
    sns.despine(trim=True)


def plotparams(smiles: List[str], models: List[GNNePCSAFT], xlabel: str = "CnHn+2"):
    """
    For plotting te behaviour between parameters and chain length.
    """
    if not osp.exists("images"):
        os.mkdir("images")

    list_array_params = predparams2(smiles, models)

    x = np.arange(2, 51)

    for array_params in list_array_params:
        marker = next(iter(markers))
        y = array_params[:, 0]
        pltscatter(x, y, marker)
        plotlinearfit(x, y, marker)

    pltcustom2(xlabel=xlabel, ylabel=r"$m$", n=len(models))

    saveplot("m_" + xlabel + ".png")

    for array_params in list_array_params:
        marker = next(iter(markers))
        y = array_params[:, 0] * array_params[:, 1] ** 3
        pltscatter(x, y, marker)
        plotlinearfit(x, y, marker)

    pltcustom2(xlabel=xlabel, ylabel=r"$m \cdot \sigma³ (Å³)$", n=len(models))

    saveplot("sigma_" + xlabel + ".png")
    for array_params in list_array_params:
        marker = next(iter(markers))
        y = array_params[:, 0] * array_params[:, 2]
        pltscatter(x, y, marker)
        plotlinearfit(x, y, marker)
    pltcustom2(xlabel=xlabel, ylabel=r"$m \cdot \mu k_b^{-1} (K)$", n=len(models))

    saveplot("e_" + xlabel + ".png")


def saveplot(filename: str):
    "savel current plt figure"
    img_path = osp.join("images", filename)
    plt.savefig(img_path, dpi=300, format="png", bbox_inches="tight", transparent=True)
    plt.show()


def plotlinearfit(x, y, marker):
    """Plot linear fit."""
    plt.plot(
        x,
        np.poly1d(np.polyfit(x, y, 1))(x),
        color="red",
        marker=marker,
        linewidth=0.5,
        markersize=3,
    )


def predparams2(smiles: List[str], models: List[GNNePCSAFT]) -> List[np.ndarray]:
    "Use models to predict ePC-SAFT parameters from smiles."
    list_array_params = []
    for model in models:
        model.eval()
        with torch.no_grad():
            list_params = []
            for smile in smiles:
                graphs = from_smiles(smile)
                graphs = graphs.to(device)
                parameters = model.pred_with_bounds(graphs)
                params = parameters.squeeze().to(torch.float64).numpy()
                list_params.append(params)
        array_params = np.asarray(list_params)
        list_array_params.append(array_params)
    return list_array_params


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

    exportedprogram = torch.export.export(
        model,
        example_input,
        dynamic_shapes=dynamic_shapes,
    )

    torch.onnx.export(
        exportedprogram,
        example_input,
        path,
        verbose=True,
        external_data=False,
        optimize=True,
        verify=True,
    )
    return exportedprogram


def binary_test(
    model: GNNePCSAFT, model_msigmae: Optional[GNNePCSAFT] = None
) -> List[Tuple[Tuple[str, str], Tuple[float, float]]]:
    "for testing models performance on binary data"

    binary_data = pl.read_parquet(
        osp.join(real_path, "../data/thermoml/raw/binary.parquet")
    )

    inchi_list = [
        (row["inchi1"], row["inchi2"])
        for row in binary_data.filter(pl.col("tp") == 1)
        .unique(("inchi1", "inchi2"))
        .to_dicts()
    ]

    with torch.no_grad():
        all_predictions = []
        for inchi1, inchi2 in inchi_list:
            mix_params = get_mix_params(model, model_msigmae, [inchi1, inchi2])

            rho_data = (
                binary_data.filter(
                    (pl.col("inchi1") == inchi1)
                    & (pl.col("inchi2") == inchi2)
                    & (pl.col("tp") == 1)
                )
                .select("m", "TK", "PPa", "mlc1", "mlc2")
                .to_numpy()
            )

            all_rho = []
            for state in rho_data:
                rho_for_state = mix_den_feos(mix_params, state[1:])
                ref_rho = (
                    state[0]
                    * 1000
                    / (mix_params[0][-1] * state[3] + mix_params[1][-1] * state[4])
                ).item()
                all_rho.append((rho_for_state, ref_rho))
            all_predictions.append(((inchi1, inchi2), all_rho))
    return all_predictions


def get_mix_params(
    model: GNNePCSAFT, model_msigmae: Optional[GNNePCSAFT], inchis: List[str]
) -> List[List[float]]:
    "to organize the parameters for the mixture"
    mix_params = []
    for inchi in inchis:
        gh = from_InChI(inchi)
        para_for_inchi = get_params(model, model_msigmae, gh).tolist()
        para_for_inchi.append(gh.smiles)
        para_for_inchi.append(gh.InChI)
        para_for_inchi.append(gh.mw.item())
        mix_params.append(para_for_inchi)
    return mix_params
