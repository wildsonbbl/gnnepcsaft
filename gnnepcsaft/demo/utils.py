"""Module for functions used in model demonstration."""

import itertools
import os
import os.path as osp
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from rdkit import Chem
from torch.export.dynamic_shapes import Dim

from ..configs.default import get_config
from ..data.graph import assoc_number, from_InChI, from_smiles
from ..data.graphdataset import Esper, Ramirez, ThermoMLDataset
from ..epcsaft.utils import parameters_gc_pcsaft
from ..train.models import PNAPCSAFT, PNApcsaftL
from ..train.utils import mape, rhovp_data

sns.set_theme(style="ticks")

markers = itertools.cycle(("o", "v", "^", "<", ">", "*", "s", "p", "P", "D"))

config = get_config()
device = torch.device("cpu")
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

device = torch.device("cpu")


def loadckp(ckp_path: str, model: Union[PNAPCSAFT, PNApcsaftL]):
    """Loads save checkpoint."""
    if osp.exists(ckp_path):
        state = "model_state_dict" if isinstance(model, PNAPCSAFT) else "state_dict"
        checkpoint = torch.load(
            ckp_path, map_location=torch.device("cpu"), weights_only=False
        )
        model.load_state_dict(checkpoint[state])
        del checkpoint


def plotdata(
    inchi: str, molecule_name: str, models: list[PNAPCSAFT], model_msigmae: PNAPCSAFT
):
    """Plots ThermoML Archive experimental density and/or vapor pressure
    and compares with predicted values by ePC-SAFT with model estimated
    parameters"""
    # pylint: disable=C0415
    from rdkit.Chem import Draw

    if not osp.exists("images"):
        os.mkdir("images")

    if inchi in tml_para:
        gh = tml_para[inchi]
    else:
        return
    list_params = predparams(inchi, models, model_msigmae)

    rho = gh.rho.view(-1, 5).to(torch.float64).numpy()
    vp = gh.vp.view(-1, 5).to(torch.float64).numpy()

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


def pltscatter(x, y):
    "Scatter plot."
    return plt.scatter(x, y, marker="x", s=10, c="black", zorder=10)


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


def predparams(inchi: str, models: list[PNAPCSAFT], model_msigmae: PNAPCSAFT):
    "Use models to predict ePC-SAFT parameters from InChI."
    with torch.no_grad():
        gh = from_InChI(inchi)
        graphs = gh.to(device)
        list_params = []
        for model in models:
            model.eval()
            parameters = model.pred_with_bounds(graphs)
            params = parameters.squeeze().to(torch.float64)
            if params.size(0) == 2:
                if inchi in es_para:
                    munanb = es_para[inchi][2]
                else:
                    munanb = torch.tensor(
                        (0,) + assoc_number(inchi), dtype=torch.float32
                    )
                msigmae = (
                    model_msigmae.pred_with_bounds(graphs).squeeze().to(torch.float64)
                )
                params = torch.hstack(
                    (msigmae, 10 ** (params * torch.tensor([-1.0, 1.0])), munanb)
                )
            elif params.size(0) == 3:
                params = torch.hstack((params, torch.zeros(5)))
            list_params.append(params.numpy().round(decimals=4))
        if inchi in es_para:
            list_params.append(np.hstack(es_para[inchi]).round(decimals=4))
        try:
            list_params.append(
                np.asarray(parameters_gc_pcsaft(gh.smiles)).round(decimals=4)
            )
        # pylint: disable=W0702
        except:
            pass

    return list_params


def pred_rhovp(inchi, list_params, rho, vp):
    "Predicted density and vapor pressure with ePC-SAFT."
    pred_den_list, pred_vp_list = [], []
    print(f"#### {inchi} ####")
    for i, params in enumerate(list_params):
        print(f"#### Parameters for model {i+1} ####")
        print(params)
        pred_den, pred_vp = rhovp_data(params, rho, vp)
        pred_den_list.append(pred_den)
        pred_vp_list.append(pred_vp)
    return pred_den_list, pred_vp_list


def plotden(inchi, molecule_name, models, data):
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
            img_path = osp.join("images", "den_" + molecule_name + ".png")
            plt.savefig(
                img_path, dpi=300, format="png", bbox_inches="tight", transparent=True
            )
            plt.show()


def plotvp(inchi, molecule_name, models, data):
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
        img_path = osp.join("images", "vp_" + molecule_name + ".png")
        plt.savefig(
            img_path, dpi=300, format="png", bbox_inches="tight", transparent=True
        )
        plt.show()


def model_para_fn(model: PNAPCSAFT, model_msigmae: PNAPCSAFT):
    """Calculates density and/or vapor pressure mean absolute percentage error
    between ThermoML Archive experimental data and predicted data with ePC-SAFT
    using the model estimated parameters."""
    model_para = {}
    model_array = {}
    model.eval()
    with torch.no_grad():
        for graphs in tml_loader:
            graphs = graphs.to(device)
            parameters = model.pred_with_bounds(graphs)
            params = parameters.squeeze().to(torch.float64)
            if params.size(0) == 2:
                if graphs.InChI in es_para:
                    munanb = es_para[graphs.InChI][2]
                else:
                    munanb = torch.tensor(
                        (0,) + assoc_number(graphs.InChI), dtype=torch.float32
                    )
                msigmae = (
                    model_msigmae.pred_with_bounds(graphs).squeeze().to(torch.float64)
                )
                params = torch.hstack(
                    (msigmae, 10 ** (params * torch.tensor([-1.0, 1.0])), munanb)
                )
            elif params.size(0) == 3:
                params = torch.hstack((params, torch.zeros(5)))
            params = params.numpy()
            rho = graphs.rho.view(-1, 5).to(torch.float64).numpy()
            vp = graphs.vp.view(-1, 5).to(torch.float64).numpy()
            mden_array, mvp_array = mape(params, rho, vp, False)
            mden, mvp = mden_array.mean(), mvp_array.mean()
            parameters = parameters.tolist()[0]
            model_para[graphs.InChI] = (parameters, mden, mvp)
            model_array[graphs.InChI] = (mden_array, mvp_array)
    return model_para, model_array


def datacsv(model_para):
    """Builds a dataset of InChI, density mape, vapor pressure mape."""
    data = {"inchis": [], "mden": [], "mvp": []}
    for inchi in model_para:
        data["inchis"].append(inchi)
        data["mden"].append(model_para[inchi][1])
        data["mvp"].append(model_para[inchi][2])
    return data


def pltcustom2(scale="linear", xlabel="", ylabel="", n=2):
    """Add legend and lable to `plotparams`"""
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("")
    plt.grid(False)
    plt.yscale(scale)
    legend = []
    for i in range(1, n + 1):
        legend += [f"Modelo {i}"]
    plt.legend(legend, loc=(1.01, 0.75))
    sns.despine(trim=True)


def plotparams(smiles: list[str], models: list[PNAPCSAFT], xlabel: str = "CnHn+2"):
    """
    For plotting te behaviour between parameters and chain length.
    """
    if not osp.exists("images"):
        os.mkdir("images")

    list_array_params = predparams2(smiles, models)

    chain_array = [range(1, len(smiles) + 1)]

    for array_params in list_array_params:
        pltscatter(chain_array, array_params[:, 0])

    pltcustom2(xlabel=xlabel, ylabel="m", n=len(models))

    img_path = osp.join("images", "m_" + xlabel + ".png")
    plt.savefig(img_path, dpi=300, format="png", bbox_inches="tight", transparent=True)
    plt.show()

    for array_params in list_array_params:
        pltscatter(chain_array, array_params[:, 0] * array_params[:, 1])

    pltcustom2(xlabel=xlabel, ylabel="m * sigma (Å)", n=len(models))

    img_path = osp.join("images", "sigma_" + xlabel + ".png")
    plt.savefig(img_path, dpi=300, format="png", bbox_inches="tight", transparent=True)
    plt.show()
    for array_params in list_array_params:
        pltscatter(chain_array, array_params[:, 0] * array_params[:, 2])
    pltcustom2(xlabel=xlabel, ylabel="m * e (K)", n=len(models))

    img_path = osp.join("images", "e_" + xlabel + ".png")
    plt.savefig(img_path, dpi=300, format="png", bbox_inches="tight", transparent=True)
    plt.show()


def predparams2(smiles, models):
    "Use models to predict ePC-SAFT parameters from smiles."
    list_array_params = []
    for model in models:
        model.eval()
        with torch.no_grad():
            list_params = []
            for smile in smiles:
                graphs = from_smiles(smile)
                graphs = graphs.to(device)
                parameters = model(graphs)
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
        "x": (Dim.AUTO, 9),
        "edge_index": (2, Dim.AUTO),
        "edge_attr": (Dim.AUTO, 3),
        "batch": None,
    }

    exportedprogram = torch.export.export(
        model,
        example_input,
        dynamic_shapes=dynamic_shapes,
    )

    torch.export.save(exportedprogram, path)
    return exportedprogram
