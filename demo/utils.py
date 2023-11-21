"""Module for functions used in model demonstration."""
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import torch
from rdkit import Chem

from ..configs.default import get_config
from ..data.graph import from_InChI, from_smiles
from ..data.graphdataset import Ramirez, ThermoMLDataset
from ..train.models import PNAPCSAFT
from ..train.utils import mape, rhovp_data

os.environ["CUDA_VISIBLE_DEVICES"] = ""

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

testloader = ThermoMLDataset(osp.join(real_path, "../data/thermoml"))
device = torch.device("cpu")


def loadckp(ckp_path: str, model: PNAPCSAFT):
    """Loads save checkpoint."""
    if osp.exists(ckp_path):
        checkpoint = torch.load(ckp_path, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"model checkpoint step {checkpoint['step']}")
        del checkpoint


def plotdata(inchi: str, molecule_name: str, models: list[PNAPCSAFT]):
    """Plots ThermoML Archive experimental density and/or vapor pressure
    and compares with predicted values by ePC-SAFT with model estimated
    parameters"""
    # pylint: disable=C0415
    from rdkit.Chem import Draw

    if inchi in tml_para:
        gh = tml_para[inchi]
    else:
        return
    list_params = predparams(inchi, models)

    rho = gh.rho.view(-1, 5).to(torch.float64).numpy()
    vp = gh.vp.view(-1, 5).to(torch.float64).numpy()

    pred_den_list, pred_vp_list, ra_den, ra_vp = pred_rhovp(inchi, list_params, rho, vp)

    plotvp(
        inchi,
        molecule_name,
        models,
        (
            vp,
            pred_vp_list,
            ra_vp,
        ),
    )

    plotden(
        inchi,
        molecule_name,
        models,
        (
            rho,
            pred_den_list,
            ra_den,
        ),
    )

    mol = Chem.MolFromInchi(inchi)
    img = Draw.MolToImage(mol, size=(600, 600))
    img.show()
    img_path = osp.join("images", "mol_" + molecule_name + ".png")
    img.save(img_path, dpi=(300, 300), format="png", bitmap_format="png")


def pltline(x, y):
    "Line plot."
    return plt.plot(x, y, linewidth=0.5)


def pltscatter(x, y):
    "Scatter plot."
    return plt.scatter(x, y, marker="x", c="black", s=10)


def plterr(x, y, m):
    "Add mean absolute percentage error to plot."
    tb = 0
    for i, maperror in enumerate(np.round(m, decimals=1)):
        ta = x[i]
        if (maperror > 1) & (ta - tb > 2):
            tb = ta
            plt.text(x[i], y[i], f"{maperror} %", ha="center", va="center", fontsize=8)


def pltcustom(ra, scale="linear", ylabel="", n=2):
    """
    Add legend and lables for `plotdata`.
    """
    plt.xlabel("T (K)")
    plt.ylabel(ylabel)
    plt.title("")
    legend = ["Pontos experimentais"]
    for i in range(1, n + 1):
        legend += [f"Modelo {i}"]
    if ra:
        legend += ["Ramírez-Vélez et al. (2022)"]
    plt.legend(legend, loc=(1.01, 0.75))
    plt.grid(False)
    plt.yscale(scale)


def predparams(inchi, models):
    "Use models to predict ePC-SAFT parameters from InChI."
    with torch.no_grad():
        gh = from_InChI(inchi)
        graphs = gh.to(device)
        list_params = []
        for model in models:
            model.eval()
            parameters = model(graphs)
            params = parameters.squeeze().to(torch.float64).numpy()
            list_params.append(params)
    return list_params


def pred_rhovp(inchi, list_params, rho, vp):
    "Predicted density and vapor pressure with ePC-SAFT."
    pred_den_list, pred_vp_list = [], []
    for i, params in enumerate(list_params):
        pred_den, pred_vp = rhovp_data(params, rho, vp)
        pred_den_list.append(pred_den)
        pred_vp_list.append(pred_vp)
        print(f"#### Parameters for model {i + 1} ####")
        print(params)
    if inchi in ra_para:
        params = np.asarray(ra_para[inchi])
        ra_den, ra_vp = rhovp_data(params, rho, vp)
    else:
        ra_den, ra_vp = [], []
    return pred_den_list, pred_vp_list, ra_den, ra_vp


def plotden(inchi, molecule_name, models, data):
    "Plot density data."

    rho, pred_den_list, ra_den = data
    if ~np.all(rho == np.zeros_like(rho)):
        idx_p = abs(rho[:, 1] - 101325) < 15_000
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

            if inchi in ra_para:
                ra_den = ra_den[idx_p]
                y = ra_den[idx]
                pltline(x, y)
                # mden_ra = 100 * np.abs(rho[idx, -1] - ra_den[idx]) / rho[idx, -1]
                # plterr(x, y, mden_ra)

            # Customize the plot appearance
            pltcustom(inchi in ra_para, "linear", "Densidade (mol / m³)", len(models))
            img_path = osp.join("images", "den_" + molecule_name + ".png")
            plt.savefig(
                img_path, dpi=300, format="png", bbox_inches="tight", transparent=True
            )
            plt.show()


def plotvp(inchi, molecule_name, models, data):
    "Plot vapor pressure data."
    vp, pred_vp_list, ra_vp = data
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

        if inchi in ra_para:
            y = ra_vp[idx] / 100000
            pltline(x, y)
            # mvp_ra = 100 * np.abs(vp[idx, -1] - ra_vp[idx]) / vp[idx, -1]
            # plterr(x, y * 1.01, mvp_ra)

        # Customize the plot appearance
        pltcustom(inchi in ra_para, "log", "Pressão de vapor (bar)", len(models))

        # Save the plot as a high-quality image file
        img_path = osp.join("images", "vp_" + molecule_name + ".png")
        plt.savefig(
            img_path, dpi=300, format="png", bbox_inches="tight", transparent=True
        )
        plt.show()


def model_para_fn(model: PNAPCSAFT):
    """Calculates density and/or vapor pressure mean absolute percentage error
    between ThermoML Archive experimental data and predicted data with ePC-SAFT
    using the model estimated parameters."""
    model_para = {}
    model_array = {}
    model.eval()
    with torch.no_grad():
        for graphs in testloader:
            graphs = graphs.to(device)
            parameters = model(graphs)
            params = parameters.squeeze().to(torch.float64).numpy()
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


def plotparams(smiles: list[str], models: list[PNAPCSAFT], xlabel: str = "CnHn+2"):
    """
    For plotting te behaviour between parameters and chain length.
    """

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
