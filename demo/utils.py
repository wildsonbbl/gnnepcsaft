import os
import os.path as osp

# pylint: disable=all
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from configs.default import get_config
from data.graph import from_InChI, from_smiles
from data.graphdataset import ThermoMLDataset, ThermoMLpara, ramirez
from rdkit import Chem
from rdkit.Chem import Draw
from train.model_deg import calc_deg
from train.models import PNAPCSAFT
from train.parametrisation import MAPE, rhovp_data

sns.set_theme(style="ticks")
from itertools import cycle

config = get_config()
device = torch.device("cpu")
model_dtype = torch.float64
deg_model2 = calc_deg("thermoml", "./")
deg_model1 = calc_deg("ramirez", "./")

ra_loader = ramirez("./data/ramirez2022")
ra_para = {}
for graph in ra_loader:
    inchi, para = graph.InChI, graph.para.view(-1, 3).round(decimals=2)
    ra_para[inchi] = para.tolist()[0]
tml_loader = ThermoMLpara("./data/thermoml")
tml_para = {}
for graph in tml_loader:
    inchi, para = graph.InChI, graph.para.view(-1, 3).round(decimals=2)
    tml_para[inchi] = para.tolist()[0]
path = osp.join("data", "thermoml")
testloader = ThermoMLDataset(path)
device = torch.device("cpu")


def loadckp(ckp_path: str, model: PNAPCSAFT):
    if osp.exists(ckp_path):
        checkpoint = torch.load(ckp_path, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"model checkpoint step {checkpoint['step']}")
        del checkpoint


def plotdata(inchi: str, molecule_name: str, models: list[PNAPCSAFT]):
    plotletter = ["A", "B"]

    markers = ["x", "v", "^", "o", "*", "+", "D", "H", "s", "<", ">", "8", "p"]
    marker_cycle = cycle(markers)

    def pltline(x, y, fig=1):
        plt.figure(fig)
        return plt.plot(x, y, marker=next(marker_cycle), linewidth=0.5, ms=2.5)

    def pltscatter(x, y, fig=1):
        plt.figure(fig)
        return plt.scatter(x, y, s=15, marker=next(marker_cycle), c="black")

    def plterr(x, y, m):
        tb = 0
        for i, mape in enumerate(np.round(m, decimals=1)):
            ta = x[i]
            if (mape > 1) & (ta - tb > 2):
                tb = ta
                plt.text(x[i], y[i], f"{mape} %", ha="center", va="center", fontsize=8)

    def pltcustom(ra, scale="linear", ylabel="", n=2, xlabel="T (K)", fig=1):
        plt.figure(fig)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title("")
        legend = ["Pontos experimentais"]
        for i in range(1, n + 1):
            legend += [f"Modelo {i}"]
        if ra:
            legend += [f"Ramírez-Vélez et al. (2022)"]
        plt.legend(legend, loc=(1.01, 0.75))
        plt.grid(False)
        plt.yscale(scale)
        sns.despine(trim=True)
        plt.figtext(1.01, 0.5, plotletter[fig - 1], fontsize=44)

    with torch.no_grad():
        for graphs in testloader:
            if inchi == graphs.InChI:
                break
        graphs.x = graphs.x.to(model_dtype)
        graphs.edge_attr = graphs.edge_attr.to(model_dtype)
        graphs.edge_index = graphs.edge_index.to(torch.int64)
        graphs = graphs.to(device)
        list_params = []
        for model in models:
            model.eval()
            parameters = model(graphs)
            params = parameters.squeeze().to(torch.float64).numpy()
            list_params.append(params)

    rho = graphs.rho.view(-1, 5).to(torch.float64).numpy()
    vp = graphs.vp.view(-1, 5).to(torch.float64).numpy()
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

    if ~np.all(vp == np.zeros_like(vp)):
        idx = np.argsort(vp[:, 0], 0)
        x = vp[idx, 0]
        y = vp[idx, -1] / 100000
        pltscatter(x, y, 1)
        plt.figure(2)
        plt.plot(y, y)

        for pred_vp in pred_vp_list:
            pred_y = pred_vp[idx] / 100000
            pltline(x, pred_y, 1)
            pltscatter(y, pred_y, 2)
            # mvp_model = 100 * np.abs(vp[idx, -1] - pred_vp[idx]) / vp[idx, -1]
            # plterr(x, y, mvp_model)

        if inchi in ra_para:
            ra_vp = ra_vp
            ra_y = ra_vp[idx] / 100000
            pltline(x, ra_y, 1)
            pltscatter(y, ra_y, 2)

            # mvp_ra = 100 * np.abs(vp[idx, -1] - ra_vp[idx]) / vp[idx, -1]
            # plterr(x, y * 1.01, mvp_ra)

        # Customize the plot appearance
        pltcustom(inchi in ra_para, "log", "Pressão de vapor (bar)", len(models))
        pltcustom(
            inchi in ra_para,
            "linear",
            "Pressão de vapor predita (bar)",
            len(models),
            "Pressão de vapor experimental (bar)",
            2,
        )

        # Save the plot as a high-quality image file
        path = osp.join("images", "vp_" + molecule_name + ".png")
        plt.figure(1)
        plt.savefig(path, dpi=300, format="png", bbox_inches="tight", transparent=True)

        path = osp.join("images", "normal_vp_" + molecule_name + ".png")
        plt.figure(2)
        plt.savefig(path, dpi=300, format="png", bbox_inches="tight", transparent=True)
        plt.show()

    if ~np.all(rho == np.zeros_like(rho)):
        idx_p = abs(rho[:, 1] - 101325) < 15_000
        rho = rho[idx_p]
        if rho.shape[0] != 0:
            idx = np.argsort(rho[:, 0], 0)
            x = rho[idx, 0]
            y = rho[idx, -1]
            pltscatter(x, y)
            plt.figure(2)
            plt.plot(y, y)

            for pred_den in pred_den_list:
                pred_den = pred_den[idx_p]
                pred_y = pred_den[idx]
                pltline(x, pred_y)
                pltscatter(y, pred_y, 2)
                # mden_model = 100 * np.abs(rho[idx, -1] - pred_den[idx]) / rho[idx, -1]
                # plterr(x, y, mden_model)

            if inchi in ra_para:
                ra_den = ra_den[idx_p]
                ra_y = ra_den[idx]
                pltline(x, ra_y)
                pltscatter(y, ra_y, 2)
                # mden_ra = 100 * np.abs(rho[idx, -1] - ra_den[idx]) / rho[idx, -1]
                # plterr(x, y, mden_ra)

            # Customize the plot appearance
            pltcustom(inchi in ra_para, "linear", "Densidade (mol / m³)", len(models))
            pltcustom(
                inchi in ra_para,
                "linear",
                "Densidade predita (mol / m³)",
                len(models),
                "Densidade experimental (mol / m³)",
                2,
            )

            path = osp.join("images", "den_" + molecule_name + ".png")
            plt.figure(1)
            plt.savefig(
                path, dpi=300, format="png", bbox_inches="tight", transparent=True
            )

            path = osp.join("images", "normal_den_" + molecule_name + ".png")
            plt.figure(2)
            plt.savefig(
                path, dpi=300, format="png", bbox_inches="tight", transparent=True
            )
            plt.show()
    plt.figure(3)
    mol = Chem.MolFromInchi(inchi)
    img = Draw.MolToImage(mol, size=(600, 600))
    path = osp.join("images", "mol_" + molecule_name + ".png")
    img.save(path, dpi=(300, 300), format="png", bitmap_format="png")


def model_para_fn(model: PNAPCSAFT):
    model_para = {}
    model_array = {}
    model.eval()
    with torch.no_grad():
        for graphs in testloader:
            graphs.x = graphs.x.to(model_dtype)
            graphs.edge_attr = graphs.edge_attr.to(model_dtype)
            graphs.edge_index = graphs.edge_index.to(torch.int64)

            graphs = graphs.to(device)
            parameters = model(graphs)
            params = parameters.squeeze().to(torch.float64).numpy()
            rho = graphs.rho.view(-1, 5).to(torch.float64).numpy()
            vp = graphs.vp.view(-1, 5).to(torch.float64).numpy()
            mden_array, mvp_array = MAPE(params, rho, vp, False)
            mden, mvp = mden_array.mean(), mvp_array.mean()
            parameters = parameters.tolist()[0]
            model_para[graphs.InChI] = (parameters, mden, mvp)
            model_array[graphs.InChI] = (mden_array, mvp_array)
    return model_para, model_array


def datacsv(model_para):
    data = {"inchis": [], "mden": [], "mvp": []}
    for inchi in model_para:
        data["inchis"].append(inchi)
        data["mden"].append(model_para[inchi][1])
        data["mvp"].append(model_para[inchi][2])
    return data


def plotparams(smiles: list[str], models: list[PNAPCSAFT], xlabel: str = "CnHn+2"):
    def pltline(x, y):
        return plt.plot(x, y, linewidth=0.5)

    def pltscatter(x, y):
        return plt.scatter(x, y, marker="x", s=10)

    def pltcustom(scale="linear", ylabel="", n=2):
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title("")
        plt.grid(False)
        plt.yscale(scale)
        legend = []
        for i in range(1, n + 1):
            legend += [f"Modelo {i}"]
        plt.legend(legend, loc=(1.01, 0.75))

    list_array_params = []
    list_chain_array = []
    for model in models:
        model.eval()
        with torch.no_grad():
            list_params = []
            for smile in smiles:
                mol = Chem.MolFromSmiles(smile)
                inchi = Chem.MolToInchi(mol)
                graphs = from_InChI(inchi, sanitize=False)
                graphs.x = graphs.x.to(model_dtype)
                graphs.edge_attr = graphs.edge_attr.to(model_dtype)
                graphs.edge_index = graphs.edge_index.to(torch.int64)
                graphs = graphs.to(device)

                parameters = model(graphs)
                params = parameters.squeeze().to(torch.float64).numpy()
                list_params.append(params)
        array_params = np.asarray(list_params)
        chain_array = np.arange(1, array_params.shape[0] + 1)
        list_array_params.append(array_params)
        list_chain_array.append(chain_array)
    for array_params, chain_array in zip(list_array_params, list_chain_array):
        pltscatter(chain_array, array_params[:, 0])
    pltcustom(ylabel="m", n=len(models))
    path = osp.join("images", "m_" + xlabel + ".png")
    plt.savefig(path, dpi=300, format="png", bbox_inches="tight", transparent=True)
    plt.show()
    for array_params, chain_array in zip(list_array_params, list_chain_array):
        pltscatter(chain_array, array_params[:, 0] * array_params[:, 1])
    pltcustom(ylabel="m * sigma (Å)", n=len(models))
    path = osp.join("images", "sigma_" + xlabel + ".png")
    plt.savefig(path, dpi=300, format="png", bbox_inches="tight", transparent=True)
    plt.show()
    for array_params, chain_array in zip(list_array_params, list_chain_array):
        pltscatter(chain_array, array_params[:, 0] * array_params[:, 2])
    pltcustom(ylabel="m * e (K)", n=len(models))
    path = osp.join("images", "e_" + xlabel + ".png")
    plt.savefig(path, dpi=300, format="png", bbox_inches="tight", transparent=True)
    plt.show()
