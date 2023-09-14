
import os.path as osp, os
os.environ["CUDA_VISIBLE_DEVICES"] = ''
import torch, numpy as np
from data.graphdataset import ThermoMLDataset
from train.models import PNAPCSAFT
from train.model_deg import calc_deg
from train.parametrisation import rhovp_data
import matplotlib.pyplot as plt
from configs.default import get_config
config = get_config()
device = torch.device("cpu")
model_dtype = torch.float64
deg_model2 = calc_deg("thermoml", './')
deg_model1 = calc_deg("ramirez", './')


def loadckp(ckp_path: str, model: PNAPCSAFT):
    if osp.exists(ckp_path):
        checkpoint = torch.load(ckp_path, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"model checkpoint step {checkpoint['step']}")
        del checkpoint

def plotdata(inchi: str, model_name: str, molecule_name: str, model: PNAPCSAFT, testloader: ThermoMLDataset, ra_para: dict):
    def pltline(x, y):
        return plt.plot(x, y, linewidth=0.5)

    def pltscatter(x, y):
        return plt.scatter(x, y, marker="x", c="black", s=10)

    def plterr(x, y, m):
        tb = 0
        for i, mape in enumerate(np.round(m, decimals=1)):
            ta = x[i]
            if (mape > 1) & (ta - tb > 2):
                tb = ta
                plt.text(x[i], y[i], f"{mape} %", ha="center", va="center", fontsize=4)

    model.eval()
    with torch.no_grad():
        for graphs in testloader:
            if inchi == graphs.InChI:
                break
        graphs.x = graphs.x.to(model_dtype)
        graphs.edge_attr = graphs.edge_attr.to(model_dtype)
        graphs.edge_index = graphs.edge_index.to(torch.int64)

        graphs = graphs.to(device)
        parameters = model(graphs)
        params = parameters.squeeze().to(torch.float64).numpy()
        # params[0] -= 0.0112
        rho = graphs.rho.view(-1, 5).to(torch.float64).numpy()
        vp = graphs.vp.view(-1, 5).to(torch.float64).numpy()
        pred_den, pred_vp = rhovp_data(params, rho, vp)
        params = np.asarray(ra_para[inchi])
        ra_den, ra_vp = rhovp_data(params, rho, vp)

        idx_vp = (pred_vp != 0) & (ra_vp != 0)
        vp = vp[idx_vp]
        ra_vp = ra_vp[idx_vp]
        pred_vp = pred_vp[idx_vp]

        idx = np.argsort(vp[:, 0], 0)
        x = vp[idx, 0]
        y = vp[idx, -1] / 100000
        pltscatter(x, y)

        y = pred_vp[idx] / 100000
        pltline(x, y)
        mvp_model = 100 * np.abs(vp[idx, -1] - pred_vp[idx]) / vp[idx, -1]
        plterr(x, y * 0.99, mvp_model)

        y = ra_vp[idx] / 100000
        pltline(x, y)
        mvp_ra = 100 * np.abs(vp[idx, -1] - ra_vp[idx]) / vp[idx, -1]
        plterr(x, y * 1.01, mvp_ra)

        # Customize the plot appearance
        pltcustom(model_name, mvp_model, mvp_ra, "linear", "Pressão de vapor (Bar)")

        # Save the plot as a high-quality image file
        path = osp.join(
            "images", "vp_" + model_name.strip() + "_" + molecule_name + ".png"
        )
        plt.savefig(path, dpi=300)
        plt.show()

        idx_p = abs(rho[:, 1] - 101325) < 10000
        rho = rho[idx_p]
        pred_den = pred_den[idx_p]
        ra_den = ra_den[idx_p]
        idx = np.argsort(rho[:, 0], 0)

        x = rho[idx, 0]
        y = rho[idx, -1]
        pltscatter(x, y)

        y = pred_den[idx]
        pltline(x, y)
        mden_model = 100 * np.abs(rho[idx, -1] - pred_den[idx]) / rho[idx, -1]
        plterr(x, y, mden_model)

        y = ra_den[idx]
        pltline(x, y)
        mden_ra = 100 * np.abs(rho[idx, -1] - ra_den[idx]) / rho[idx, -1]
        plterr(x, y, mden_ra)

        # Customize the plot appearance
        pltcustom(model_name, mden_model, mden_ra, "linear", "Densidade (mol / m³)")
        path = osp.join("images", "den_" + model_name.strip()+ "_" + molecule_name + ".png")
        plt.savefig(path, dpi=300)
        plt.show()


def pltcustom(modelname, m_model, m_ra, scale="linear", ylabel=""):
    plt.xlabel("T (K)")
    plt.ylabel(ylabel)
    plt.title("")
    plt.legend(
        [
            "ThermoML",
            modelname + f" ({m_model.mean().round(decimals=2)} %)",
            f"Ramírez-Vélez et al. ({m_ra.mean().round(decimals=2)} %)",
        ]
    )
    plt.grid(False)
    plt.yscale(scale)