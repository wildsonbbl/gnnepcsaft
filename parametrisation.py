from scipy.optimize import least_squares
from pcsaft import pcsaft_den, flashTQ
import numpy as np

import os.path as osp, pickle
import torch
from torch_geometric.loader import DataLoader
from graphdataset import ThermoMLDataset


path = osp.join("data", "thermoml")
dataset = ThermoMLDataset(path)
loader = DataLoader(dataset, batch_size=1, shuffle=False)
device = torch.device("cpu")

with open("./data/thermoml/processed/para3.pkl", "rb") as file:
    init_para = pickle.load(file)


def loss(parameters, rho, vp):
    m = parameters[0]
    s = parameters[1]
    e = parameters[2]
    loss = []

    if ~np.all(rho == np.zeros_like(rho)):
        for state in rho:
            x = np.asarray([1.0])
            t = state[0]
            p = state[1]
            phase = ["liq" if state[2] == 1 else "vap"][0]
            params = {"m": m, "s": s, "e": e}
            den = pcsaft_den(t, p, x, params, phase=phase)
            loss += [((state[-1] - den) / state[-1]) ** 2 * 3]

    if ~np.all(vp == np.zeros_like(vp)):
        for state in vp:
            x = np.asarray([1.0])
            t = state[0]
            p = state[1]
            phase = ["liq" if state[2] == 1 else "vap"][0]
            params = {"m": m, "s": s, "e": e}
            try:
                vp, xl, xv = flashTQ(t, 0, x, params, p)
            except:
                vp = 0
            loss += [((state[-1] - vp) / state[-1]) ** 2 * 2]
    loss = np.asarray(loss).flatten()

    return loss


if __name__ == "__main__":
    fitted_para = {}

    for graph in loader:
        rho = graph.rho.view(-1, 5).numpy()
        vp = graph.vp.view(-1, 5).numpy()
        n_datapoints = rho.shape[0] + vp.shape[0]
        if n_datapoints < 3:
            print(f"skipping {graph.InChI[0]} for having {n_datapoints} datapoints")
            continue
        params = np.asarray(init_para[graph.InChI[0]][0])
        res = least_squares(loss, params, method="lm", args=(rho, vp))
        fit_para = res.x.tolist()
        cost = res.cost
        print(cost, fit_para)
        fitted_para[graph.InChI[0]] = (fit_para, cost)
        with open("./data/thermoml/processed/para3_fitted.pkl", "wb") as file:
            pickle.dump(fitted_para, file)
