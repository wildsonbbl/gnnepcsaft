import os.path as osp
import pickle

import ml_collections
import numpy as np
import torch
import wandb
from absl import app, flags, logging
from ml_collections import config_flags
from pcsaft import flashTQ, pcsaft_den
from scipy.optimize import least_squares

from data.graphdataset import ThermoMLDataset

path = osp.join("data", "thermoml")
loader = ThermoMLDataset(path)
device = torch.device("cpu")

with open("./data/thermoml/processed/para3.pkl", "rb") as file:
    init_para = pickle.load(file)
with open("./data/thermoml/raw/para3_fitted.pkl", "rb") as file:
    fitted_para = pickle.load(file)


def MAPE(parameters: np.ndarray, rho: np.ndarray, vp: np.ndarray, mean: bool = True):
    parameters = np.abs(parameters)
    m = parameters[0]
    s = parameters[1]
    e = parameters[2]
    mape = 0.0

    if ~np.all(rho == np.zeros_like(rho)):
        mape = []
        for state in rho:
            x = np.asarray([1.0])
            t = state[0]
            p = state[1]
            phase = "liq" if state[2] == 1 else "vap"
            params = {"m": m, "s": s, "e": e}
            den = pcsaft_den(t, p, x, params, phase=phase)
            mape += [np.abs((state[-1] - den) / state[-1])]

    den = np.asarray(mape)
    if mean:
        den = den.mean()

    mape = 0.0
    if ~np.all(vp == np.zeros_like(vp)):
        mape = []
        for state in vp:
            x = np.asarray([1.0])
            t = state[0]
            p = state[1]
            phase = "liq" if state[2] == 1 else "vap"
            params = {"m": m, "s": s, "e": e}
            try:
                vp, xl, xv = flashTQ(t, 0, x, params, p)
                mape += [np.abs((state[-1] - vp) / state[-1])]
            except:
                pass

    vp = np.asarray(mape)
    if mean:
        vp = vp.mean()

    return den, vp


def rhovp_data(parameters: np.ndarray, rho: np.ndarray, vp: np.ndarray):
    parameters = np.abs(parameters)
    m = parameters[0]
    s = parameters[1]
    e = parameters[2]
    den = []

    if ~np.all(rho == np.zeros_like(rho)):
        for state in rho:
            x = np.asarray([1.0])
            t = state[0]
            p = state[1]
            phase = ["liq" if state[2] == 1 else "vap"][0]
            params = {"m": m, "s": s, "e": e}
            den += [pcsaft_den(t, p, x, params, phase=phase)]

    den = np.asarray(den)

    vpl = []
    if ~np.all(vp == np.zeros_like(vp)):
        for state in vp:
            x = np.asarray([1.0])
            t = state[0]
            p = state[1]
            phase = ["liq" if state[2] == 1 else "vap"][0]
            params = {"m": m, "s": s, "e": e}
            try:
                vp, xl, xv = flashTQ(t, 0, x, params, p)
                vpl += [vp]
            except:
                vpl += [np.nan]

    vp = np.asarray(vpl)

    return den, vp


def parametrisation(weight_decay):
    def loss(parameters: np.ndarray, rho: np.ndarray, vp: np.ndarray):
        parameters = np.abs(parameters)
        m = parameters[0]
        s = parameters[1]
        e = parameters[2]
        loss = []
        x_scale = np.array([1.0, 1.0, 100.0])
        n = rho.shape[0] + vp.shape[0]
        l2penalty = np.sum((parameters / x_scale) ** 2) * weight_decay / n

        if ~np.all(rho == np.zeros_like(rho)):
            for state in rho:
                x = np.asarray([1.0])
                t = state[0]
                p = state[1]
                phase = "liq" if state[2] == 1 else "vap"
                params = {"m": m, "s": s, "e": e}
                den = pcsaft_den(t, p, x, params, phase=phase)
                loss += [((state[-1] - den) / state[-1]) * np.sqrt(2)]

        if ~np.all(vp == np.zeros_like(vp)):
            for state in vp:
                x = np.asarray([1.0])
                t = state[0]
                p = state[1]
                phase = "liq" if state[2] == 1 else "vap"
                params = {"m": m, "s": s, "e": e}
                try:
                    vp, xl, xv = flashTQ(t, 0, x, params, p)
                    loss += [((state[-1] - vp) / state[-1]) * np.sqrt(3)]
                except:
                    loss += [1e6]

        loss = np.asarray(loss).flatten() + np.sqrt(l2penalty)

        return loss

    run = wandb.init(
        # Set the project where this run will be logged
        project="gnn-pc-saft",
        config={"weight decay": weight_decay},
        tags=["para"],
    )

    x_scale = np.array([10.0, 10.0, 1000.0])
    for graph in loader:
        if graph.InChI not in init_para:
            continue
        rho = graph.rho.view(-1, 5).numpy()
        vp = graph.vp.view(-1, 5).numpy()
        params = np.asarray(init_para[graph.InChI][0])
        res = least_squares(loss, params, method="lm", x_scale=x_scale, args=(rho, vp))
        fit_para = np.abs(res.x).tolist()
        cost = res.cost
        mden, mvp = MAPE(res.x, rho, vp)
        wandb.log(
            {
                "cost": cost,
                "m": fit_para[0],
                "s": fit_para[1],
                "e": fit_para[2],
                "mape_den": mden,
                "mape_vp": mvp,
                "success": int(res.success),
            },
        )
        _, saved_mden, saved_mvp = fitted_para[graph.InChI]
        if (
            ((saved_mden == 0) & (saved_mvp > mvp))
            & (np.isfinite(mden))
            & (np.isfinite(mvp))
        ):
            fitted_para[graph.InChI] = (fit_para, mden, mvp)
        elif (
            ((saved_mden > mden) & (saved_mvp > mvp))
            & (np.isfinite(mden))
            & (np.isfinite(mvp))
        ):
            fitted_para[graph.InChI] = (fit_para, mden, mvp)
        elif (
            ((saved_mden > mden) & (saved_mvp == 0))
            & (np.isfinite(mden))
            & (np.isfinite(mvp))
        ):
            fitted_para[graph.InChI] = (fit_para, mden, mvp)

    with open("./data/thermoml/raw/para3_fitted.pkl", "wb") as file:
        pickle.dump(fitted_para, file)
    wandb.finish()


FLAGS = flags.FLAGS

flags.DEFINE_float("weight_decay", None, "For L2 penalty.")


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    logging.info("Calling parametrisation")

    parametrisation(FLAGS.weight_decay)


if __name__ == "__main__":
    flags.mark_flags_as_required(["weight_decay"])
    app.run(main)
