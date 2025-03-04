"Module for training with ECFP"

import os

import joblib
import numpy as np
import torch
from absl import app, flags, logging
from sklearn.ensemble import RandomForestRegressor
from torch_geometric.loader import DataLoader

from .utils import build_test_dataset, build_train_dataset, rho_batch, vp_batch

# PCSAFT parameters bounds
params_lower_bound = np.array([1.0, 1.9, 50.0, 1e-4, 200.0, 0, 0, 0])
params_upper_bound = np.array([25.0, 4.5, 550.0, 0.9, 5000.0, np.inf, np.inf, np.inf])


def training(workdir: str, config: dict):
    """Training function"""
    # Load the data
    train_dataset = build_train_dataset(workdir, "esper")
    test_dts = build_test_dataset(workdir, train_dataset)
    # Create the Dataloader
    train_loader = DataLoader(
        train_dataset, batch_size=len(train_dataset), shuffle=True
    )
    val_loader = DataLoader(test_dts[0], batch_size=len(test_dts[0]))
    train_val_dataloader = DataLoader(test_dts[1], batch_size=len(test_dts[1]))
    # Create the XGBoost dataset
    for graphs_train in train_loader:
        x = torch.hstack(
            (
                graphs_train.ecfp,
                graphs_train.mw,
                graphs_train.atom_count,
                graphs_train.ring_count,
                graphs_train.rbond_count,
            )
        ).numpy()
        y = graphs_train.para.numpy()

    rf_model = RandomForestRegressor(
        criterion="squared_error",
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=config["max_features"],
        bootstrap=True,
        oob_score=True,
    )

    rf_model.fit(x, y)

    para_pred = rf_model.predict(x)

    para_pred_mape = np.mean(np.abs(para_pred - y) / y)

    print(f"Train-mape: {para_pred_mape.item()}")
    # evaluate on validation set
    mape_den, mape_vp = evaluation(val_loader, rf_model)

    print(f"MAPE den/val: {mape_den}, MAPE vp/val: {mape_vp}")

    mape_den, mape_vp = evaluation(train_val_dataloader, rf_model)

    print(f"MAPE den/train_val: {mape_den}, MAPE vp/train_val: {mape_vp}")

    # save model
    joblib.dump(
        rf_model,
        os.path.join(workdir, "train/checkpoints/rf_model.joblib"),
        compress=3,
    )
    return mape_den, mape_vp


def evaluation(val_loader: DataLoader, rf_model: RandomForestRegressor):
    """Evaluation function"""
    for graphs in val_loader:
        x = torch.hstack(
            (
                graphs.ecfp,
                graphs.mw,
                graphs.atom_count,
                graphs.ring_count,
                graphs.rbond_count,
            )
        )
        x = x.numpy()
        pred_msigmae = rf_model.predict(x)
        assert isinstance(pred_msigmae, np.ndarray)
        assert pred_msigmae.shape == graphs.para.numpy().shape
        para_assoc = 10 ** (graphs.assoc.numpy() * np.array([-1.0, 1.0]))
        pred_params = np.hstack([pred_msigmae, para_assoc, graphs.munanb.numpy()])
        pred_params.clip(params_lower_bound, params_upper_bound, out=pred_params)
        assert pred_params.shape == (len(graphs.rho), 8)
        assert isinstance(graphs.rho[0], np.ndarray)
        assert isinstance(graphs.vp[0], np.ndarray)
        assert isinstance(graphs.rho, list)
        assert isinstance(graphs.vp, list)
        pred_rho = rho_batch(pred_params, graphs.rho)
        pred_vp = vp_batch(pred_params, graphs.vp)
        assert isinstance(pred_rho[0], np.ndarray)
        assert isinstance(pred_vp[0], np.ndarray)
        assert isinstance(pred_rho, list)
        assert isinstance(pred_vp, list)
        rho = [rho[:, -1] for rho in graphs.rho if rho.shape[0] > 0]
        vp = [vp[:, -1] for vp in graphs.vp if vp.shape[0] > 0]
        mape_den = []
        for pred, exp in zip(pred_rho, rho):
            assert pred.shape == exp.shape
            mape_den += [np.mean(np.abs(pred - exp) / exp).item()]
        mape_den = np.asarray(mape_den).mean().item()
        mape_vp = []
        for pred, exp in zip(pred_vp, vp):
            assert pred.shape == exp.shape
            mape_vp += [np.mean(np.abs(pred - exp) / exp).item()]
        mape_vp = np.asarray(mape_vp).mean().item()
        return mape_den, mape_vp


FLAGS = flags.FLAGS

flags.DEFINE_string("workdir", None, "Working Directory.")


def main(argv):
    """Execution from command line"""
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    logging.info("Calling training!")

    training(
        FLAGS.workdir,
        {
            "n_estimators": 1000,
            "max_depth": None,
            "max_features": "log2",
        },
    )


if __name__ == "__main__":
    flags.mark_flags_as_required(["workdir"])
    app.run(main)
