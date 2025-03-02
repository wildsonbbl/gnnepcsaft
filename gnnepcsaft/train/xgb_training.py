"Module for training with ECFP"

import os

import numpy as np
import torch
import xgboost as xgb
from absl import app, flags, logging
from torch_geometric.loader import DataLoader

from .utils import build_test_dataset, build_train_dataset, rho_batch, vp_batch

# PCSAFT parameters bounds
params_lower_bound = np.array([1.0, 1.9, 50.0, 1e-4, 200.0, 0, 0, 0])
params_upper_bound = np.array([25.0, 4.5, 550.0, 0.9, 5000.0, np.inf, np.inf, np.inf])


def training(workdir: str, config: dict):
    """Training function"""
    # Load the data
    train_dataset = build_train_dataset(workdir, "esper")
    val_dataset, train_val_dataset, _ = build_test_dataset(workdir, train_dataset)
    # Create the Dataloader
    train_loader = DataLoader(
        train_dataset, batch_size=len(train_dataset), shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset))
    train_val_dataloader = DataLoader(
        train_val_dataset, batch_size=len(train_val_dataset)
    )
    # Create the XGBoost dataset
    for graphs_train in train_loader:
        train_dmatrix = xgb.DMatrix(
            torch.hstack(
                (
                    graphs_train.ecfp,
                    graphs_train.mw,
                    graphs_train.atom_count,
                    graphs_train.ring_count,
                    graphs_train.rbond_count,
                )
            ).numpy(),
            label=graphs_train.para.numpy(),
        )

    xgb_param = {
        "booster": "gblinear",
        "objective": "reg:squaredlogerror",
        "eval_metric": ["mape", "rmsle"],
        "device": "cuda",
        "lambda": config["lambda"],
        "alpha": config["alpha"],
        "eta": config["eta"],
    }
    # Train the model
    results = {}
    xgb_model = xgb.train(
        xgb_param,
        train_dmatrix,
        num_boost_round=config["num_boost_round"],
        evals=[(train_dmatrix, "train")],
        evals_result=results,
        verbose_eval=False,
    )

    print(
        f"Train-mape: {results['train']['mape'][-1]} "
        f"|| Train-rmsle: {results['train']['rmsle'][-1]}"
    )
    # evaluate on validation set
    mape_den, mape_vp = evaluation(val_loader, xgb_model)

    print(f"MAPE den/val: {mape_den}, MAPE vp/val: {mape_vp}")

    mape_den, mape_vp = evaluation(train_val_dataloader, xgb_model)

    print(f"MAPE den/train_val: {mape_den}, MAPE vp/train_val: {mape_vp}")

    # save model
    xgb_model.save_model(os.path.join(workdir, "train/checkpoints/xgb_model.json"))
    return mape_den, mape_vp


def evaluation(val_loader, xgb_model):
    """Evaluation function"""
    for graphs in val_loader:
        test_dmatrix = xgb.DMatrix(
            torch.hstack(
                (
                    graphs.ecfp,
                    graphs.mw,
                    graphs.atom_count,
                    graphs.ring_count,
                    graphs.rbond_count,
                )
            ).numpy()
        )
        pred_msigmae = xgb_model.predict(test_dmatrix)
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
flags.DEFINE_integer("num_boost_round", 1000, "Number of boosting rounds.")


def main(argv):
    """Execution from command line"""
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    logging.info("Calling training!")

    training(
        FLAGS.workdir,
        {
            "lambda": 1.92e-6,
            "alpha": 2.77e-5,
            "num_boost_round": FLAGS.num_boost_round,
            "eta": 0.039,
        },
    )


if __name__ == "__main__":
    flags.mark_flags_as_required(["workdir"])
    app.run(main)
