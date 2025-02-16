"Module for training with ECFP"

import os

import numpy as np
import xgboost as xgb
from absl import app, flags, logging
from torch_geometric.loader import DataLoader

from .utils import build_test_dataset, build_train_dataset, rho_batch, vp_batch

# PCSAFT parameters bounds
params_lower_bound = np.array([1.0, 1.9, 50.0, 1e-4, 200.0, 0, 0, 0])
params_upper_bound = np.array([25.0, 4.5, 550.0, 0.9, 5000.0, np.inf, np.inf, np.inf])


def training(workdir):
    """Training function"""
    # Load the data
    train_dataset = build_train_dataset(workdir, "esper")
    val_dataset, _ = build_test_dataset(workdir, train_dataset)
    # Create the Dataloader
    train_loader = DataLoader(
        train_dataset, batch_size=len(train_dataset), shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
    # Create the XGBoost dataset
    for graphs_train in train_loader:
        train_dmatrix = xgb.DMatrix(
            np.asarray(graphs_train.ecfp), label=graphs_train.para.numpy()
        )

        xgb_param = {
            "booster": "gblinear",
            "objective": "reg:squarederror",
            "eval_metric": "mape",
            "device": "cuda",
        }
        # Train the model
        xgb_model = xgb.train(
            xgb_param,
            train_dmatrix,
            num_boost_round=10000,
            evals=[(train_dmatrix, "train")],
        )

        # evaluate on validation set
        evaluation(val_loader, xgb_model)

        # save model
        xgb_model.save_model(os.path.join(workdir, "train/checkpoints/xgb_model.json"))


def evaluation(val_loader, xgb_model):
    """Evaluation function"""
    for graphs in val_loader:
        test_dmatrix = xgb.DMatrix(np.asarray(graphs.ecfp))
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
        print(f"MAPE den: {mape_den}, MAPE vp: {mape_vp}")


FLAGS = flags.FLAGS

flags.DEFINE_string("workdir", None, "Working Directory.")


def main(argv):
    """Execution from command line"""
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    logging.info("Calling training!")

    training(FLAGS.workdir)


if __name__ == "__main__":
    flags.mark_flags_as_required(["workdir"])
    app.run(main)
