"Module for training with ECFP"

import os

import joblib
import numpy as np
import torch
from absl import app, flags, logging
from sklearn.ensemble import RandomForestRegressor
from torch_geometric.loader import DataLoader

from .utils import build_test_dataset, build_train_dataset, rf_xgb_evaluation


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

    graphs_train = next(iter(train_loader))
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

    rf_model.fit(x, y)

    para_pred = rf_model.predict(x)

    para_pred_mape = np.mean(np.abs(para_pred - y) / y)

    print(f"Train-mape: {para_pred_mape.item()}")
    # evaluate on validation set
    mape_den, mape_vp = rf_xgb_evaluation(next(iter(val_loader)), rf_model)

    print(f"MAPE den/val: {mape_den}, MAPE vp/val: {mape_vp}")

    mape_den, mape_vp = rf_xgb_evaluation(next(iter(train_val_dataloader)), rf_model)

    print(f"MAPE den/train_val: {mape_den}, MAPE vp/train_val: {mape_vp}")

    # save model
    joblib.dump(
        rf_model,
        os.path.join(workdir, "train/checkpoints/rf_model.joblib"),
        compress=3,
    )
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
