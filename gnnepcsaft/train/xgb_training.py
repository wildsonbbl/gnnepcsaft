"Module for training with ECFP"

import os

import torch
import xgboost as xgb
from absl import app, flags, logging
from torch_geometric.loader import DataLoader

from .utils import build_test_dataset, build_train_dataset, rf_xgb_evaluation


def training(workdir: str, config: dict):
    """Training function"""
    # Load the data
    train_dataset = build_train_dataset(workdir, "esper")
    val_dataset, train_val_dataset = build_test_dataset(workdir, train_dataset)
    # Create the Dataloader
    train_loader = DataLoader(
        train_dataset, batch_size=len(train_dataset), shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset))
    train_val_dataloader = DataLoader(
        train_val_dataset, batch_size=len(train_val_dataset)
    )
    graphs_train = next(iter(train_loader))
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
        "booster": "gbtree",
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
    mape_den, mape_vp = rf_xgb_evaluation(next(iter(val_loader)), xgb_model)

    print(f"MAPE den/val: {mape_den}, MAPE vp/val: {mape_vp}")

    mape_den, mape_vp = rf_xgb_evaluation(next(iter(train_val_dataloader)), xgb_model)

    print(f"MAPE den/train_val: {mape_den}, MAPE vp/train_val: {mape_vp}")

    # save model
    xgb_model.save_model(os.path.join(workdir, "train/checkpoints/xgb_model.json"))
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
