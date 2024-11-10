"""Module to be used for evaluation of an emsemble of models.

At `evalmodels.ipynb` a working example can be found.
"""

import os.path as osp

import ml_collections
import torch
import wandb
from absl import app, flags, logging
from ml_collections import config_flags
from torch.nn import HuberLoss
from torchmetrics import MeanAbsolutePercentageError

from ..epcsaft import utils
from ..train.utils import (
    build_datasets_loaders,
    calc_deg,
    create_model,
    input_artifacts,
)

device = torch.device("cpu")
MODEL_DTYPE = torch.float64

# pylint: disable=no-member
pcsaft_den = utils.DenFromTensor.apply
pcsaft_vp = utils.VpFromTensor.apply
hloss = HuberLoss("mean")
mape = MeanAbsolutePercentageError()


def evaluate(
    config: ml_collections.ConfigDict, workdir: str, dataset: str, modelname: str
):
    """Execute model training and evaluation loop.

    Args:
      config: Hyperparameter configuration for training and evaluation.
      workdir: Working Directory.
      modelname: One or more model checkpoint file name separated by `.`.
    """
    deg = calc_deg(dataset, workdir)
    modelnames = modelname.split(".")
    logging.info(f"evaluating models {modelnames}")
    # Create writer for logs.
    wandb.login()
    wandb.init(
        # Set the project where this run will be logged
        project="gnn-pc-saft",
        # Track hyperparameters and run metadata
        config=config.to_dict(),
        name=modelname,
        group=dataset,
        tags=[dataset, "eval"],
        job_type="eval",
    )

    # Get datasets, organized by split.
    logging.info("Obtaining datasets.")

    _, test_loader, para_data = build_datasets_loaders(config, workdir, dataset)

    # Create and initialize the network.
    logging.info("Initializing network.")
    model_dict = {
        name: create_model(config, deg).to(device, MODEL_DTYPE) for name in modelnames
    }

    # Set up checkpointing of the model.
    for name in model_dict:
        input_artifacts(workdir, dataset, name)
        ckp_path = osp.join(workdir, "train/checkpoints", f"{name}.pth")
        checkpoint = torch.load(ckp_path, map_location=torch.device("cpu"))
        model_dict[name].load_state_dict(checkpoint["model_state_dict"])
        model_dict[name].eval()

    # Evaluate on validation or test, if required.
    val = test_den(
        test_loader=test_loader,
        para_data=para_data,
        model_dict=model_dict,
        test="val",
    )
    test = test_den(
        test_loader=test_loader,
        para_data=para_data,
        model_dict=model_dict,
        test="test",
    )
    wandb.log(
        {
            "val_mape_den": val[0],
            "val_huber_den": val[1],
            "test_huber_den": test[1],
            "test_mape_den": test[0],
        }
    )

    # Evaluate on validation or test, if required.
    val = test_vp(
        test_loader=test_loader,
        para_data=para_data,
        model_dict=model_dict,
        test="val",
    )
    test = test_vp(
        test_loader=test_loader,
        para_data=para_data,
        model_dict=model_dict,
        test="test",
    )

    wandb.log(
        {
            "val_mape_vp": val[0],
            "val_huber_vp": val[1],
            "test_huber_vp": test[1],
            "test_mape_vp": test[0],
        }
    )

    wandb.finish()


# test fn
@torch.no_grad()
def test_den(test_loader, para_data, model_dict, test="test"):
    "Evaluates density prediction."
    total_loss = ([], [])
    for graphs in test_loader:
        if test == "test":
            if graphs.InChI in para_data:
                continue
        if test == "val":
            if graphs.InChI not in para_data:
                continue
        datapoints = graphs.rho.to(device, MODEL_DTYPE)
        if torch.all(datapoints == torch.zeros_like(datapoints)):
            continue
        graphs = graphs.to(device)
        pred_para = []
        for name in model_dict:
            pred_params = model_dict[name](graphs)
            pred_para += [pred_params]
        pred_para = torch.concat(pred_para, dim=0).mean(0).to(MODEL_DTYPE)
        pred = pcsaft_den(pred_para, datapoints)
        target = datapoints[:, -1]
        # pylint: disable = not-callable
        loss_mape = mape(pred, target)
        loss_huber = hloss(pred, target)
        wandb.log(
            {
                "mape_den": loss_mape.item(),
                "huber_den": loss_huber.item(),
            },
        )
        total_loss[0].append(loss_mape.item())
        total_loss[1].append(loss_huber.item())

    return (
        torch.tensor(total_loss[0]).nanmean().item(),
        torch.tensor(total_loss[1]).nanmean().item(),
    )


@torch.no_grad()
def test_vp(test_loader, para_data, model_dict, test="test"):
    "Evaluates vapor pressure prediction."
    total_loss = ([], [])
    for graphs in test_loader:
        if test == "test":
            if graphs.InChI in para_data:
                continue
        if test == "val":
            if graphs.InChI not in para_data:
                continue
        datapoints = graphs.vp.to(device, MODEL_DTYPE)
        if torch.all(datapoints == torch.zeros_like(datapoints)):
            continue
        graphs = graphs.to(device)
        pred_para = []
        for model in model_dict.values():
            pred_params = model(graphs)
            pred_para.append(pred_params)
        pred_para = torch.concat(pred_para, dim=0).mean(0).to(MODEL_DTYPE)
        pred = pcsaft_vp(pred_para, datapoints)
        target = datapoints[:, -1]
        result_filter = ~torch.isnan(pred)
        # pylint: disable = not-callable
        loss_mape = mape(pred[result_filter], target[result_filter])
        loss_huber = hloss(pred[result_filter], target[result_filter])
        wandb.log(
            {
                "mape_vp": loss_mape.item(),
                "huber_vp": loss_huber.item(),
            },
        )
        if loss_mape.item() >= 0.9:
            continue
        total_loss[0].append(loss_mape.item())
        total_loss[1].append(loss_huber.item())

    return (
        torch.tensor(total_loss[0]).nanmean().item(),
        torch.tensor(total_loss[1]).nanmean().item(),
    )


FLAGS = flags.FLAGS

flags.DEFINE_string("workdir", None, "Working Directory.")
flags.DEFINE_string("dataset", None, "Dataset to test model on")
flags.DEFINE_string("modelname", None, "Model name to test on")
config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=True,
)


def main(argv):
    """Execution from command line"""
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    logging.info("Calling evaluate!")

    evaluate(FLAGS.config, FLAGS.workdir, FLAGS.dataset, FLAGS.modelname)


if __name__ == "__main__":
    flags.mark_flags_as_required(["config", "workdir"])
    app.run(main)
