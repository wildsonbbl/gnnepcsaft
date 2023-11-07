import os.path as osp
from absl import app
from absl import flags
from ml_collections import config_flags

from absl import logging
import ml_collections

from train import models

import torch
from torch.nn import HuberLoss
from torchmetrics import MeanAbsolutePercentageError

from epcsaft import epcsaft_cython

import wandb

from data.graphdataset import ThermoMLDataset, ramirez, ThermoMLpara

from train.utils import calc_deg, create_model

device = "cpu"


def evaluate(
    config: ml_collections.ConfigDict, workdir: str, dataset: str, modelname: str
):
    """Execute model training and evaluation loop.

    Args:
      config: Hyperparameter configuration for training and evaluation.
      workdir: Working Directory.
    """
    deg = calc_deg(dataset, workdir)
    # Create writer for logs.
    wandb.login()

    # Get datasets, organized by split.
    logging.info("Obtaining datasets.")
    model_dtype = torch.float64

    if dataset == "ramirez":
        path = osp.join(workdir, "data/ramirez2022")
        train_loader = ramirez(path)
    elif dataset == "thermoml":
        path = osp.join(workdir, "data/thermoml")
        train_loader = ThermoMLpara(path)
    else:
        ValueError(
            f"dataset is either ramirez or thermoml, got >>> {dataset} <<< instead"
        )

    test_loader = ThermoMLDataset(osp.join(workdir, "data/thermoml"))

    para_data = {}
    for graph in train_loader:
        inchi, para = graph.InChI, graph.para.view(-1, 3)
        para_data[inchi] = para

    # Create and initialize the network.
    logging.info("Initializing network.")
    model = create_model(config, deg).to(device, model_dtype)
    pcsaft_den = epcsaft_cython.PCSAFT_den.apply
    pcsaft_vp = epcsaft_cython.PCSAFT_vp.apply
    HLoss = HuberLoss("mean").to(device)
    mape = MeanAbsolutePercentageError().to(device)

    # Set up checkpointing of the model.
    if modelname:
        ckp_path = osp.join(workdir, "train/checkpoints", f"{modelname}.pth")
    elif dataset == "ramirez":
        ckp_path = osp.join(workdir, "train/checkpoints/ra_last_checkpoint.pth")
    else:
        ckp_path = osp.join(workdir, "train/checkpoints/tml_last_checkpoint.pth")
    checkpoint = torch.load(ckp_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state_dict"])

    # test fn
    @torch.no_grad()
    def test_den(test="test"):
        model.eval()
        total_loss_mape = []
        total_loss_huber = []
        for graphs in test_loader:
            if test == "test":
                if graphs.InChI in para_data:
                    continue
            if test == "val":
                if graphs.InChI not in para_data:
                    continue
            datapoints = graphs.rho.to(device, model_dtype)
            if torch.all(datapoints == torch.zeros_like(datapoints)):
                continue
            graphs.x = graphs.x.to(model_dtype)
            graphs.edge_attr = graphs.edge_attr.to(model_dtype)
            graphs.edge_index = graphs.edge_index.to(torch.int64)
            graphs = graphs.to(device)
            pred_para = model(graphs).squeeze()
            pred = pcsaft_den(pred_para, datapoints)
            target = datapoints[:, -1]
            loss_mape = mape(pred, target)
            loss_huber = HLoss(pred, target)
            wandb.log(
                {
                    "mape_den": loss_mape.item(),
                    "huber_den": loss_huber.item(),
                },
            )
            total_loss_mape += [loss_mape.item()]
            total_loss_huber += [loss_huber.item()]

        return (
            torch.tensor(total_loss_mape).nanmean().item(),
            torch.tensor(total_loss_huber).nanmean().item(),
        )

    @torch.no_grad()
    def test_vp(test="test"):
        model.eval()
        total_loss_mape = []
        total_loss_huber = []
        for graphs in test_loader:
            if test == "test":
                if graphs.InChI in para_data:
                    continue
            if test == "val":
                if graphs.InChI not in para_data:
                    continue
            datapoints = graphs.vp.to(device, model_dtype)
            if torch.all(datapoints == torch.zeros_like(datapoints)):
                continue
            graphs.x = graphs.x.to(model_dtype)
            graphs.edge_attr = graphs.edge_attr.to(model_dtype)
            graphs.edge_index = graphs.edge_index.to(torch.int64)
            graphs = graphs.to(device)
            pred_para = model(graphs).squeeze()
            pred = pcsaft_vp(pred_para, datapoints)
            target = datapoints[:, -1]
            result_filter = ~torch.isnan(pred)
            loss_mape = mape(pred[result_filter], target[result_filter])
            loss_huber = HLoss(pred[result_filter], target[result_filter])
            wandb.log(
                {
                    "mape_vp": loss_mape.item(),
                    "huber_vp": loss_huber.item(),
                },
            )
            if loss_mape.item() >= 0.9:
                continue
            total_loss_mape += [loss_mape.item()]
            total_loss_huber += [loss_huber.item()]

        return (
            torch.tensor(total_loss_mape).nanmean().item(),
            torch.tensor(total_loss_huber).nanmean().item(),
        )

    run = wandb.init(
        # Set the project where this run will be logged
        project="gnn-pc-saft",
        # Track hyperparameters and run metadata
        config=config.to_dict(),
        name=modelname,
        group=dataset,
        tags=[dataset, "eval"],
    )
    # Evaluate on validation or test, if required.
    val_mape_den, val_huber_den = test_den("val")
    test_mape_den, test_huber_den = test_den("test")

    wandb.log(
        {
            "val_mape_den": val_mape_den,
            "val_huber_den": val_huber_den,
            "test_huber_den": test_huber_den,
            "test_mape_den": test_mape_den,
        }
    )

    # Evaluate on validation or test, if required.
    val_mape_vp, val_huber_vp = test_vp("val")
    test_mape_vp, test_huber_vp = test_vp("test")

    wandb.log(
        {
            "val_mape_vp": val_mape_vp,
            "val_huber_vp": val_huber_vp,
            "test_huber_vp": test_huber_vp,
            "test_mape_vp": test_mape_vp,
        }
    )

    wandb.finish()


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
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    logging.info("Calling evaluate")

    evaluate(FLAGS.config, FLAGS.workdir, FLAGS.dataset, FLAGS.modelname)


if __name__ == "__main__":
    flags.mark_flags_as_required(["config", "workdir"])
    app.run(main)
