import os.path as osp
from absl import app
from absl import flags
from ml_collections import config_flags

from absl import logging
import ml_collections

from train import models

import torch
from torch.nn import HuberLoss
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
)
from torch_geometric.loader import DataLoader
from torchmetrics import MeanAbsolutePercentageError

from epcsaft import epcsaft_cython

import wandb

from data.graphdataset import ThermoMLDataset, ramirez, ThermoMLpara

from train.model_deg import calc_deg


def create_model(
    config: ml_collections.ConfigDict, deg: torch.Tensor
) -> torch.nn.Module:
    """Creates a model, as specified by the config."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.half_precision:
        model_dtype = torch.float16
    else:
        model_dtype = torch.float32
    if config.model == "PNA":
        return models.PNAPCSAFT(
            hidden_dim=config.hidden_dim,
            propagation_depth=config.propagation_depth,
            pre_layers=config.pre_layers,
            post_layers=config.post_layers,
            num_mlp_layers=config.num_mlp_layers,
            num_para=config.num_para,
            deg=deg,
            dtype=model_dtype,
            device=device,
        )
    raise ValueError(f"Unsupported model: {config.model}.")


def create_optimizer(config: ml_collections.ConfigDict, params):
    """Creates an optimizer, as specified by the config."""
    if config.optimizer == "adam":
        return torch.optim.AdamW(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            amsgrad=True,
            eps=1e-5,
        )
    if config.optimizer == "sgd":
        return torch.optim.SGD(
            params,
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            nesterov=True,
        )
    raise ValueError(f"Unsupported optimizer: {config.optimizer}.")


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str, dataset: str):
    """Execute model training and evaluation loop.

    Args:
      config: Hyperparameter configuration for training and evaluation.
      workdir: Working Directory.
    """

    deg = calc_deg(dataset, workdir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create writer for logs.
    wandb.login()
    run = wandb.init(
        # Set the project where this run will be logged
        project="gnn-pc-saft",
        # Track hyperparameters and run metadata
        config=config.to_dict(),
    )

    # Get datasets, organized by split.
    logging.info("Obtaining datasets.")
    if config.half_precision:
        model_dtype = torch.float16
    else:
        model_dtype = torch.float32

    if dataset == "ramirez":
        path = osp.join(workdir, "data/ramirez2022")
        train_dataset = ramirez(path)
    elif dataset == "thermoml":
        path = osp.join(workdir, "data/thermoml")
        train_dataset = ThermoMLpara(path)
    else:
        ValueError(
            f"dataset is either ramirez or thermoml, got >>> {dataset} <<< instead"
        )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    test_dataset = ThermoMLDataset(osp.join(workdir, "data/thermoml"))
    test_loader = DataLoader(test_dataset)

    para_data = {}
    for graph in train_loader:
        for inchi, para in zip(graph.InChI, graph.para.view(-1, 3)):
            para_data[inchi] = para

    # Create and initialize the network.
    logging.info("Initializing network.")
    model = create_model(config, deg).to(device, model_dtype)
    pcsaft_den = epcsaft_cython.PCSAFT_den.apply
    pcsaft_vp = epcsaft_cython.PCSAFT_vp.apply
    HLoss = HuberLoss("mean").to(device)
    mape = MeanAbsolutePercentageError().to(device)

    # Create the optimizer.
    optimizer = create_optimizer(config, model.parameters())

    # Set up checkpointing of the model.
    if dataset == "ramirez":
        ckp_path = osp.join(workdir, "train/checkpoints/ra_last_checkpoint.pth")
    else:
        ckp_path = osp.join(workdir, "train/checkpoints/tml_last_checkpoint.pth")
    initial_step = 1
    if osp.exists(ckp_path):
        checkpoint = torch.load(ckp_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        step = checkpoint["step"]
        initial_step = int(step) + 1
        del checkpoint

    # Scheduler
    scheduler = CosineAnnealingWarmRestarts(optimizer, config.warmup_steps)
    # scheduler = ReduceLROnPlateau(optimizer, patience = config.patience)

    @torch.no_grad()
    def test(test="test"):
        model.eval()
        total_mape_den = []
        total_huber_den = []
        total_mape_vp = []
        total_huber_vp = []
        for graphs in test_loader:
            if test == "test":
                if graphs.InChI[0] in para_data:
                    continue
            if test == "val":
                if graphs.InChI[0] not in para_data:
                    continue
            graphs = graphs.to(device)
            pred_para = model(graphs).squeeze().to("cpu", torch.float64)

            datapoints = graphs.rho.to("cpu", torch.float64).view(-1, 5)
            if ~torch.all(datapoints == torch.zeros_like(datapoints)):
                pred = pcsaft_den(pred_para, datapoints)
                target = datapoints[:, -1]
                loss_mape = mape(pred, target)
                loss_huber = HLoss(pred, target)
                total_mape_den += [loss_mape.item()]
                total_huber_den += [loss_huber.item()]

            datapoints = graphs.vp.to("cpu", torch.float64).view(-1, 5)
            if ~torch.all(datapoints == torch.zeros_like(datapoints)):
                pred = pcsaft_vp(pred_para, datapoints)
                target = datapoints[:, -1]
                result_filter = ~torch.isnan(pred)
                loss_mape = mape(pred[result_filter], target[result_filter])
                loss_huber = HLoss(pred[result_filter], target[result_filter])
                total_mape_vp += [loss_mape.item()]
                total_huber_vp += [loss_huber.item()]

        return (
            torch.tensor(total_mape_den).nanmean().item(),
            torch.tensor(total_huber_den).nanmean().item(),
            torch.tensor(total_mape_vp).nanmean().item(),
            torch.tensor(total_huber_vp).nanmean().item(),
        )

    # Begin training loop.
    logging.info("Starting training.")
    step = initial_step
    total_loss_mape = []
    total_loss_huber = []
    lr = []

    model.train()
    while step < config.num_train_steps + 1:
        for graphs in train_loader:
            target = graphs.para.to(device, model_dtype).view(-1, 3)
            graphs = graphs.to(device)
            optimizer.zero_grad()
            pred = model(graphs)
            loss_mape = mape(pred, target)
            loss_huber = HLoss(pred, target)
            loss_mape.backward()
            optimizer.step()
            total_loss_mape += [loss_mape.item()]
            total_loss_huber += [loss_huber.item()]
            lr += scheduler.get_last_lr()
            scheduler.step(step)

            # Quick indication that training is happening.
            logging.log_first_n(logging.INFO, "Finished training step %d.", 10, step)

            # Log, if required.
            is_last_step = step == config.num_train_steps - 1
            if step % config.log_every_steps == 0 or is_last_step:
                wandb.log(
                    {
                        "train_mape": torch.tensor(total_loss_mape).mean().item(),
                        "train_huber": torch.tensor(total_loss_huber).mean().item(),
                        "train_lr": torch.tensor(lr).mean().item(),
                    },
                    step=step,
                )
                total_loss_mape = []
                total_loss_huber = []
                lr = []

            # Checkpoint model, if required.
            if step % config.checkpoint_every_steps == 0 or is_last_step:
                savemodel(model, optimizer, ckp_path, step)

            # Evaluate on validation.
            if step % config.eval_every_steps == 0 or (is_last_step):
                mape_den, huber_den, mape_vp, huber_vp = test(test="val")
                wandb.log(
                    {
                        "mape_den": mape_den,
                        "huber_den": huber_den,
                        "mape_vp": mape_vp,
                        "huber_vp": huber_vp,
                    },
                    step=step,
                )
                model.train()

            step += 1
            if step > config.num_train_steps:
                break
    wandb.finish()


def savemodel(model, optimizer, path, step):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
        },
        path,
    )


FLAGS = flags.FLAGS

flags.DEFINE_string("workdir", None, "Working Directory.")
flags.DEFINE_string("dataset", None, "Dataset to train model on")
config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=True,
)


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    logging.info("Calling train and evaluate")

    train_and_evaluate(FLAGS.config, FLAGS.workdir, FLAGS.dataset)


if __name__ == "__main__":
    flags.mark_flags_as_required(["config", "workdir", "dataset"])
    app.run(main)
