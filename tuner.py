import os.path as osp
from absl import app
from absl import flags
from ml_collections import config_flags
from functools import partial

from absl import logging
import ml_collections

import models

import torch
from torch.nn import HuberLoss
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
    ChainedScheduler,
)
from torch_geometric.loader import DataLoader
from torchmetrics import MeanAbsolutePercentageError

import epcsaft_cython
import jax

from graphdataset import ThermoMLDataset, ThermoML_padded, ramirez
import pickle

from ray import tune, air
import ray
from ray.air import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler

from model_deg import deg


def create_model(config: ml_collections.ConfigDict) -> torch.nn.Module:
    """Creates a model, as specified by the config."""
    platform = jax.local_devices()[0].platform
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.half_precision:
        if platform == "tpu":
            model_dtype = torch.bfloat16
        else:
            model_dtype = torch.float16
    else:
        model_dtype = torch.float32
    if config.model == "PNA":
        return models.PNA(
            hidden_dim=config.hidden_dim,
            propagation_depth=config.propagation_depth,
            pre_layers=config.pre_layers,
            post_layers=config.post_layers,
            num_mlp_layers=config.num_mlp_layers,
            num_para=config.num_para,
            deg=deg,
            layer_norm=config.layer_norm,
            dtype=model_dtype,
            device=device,
        )
    raise ValueError(f"Unsupported model: {config.model}.")


def create_optimizer(config: ml_collections.ConfigDict, params):
    """Creates an optimizer, as specified by the config."""
    if config.optimizer == "adam":
        return torch.optim.AdamW(
            params, lr=config.learning_rate, weight_decay=1e-2, amsgrad=True, eps=1e-5
        )
    if config.optimizer == "sgd":
        return torch.optim.SGD(
            params,
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=1e-2,
            nesterov=True,
        )
    raise ValueError(f"Unsupported optimizer: {config.optimizer}.")


def train_and_evaluate(
    config_tuner: dict, config: ml_collections.ConfigDict, workdir: str
):
    """Execute model training and evaluation loop.

    Args:
      config: Hyperparameter configuration for training and evaluation.
      workdir: Working Directory.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    platform = jax.local_devices()[0].platform

    config.propagation_depth = config_tuner["propagation_depth"]
    config.hidden_dim = config_tuner["hidden_dim"]
    config.num_mlp_layers = config_tuner["num_mlp_layers"]
    config.pre_layers = config_tuner["pre_layers"]
    config.post_layers = config_tuner["post_layers"]
    config.warmup_steps = config_tuner["warmup_steps"]

    # Get datasets, organized by split.
    logging.info("Obtaining datasets.")
    if config.half_precision:
        model_dtype = torch.float16
    else:
        model_dtype = torch.float32

    path = osp.join(workdir, "data/ramirez2022")
    train_dataset = ramirez(path)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    test_dataset = ThermoMLDataset(osp.join(workdir, "data/thermoml"))
    test_dataset = ThermoML_padded(test_dataset, 4096 * 2)
    test_loader = DataLoader(test_dataset)

    ra_data = {}
    for graph in train_loader:
        for inchi, para in zip(graph.InChI, graph.para):
            ra_data[inchi] = para

    # Create and initialize the network.
    logging.info("Initializing network.")
    model = create_model(config).to(device, model_dtype)
    pcsaft_den = epcsaft_cython.PCSAFT_den.apply
    pcsaft_vp = epcsaft_cython.PCSAFT_vp.apply
    HLoss = HuberLoss("mean").to(device)
    mape = MeanAbsolutePercentageError().to(device)

    # Create the optimizer.
    optimizer = create_optimizer(config, model.parameters())

    # Set up checkpointing of the model.
    ckp_path = osp.join(workdir, "training/last_checkpoint.pth")
    initial_step = 1

    # Scheduler
    scheduler = CosineAnnealingWarmRestarts(optimizer, config.warmup_steps)

    @torch.no_grad()
    def test(test="test"):
        model.eval()
        total_mape_den = []
        total_huber_den = []
        total_mape_vp = []
        total_huber_vp = []
        for graphs in test_loader:
            if test == "test":
                if graphs.InChI[0] in ra_data:
                    continue
            if test == "val":
                if graphs.InChI[0] not in ra_data:
                    continue
            datapoints = graphs.rho.to("cpu", model_dtype)
            graphs = graphs.to(device)
            pred_para = model(graphs).squeeze().to("cpu")
            pred = pcsaft_den(pred_para, datapoints)
            target = datapoints[:, -1]
            loss_mape = mape(pred, target)
            loss_huber = HLoss(pred, target)
            total_mape_den += [loss_mape.item()]
            total_huber_den += [loss_huber.item()]

            datapoints = graphs.vp.to(device, model_dtype)
            if torch.all(datapoints == torch.zeros_like(datapoints)):
                continue
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
            target = graphs.para.to(device, model_dtype)
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
            scheduler.step()

            # Log
            if step % config.log_every_steps == 0:
                mape_den, huber_den, mape_vp, huber_vp = test(test="val")
                session.report(
                    {
                        "train_mape": torch.tensor(total_loss_mape).mean().item(),
                        "train_huber": torch.tensor(total_loss_huber).mean().item(),
                        "train_lr": torch.tensor(lr).mean().item(),
                        "mape_den": mape_den,
                        "huber_den": huber_den,
                        "mape_vp": mape_vp,
                        "huber_vp": huber_vp,
                    },
                )
                total_loss_mape = []
                total_loss_huber = []
                lr = []

            step += 1
            if step > config.num_train_steps:
                break


FLAGS = flags.FLAGS

flags.DEFINE_string("workdir", None, "Working Directory")
config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=True,
)


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    logging.info("JAX host: %d / %d", jax.process_index(), jax.process_count())
    logging.info("JAX local devices: %r", jax.local_devices())

    logging.info("Calling tuner")

    ptrain = partial(train_and_evaluate, config=FLAGS.config, workdir=FLAGS.workdir)
    search_space = {
        "propagation_depth": tune.choice([3, 4, 5, 6, 7]),
        "hidden_dim": tune.choice([64, 128, 256, 512]),
        "num_mlp_layers": tune.choice([1, 2, 3]),
        "pre_layers": tune.choice([1, 2, 3]),
        "post_layers": tune.choice([1, 2, 3]),
        "warmup_steps": tune.choice([100, 500, 1000, 2000]),
    }
    scheduler = ASHAScheduler(
        metric="mape_den",
        mode="min",
        max_t=100,
        grace_period=30,
        reduction_factor=2,
    )

    ray.init(num_gpus=1)

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(ptrain), resources={"cpu": 2, "gpu": 1}
        ),
        param_space=search_space,
        tune_config=tune.TuneConfig(scheduler=scheduler, num_samples=100),
        run_config=air.RunConfig(storage_path="./ray", verbose=1),
    )

    result = tuner.fit()

    best_trial = result.get_best_result(
        metric="mape_den",
        mode="min",
    )
    print(f"\nBest trial config:\n {best_trial.config}")
    print(f"\nBest trial final metrics:\n {best_trial.metrics}")


if __name__ == "__main__":
    flags.mark_flags_as_required(["config", "workdir"])
    app.run(main)
