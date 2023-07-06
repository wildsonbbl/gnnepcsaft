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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch_geometric.loader import DataLoader

import ml_pc_saft
import jax

from graphdataset import ThermoMLDataset, ThermoML_padded, ramirez
import pickle

from ray import tune
from ray.air import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler

deg = torch.tensor([78, 5572, 8525, 2569, 602, 1, 2])


def create_model(config: ml_collections.ConfigDict) -> torch.nn.Module:
    """Creates a Flax model, as specified by the config."""
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
      workdir: Directory where the TensorBoard summaries are written to.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    platform = jax.local_devices()[0].platform
    # Create writer for logs.
    config.propagation_depth = config_tuner["propagation_depth"]
    config.hidden_dim = config_tuner["hidden_dim"]
    config.num_mlp_layers = config_tuner["num_mlp_layers"]
    config.pre_layers = config_tuner["pre_layers"]
    config.post_layers = config_tuner["post_layers"]

    # Get datasets, organized by split.

    if config.half_precision:
        model_dtype = torch.float16
    else:
        model_dtype = torch.float32

    path = "./data/thermoml"
    val_dataset = ThermoMLDataset(path, subset="train")
    test_dataset = ThermoMLDataset(path, subset="test")

    val_dataset = ThermoML_padded(val_dataset, config.pad_size)
    test_dataset = ThermoML_padded(test_dataset, 16)

    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    train_dataset = ramirez("./data/ramirez2022")
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    model = create_model(config).to(device, model_dtype)
    pcsaft_den = ml_pc_saft.PCSAFT_den.apply
    pcsaft_vp = ml_pc_saft.PCSAFT_vp.apply
    lossfn = HuberLoss("mean").to(device)

    # Create the optimizer.
    optimizer = create_optimizer(config, model.parameters())

    # Scheduler
    warm_up = config.warmup_steps
    scheduler = CosineAnnealingWarmRestarts(optimizer, warm_up)

    # test fn
    @torch.no_grad()
    def test(loader):
        model.eval()
        total_loss = []
        pad_size = config.pad_size
        para = config.num_para
        for graphs in loader:
            graphs = graphs.to(device)
            datapoints = graphs.vp.view(-1, 5)
            datapoints = datapoints.to(device)
            parameters = model(graphs).to(torch.float64)
            pred_y = pcsaft_vp(parameters, datapoints)
            y = datapoints[:, -1]
            loss = torch.square(pred_y - y).mean()
            total_loss += [loss.item()]

        return torch.tensor(total_loss).nanmean().item()

    # Begin training loop.
    step = 1
    lr = []
    model.train()
    unitscale = torch.tensor(
        [[1.0, 1.0, 1.0e2, 1.0e-3, 1.0e3, 1.0, 1.0, 1.0, 1.0]], device=device
    )
    while step < config.num_train_steps + 1:
        for graphs in train_loader:
            graphs = graphs.to(device)
            optimizer.zero_grad()
            pred = model(graphs)
            target = graphs.para.view(-1, 3).to(model_dtype)
            loss = lossfn(pred, target)
            loss.backward()
            optimizer.step()
            lr += scheduler.get_last_lr()
            scheduler.step()

            # Log
            session.report(
                {"train_HuberLoss": loss.item()},
            )

            step += 1
            if step > config.num_train_steps:
                break


FLAGS = flags.FLAGS

flags.DEFINE_string("workdir", None, "Directory to store model data.")
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
    config = {
        "propagation_depth": tune.choice([3, 4, 5, 6, 7]),
        "hidden_dim": tune.choice([64, 128, 256, 512]),
        "num_mlp_layers": tune.choice([1, 2, 3]),
        "pre_layers": tune.choice([1, 2, 3]),
        "post_layers": tune.choice([1, 2, 3]),
    }
    scheduler = ASHAScheduler(
        metric="train_HuberLoss",
        mode="min",
        max_t=6000,
        grace_period=1000,
        reduction_factor=2,
    )

    result = tune.run(
        ptrain,
        resources_per_trial={"cpu": 8, "gpu": 1},
        scheduler=scheduler,
        config=config,
        num_samples=20,
        storage_path="./ray",
        verbose=1,
    )

    best_trial = result.get_best_trial("train_HuberLoss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(
        f"Best trial final validation loss: {best_trial.last_result['train_HuberLoss']}"
    )


if __name__ == "__main__":
    flags.mark_flags_as_required(["config", "workdir"])
    app.run(main)
