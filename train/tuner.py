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

from data.graphdataset import ThermoMLDataset, ramirez, ThermoMLpara

from train.model_deg import calc_deg

from ray import tune, air
import ray
from ray.air import session
from ray.air.checkpoint import Checkpoint
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
from functools import partial


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


def train_and_evaluate(
    config_tuner: dict, config: ml_collections.ConfigDict, workdir: str, dataset: str
):
    """Execute model training and evaluation loop.

    Args:
      config: Hyperparameter configuration for training and evaluation.
      workdir: Working Directory.
    """
    deg = calc_deg(dataset, workdir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config.propagation_depth = config_tuner["propagation_depth"]
    config.hidden_dim = config_tuner["hidden_dim"]
    config.num_mlp_layers = config_tuner["num_mlp_layers"]
    config.pre_layers = config_tuner["pre_layers"]
    config.post_layers = config_tuner["post_layers"]
    config.warmup_steps = config_tuner["warmup_steps"]
    config.weight_decay = config_tuner["weight_decay"]
    config.batch_size = config_tuner["batch_size"]
    config.learning_rate = config_tuner["learning_rate"]

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
    initial_step = 1
    checkpoint = session.get_checkpoint()
    if checkpoint:
        checkpoint = checkpoint.to_dict()
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        step = checkpoint["step"]
        initial_step = int(step) + 1

    # Scheduler
    scheduler = CosineAnnealingWarmRestarts(optimizer, config.warmup_steps)

    @torch.no_grad()
    def test(test="test"):
        model.eval()
        total_mape_den = []
        total_huber_den = []
        for graphs in test_loader:
            if test == "test":
                if graphs.InChI[0] in para_data:
                    continue
            if test == "val":
                if graphs.InChI[0] not in para_data:
                    continue
            datapoints = graphs.rho.to("cpu", torch.float64).view(-1, 5)
            if ~torch.all(datapoints == torch.zeros_like(datapoints)):
                graphs = graphs.to(device)
                pred_para = model(graphs).squeeze().to("cpu", torch.float64)
                pred = pcsaft_den(pred_para, datapoints)
                target = datapoints[:, -1]
                loss_mape = mape(pred, target)
                loss_huber = HLoss(pred, target)
                total_mape_den += [loss_mape.item()]
                total_huber_den += [loss_huber.item()]

        return (
            torch.tensor(total_mape_den).nanmean().item(),
            torch.tensor(total_huber_den).nanmean().item(),
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
            scheduler.step()

            # Log
            if step % config.log_every_steps == 0:
                mape_den, huber_den = test(test="val")
                checkpoint = Checkpoint.from_dict(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "step": step,
                    }
                )
                session.report(
                    {
                        "train_mape": torch.tensor(total_loss_mape).mean().item(),
                        "train_huber": torch.tensor(total_loss_huber).mean().item(),
                        "train_lr": torch.tensor(lr).mean().item(),
                        "mape_den": mape_den,
                        "huber_den": huber_den,
                    },
                    checkpoint=checkpoint,
                )
                total_loss_mape = []
                total_loss_huber = []
                lr = []

            step += 1
            if step > config.num_train_steps:
                break


FLAGS = flags.FLAGS

flags.DEFINE_string("workdir", None, "Working Directory")
flags.DEFINE_string("restoredir", None, "Restore Directory")
flags.DEFINE_string("dataset", None, "Dataset to train model on")
flags.DEFINE_integer("num_cpu", 1, "Number of CPU threads")
flags.DEFINE_integer("num_gpus", 1, "Number of GPUs")
flags.DEFINE_integer("num_samples", 100, "Number of trials")
config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=True,
)


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    logging.info(f"config file below: \n{FLAGS.config}")

    logging.info("Calling tuner")

    config = FLAGS.config

    ptrain = partial(
        train_and_evaluate, config=config, workdir=FLAGS.workdir, dataset=FLAGS.dataset
    )
    search_space = {
        "propagation_depth": tune.choice([3, 4, 5, 6, 7]),
        "hidden_dim": tune.choice([64, 128, 256, 512]),
        "num_mlp_layers": tune.choice([1, 2, 3]),
        "pre_layers": tune.choice([1, 2, 3]),
        "post_layers": tune.choice([1, 2, 3]),
        "warmup_steps": tune.choice([100, 500, 1000]),
        "batch_size": tune.choice([64, 128, 256, 512]),
        "weight_decay": tune.loguniform(1e-9, 1e-2),
        "learning_rate": tune.loguniform(1e-9, 1e-2),
    }
    max_t = config.num_train_steps // config.log_every_steps - 1

    search_alg = TuneBOHB(metric="mape_den", mode="min", seed=77)
    scheduler = HyperBandForBOHB(
        metric="mape_den",
        mode="min",
        max_t=max_t,
        stop_last_trials=True,
    )

    ray.init(num_gpus=FLAGS.num_gpus, num_cpus=FLAGS.num_cpu)
    resources = {"cpu": FLAGS.num_cpu, "gpu": FLAGS.num_gpus}

    if FLAGS.restoredir:
        tuner = tune.Tuner.restore(
            FLAGS.restoredir,
            tune.with_resources(tune.with_parameters(ptrain), resources=resources),
            resume_unfinished=True,
            resume_errored=False,
            restart_errored=False,
        )
    else:
        tuner = tune.Tuner(
            tune.with_resources(tune.with_parameters(ptrain), resources=resources),
            param_space=search_space,
            tune_config=tune.TuneConfig(
                search_alg=search_alg,
                scheduler=scheduler,
                num_samples=FLAGS.num_samples,
            ),
            run_config=air.RunConfig(
                storage_path="./ray",
                verbose=1,
                checkpoint_config=air.CheckpointConfig(
                    num_to_keep=1, checkpoint_at_end=False
                ),
            ),
        )

    result = tuner.fit()

    best_trial = result.get_best_result(
        metric="mape_den",
        mode="min",
    )
    print(f"\nBest trial config:\n {best_trial.config}")
    print(f"\nBest trial final metrics:\n {best_trial.metrics}")


if __name__ == "__main__":
    flags.mark_flags_as_required(["config", "workdir", "dataset"])
    app.run(main)
