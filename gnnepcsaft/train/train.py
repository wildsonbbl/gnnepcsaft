"""Module to be used for model training"""

import os
import os.path as osp
from functools import partial
from pathlib import Path
from typing import Any, Union

import lightning as L
import numpy as np
import torch
import wandb
from absl import app, flags, logging
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from ml_collections import config_flags
from ray import train, tune
from ray.train.torch import TorchTrainer
from torch_geometric.loader import DataLoader

from ..configs.configs_parallel import get_configs
from .models import GNNePCSAFTL, HabitchNNL, create_model
from .utils import (
    CustomRayTrainReportCallback,
    EpochTimer,
    build_test_dataset,
    build_train_dataset,
    calc_deg,
)


def create_logger(config, dataset):
    "Creates wandb logging or equivalent."
    wandb.init(
        # Set the project where this run will be logged
        project="gnn-pc-saft",
        # Track hyperparameters and run metadata
        config=config.to_dict(),
        group=dataset,
        tags=[dataset, "train"],
        job_type="train",
    )


def ltrain_and_evaluate(  # pylint:  disable=too-many-locals
    config: dict[str, Any], workdir: str
):
    """Execute model training and evaluation loop with lightning.

    Args:
      config: Hyperparameter configuration for training and evaluation.
      workdir: Working Directory.
    """
    torch.set_float32_matmul_precision("medium")
    # Dataset building
    train_dataset = build_train_dataset(workdir, config["dataset"])
    val_dataset, train_val_dataset = build_test_dataset(workdir, train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=os.cpu_count(),
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=len(val_dataset),
        num_workers=os.cpu_count(),
    )
    train_val_dataloader = DataLoader(
        train_val_dataset,
        batch_size=len(train_val_dataset),
        num_workers=os.cpu_count(),
    )

    # trainer callback and logger
    callbacks, logger = get_callbacks_and_logger(config, workdir)

    # creating model from config
    model = create_model(config, calc_deg(config["dataset"], workdir))
    # model = torch.compile(model, dynamic=True) off for old gpus

    # creating Lighting trainer function
    trainer = L.Trainer(
        devices="auto",
        accelerator=config["accelerator"],
        strategy="auto",
        max_steps=config["num_train_steps"],
        log_every_n_steps=config["log_every_steps"],
        val_check_interval=config["eval_every_steps"],
        check_val_every_n_epoch=None,
        num_sanity_val_steps=0,
        callbacks=callbacks,
        logger=logger,
        plugins=None,
        enable_progress_bar=False,
        enable_checkpointing=config["job_type"] == "train",
    )

    ckpt_path = get_ckpt_path(config, workdir, model, logger)

    # training run
    logging.info("Training run!")
    trainer.fit(
        model,
        train_loader,
        (
            [train_val_dataloader, val_dataloader]
            if config["dataset"] == "esper"
            else [train_val_dataloader, train_val_dataloader]
        ),
        ckpt_path=ckpt_path,
    )

    if config["job_type"] == "train":
        wandb.finish()


def get_ckpt_path(
    config: dict[str, Any],
    workdir: str,
    model: Union[GNNePCSAFTL, HabitchNNL],
    logger: Union[WandbLogger, None],
):
    "gets checkpoint path for resuming training"
    ckpt_path = None
    if config["job_type"] == "tuning":

        checkpoint: tune.Checkpoint = tune.get_checkpoint()
        trial_id = tune.get_context().get_trial_id()

        if checkpoint:
            with checkpoint.as_directory() as ckpt_dir:
                ckpt_path = Path(ckpt_dir) / f"{trial_id}.ckpt"
        elif config["checkpoint"]:
            ckpt_path = Path(workdir) / f"train/checkpoints/{config['checkpoint']}"
    elif config["checkpoint"]:
        if logger:
            ckpt_dir = Path(workdir) / f"train/checkpoints/{config['model_name']}"
            artifact = logger.use_artifact(config["checkpoint"], "model")
            artifact.download(ckpt_dir)
            ckpt_path = ckpt_dir / "model.ckpt"
            if config["change_opt"]:

                ckpt = torch.load(ckpt_path, weights_only=True)
                model.load_state_dict(ckpt["state_dict"])

                ckpt_path = None
    return ckpt_path


def get_callbacks_and_logger(config, workdir):
    """Creates callbacks and logger for training."""
    callbacks = []
    job_type = config.job_type
    dataset = config.dataset

    if job_type == "train":
        # Checkpointing from val loss (mape_den) and train loss (train_mape)
        checkpoint_mape_den = ModelCheckpoint(
            dirpath=osp.join(workdir, "train/checkpoints"),
            filename=config.model_name + "-{epoch}-mape_den_train",
            save_last=False,
            monitor="mape_den/dataloader_idx_0",
            save_top_k=1,
            verbose=True,
        )
        callbacks.append(checkpoint_mape_den)

        checkpoint_mape_den_2 = ModelCheckpoint(
            dirpath=osp.join(workdir, "train/checkpoints"),
            filename=config.model_name + "-{epoch}-mape_den_val",
            save_last=False,
            monitor="mape_den/dataloader_idx_1",
            save_top_k=1,
            verbose=True,
        )
        callbacks.append(checkpoint_mape_den_2)

        checkpoint_train_loss = ModelCheckpoint(
            dirpath=osp.join(workdir, "train/checkpoints"),
            filename=config.model_name + "-{epoch}-{train_mape:.4f}",
            save_last=False,
            monitor="train_mape",
            save_top_k=1,
            verbose=True,
        )
        callbacks.append(checkpoint_train_loss)

        checkpoint_last_epoch = ModelCheckpoint(
            dirpath=osp.join(workdir, "train/checkpoints"),
            filename=config.model_name + "-{epoch}-{step}",
            verbose=True,
        )
        callbacks.append(checkpoint_last_epoch)

        callbacks.append(EpochTimer())

        # Logging training results at wandb
        logger = WandbLogger(
            log_model=True,
            # Set the project where this run will be logged
            project="gnn-pc-saft",
            id=config.resume_id if config.resume_id else None,
            # Track hyperparameters and run metadata
            config=config.to_dict(),
            group=dataset,
            tags=[dataset, "train", config.model_name],
            job_type="train",
            resume="must" if config.resume_id else "allow",
        )
    else:
        callbacks.append(CustomRayTrainReportCallback())
        logger = None
    return callbacks, logger


def training_parallel(
    train_loop_config: dict[str, Any],
    config: dict[str, Any],
    workdir: str,
):
    """Execute model training and evaluation loop in parallel with ray.

    Args:
      train_loop_config: Dict with hyperparameter configuration
      for training and evaluation in each worker.
    """

    local_rank = str(train.get_context().get_local_rank())
    train_config = train_loop_config[local_rank]
    # selected hyperparameters to test
    training_updated(train_config, config, workdir)


def training_updated(
    train_config: dict[str, Any],
    config: dict[str, Any],
    workdir: str,
):
    """Execute model training and evaluation loop with updated config.

    Args:
      train_config: Updated hyperparameter configuration for training and evaluation.

    """

    for hparam in train_config:
        value = train_config[hparam]
        if isinstance(value, (np.signedinteger,)):
            value = int(value)
        if isinstance(value, (np.floating,)):
            value = float(value)
        config[hparam] = value

    ltrain_and_evaluate(config, workdir)


FLAGS = flags.FLAGS

flags.DEFINE_string("workdir", None, "Working Directory.")
flags.DEFINE_string(
    "framework", "lightning", "Define framework to run: lightning or ray."
)
flags.DEFINE_list("tags", [], "wandb tags")
flags.DEFINE_float("num_cpu", 1.0, "Fraction of CPU threads per trial for ray")
flags.DEFINE_float("num_gpus", 1.0, "Fraction of GPUs per trial for ray")
flags.DEFINE_integer("num_workers", 1, "Number of workers for ray")
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

    logging.info("Calling train and evaluate!")
    torch.set_float32_matmul_precision("medium")

    if FLAGS.framework == "lightning":
        ltrain_and_evaluate(FLAGS.config, FLAGS.workdir)
    elif FLAGS.framework == "ray":
        test_configs = get_configs()
        train_loop_config = {
            str(local_rank): test_configs[local_rank]
            for local_rank in range(FLAGS.num_workers)
        }
        scaling_config = train.ScalingConfig(
            num_workers=FLAGS.num_workers,
            use_gpu=True,
            resources_per_worker={"CPU": FLAGS.num_cpu, "GPU": FLAGS.num_gpus},
        )
        run_config = train.RunConfig(
            name="gnnpcsaft",
            storage_path=None,
            callbacks=(None),
        )
        trainer = TorchTrainer(
            partial(
                training_parallel,
                config=FLAGS.config,
                workdir=FLAGS.workdir,
            ),
            train_loop_config=train_loop_config,
            scaling_config=scaling_config,
            run_config=run_config,
        )

        result = trainer.fit()
        print(result.metrics_dataframe)


if __name__ == "__main__":
    flags.mark_flags_as_required(["config", "workdir"])
    app.run(main)
