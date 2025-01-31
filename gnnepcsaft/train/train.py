"""Module to be used for model training"""

import os
import os.path as osp
from functools import partial

import lightning as L
import ml_collections
import torch
import wandb
from absl import app, flags, logging
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from ml_collections import config_flags
from ray import train
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.train import Checkpoint
from ray.train.lightning import RayDDPStrategy, RayLightningEnvironment, prepare_trainer
from ray.train.torch import TorchTrainer
from torch_geometric.loader import DataLoader

from ..configs.configs_parallel import get_configs
from . import models
from .utils import (
    CustomRayTrainReportCallback,
    EpochTimer,
    VpOff,
    build_test_dataset,
    build_train_dataset,
    calc_deg,
    create_model,
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


# pylint: disable=R0914,R0915
def ltrain_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
    """Execute model training and evaluation loop with lightning.

    Args:
      config: Hyperparameter configuration for training and evaluation.
      workdir: Working Directory.
    """
    job_type = config.job_type
    dataset = config.dataset
    # Dataset building
    transform = None if job_type == "train" else VpOff()
    train_dataset = build_train_dataset(workdir, dataset)
    val_dataset, _ = build_test_dataset(workdir, train_dataset, transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
    )

    val_dataset = DataLoader(val_dataset)

    # trainer callback and logger
    callbacks = []

    if job_type == "train":
        # Checkpointing from val loss (mape_den) and train loss (train_mape)
        checkpoint_mape_den = ModelCheckpoint(
            dirpath=osp.join(workdir, "train/checkpoints"),
            filename=config.model_name + "-{epoch}-{mape_den:.4f}",
            save_last=False,
            monitor="mape_den",
            save_top_k=1,
            every_n_train_steps=config.checkpoint_every_steps,
            verbose=True,
            save_on_train_epoch_end=False,
        )
        callbacks.append(checkpoint_mape_den)

        checkpoint_train_loss = ModelCheckpoint(
            dirpath=osp.join(workdir, "train/checkpoints"),
            filename=config.model_name + "-{epoch}-{train_mape:.4f}",
            save_last=False,
            monitor="train_mape",
            save_top_k=1,
            every_n_train_steps=config.log_every_steps,
            verbose=True,
            save_on_train_epoch_end=True,
        )
        callbacks.append(checkpoint_train_loss)

        epoch_timer = EpochTimer()
        callbacks.append(epoch_timer)

        # Logging training results at wandb
        logger = WandbLogger(
            log_model=True,
            # Set the project where this run will be logged
            project="gnn-pc-saft",
            # Track hyperparameters and run metadata
            config=config.to_dict(),
            group=dataset,
            tags=[dataset, "train", config.model_name],
            job_type="train",
        )
    else:
        callbacks.append(CustomRayTrainReportCallback())
        logger = None

    # creating model from config
    deg = calc_deg(dataset, workdir)
    model: models.PNApcsaftL = create_model(config, deg)
    # model = torch.compile(model, dynamic=True) off for old gpus

    # Trainer configs
    if job_type == "train":
        strategy = "auto"
        plugins = None
        enable_checkpointing = True
    else:
        strategy = RayDDPStrategy()
        plugins = [RayLightningEnvironment()]
        enable_checkpointing = False

    # creating Lighting trainer function
    trainer = L.Trainer(
        devices="auto",
        accelerator=config.accelerator,
        strategy=strategy,
        max_steps=config.num_train_steps,
        log_every_n_steps=config.log_every_steps,
        val_check_interval=config.eval_every_steps,
        check_val_every_n_epoch=None,
        num_sanity_val_steps=0,
        callbacks=callbacks,
        logger=logger,
        plugins=plugins,
        enable_progress_bar=False,
        enable_checkpointing=enable_checkpointing,
    )

    ckpt_path = None
    if job_type == "tuning":
        trainer: L.Trainer = prepare_trainer(trainer)

        checkpoint: Checkpoint = train.get_checkpoint()
        trial_id = train.get_context().get_trial_id()

        if checkpoint:
            with checkpoint.as_directory() as ckpt_dir:
                ckpt_path = osp.join(ckpt_dir, f"{trial_id}.pt")
    elif config.checkpoint:
        ckpt_path = osp.join(workdir, f"train/checkpoints/{config.checkpoint}")
        if config.change_opt:
            # pylint: disable=E1120
            ckpt = torch.load(ckpt_path, weights_only=False)
            model.load_state_dict(ckpt["state_dict"])
            # pylint: enable=E1120
            ckpt_path = None

    # training run
    logging.info("Training run!")
    trainer.fit(
        model,
        train_loader,
        val_dataset,
        ckpt_path=ckpt_path,
    )
    # if job_type == "train":
    #     wandb.finish()

    #     logging.info("Testing run!")
    #     # Logging test results at wandb
    #     trainer.logger = WandbLogger(
    #         log_model=True,
    #         # Set the project where this run will be logged
    #         project="gnn-pc-saft",
    #         # Track hyperparameters and run metadata
    #         config=config.to_dict(),
    #         group=dataset,
    #         tags=[dataset, "eval", config.model_name],
    #         job_type="eval",
    #     )

    #     trainer.test(model, test_dataset)
    #     wandb.finish()


def training_parallel(
    train_loop_config: dict,
    config: ml_collections.ConfigDict,
    workdir: str,
):
    """Execute model training and evaluation loop in parallel with ray.

    Args:
      train_loop_config: Dict with hyperparameter configuration
      for training and evaluation in each worker.
    """

    local_rank = str(train.get_context().get_local_rank())
    train_config = train_loop_config[local_rank]
    config.model_name = config.model_name + "_" + local_rank
    # selected hyperparameters to test
    training_updated(train_config, config, workdir)


def training_updated(
    train_config: dict,
    config: ml_collections.ConfigDict,
    workdir: str,
):
    """Execute model training and evaluation loop with updated config.

    Args:
      train_config: Updated hyperparameter configuration for training and evaluation.

    """

    config.propagation_depth = int(train_config["propagation_depth"])
    config.hidden_dim = int(train_config["hidden_dim"])
    config.pre_layers = int(train_config["pre_layers"])
    config.post_layers = int(train_config["post_layers"])
    config.heads = int(train_config["heads"])

    ltrain_and_evaluate(config, workdir)


# pylint: disable=R0913,R0917
def torch_trainer_config(
    num_workers: int,
    num_cpu: float,
    num_gpus: float,
    num_cpu_trainer: float,
    verbose: int,
    config: ml_collections.ConfigDict,
    tags: list,
):
    """
    Builds torch trainer configs from ray train to run training in parallel.
    """
    scaling_config = train.ScalingConfig(
        num_workers=num_workers,
        use_gpu=True,
        resources_per_worker={"CPU": num_cpu, "GPU": num_gpus},
        trainer_resources={"CPU": num_cpu_trainer},
    )
    run_config = train.RunConfig(
        name="gnnpcsaft",
        storage_path=None,
        verbose=verbose,
        checkpoint_config=train.CheckpointConfig(
            num_to_keep=1,
        ),
        progress_reporter=None,
        log_to_file=False,
        stop=None,
        callbacks=(
            [
                WandbLoggerCallback(
                    "gnn-pc-saft",
                    config.dataset,
                    tags=["tuning", config.dataset] + tags,
                )
            ]
            if config.job_type == "tuning"
            else None
        ),
    )

    return scaling_config, run_config


FLAGS = flags.FLAGS

flags.DEFINE_string("workdir", None, "Working Directory.")
flags.DEFINE_string(
    "framework", "lightning", "Define framework to run: lightning or ray."
)
flags.DEFINE_list("tags", [], "wandb tags")
flags.DEFINE_float("num_cpu", 1.0, "Fraction of CPU threads per trial for ray")
flags.DEFINE_float("num_gpus", 1.0, "Fraction of GPUs per trial for ray")
flags.DEFINE_float(
    "num_cpu_trainer", 1.0, "Fraction of CPUs for trainer resources for ray"
)
flags.DEFINE_integer("num_workers", 1, "Number of workers for ray")
flags.DEFINE_integer("verbose", 0, "Ray train verbose")
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
        scaling_config, run_config = torch_trainer_config(
            FLAGS.num_workers,
            FLAGS.num_cpu,
            FLAGS.num_gpus,
            FLAGS.num_cpu_trainer,
            FLAGS.verbose,
            FLAGS.config,
            FLAGS.tags,
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
