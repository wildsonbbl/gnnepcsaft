"""Module to be used for model training"""

import os
import os.path as osp
import time
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
from torch.nn import HuberLoss
from torch_geometric.loader import DataLoader
from torch_geometric.nn import summary
from torchmetrics import MeanAbsolutePercentageError

from ..configs.configs_parallel import get_configs
from ..epcsaft import epcsaft_cython
from . import models
from .utils import (
    CustomRayTrainReportCallback,
    EpochTimer,
    VpOff,
    build_datasets_loaders,
    build_test_dataset,
    build_train_dataset,
    calc_deg,
    create_model,
    create_optimizer,
    create_schedulers,
    input_artifacts,
    load_checkpoint,
    output_artifacts,
    savemodel,
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


# pylint: disable=R0914
# pylint: disable=R0915
def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str, dataset: str):
    """Execute model training and evaluation loop.

    Args:
      config: Hyperparameter configuration for training and evaluation.
      workdir: Working Directory.
      dataset: dataset name (ramirez or thermoml)
    """

    deg = calc_deg(dataset, workdir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = config.amp
    # Create writer for logs.
    create_logger(config, dataset)
    input_artifacts(workdir, dataset)

    # Get datasets, organized by split.
    logging.info("Obtaining datasets.")
    train_loader, test_loader, para_data = build_datasets_loaders(
        config, workdir, dataset
    )

    # Create and initialize the network.
    logging.info("Initializing network.")
    model = create_model(config, deg)
    model.to(device)
    wandb.watch(model, log="all", log_graph=True)
    # pylint: disable=no-member
    pcsaft_den = epcsaft_cython.DenFromTensor.apply
    pcsaft_vp = epcsaft_cython.VpFromTensor.apply
    hloss = HuberLoss("mean")
    hloss.to(device)
    mape = MeanAbsolutePercentageError()
    mape.to(device)
    dummy = test_loader[0]
    dummy.to(device)
    logging.info(f"Model summary: \n {summary(model, dummy)}")

    # Create the optimizer.
    optimizer = create_optimizer(config, model.parameters())
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Set up checkpointing of the model.
    ckp_path, step = load_checkpoint(config, workdir, model, optimizer, scaler, device)

    # Scheduler
    scheduler, scheduler2 = create_schedulers(config, optimizer)

    @torch.no_grad()
    def test(test="test"):
        model.eval()
        total_mape_den = []
        total_huber_den = []
        total_mape_vp = []
        total_huber_vp = []
        for graphs in test_loader:
            if test == "test":
                if graphs.InChI in para_data:
                    continue
            if test == "val":
                if graphs.InChI not in para_data:
                    continue
            graphs = graphs.to(device)
            pred_para = model(graphs).squeeze().to("cpu", torch.float64)

            datapoints = graphs.rho.to("cpu", torch.float64).view(-1, 5)
            if ~torch.all(datapoints == torch.zeros_like(datapoints)):
                pred = pcsaft_den(pred_para, datapoints)
                target = datapoints[:, -1]
                # pylint: disable = not-callable
                loss_mape = mape(pred, target)
                loss_huber = hloss(pred, target)
                total_mape_den += [loss_mape.item()]
                total_huber_den += [loss_huber.item()]

            datapoints = graphs.vp.to("cpu", torch.float64).view(-1, 5)
            if ~torch.all(datapoints == torch.zeros_like(datapoints)):
                pred = pcsaft_vp(pred_para, datapoints)
                target = datapoints[:, -1]
                result_filter = ~torch.isnan(pred)
                # pylint: disable = not-callable
                loss_mape = mape(pred[result_filter], target[result_filter])
                loss_huber = hloss(pred[result_filter], target[result_filter])
                if loss_mape.item() >= 0.9:
                    continue
                total_mape_vp += [loss_mape.item()]
                total_huber_vp += [loss_huber.item()]

        return (
            torch.tensor(total_mape_den).nanmean().item(),
            torch.tensor(total_huber_den).nanmean().item(),
            torch.tensor(total_mape_vp).nanmean().item(),
            torch.tensor(total_huber_vp).nanmean().item(),
        )

    def test_logging():
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

    # Begin training loop.
    logging.info("Starting training.")
    total_loss_mape = []
    total_loss_huber = []
    lr = []
    start_time = time.time()

    def train_step(graphs):
        target = graphs.para.to(device).view(-1, 3)
        graphs = graphs.to(device)
        optimizer.zero_grad()
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
            enabled=use_amp,
        ):
            pred = model(graphs)
            loss_mape = mape(pred, target)
            loss_huber = hloss(pred, target)
        scaler.scale(loss_mape).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss_mape.append(loss_mape.item())
        total_loss_huber.append(loss_huber.item())
        if config.change_sch:
            lr.append(optimizer.param_groups[0]["lr"])
        else:
            lr.append(scheduler.get_last_lr()[0])
        scheduler.step(step)
        return pred

    def train_logging():
        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.log_first_n(
            logging.INFO, "Elapsed time %.4f min.", 20, elapsed_time / 60
        )
        wandb.log(
            {
                "train_mape": torch.tensor(total_loss_mape).mean().item(),
                "train_huber": torch.tensor(total_loss_huber).mean().item(),
                "train_lr": torch.tensor(lr).mean().item(),
            },
            step=step,
        )
        scheduler2.step(torch.tensor(total_loss_huber).mean())

    model.train()
    while step < config.num_train_steps + 1:
        for graphs in train_loader:
            pred = train_step(graphs)

            # Quick indication that training is happening.
            logging.log_first_n(logging.INFO, "Finished training step %d.", 10, step)

            # Log, if required.
            is_last_step = step == config.num_train_steps
            if step % config.log_every_steps == 0 or is_last_step:
                train_logging()
                total_loss_mape = []
                total_loss_huber = []
                lr = []
                start_time = time.time()

            # Checkpoint model, if required.
            if (step % config.checkpoint_every_steps == 0 or is_last_step) and (
                ~torch.any(torch.isnan(pred))
            ):
                savemodel(model, optimizer, scaler, ckp_path, step)

            # Evaluate on validation.
            if step % config.eval_every_steps == 0 or is_last_step:
                test_logging()
                model.train()

            step += 1
            if step > config.num_train_steps or (torch.any(torch.isnan(pred))):
                output_artifacts(workdir)
                wandb.finish()
                break
        if torch.any(torch.isnan(pred)):
            break


def ltrain_and_evaluate(config: ml_collections.ConfigDict, workdir: str, dataset: str):
    """Execute model training and evaluation loop with lightning."""
    job_type = config.job_type
    # Dataset building
    transform = None if job_type == "train" else VpOff()
    train_dataset = build_train_dataset(workdir, dataset)
    tml_dataset, para_data = build_test_dataset(
        workdir, train_dataset, transform=transform
    )
    test_idx = []
    val_idx = []
    # separate test and val dataset
    for idx, graph in enumerate(tml_dataset):
        if graph.InChI in para_data:
            val_idx.append(idx)
        else:
            test_idx.append(idx)
    test_dataset = tml_dataset[test_idx]
    val_dataset = tml_dataset[val_idx]
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
    )

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
            ckpt = torch.load(ckpt_path)
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
    if job_type == "train":
        wandb.finish()

        logging.info("Testing run!")
        # Logging test results at wandb
        trainer.logger = WandbLogger(
            log_model=True,
            # Set the project where this run will be logged
            project="gnn-pc-saft",
            # Track hyperparameters and run metadata
            config=config.to_dict(),
            group=dataset,
            tags=[dataset, "eval", config.model_name],
            job_type="eval",
        )

        trainer.test(model, test_dataset)
        wandb.finish()


def training_parallel(
    train_loop_config: dict,
    config: ml_collections.ConfigDict,
    workdir: str,
    dataset: str,
):
    """Execute model training and evaluation loop in parallel.

    Args:
      train_loop_config: Dict with hyperparameter configuration
      for training and evaluation in each worker.
    """

    local_rank = str(train.get_context().get_local_rank())
    train_config = train_loop_config[local_rank]
    config.model_name = config.model_name + "_" + local_rank
    # selected hyperparameters to test
    training_updated(train_config, config, workdir, dataset)


def training_updated(
    train_config: dict, config: ml_collections.ConfigDict, workdir: str, dataset: str
):
    """Execute model training and evaluation loop with updated config.

    Args:
      train_config: Updated hyperparameter configuration for training and evaluation.

    """

    config.propagation_depth = train_config["propagation_depth"]
    config.hidden_dim = train_config["hidden_dim"]
    config.num_mlp_layers = train_config["num_mlp_layers"]
    config.pre_layers = train_config["pre_layers"]
    config.post_layers = train_config["post_layers"]
    config.skip_connections = train_config["skip_connections"]
    config.add_self_loops = train_config["add_self_loops"]

    ltrain_and_evaluate(config, workdir, dataset)


# pylint: disable=R0913
def torch_trainer_config(
    num_workers: int,
    num_cpu: float,
    num_gpus: float,
    num_cpu_trainer: float,
    verbose: int,
    dataset: str,
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
        log_to_file=True,
        stop=None,
        callbacks=(
            [
                WandbLoggerCallback(
                    "gnn-pc-saft",
                    dataset,
                    tags=["tuning", dataset] + tags,
                    config=config.to_dict(),
                )
            ]
            if config.job_type == "tuning"
            else None
        ),
    )

    return scaling_config, run_config


FLAGS = flags.FLAGS

flags.DEFINE_string("workdir", None, "Working Directory.")
flags.DEFINE_string("dataset", None, "Dataset to train model on")
flags.DEFINE_string(
    "framework", "lightning", "Define framework to run: pytorch, lightning or ray."
)
flags.DEFINE_list("tags", [], "wandb tags")
flags.DEFINE_float("num_cpu", 1.0, "Fraction of CPU threads per trial")
flags.DEFINE_float("num_gpus", 1.0, "Fraction of GPUs per trial")
flags.DEFINE_float("num_cpu_trainer", 1.0, "Fraction of CPUs for trainer resources")
flags.DEFINE_integer("num_workers", 1, "number of workers")
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
        ltrain_and_evaluate(FLAGS.config, FLAGS.workdir, FLAGS.dataset)
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
            FLAGS.dataset,
            FLAGS.config,
            FLAGS.tags,
        )
        trainer = TorchTrainer(
            partial(
                training_parallel,
                config=FLAGS.config,
                workdir=FLAGS.workdir,
                dataset=FLAGS.dataset,
            ),
            train_loop_config=train_loop_config,
            scaling_config=scaling_config,
            run_config=run_config,
        )

        result = trainer.fit()
        print(result.metrics_dataframe)
    else:
        train_and_evaluate(FLAGS.config, FLAGS.workdir, FLAGS.dataset)


if __name__ == "__main__":
    flags.mark_flags_as_required(["config", "workdir", "dataset"])
    app.run(main)
