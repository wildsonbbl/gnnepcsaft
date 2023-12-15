"""Module to be used for model training"""
import os
import os.path as osp
import time

import lightning as L
import ml_collections
import torch
from absl import app, flags, logging
from lightning.pytorch.callbacks import ModelCheckpoint
from ml_collections import config_flags
from torch.nn import HuberLoss
from torch_geometric.loader import DataLoader
from torch_geometric.nn import summary
from torchmetrics import MeanAbsolutePercentageError

import wandb

from ..epcsaft import epcsaft_cython
from .utils import (
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
    wandb.login()
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


FLAGS = flags.FLAGS

flags.DEFINE_string("workdir", None, "Working Directory.")
flags.DEFINE_string("dataset", None, "Dataset to train model on")
flags.DEFINE_bool("lightning", False, "Training run with pytorch lighting.")
flags.DEFINE_string("device", "gpu", "Device to run training.")
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

    if FLAGS.lightning:
        train_dataset = build_train_dataset(FLAGS.workdir, FLAGS.dataset)
        tml_dataset, para_data = build_test_dataset(FLAGS.workdir, train_dataset)
        test_idx = []
        val_idx = []
        for idx, graph in enumerate(tml_dataset):
            if graph.InChI in para_data:
                val_idx.append(idx)
            else:
                test_idx.append(idx)
        test_dataset = tml_dataset[test_idx]
        val_dataset = tml_dataset[val_idx]
        train_loader = DataLoader(
            train_dataset,
            batch_size=FLAGS.config.batch_size,
            shuffle=True,
            num_workers=os.cpu_count(),
        )
        checkpoint = ModelCheckpoint(
            dirpath=osp.join(FLAGS.workdir, "train/checkpoints"),
            filename=FLAGS.config.model_name + "-{step}",
            save_last=True,
            monitor="mape_den",
            save_top_k=1,
            every_n_epochs=int(
                FLAGS.config.checkpoint_every_steps
                // (len(train_dataset) / FLAGS.config.batch_size)
            ),
            verbose=True,
        )
        deg = calc_deg(FLAGS.dataset, FLAGS.workdir)
        model = create_model(FLAGS.config, deg)
        trainer = L.Trainer(
            max_steps=FLAGS.config.num_train_steps,
            log_every_n_steps=FLAGS.config.log_every_steps,
            check_val_every_n_epoch=int(
                FLAGS.config.eval_every_steps
                // (len(train_dataset) / FLAGS.config.batch_size)
            ),
            callbacks=[checkpoint],
        )
        trainer.fit(
            model,
            train_loader,
            val_dataset,
            # ckpt_path=osp.join(
            #     FLAGS.workdir, f"train/checkpoints/{FLAGS.config.checkpoint}"
            # )
            # if FLAGS.config.checkpoint
            # else None,
        )
        trainer.test(model, test_dataset)

    else:
        train_and_evaluate(FLAGS.config, FLAGS.workdir, FLAGS.dataset)


if __name__ == "__main__":
    flags.mark_flags_as_required(["config", "workdir", "dataset"])
    app.run(main)
