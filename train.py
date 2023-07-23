import os.path as osp
from absl import app
from absl import flags
from ml_collections import config_flags

from absl import logging
import ml_collections

import models

import torch
from torch.nn import HuberLoss
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch_geometric.loader import DataLoader
from torchmetrics import MeanAbsolutePercentageError

import ml_pc_saft
import jax

import wandb

from graphdataset import ThermoMLDataset, ThermoML_padded, ramirez
import pickle

deg = torch.tensor([228, 10903, 14978, 3205, 2177, 0, 34])


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


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
    """Execute model training and evaluation loop.

    Args:
      config: Hyperparameter configuration for training and evaluation.
      workdir: Directory where the TensorBoard summaries are written to.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    platform = jax.local_devices()[0].platform
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

    path = osp.join(workdir, "data/thermoml")
    train_dataset = ThermoMLDataset(path, subset="train")
    test_dataset = ThermoMLDataset(path, subset="test")

    train_dataset = ThermoML_padded(train_dataset, config.pad_size)
    test_dataset = ThermoML_padded(test_dataset, config.pad_size)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    path = osp.join(workdir, "data/ramirez2022")
    # train_dataset = ramirez(path)
    # train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    if osp.exists("./data/thermoml/processed/parameters.pkl"):
        parameters = pickle.load(open("./data/thermoml/processed/parameters.pkl", "rb"))
        print(f"inchis saved: {len(parameters.keys())}")
    else:
        print("missing parameters.pkl")

    # Create and initialize the network.
    logging.info("Initializing network.")
    model = create_model(config).to(device, model_dtype)
    pcsaft_den = ml_pc_saft.PCSAFT_den.apply
    pcsaft_vp = ml_pc_saft.PCSAFT_vp.apply
    HLoss = HuberLoss("mean").to(device)
    mape = MeanAbsolutePercentageError().to(device)

    # Create the optimizer.
    optimizer = create_optimizer(config, model.parameters())

    # Set up checkpointing of the model.
    ckp_path = osp.join(workdir, "training/last_checkpoint.pth")
    initial_step = 1
    if osp.exists(ckp_path):
        checkpoint = torch.load(ckp_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        #optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        step = checkpoint["step"]
        initial_step = int(step) + 1

    # Scheduler
    warm_up = config.warmup_steps
    scheduler = CosineAnnealingWarmRestarts(optimizer, warm_up)

    # test fn
    @torch.no_grad()
    def test(loader):
        model.eval()
        total_loss = []
        for graphs in loader:
            graphs = graphs.to(device)
            datapoints = graphs.rho.view(-1, 5)
            datapoints = datapoints.to(device)
            parameters = model(graphs)
            pred_y = pcsaft_den(parameters, datapoints)
            y = datapoints[:, -1]
            if torch.all(
                graphs.vp.view(-1, 5) != torch.zeros_like(graphs.vp.view(-1, 5))
            ):
                datapoints = graphs.vp.view(-1, 5)
                datapoints = datapoints.to(device)
                pred_y_vp = pcsaft_vp(parameters, datapoints)
                y_vp = datapoints[:, -1]
                pred_y = torch.concat([pred_y, pred_y_vp])
                y = torch.concat([y, y_vp])
            loss = mape(pred_y, y)
            total_loss += [loss.item()]

        return torch.tensor(total_loss).nanmean().item()

    # Begin training loop.
    logging.info("Starting training.")
    step = initial_step
    total_loss = []
    lr = []

    model.train()
    while step < config.num_train_steps + 1:
        for graphs in train_loader:
            target = [parameters[inchi][0] for inchi in graphs.InChI]
            target = torch.tensor(target, device=device, dtype=model_dtype)
            graphs = graphs.to(device)
            optimizer.zero_grad()
            pred = model(graphs)
            loss = mape(pred, target)
            loss.backward()
            optimizer.step()
            total_loss += [loss.item()]
            lr += scheduler.get_last_lr()
            scheduler.step()

            # Quick indication that training is happening.
            logging.log_first_n(logging.INFO, "Finished training step %d.", 10, step)

            # Log, if required.
            is_last_step = step == config.num_train_steps - 1
            if step % config.log_every_steps == 0 or is_last_step:
                wandb.log(
                    {
                        "train_mape": torch.tensor(total_loss).mean().item(),
                        "train_lr": torch.tensor(lr).mean().item(),
                    },
                    step=step,
                )
                total_loss = []
                lr = []

            # Checkpoint model, if required.
            if step % config.checkpoint_every_steps == 0 or is_last_step:
                savemodel(model, optimizer, ckp_path, step)

            # Evaluate on validation or test, if required.
            # if step % config.eval_every_steps == 0 or (is_last_step):
            #    test_msle = test(val_loader)
            #    wandb.log({"val_msle": test_msle}, step=step)
            #    model.train()

            # if is_last_step:
            #    test_msle = test(test_loader)
            #    wandb.log({"test_msle": test_msle}, step=step)
            #    model.train()

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

    logging.info("Calling train and evaluate")

    train_and_evaluate(FLAGS.config, FLAGS.workdir)


if __name__ == "__main__":
    flags.mark_flags_as_required(["config", "workdir"])
    app.run(main)
