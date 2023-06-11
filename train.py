import os.path as osp

from absl import logging
import ml_collections

import models

import torch
from torchmetrics import MeanSquaredLogError
from torch.optim.lr_scheduler import CyclicLR, LinearLR
from torch_geometric.loader import DataLoader

import ml_pc_saft

import wandb

from graphdataset import ThermoMLDataset, get_padded_array
from graph import from_InChI


deg = torch.tensor([228, 10738, 15049, 3228, 2083, 0, 34])


def create_model(config: ml_collections.ConfigDict) -> torch.nn.Module:
    """Creates a Flax model, as specified by the config."""
    platform = "gpu"
    if config.half_precision:
        if platform == "tpu":
            model_dtype = torch.float16
        else:
            model_dtype = torch.float32
    else:
        model_dtype = torch.float32
    if config.model == "PNA":
        return models.PNA(
            hidden_dim=config.hidden_dim,
            propagation_depth=config.propagation_depth,
            num_mlp_layers=config.num_mlp_layers,
            num_para=config.num_para,
            deg=deg,
        )
    raise ValueError(f"Unsupported model: {config.model}.")


def create_optimizer(config: ml_collections.ConfigDict, params):
    """Creates an optimizer, as specified by the config."""
    if config.optimizer == "adam":
        return torch.optim.AdamW(params, lr=config.learning_rate)
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
    platform = "gpu"
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
        input_dtype = torch.int16
    else:
        input_dtype = torch.int32

    path = osp.join("data", "thermoml")
    train_dataset = ThermoMLDataset(path, subset="train", graph_dtype=input_dtype)
    val_dataset = ThermoMLDataset(path, subset="val", graph_dtype=input_dtype)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # Create and initialize the network.
    logging.info("Initializing network.")
    model = create_model(config).to(device)
    pcsaft_layer = ml_pc_saft.PCSAFT_layer.apply
    pcsaft_layer_test = ml_pc_saft.PCSAFT_layer_test.apply
    lossfn = MeanSquaredLogError().to(device)

    # Create the optimizer.
    optimizer = create_optimizer(config, model.parameters())
    scheduler = CyclicLR(
        optimizer, 
        0.00001, 
        config.learning_rate, 
        config.patience, 
        cycle_momentum=False
        )

    warm_up = config.warmup_steps
    scheduler_warmup = LinearLR(optimizer, 1 / 10, 1, warm_up)

    # Set up checkpointing of the model.
    ckp_path = "./training/last_checkpoint.pth"
    initial_step = 0
    if osp.exists(ckp_path):
        checkpoint = torch.load(ckp_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        step = checkpoint["step"]
        initial_step = int(step) + 1

    # test fn
    @torch.no_grad()
    def test(val_dict):
        model.eval()
        total_loss = []
        for key in val_dict:
            graphs = from_InChI(key, dtype=input_dtype)
            graphs = graphs.to(device)
            datapoints, _ = get_padded_array(val_dict[key][1], max_pad)
            datapoints = datapoints.to(device)
            parameters = model(graphs)
            pred_y = pcsaft_layer(parameters, datapoints)
            y = datapoints[:, -1]
            loss = lossfn(pred_y[~pred_y.isnan()], y[~pred_y.isnan()])
            total_loss += [loss.item()]

        return torch.tensor(total_loss).mean().item()

    # Begin training loop.
    logging.info("Starting training.")
    max_pad = config.max_pad
    step = initial_step
    total_loss = []
    errp = []
    nan_number = []
    lr = []
    repeat_steps = config.repeat_steps
    while step < config.num_train_steps + 1:
        model.train()
        for graphs in train_loader:
            graphs = graphs.to(device)
            for _ in range(repeat_steps):
                datapoints = graphs.states.view(-1, 5)
                datapoints = get_padded_array(datapoints, max_pad)
                datapoints = datapoints.to(device)
                optimizer.zero_grad()
                parameters = model(graphs).to(torch.float64).squeeze()
                pred_y = pcsaft_layer(parameters, datapoints)
                y = datapoints[:, -1]
                loss = lossfn(pred_y[~pred_y.isnan()], y[~pred_y.isnan()])
                loss.backward()
                optimizer.step()
                total_loss += [loss.item()]
                errp += [(pred_y / y).mean().item()]
                nan_number += [pred_y.isnan().sum().item()]
                if step < warm_up:
                    scheduler_warmup.step()
                    lr += scheduler_warmup.get_last_lr()
                else:
                    scheduler.step()
                    lr += scheduler.get_last_lr()

                # Quick indication that training is happening.
                logging.log_first_n(
                    logging.INFO, "Finished training step %d.", 10, step
                )

                # Log, if required.
                is_last_step = step == config.num_train_steps - 1
                if step % config.log_every_steps == 0 or is_last_step:
                    wandb.log(
                        {
                            "train_msle": torch.tensor(total_loss).mean().item(),
                            "train_errp": torch.tensor(errp).mean().item(),
                            "train_nan_number": torch.tensor(nan_number).mean().item(),
                            'train_lr:': torch.tensor(lr).mean().item()
                        },
                        step=step,
                    )

                # Checkpoint model, if required.
                if step % config.checkpoint_every_steps == 0 or is_last_step:
                    savemodel(model, optimizer, ckp_path, step)
                step += 1
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
