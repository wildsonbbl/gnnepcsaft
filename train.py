import os.path as osp

from absl import logging
import ml_collections

import models

import torch
from torchmetrics import MeanSquaredLogError
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch_geometric.loader import DataLoader

import ml_pc_saft
import jax

import wandb

from graphdataset import ThermoMLDataset, ThermoML_padded

deg = torch.tensor([228, 10738, 15049, 3228, 2083, 0, 34])


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
            params, lr=config.learning_rate, weight_decay=0, amsgrad=True
        )
    if config.optimizer == "sgd":
        return torch.optim.SGD(
            params,
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=0,
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

    path = osp.join("data", "thermoml")
    train_dataset = ThermoMLDataset(path, subset="train")
    val_dataset = ThermoMLDataset(path, subset="val")
    test_dataset = ThermoMLDataset(path, subset="test")

    train_dataset = ThermoML_padded(train_dataset, config.pad_size)
    val_dataset = ThermoML_padded(val_dataset, 16)
    test_dataset = ThermoML_padded(test_dataset, 16)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Create and initialize the network.
    logging.info("Initializing network.")
    model = create_model(config).to(device, model_dtype)
    pcsaft_layer = ml_pc_saft.PCSAFT_layer.apply
    pcsaft_layer_test = ml_pc_saft.PCSAFT_layer_test.apply
    lossfn = MeanSquaredLogError().to(device)

    # Create the optimizer.
    optimizer = create_optimizer(config, model.parameters())

    # Set up checkpointing of the model.
    ckp_path = "./training/last_checkpoint.pth"
    initial_step = 1
    if osp.exists(ckp_path):
        checkpoint = torch.load(ckp_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
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
        pad_size = config.pad_size
        para = config.num_para
        for graphs in loader:
            graphs = graphs.to(device)
            datapoints = graphs.states.view(-1, 5)
            datapoints = datapoints.to(device)
            parameters = model(graphs).to(torch.float64)
            parameters = parameters.repeat(1, pad_size).reshape(-1, para)
            pred_y = pcsaft_layer_test(parameters, datapoints)
            y = datapoints[:, -1]
            loss = lossfn(pred_y[~pred_y.isnan()], y[~pred_y.isnan()])
            total_loss += [loss.item()]

        return torch.tensor(total_loss).nanmean().item()

    # Begin training loop.
    logging.info("Starting training.")
    step = initial_step
    total_loss = []
    errp = []
    lr = []
    repeat_steps = config.repeat_steps
    model.train()
    pad_size = config.pad_size
    batch_size = config.batch_size
    para = config.num_para
    while step < config.num_train_steps + 1:
        for graphs in train_loader:
            graphs = graphs.to(device)
            for _ in range(repeat_steps):
                datapoints = graphs.states.view(-1, 5)
                datapoints = datapoints.to(device)
                optimizer.zero_grad()
                parameters = model(graphs).to(torch.float64)
                parameters = parameters.repeat(1, pad_size).reshape(-1, para)
                pred_y = pcsaft_layer(parameters, datapoints)
                y = datapoints[:, -1]
                loss = lossfn(pred_y, y)
                loss.backward()
                optimizer.step()
                total_loss += [loss.item()]
                errp += [(pred_y / y * 100).mean().item()]
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
                            "train_lr": torch.tensor(lr).mean().item(),
                        },
                        step=step,
                    )
                    total_loss = []
                    errp = []
                    lr = []
                    
                # Checkpoint model, if required.
                if step % config.checkpoint_every_steps == 0 or is_last_step:
                    savemodel(model, optimizer, ckp_path, step)

                # Evaluate on validation or test, if required.
                if step % config.eval_every_steps == 0 or (is_last_step):
                    test_msle = test(val_loader)
                    wandb.log({"val_msle": test_msle}, step=step)
                    model.train()

                if is_last_step:
                    test_msle = test(test_loader)
                    wandb.log({"test_msle": test_msle}, step=step)
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
