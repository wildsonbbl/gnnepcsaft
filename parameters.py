import os.path as osp
import pickle

from absl import logging

import torch
from torchmetrics import MeanAbsolutePercentageError
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import ml_pc_saft

import wandb

from graphdataset import ThermoMLDataset, ThermoML_padded

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fit_para():
    wandb.login()
    run = wandb.init(
        # Set the project where this run will be logged
        project="gnn-pc-saft",
    )

    # Get datasets, organized by split.
    logging.info("Obtaining datasets.")
    path = osp.join("data", "thermoml")
    train_dataset = ThermoMLDataset(path, subset="train")
    test_dataset = ThermoMLDataset(path, subset="test")

    train_dataset = ThermoML_padded(train_dataset, 4096)
    test_dataset = ThermoML_padded(test_dataset, 16)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Create and initialize the network.
    logging.info("Initializing network.")
    pcsaft_layer = ml_pc_saft.PCSAFT_den.apply
    pcsaft_layer_test = ml_pc_saft.PCSAFT_vp.apply
    lossfn = MeanAbsolutePercentageError().to(device)

    # Create the optimizer.
    unitscale = torch.tensor(
        [
            10.0,
            10.0,
            1e3,
            1e-2,
            1e4,
            10.0,
            10.0,
        ]
    )
    unitscale = unitscale.to(device)
    para = torch.tensor([0.152, 0.323, 0.1889, 0.351, 0.28995, 0.125, 0.251])
    para = para.to(device)
    para.requires_grad_()
    optimizer = torch.optim.AdamW(
        [para],
        lr=0.01,
        eps=1e-5,
        weight_decay=0,
        amsgrad=True,
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, 10)

    print("### starting iterations ###")
    if osp.exists("./data/thermoml/processed/parameters.pkl"):
        parameters = pickle.load(open("./data/thermoml/processed/parameters.pkl", "rb"))
        print(f"inchis saved: {len(parameters.keys())}")
    else:
        parameters = {}

    # Begin training loop.
    logging.info("Starting training.")
    for graphs in train_loader:
        graphs = graphs.to(device)
        # print(graphs)
        if graphs.InChI[0] in parameters:
            continue
        # print(f"for inchi: {graphs.InChI[0]}")
        loss = 2
        step = 1
        while (loss > 1.0 / 100) & (step < 100):
            optimizer.zero_grad()
            pcsaft_params = para.abs() * unitscale + 1e-5
            pred = pcsaft_layer(pcsaft_params, graphs.rho)
            loss = lossfn(pred, graphs.rho[:, -1])
            if loss.isnan():
                print("nan loss")
                break
            loss.backward()
            optimizer.step()
            lr = scheduler.get_last_lr()[0]
            scheduler.step(step)
            # Quick indication that training is happening.
            logging.log_first_n(logging.INFO, "Finished training step %d.", 10, step)
            wandb.log({"train_mape": loss.item(), "train_lr": lr})
            step += 1
        if ~loss.isnan():
            parameters[graphs.InChI[0]] = [pcsaft_params.tolist(), loss.item()]
        # print(f"params: {pcsaft_params}")
        with open("./data/thermoml/processed/parameters.pkl", "wb") as file:
            # A new file will be created
            pickle.dump(parameters, file)
    wandb.finish()


if __name__ == "__main__":
    fit_para()
