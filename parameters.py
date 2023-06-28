import os.path as osp
import pickle

from absl import logging

import torch
from torchmetrics import MeanSquaredLogError
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
    lossfn = MeanSquaredLogError().to(device)

    # Create the optimizer.
    para = torch.tensor([1.52, 3.23, 188.9, 0.0351, 2899.5, 0.01, 1.0])
    para = para.to(device)
    para.requires_grad_()
    optimizer = torch.optim.SGD(
        [para],
        lr=0.01,
        momentum=0.9,
        weight_decay=0,
        nesterov=True,
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, 30)

    print("### starting iterations ###")
    if osp.exists("./data/thermoml/processed/parameters.pkl"):
        parameters = pickle.load(open("./data/thermoml/processed/parameters.pkl", "rb"))
        print(f"inchis saved: {len(parameters.keys)}")
    else:
        parameters = {}

    # Begin training loop.
    logging.info("Starting training.")
    for graphs in train_loader:
        graphs = graphs.to(device)
        print(graphs)
        if graphs.InChI[0] in parameters:
            continue
        print(f"for inchi: {graphs.InChI[0]}")
        loss = 2
        step = 1
        while (loss > 0.0001) & (step < 100):
            optimizer.zero_grad()
            pred = pcsaft_layer(para.abs() + 1.0e-6, graphs.rho)
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
            wandb.log({"train_msle": loss.item(), "lr": lr})
            step += 1
        if (~loss.isnan()):
            parameters[graphs.InChI[0]] = [para.tolist(), loss.item()]
            print(f"params: {para}")
        with open("./data/thermoml/processed/parameters.pkl", "wb") as file:
            # A new file will be created
            pickle.dump(parameters, file)
    wandb.finish()


if __name__ == "__main__":
    fit_para()
