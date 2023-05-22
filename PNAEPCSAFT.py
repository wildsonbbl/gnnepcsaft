import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList, ReLU, Sequential, Tanh, BatchNorm1d

from torchmetrics import MeanSquaredLogError

from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR

from graphdataset import ThermoMLDataset


from torch_geometric.loader import DataLoader
from torch_geometric.nn import BatchNorm, PNAConv, global_add_pool
from torch_geometric.data import Data

from tqdm.notebook import tqdm

import ml_pc_saft

import wandb

wandb.login()

path = osp.join("data", "thermoml", "train")
train_dataset = ThermoMLDataset(path, Notebook=True, subset="train")

path = osp.join("data", "thermoml", "val")
val_dataset = ThermoMLDataset(path, Notebook=True, subset="val")

batch_size = 2**7
lr = 0.5
patience = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=True, drop_last=True
)

# Compute the maximum in-degree in the training data.
deg = torch.tensor([67167, 3157428, 5106064, 885236, 453935, 0, 11152])


class PNAEPCSAFT(torch.nn.Module):
    def __init__(self):
        super().__init__()

        aggregators = ["mean", "min", "max", "std"]
        scalers = ["identity", "amplification", "attenuation"]
        self.unitscale = torch.tensor(
            [[[1, 1, 100, 0.01, 1e3, 0, 1e-5]]],
            dtype=torch.float,
            device=device,
        )

        self.convs = ModuleList()
        self.batch_norms = ModuleList()

        conv = PNAConv(
            in_channels=9,
            out_channels=8 * 9,
            aggregators=aggregators,
            scalers=scalers,
            deg=deg,
            edge_dim=3,
            towers=4,
            pre_layers=1,
            post_layers=1,
            divide_input=False,
        )
        self.convs.append(conv)
        self.batch_norms.append(BatchNorm(8 * 9))

        for _ in range(3):
            conv = PNAConv(
                in_channels=8 * 9,
                out_channels=8 * 9,
                aggregators=aggregators,
                scalers=scalers,
                deg=deg,
                edge_dim=3,
                towers=4,
                pre_layers=1,
                post_layers=1,
                divide_input=False,
            )
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(8 * 9))

        self.mlp1 = Sequential(
            Linear(8 * 9, 50),
            BatchNorm1d(50),
            ReLU(),
            Linear(50, 25),
            BatchNorm1d(25),
            ReLU(),
            Linear(25, 3),
            Tanh(),
        )
        self.mlp2 = Sequential(
            Linear(8 * 9, 50),
            BatchNorm1d(50),
            ReLU(),
            Linear(50, 25),
            BatchNorm1d(25),
            ReLU(),
            Linear(25, 7),
            ReLU(),
        )
        self.mlp3 = Sequential(
            Linear(8 * 9, 50),
            BatchNorm1d(50),
            ReLU(),
            Linear(50, 25),
            BatchNorm1d(25),
            ReLU(),
            Linear(25, 7),
            ReLU(),
        )

    def forward(
        self,
        data: Data,
    ):
        x, edge_index, edge_attr, batch = (
            data.x.to(torch.float),
            data.edge_index,
            data.edge_attr.to(torch.float),
            data.batch,
        )

        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index, edge_attr)))

        x = global_add_pool(x, batch)
        kij = self.mlp1(x).unsqueeze(1).expand(-1, 2, -1)
        c1 = self.mlp2(x).unsqueeze(1)
        c2 = self.mlp3(x).unsqueeze(1)
        x = torch.concat((c1, c2), 1) + self.unitscale
        x = torch.concat((x, kij), -1)
        return x


model = PNAEPCSAFT().to(device)
# model = compile(model)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# scheduler = ReduceLROnPlateau(
#    optimizer, mode="min", factor=0.1, patience=patience, min_lr=0.00001
# )

scheduler = CyclicLR(optimizer, 0.001, 1.0, 500)

if osp.exists("training/last_checkpoint.pth"):
    PATH = "training/last_checkpoint.pth"
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    # scheduler.step(checkpoint['loss'])


pcsaft_layer = ml_pc_saft.PCSAFT_layer.apply
pcsaft_layer_test = ml_pc_saft.PCSAFT_layer_test.apply
lossfn = MeanSquaredLogError().to(device)

run = wandb.init(
    # Set the project where this run will be logged
    project="gnn-pc-saft",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": lr,
        "batch": batch_size,
        "LR_patience": patience,
        "checkpoint_save": "1 epoch",
        "Loss_function": "MSLE",
        "scheduler_step": 500,
    },
)


def train(epoch, path):
    model.train()
    step = 0
    total_loss = 0
    for data in tqdm(train_loader, desc="step "):
        data = data.to(device)
        y = data.y.view(-1, 7)[:, -1].squeeze()
        state = data.y.view(-1, 7)
        optimizer.zero_grad()
        para = model(data)
        pred = pcsaft_layer(para, state)
        loss = lossfn(pred[~pred.isnan()], y[~pred.isnan()])
        if loss.isnan():
            continue
        loss.backward()
        step += 1
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
        errp = (pred / y).nanmean().item() * 100.0
        wandb.log(
            {
                "Loss_train": loss.item(),
                "nan_number": pred.isnan().sum().item(),
                "pred/target_fraction": errp,
            }
        )
        scheduler.step()
    loss_train = total_loss / len(train_loader.dataset)
    loss_val = loss_train  # test(val_loader)
    wandb.log({"Loss_val": loss_val, "Loss_train_ep": loss_train})

    savemodel(
        model,
        optimizer,
        path,
        epoch,
        loss_val,
    )


@torch.no_grad()
def test(loader):
    model.eval()
    total_error = 0
    step = 0
    for data in loader:
        data = data.to(device)
        y = data.y.view(-1, 7)[:, -1].squeeze()
        state = data.y.view(-1, 7)
        para = model(data)
        pred = pcsaft_layer_test(para, state)
        loss = lossfn(pred[~pred.isnan()], y[~pred.isnan()])
        if loss.isnan():
            continue
        total_error += loss.item() * data.num_graphs
        step += 1
    return total_error / len(loader.dataset)


def savemodel(model, optimizer, path, epoch, loss):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        path,
    )
