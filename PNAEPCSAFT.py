import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList, ReLU, Sequential, Tanh

from torchmetrics import MeanSquaredLogError

from torch.optim.lr_scheduler import ReduceLROnPlateau

from graphdataset import ThermoMLDataset


from torch_geometric.loader import DataLoader
from torch_geometric.nn import BatchNorm, PNAConv, global_add_pool

from tqdm.notebook import tqdm

import ml_pc_saft

import wandb

wandb.login()

path = osp.join("data", "thermoml", "train")
train_dataset = ThermoMLDataset(path, Notebook=True, subset="train")

path = osp.join("data", "thermoml", "val")
val_dataset = ThermoMLDataset(path, Notebook=True, subset="val")


batch_size = 2**7

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
            device="cuda",
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
            Linear(8 * 9, 50), ReLU(), Linear(50, 25), ReLU(), Linear(25, 3), Tanh()
        )
        self.mlp2 = Sequential(
            Linear(8 * 9, 50),
            ReLU(),
            Linear(50, 25),
            ReLU(),
            Linear(25, 7),
            ReLU(),
        )
        self.mlp3 = Sequential(
            Linear(8 * 9, 50), ReLU(), Linear(50, 25), ReLU(), Linear(25, 7), ReLU()
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ):
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index, edge_attr)))

        x = global_add_pool(x, batch)
        kij = self.mlp1(x).unsqueeze(1).expand(-1, 2, -1)
        c1 = self.mlp2(x).unsqueeze(1)
        c2 = self.mlp3(x).unsqueeze(1)
        x = torch.concat((c1, c2), 1) + self.unitscale
        x = torch.concat((x, kij), -1)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PNAEPCSAFT().to(device)
# model = compile(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


if osp.exists("training/last_checkpoint.pth"):
    PATH = "training/last_checkpoint.pth"
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]

scheduler = ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=20, min_lr=0.00001
)
pcsaft_layer = ml_pc_saft.PCSAFT_layer.apply
pcsaft_layer_test = ml_pc_saft.PCSAFT_layer_test.apply
lossfn = MeanSquaredLogError().to(device)

run = wandb.init(
    # Set the project where this run will be logged
    project="gnn-pc-saft",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": 0.001,
        "epochs": 5,
        "batch": batch_size,
        "LR_patience": 20,
        "checkpoint_log": 500,
        "Loss_function": "MSLE"
    },
)


def train(epoch, path):
    model.train()
    step = 0
    for data in tqdm(train_loader, desc="step "):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(
            data.x.to(torch.float),
            data.edge_index,
            data.edge_attr.to(torch.float),
            data.batch,
        )
        pred = pcsaft_layer(out, data.y.view(-1, 7))
        loss = lossfn(pred[~pred.isnan()], data.y.view(-1,7)[~pred.isnan(),6])
        if loss.isnan():
            continue
        loss.backward()
        step += 1
        optimizer.step()
        if step % 1000 == -1:
            loss_val = test(val_loader)
            wandb.log({"Loss_val": loss_val})
        else:
            loss_val = loss.item()
        wandb.log({"Loss_train": loss.item()})
        wandb.log({"nan_number": pred.isnan().sum().item()})
        if step % 500 == 0:
            savemodel(
                model,
                optimizer,
                path,
                epoch,
                loss,
                step,
            )
            scheduler.step(loss_val)


@torch.no_grad()
def test(loader):
    model.eval()
    total_error = 0
    step = 0
    for data in loader:
        if step >= 100:
            break
        data = data.to(device)
        out = model(
            data.x.to(torch.float),
            data.edge_index,
            data.edge_attr.to(torch.float),
            data.batch,
        )
        pred = pcsaft_layer_test(out, data.y.view(-1, 7))
        loss = lossfn(pred[~pred.isnan()], data.y.view(-1,7)[~pred.isnan(),6])
        if loss.isnan():
            continue
        total_error += loss.item()
        step += 1
    return total_error / step


def savemodel(model, optimizer, path, epoch, loss, step):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "step": step,
        },
        path,
    )
