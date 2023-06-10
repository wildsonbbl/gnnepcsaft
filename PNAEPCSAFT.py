import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList, ReLU, Sequential, Tanh, BatchNorm1d

from torchmetrics import MeanSquaredLogError

from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR, LinearLR

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
lr = 1e-4
patience = 3700
hidden_dim = 80
propagation_depth = 7
warmup = 700
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, drop_last=True
)

# Compute the maximum in-degree in the training data.
deg = torch.tensor([67167, 3157428, 5106064, 885236, 453935, 0, 11152])


class PNAEPCSAFT(torch.nn.Module):
    def __init__(self):
        super().__init__()

        aggregators = ["mean", "min", "max", "std"]
        scalers = ["identity", "amplification", "attenuation"]
        self.unitscale = torch.tensor(
            [1, 1, 10, 0, 0, 0, 0, 1, 1, 10, 0, 0, 0, 0, 0,0,0],
            dtype=torch.float,
            device=device,
        )

        self.convs = ModuleList()
        self.batch_norms = ModuleList()

        conv = PNAConv(
            in_channels=9,
            out_channels=hidden_dim,
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
        self.batch_norms.append(BatchNorm(hidden_dim))

        for _ in range(propagation_depth - 1):
            conv = PNAConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
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
            self.batch_norms.append(BatchNorm(hidden_dim))

        self.mlp = Sequential(
            Linear(hidden_dim, hidden_dim // 2),
            BatchNorm1d(hidden_dim // 2),
            ReLU(),
            Linear(hidden_dim // 2, hidden_dim // 4),
            BatchNorm1d(hidden_dim // 4),
            ReLU(),
            Linear(hidden_dim // 4, 17),
            ReLU(),
        )

    def forward(
        self,
        data: Data,
    ):
        x, edge_index, edge_attr, batch, TK = (
            data.x.to(torch.float),
            data.edge_index,
            data.edge_attr.to(torch.float),
            data.batch,
            data.y.view(-1,7)[:,2],
        )

        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index, edge_attr)))

        x = global_add_pool(x, batch)
        x = self.mlp(x) + self.unitscale
        x[:,-3:] = F.tanh(x[:,-3:])
        return x


model = PNAEPCSAFT().to(device)
# model = compile(model)
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-2, nesterov=True)

scheduler = ReduceLROnPlateau(
    optimizer, mode="min", factor=0.1, patience=patience, min_lr=1e-8,
    verbose=True
)

scheduler_warmup = LinearLR(optimizer, 1/10,1, warmup)

# scheduler = CyclicLR(optimizer, 0.00001, 0.001, patience, cycle_momentum=False)

if osp.exists("training/last_checkpoint.pth"):
    PATH = "training/last_checkpoint.pth"
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint["model_state_dict"])
    #optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


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
        "scheduler_step": 1,
        "hidden_dim": hidden_dim,
        "propagation_depth": propagation_depth,
        "lr_warmup": warmup,
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
        loss.backward()
        step += 1
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
        errp = (((pred / y) * 100.0 - 100.0).abs() >  50).sum().item() / data.num_graphs
        wandb.log(
            {
                "Loss_train": loss.item(),
                "nan_number": pred.isnan().sum().item(),
                "pred_target": errp,
            }
        )
        scheduler_warmup.step()
        scheduler.step(loss)
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
