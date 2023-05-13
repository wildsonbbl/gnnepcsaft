import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList, ReLU, Sequential
from torch.optim.lr_scheduler import ReduceLROnPlateau

from graphdataset import ThermoMLDataset

from torch_geometric import compile

from torch_geometric.loader import DataLoader
from torch_geometric.nn import BatchNorm, PNAConv, global_add_pool

from tqdm.notebook import tqdm

import ml_pc_saft

from torch.utils.tensorboard import SummaryWriter


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
            [[[10, 10, 100, 1, 10e3, 10, 10, 10, 10, 1, 1, 1]]], dtype=torch.float,
        device = 'cuda')

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
            Linear(8 * 9, 50), ReLU(), Linear(50, 25), ReLU(), Linear(25, 12)
        )
        self.mlp2 = Sequential(
            Linear(8 * 9, 50), ReLU(), Linear(50, 25), ReLU(), Linear(25, 12)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index, edge_attr)))

        x = global_add_pool(x, batch)
        c1 = self.mlp1(x).unsqueeze(1)
        c2 = self.mlp2(x).unsqueeze(1)
        x = torch.concat((c1, c2), 1)*self.unitscale
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PNAEPCSAFT().to(device)
# model = compile(model
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epoch = 0
best_val_loss = float("inf")

if osp.exists("training/last_checkpoint.pth"):
    PATH = "training/last_checkpoint.pth"
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    best_val_loss = checkpoint["best_val_loss"]

scheduler = ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=20, min_lr=0.00001
)
lossfn = ml_pc_saft.PCSAFTLOSS.apply
lossfn_test = ml_pc_saft.PCSAFTLOSS_test.apply

writer = SummaryWriter("runs/exp1")


def train(epoch):
    model.train()

    total_loss = 0
    step = 0
    for data in tqdm(train_loader, desc="step "):
        step += 1
        data = data.to(device)
        optimizer.zero_grad()
        out = model(
            data.x.to(torch.float),
            data.edge_index,
            data.edge_attr.to(torch.float),
            data.batch,
        )
        n = out.shape[0]
        loss = lossfn(out, data.y.view(-1, 7))
        loss.backward()
        optimizer.step()

        loss_val = test(val_loader)
        writer.add_scalar(f'Loss/train_{epoch}', loss.item()/n, step)
        writer.add_scalar(f'Loss/val{epoch}', loss_val, step)
        if loss_val < best_val_loss:
            best_val_loss = loss_val
            savemodel(
                model,
                optimizer,
                "best_checkpoint.pth",
                epoch,
                loss,
                best_val_loss,
                step,
            )
        savemodel(
            model, optimizer, "last_checkpoint.pth", epoch, loss, best_val_loss, step
        )
        scheduler.step(loss_val)
        total_loss += loss.item()

    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(loader):
    model.eval()

    total_error = 0
    step = 1
    for data in loader:
        data = data.to(device)
        out = model(
            data.x.to(torch.float),
            data.edge_index,
            data.edge_attr.to(torch.float),
            data.batch,
        )
        n = out.shape[0]
        loss = lossfn_test(out, data.y.view(-1, 7)).item()
        total_error += loss
        if step >= 1000:
            break
    return total_error / len(loader.dataset)


def savemodel(model, optimizer, type, epoch, loss, best_val_loss, step):
    path = osp.join("training", type)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "best_val_loss": best_val_loss,
            "step": step,
        },
        path,
    )
