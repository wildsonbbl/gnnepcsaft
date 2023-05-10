import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList, ReLU, Sequential
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import random_split

from graphdataset import ThermoMLDataset

from torch_geometric.loader import DataLoader
from torch_geometric.nn import BatchNorm, PNAConv, global_add_pool
from torch_geometric.utils import degree

from tqdm.notebook import tqdm

import ml_pc_saft


path = osp.join('data', 'thermoml','train')
dataset = ThermoMLDataset(path, Notebook=False)
path = osp.join('data', 'thermoml','test')
test_dataset = ThermoMLDataset(path, Notebook=False)

gen = torch.Generator().manual_seed(77)
train_dataset, val_dataset = random_split(dataset,[0.9,0.1], gen)

batch_size = 2**7

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Compute the maximum in-degree in the training data.
max_degree = -1
for data in tqdm(train_dataset,'data: ', len(train_loader.dataset)):
    d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
    max_degree = max(max_degree, int(d.max()))

# Compute the in-degree histogram tensor
deg = torch.zeros(max_degree + 1, dtype=torch.long)
for data in tqdm(train_dataset,'data: ', len(train_loader.dataset)):
    d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
    deg += torch.bincount(d, minlength=deg.numel())


class PNAEPCSAFT(torch.nn.Module):
    def __init__(self):
        super().__init__()

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        conv = PNAConv(in_channels=9, out_channels=75,
                           aggregators=aggregators, scalers=scalers, deg=deg,
                           edge_dim=3, towers=5, pre_layers=1, post_layers=1,
                           divide_input=False)
        self.convs.append(conv)
        self.batch_norms.append(BatchNorm(75))
        for _ in range(4):
            conv = PNAConv(in_channels=75, out_channels=75,
                           aggregators=aggregators, scalers=scalers, deg=deg,
                           edge_dim=3, towers=5, pre_layers=1, post_layers=1,
                           divide_input=False)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(75))

        self.mlp1 = Sequential(Linear(75, 50), ReLU(), Linear(50, 25), ReLU(),
                              Linear(25, 12))
        self.mlp2 = Sequential(Linear(75, 50), ReLU(), Linear(50, 25), ReLU(),
                              Linear(25, 12))

    def forward(self, x, edge_index, edge_attr, batch):
        

        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index, edge_attr)))

        x = global_add_pool(x, batch)
        c1 = self.mlp1(x).unsqueeze(1)
        c2 = self.mlp2(x).unsqueeze(1)
        x = torch.concat((c1,c2), 1)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PNAEPCSAFT().to(device, torch.float64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epoch = 0
best_val_loss = float('inf')

if osp.exists('training/last_checkpoint.pth'):
    PATH = 'training/last_checkpoint.pth'
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    best_val_loss = checkpoint['best_val_loss']

scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20,
                              min_lr=0.00001)
lossfn = ml_pc_saft.PCSAFTLOSS.apply


def train():
    model.train()

    total_loss = 0
    for data in tqdm(train_loader, desc = 'step: '):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        n = out.shape[0]
        loss = lossfn(out, data.y.reshape(n,7))
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(loader):
    model.eval()

    total_error = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        n = out.shape[0]
        loss = lossfn(out, data.y.reshape(n,7)).item()
        total_error += loss
    return total_error / len(loader.dataset)

def savemodel(model, optimizer, type, epoch, loss, best_val_loss):
    path = osp.join('training', type)
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'best_val_loss': best_val_loss
            }, path)











  
