import os.path as osp

import torch

from graphdataset import ThermoMLDataset

from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree

from tqdm.notebook import tqdm



path = osp.join('data', 'thermoml', 'train')
train_dataset = ThermoMLDataset(path, Notebook=True)
path = osp.join('data', 'thermoml', 'test')
test_dataset = ThermoMLDataset(path, Notebook=True, subset='test')
path = osp.join('data', 'thermoml', 'val')
val_dataset = ThermoMLDataset(path, Notebook=True, subset='val')


batch_size = 2**7

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last = True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Compute the maximum in-degree in the training data.
max_degree = -1
for data in tqdm(train_dataset, 'data: ', len(train_dataset)):
    d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
    max_degree = max(max_degree, int(d.max()))

# Compute the in-degree histogram tensor
deg = torch.zeros(max_degree + 1, dtype=torch.long)
for data in tqdm(train_dataset, 'data: ', len(train_dataset)):
    d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
    deg += torch.bincount(d, minlength=deg.numel())

print(deg)