import os.path as osp

import torch

from graphdataset import ThermoMLDataset

from torch_geometric.utils import degree

from tqdm import tqdm

path = osp.join('data', 'thermoml', 'train')
train_dataset = ThermoMLDataset(root = './data/thermoml/train', subset='train')

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