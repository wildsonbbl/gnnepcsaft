import os.path as osp

import torch

from graphdataset import ThermoMLDataset, ramirez

from torch_geometric.utils import degree

from tqdm import tqdm

train_dataset = ramirez("./data/ramirez2022")

# Compute the maximum in-degree in the training data.
max_degree = -1
for data in tqdm(train_dataset, 'data: ', len(train_dataset)):
    d = degree(data.edge_index[1].to(torch.int64), num_nodes=data.num_nodes, dtype=torch.int32)
    max_degree = max(max_degree, int(d.max()))

# Compute the in-degree histogram tensor
deg = torch.zeros(max_degree + 1, dtype=torch.int32)
for data in tqdm(train_dataset, 'data: ', len(train_dataset)):
    d = degree(data.edge_index[1].to(torch.int64), num_nodes=data.num_nodes, dtype=torch.int32)
    deg += torch.bincount(d, minlength=deg.numel())

print(deg)