import torch
import os.path as osp

from data.graphdataset import ramirez, ThermoMLpara

from torch_geometric.utils import degree

def calc_deg(dataset: str, workdir: str) -> torch.Tensor:

    if dataset == "ramirez":
        path = osp.join(workdir, "data/ramirez2022")
        train_dataset = ramirez(path)
    elif dataset == "thermoml":
        path = osp.join(workdir, "data/thermoml")
        train_dataset = ThermoMLpara(path)
    else:
        ValueError(f"dataset is either ramirez or thermoml, got >>> {dataset} <<< instead")
# Compute the maximum in-degree in the training data.
    max_degree = -1
    for data in train_dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, int(d.max()))

# Compute the in-degree histogram tensor
    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in train_dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())
    return deg
