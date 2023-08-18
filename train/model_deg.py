import torch

from data.graphdataset import ramirez, ThermoMLpara

from torch_geometric.utils import degree

from tqdm import tqdm

def calc_deg(dataset: str) -> torch.Tensor:

    if dataset == "ramirez":
        path = "./data/ramirez2022"
        train_dataset = ramirez(path)
    elif dataset == "thermoml":
        path = "./data/thermoml"
        train_dataset = ThermoMLpara(path)
    else:
        ValueError(f"dataset is either ramirez or thermoml, got >>> {dataset} <<< instead")
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
    return deg