import os.path as osp

import ml_collections
import torch
from torch_geometric.utils import degree

from data.graphdataset import ThermoMLpara, ramirez
from train import models


def calc_deg(dataset: str, workdir: str) -> torch.Tensor:
    if dataset == "ramirez":
        path = osp.join(workdir, "data/ramirez2022")
        train_dataset = ramirez(path)
    elif dataset == "thermoml":
        path = osp.join(workdir, "data/thermoml")
        train_dataset = ThermoMLpara(path)
    else:
        ValueError(
            f"dataset is either ramirez or thermoml, got >>> {dataset} <<< instead"
        )
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


def create_model(
    config: ml_collections.ConfigDict, deg: torch.Tensor
) -> torch.nn.Module:
    """Creates a model, as specified by the config."""

    if config.model == "PNA":
        return models.PNAPCSAFT(
            hidden_dim=config.hidden_dim,
            propagation_depth=config.propagation_depth,
            pre_layers=config.pre_layers,
            post_layers=config.post_layers,
            num_mlp_layers=config.num_mlp_layers,
            num_para=config.num_para,
            deg=deg,
            dropout=config.dropout_rate,
        )
    elif config.model == "PNA2":
        return models.PNAPCSAFT2(
            hidden_dim=config.hidden_dim,
            propagation_depth=config.propagation_depth,
            pre_layers=config.pre_layers,
            post_layers=config.post_layers,
            num_mlp_layers=config.num_mlp_layers,
            num_para=config.num_para,
            deg=deg,
            dropout=config.dropout_rate,
            skip_connections=config.skip_connections,
            add_self_loops=config.add_self_loops,
        )
    raise ValueError(f"Unsupported model: {config.model}.")


def create_optimizer(config: ml_collections.ConfigDict, params):
    """Creates an optimizer, as specified by the config."""
    if config.optimizer == "adam":
        return torch.optim.AdamW(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            amsgrad=True,
            eps=1e-5,
        )
    if config.optimizer == "sgd":
        return torch.optim.SGD(
            params,
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            nesterov=True,
        )
    raise ValueError(f"Unsupported optimizer: {config.optimizer}.")


def savemodel(model, optimizer, scaler, path, step):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "step": step,
        },
        path,
    )
