"""Module with all available Graph Neural Network models developed and used in the project"""

import dataclasses
import math

import lightning as L
import ml_collections
import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch.nn import BatchNorm1d, Dropout, Linear, ModuleList, ReLU, Sequential
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch_geometric.data.data import Data
from torch_geometric.nn import BatchNorm, GATConv, PNAConv, global_add_pool
from torch_geometric.utils import add_self_loops
from torchmetrics.functional import mean_absolute_percentage_error as mape

from .utils import rho_batch, vp_batch

# from typing import Any


@dataclasses.dataclass
class PnaconvsParams:
    "Parameters for pna convolutions."
    propagation_depth: int
    pre_layers: int
    post_layers: int
    deg: torch.Tensor
    dropout: float = 0.0
    self_loops: bool = False


# pylint: disable=R0902
class PNAPCSAFT(torch.nn.Module):
    """Graph neural network to predict ePCSAFT parameters"""

    def __init__(
        self,
        hidden_dim: int,
        pna_params: PnaconvsParams,
        num_para: int,
    ):
        super().__init__()

        aggregators = ["mean", "min", "max", "std"]
        scalers = ["identity", "amplification", "attenuation"]
        self.pna_params = pna_params
        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        self.lower_bounds = torch.tensor(
            [1.0, 1.9, 50.0, -1 * math.log10(0.9), math.log10(200.0)]
        )
        self.upper_bounds = torch.tensor(
            [25.0, 4.5, 550.0, -1 * math.log10(0.0001), math.log10(5000.0)]
        )
        self.num_para = num_para

        self.node_embed = AtomEncoder(hidden_dim)
        self.edge_embed = BondEncoder(hidden_dim)
        self.dropout = Dropout(p=pna_params.dropout)

        for _ in range(pna_params.propagation_depth):
            conv = PNAConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                aggregators=aggregators,
                scalers=scalers,
                deg=pna_params.deg,
                edge_dim=hidden_dim,
                towers=1,
                pre_layers=pna_params.pre_layers,
                post_layers=pna_params.post_layers,
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
            Linear(hidden_dim // 4, num_para),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the model"""

        edge_index, edge_attr = add_self_loops(
            edge_index, edge_attr, 0, num_nodes=x.size(0)
        )
        x = self.node_embed(x)
        edge_attr = self.edge_embed(edge_attr)

        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index, edge_attr)))
            x = self.dropout(x)

        x = global_add_pool(x, batch)
        x = self.mlp(x)
        return x

    def pred_with_bounds(self, data: Data):
        """Forward pass of the model with bounds."""

        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        params = self.forward(x, edge_index, edge_attr, batch)
        upper_bounds = (
            self.upper_bounds[:3] if self.num_para == 3 else self.upper_bounds[3:]
        ).to(device=x.device)
        lower_bounds = (
            self.lower_bounds[:3] if self.num_para == 3 else self.lower_bounds[3:]
        ).to(device=x.device)
        params = torch.minimum(params, upper_bounds)
        params = torch.maximum(params, lower_bounds)

        return params


class PNApcsaftL(L.LightningModule):
    """Graph neural network to predict ePCSAFT parameters with pytorch lightning."""

    def __init__(
        self,
        pna_params: PnaconvsParams,
        config: ml_collections.ConfigDict,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        self.model = PNAPCSAFT(
            config.hidden_dim, pna_params=pna_params, num_para=config.num_para
        )

    # pylint: disable=W0221
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the model"""
        return self.model(x, edge_index, edge_attr, batch)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        if self.config.optimizer == "adam":
            opt = torch.optim.AdamW(
                self.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                amsgrad=True,
                eps=1e-5,
            )
        elif self.config.optimizer == "sgd":
            opt = torch.optim.SGD(
                self.parameters(),
                lr=self.config.learning_rate,
                momentum=0.0,
                weight_decay=0.0,
                nesterov=False,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}.")
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": CosineAnnealingWarmRestarts(opt, self.config.warmup_steps),
                "interval": "step",
                "frequency": 1,
            },
        }

    # pylint: disable = W0613
    def training_step(self, graphs, batch_idx) -> STEP_OUTPUT:
        if self.config.dataset in ("esper_assoc", "esper_assoc_only"):
            target: torch.Tensor = graphs.assoc.view(-1, self.config.num_para)
        else:
            target: torch.Tensor = graphs.para.view(-1, self.config.num_para)
        x, edge_index, edge_attr, batch = (
            graphs.x,
            graphs.edge_index,
            graphs.edge_attr,
            graphs.batch,
        )
        pred: torch.Tensor = self(x, edge_index, edge_attr, batch)
        loss_mape = mape(pred, target)
        self.log(
            "train_mape",
            loss_mape,
            on_step=True,
            batch_size=target.shape[0],
            sync_dist=True,
        )
        return loss_mape

    # pylint: disable=R0914
    def validation_step(self, graphs, batch_idx) -> STEP_OUTPUT:
        metrics_dict = {}
        pred_para: torch.Tensor = (
            self.model.pred_with_bounds(graphs).squeeze().to(torch.float64).detach()
        )
        if self.config.num_para == 2:
            para_assoc = 10 ** (
                pred_para * torch.tensor([-1.0, 1.0], device=pred_para.device)
            )
            para_msigmae = graphs.para.view(-1, 3)
        else:
            para_assoc = 10 ** (
                graphs.assoc.view(-1, 2)
                * torch.tensor([-1.0, 1.0], device=pred_para.device)
            )
            para_msigmae = pred_para
        pred_para = (
            torch.hstack([para_msigmae, para_assoc, graphs.munanb.view(-1, 3)])
            .cpu()
            .numpy()
        )
        pred_rho = rho_batch(pred_para, graphs.rho)
        pred_vp = vp_batch(pred_para, graphs.vp)
        rho = [rho[:, -1] for rho in graphs.rho if rho.shape[0] > 0]
        vp = [vp[:, -1] for vp in graphs.vp if vp.shape[0] > 0]
        mape_den = []
        for pred, exp in zip(pred_rho, rho):
            mape_den += [np.mean(np.abs(pred - exp) / exp).item()]
        mape_den = np.asarray(mape_den).mean().item()
        mape_vp = []
        for pred, exp in zip(pred_vp, vp):
            mape_vp += [np.mean(np.abs(pred - exp) / exp).item()]
        mape_vp = np.asarray(mape_vp).mean().item()
        metrics_dict.update(
            {
                "mape_den": mape_den,
                "mape_vp": mape_vp,
            }
        )

        self.log_dict(metrics_dict, on_step=False, on_epoch=True, batch_size=1)
        return metrics_dict

    def test_step(self, graphs, batch_idx) -> STEP_OUTPUT:
        return self.validation_step(graphs, batch_idx)


# pylint: disable=R0902,R0913,R0917
class GATPCSAFT(torch.nn.Module):
    """Graph neural network to predict ePCSAFT parameters"""

    def __init__(
        self,
        hidden_dim: int,
        propagation_depth: int,
        num_para: int,
        dropout: float,
        heads: int,
    ):
        super().__init__()

        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        self.lower_bounds = torch.tensor(
            [1.0, 1.9, 50.0, -1 * math.log10(0.9), math.log10(200.0)]
        )
        self.upper_bounds = torch.tensor(
            [25.0, 4.5, 550.0, -1 * math.log10(0.0001), math.log10(5000.0)]
        )
        self.num_para = num_para

        self.node_embed = AtomEncoder(hidden_dim)
        self.edge_embed = BondEncoder(hidden_dim)
        self.dropout = Dropout(p=dropout)

        for _ in range(propagation_depth):
            conv = GATConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                edge_dim=hidden_dim,
                heads=heads,
                concat=False,
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
            Linear(hidden_dim // 4, num_para),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the model"""

        x = self.node_embed(x)
        edge_attr = self.edge_embed(edge_attr)

        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index, edge_attr)))
            x = self.dropout(x)

        x = x.sum(dim=0, keepdim=True) if batch is None else global_add_pool(x, batch)
        x = self.mlp(x)
        return x

    def pred_with_bounds(self, data: Data):
        """Forward pass of the model with bounds."""

        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        params = self.forward(x, edge_index, edge_attr, batch)
        upper_bounds = (
            self.upper_bounds[:3] if self.num_para == 3 else self.upper_bounds[3:]
        ).to(device=x.device)
        lower_bounds = (
            self.lower_bounds[:3] if self.num_para == 3 else self.lower_bounds[3:]
        ).to(device=x.device)
        params = torch.minimum(params, upper_bounds)
        params = torch.maximum(params, lower_bounds)

        return params


class PCsaftL(L.LightningModule):
    """Graph neural network to predict ePCSAFT parameters with pytorch lightning."""

    def __init__(
        self,
        config: ml_collections.ConfigDict,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        if config.model == "GATL":
            self.model = GATPCSAFT(
                config.hidden_dim,
                config.propagation_depth,
                config.num_para,
                config.dropout_rate,
                config.heads,
            )
        else:
            raise ValueError(f"Unsupported model: {config.model}.")

    # pylint: disable=W0221
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the model"""
        return self.model(x, edge_index, edge_attr, batch)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        if self.config.optimizer == "adam":
            opt = torch.optim.AdamW(
                self.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                amsgrad=True,
                eps=1e-5,
            )
        elif self.config.optimizer == "sgd":
            opt = torch.optim.SGD(
                self.parameters(),
                lr=self.config.learning_rate,
                momentum=0.0,
                weight_decay=0.0,
                nesterov=False,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}.")
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": CosineAnnealingWarmRestarts(opt, self.config.warmup_steps),
                "interval": "step",
                "frequency": 1,
            },
        }

    # pylint: disable = W0613
    def training_step(self, graphs, batch_idx) -> STEP_OUTPUT:
        if self.config.dataset in ("esper_assoc", "esper_assoc_only"):
            target: torch.Tensor = graphs.assoc.view(-1, self.config.num_para)
        else:
            target: torch.Tensor = graphs.para.view(-1, self.config.num_para)
        x, edge_index, edge_attr, batch = (
            graphs.x,
            graphs.edge_index,
            graphs.edge_attr,
            graphs.batch,
        )
        pred: torch.Tensor = self(x, edge_index, edge_attr, batch)
        loss_mape = mape(pred, target)
        self.log(
            "train_mape",
            loss_mape,
            on_step=True,
            batch_size=target.shape[0],
            sync_dist=True,
        )
        return loss_mape

    # pylint: disable=R0914
    def validation_step(self, graphs, batch_idx) -> STEP_OUTPUT:
        metrics_dict = {}
        pred_para: torch.Tensor = (
            self.model.pred_with_bounds(graphs).squeeze().to(torch.float64).detach()
        )
        if self.config.num_para == 2:
            para_assoc = 10 ** (
                pred_para * torch.tensor([-1.0, 1.0], device=pred_para.device)
            )
            para_msigmae = graphs.para.view(-1, 3)
        else:
            para_assoc = 10 ** (
                graphs.assoc.view(-1, 2)
                * torch.tensor([-1.0, 1.0], device=pred_para.device)
            )
            para_msigmae = pred_para
        pred_para = (
            torch.hstack([para_msigmae, para_assoc, graphs.munanb.view(-1, 3)])
            .cpu()
            .numpy()
        )
        pred_rho = rho_batch(pred_para, graphs.rho)
        pred_vp = vp_batch(pred_para, graphs.vp)
        rho = [rho[:, -1] for rho in graphs.rho if rho.shape[0] > 0]
        vp = [vp[:, -1] for vp in graphs.vp if vp.shape[0] > 0]
        mape_den = []
        for pred, exp in zip(pred_rho, rho):
            mape_den += [np.mean(np.abs(pred - exp) / exp).item()]
        mape_den = np.asarray(mape_den).mean().item()
        mape_vp = []
        for pred, exp in zip(pred_vp, vp):
            mape_vp += [np.mean(np.abs(pred - exp) / exp).item()]
        mape_vp = np.asarray(mape_vp).mean().item()
        metrics_dict.update(
            {
                "mape_den": mape_den,
                "mape_vp": mape_vp,
            }
        )

        self.log_dict(metrics_dict, on_step=False, on_epoch=True, batch_size=1)
        return metrics_dict

    def test_step(self, graphs, batch_idx) -> STEP_OUTPUT:
        return self.validation_step(graphs, batch_idx)


def create_model(
    config: ml_collections.ConfigDict, deg: torch.Tensor
) -> torch.nn.Module:
    """Creates a model, as specified by the config."""

    pna_params = PnaconvsParams(
        propagation_depth=config.propagation_depth,
        pre_layers=config.pre_layers,
        post_layers=config.post_layers,
        deg=deg,
        dropout=config.dropout_rate,
        self_loops=config.add_self_loops,
    )

    if config.model == "PNA":
        return PNAPCSAFT(
            hidden_dim=config.hidden_dim,
            pna_params=pna_params,
            num_para=config.num_para,
        )
    if config.model == "PNAL":
        return PNApcsaftL(
            pna_params=pna_params,
            config=config,
        )
    if config.model == "GATL":
        return PCsaftL(config=config)

    raise ValueError(f"Unsupported model: {config.model}.")
