"""Module with all available Graph Neural Network models developed and used in the project"""

import inspect
import math

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from ml_collections import ConfigDict
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch.nn import SELU, BatchNorm1d, Dropout, Linear, ModuleList, ReLU, Sequential
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch_geometric import nn as gnn
from torch_geometric.data.data import Data
from torch_geometric.nn import BatchNorm
from torchmetrics.functional import mean_absolute_percentage_error as mape

from .utils import rho_batch, vp_batch


class GNNePCSAFTL(L.LightningModule):
    """Graph neural network to predict ePCSAFT parameters with pytorch lightning."""

    def __init__(
        self,
        config: ConfigDict,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        self.model = GNNePCSAFT(config)

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

    def training_step(self, graphs, batch_idx) -> STEP_OUTPUT:  # pylint: disable=W0613
        if self.config.dataset in ("esper_assoc", "esper_assoc_only"):
            target: torch.Tensor = graphs.assoc
        else:
            target: torch.Tensor = graphs.para
        x, edge_index, edge_attr, batch = (
            graphs.x,
            graphs.edge_index,
            graphs.edge_attr,
            graphs.batch,
        )
        pred: torch.Tensor = self(x, edge_index, edge_attr, batch)
        ape = (pred - target) / target  # absolute percentage error
        zeros = torch.zeros_like(ape)
        loss = F.huber_loss(ape, zeros, delta=0.01)  # huber with ape
        loss_mape = mape(pred, target)
        self.log(
            "train_huber",
            loss,
            on_step=True,
            batch_size=target.shape[0],
            sync_dist=True,
        )
        self.log(
            "train_mape",
            loss_mape,
            on_step=True,
            batch_size=target.shape[0],
            sync_dist=True,
        )
        return loss

    def validation_step(  # pylint: disable=W0613,R0914
        self, graphs, batch_idx, dataloader_idx
    ) -> STEP_OUTPUT:
        metrics_dict = {}
        pred_para: torch.Tensor = self.model.pred_with_bounds(graphs).squeeze().detach()
        if self.config.num_para == 2:
            para_assoc = 10 ** (
                pred_para * torch.tensor([-1.0, 1.0], device=pred_para.device)
            )
            para_msigmae = graphs.para
        else:
            para_assoc = 10 ** (
                graphs.assoc * torch.tensor([-1.0, 1.0], device=pred_para.device)
            )
            para_msigmae = pred_para
        pred_para = (
            torch.hstack([para_msigmae, para_assoc, graphs.munanb])
            .cpu()
            .to(torch.float64)
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

        self.log_dict(
            metrics_dict, on_step=False, on_epoch=True, batch_size=1, sync_dist=True
        )
        return metrics_dict

    def test_step(self, graphs, batch_idx, dataloader_idx) -> STEP_OUTPUT:
        return self.validation_step(graphs, batch_idx, dataloader_idx)


class GNNePCSAFT(torch.nn.Module):  # pylint: disable=R0902
    """Graph neural network to predict ePCSAFT parameters"""

    def __init__(self, config: ConfigDict):
        super().__init__()

        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        self.lower_bounds = torch.tensor(
            [1.0, 1.9, 50.0, -1 * math.log10(0.9), math.log10(200.0)]
        )
        self.upper_bounds = torch.tensor(
            [25.0, 4.5, 550.0, -1 * math.log10(0.0001), math.log10(5000.0)]
        )
        self.num_para = config.num_para

        self.node_embed = AtomEncoder(config.hidden_dim)
        self.edge_embed = BondEncoder(config.hidden_dim)
        self.dropout = Dropout(p=config.dropout)
        self.global_pool = get_global_pool(config)
        self.global_pool_type = config.global_pool

        for _ in range(config.propagation_depth):
            conv = get_conv(config)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(config.hidden_dim))

        self.mlp = Sequential(
            Linear(config.hidden_dim, config.hidden_dim // 2),
            BatchNorm1d(config.hidden_dim // 2),
            ReLU(),
            Linear(config.hidden_dim // 2, config.hidden_dim // 4),
            BatchNorm1d(config.hidden_dim // 4),
            ReLU(),
            Linear(config.hidden_dim // 4, config.num_para),
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
            x = self.dropout(x)
            # check if edge_attr is a keyword argument in conv
            if "edge_attr" in inspect.signature(conv.forward).parameters:
                x = F.relu(
                    batch_norm(conv(x=x, edge_index=edge_index, edge_attr=edge_attr))
                )
            else:
                x = F.relu(batch_norm(conv(x=x, edge_index=edge_index)))

        if batch is not None:
            x = self.global_pool(x, batch)
        elif self.global_pool_type == "mean":
            x = x.mean(dim=0, keepdim=True)
        elif self.global_pool_type == "max":
            x = x.max(dim=0, keepdim=True)
        elif self.global_pool_type == "add":
            x = x.sum(dim=0, keepdim=True)
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

        return params.clip(lower_bounds, upper_bounds)


class HabitchNN(torch.nn.Module):
    """
    Neural network to predict PCSAFT parameters
    from (Habicht; Brandenbusch; Sadowski, 2023, 10.1016/j.fluid.2022.113657).
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.lower_bounds = torch.tensor([1.0, 1.9, 50.0])
        self.upper_bounds = torch.tensor([25.0, 4.5, 550.0])

        self.mlp = Sequential(
            Linear(input_dim, 2048),
            SELU(),
            Dropout(p=0.1),
            Linear(2048, 1024),
            SELU(),
            Dropout(p=0.1),
            Linear(1024, 1024),
            SELU(),
            Dropout(p=0.1),
            Linear(1024, 512),
            SELU(),
            Dropout(p=0.1),
            Linear(512, 128),
            SELU(),
            Dropout(p=0.1),
            Linear(128, 32),
            SELU(),
            Dropout(p=0.1),
            Linear(32, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model"""
        return self.mlp(x)

    def pred_with_bounds(self, graphs: Data):
        """Forward pass of the model with bounds."""

        ecfp, mw, atom_count, ring_count, rbonds_count = (
            graphs.ecfp,
            graphs.mw,
            graphs.atom_count,
            graphs.ring_count,
            graphs.rbond_count,
        )
        x = torch.hstack((ecfp, mw, atom_count, ring_count, rbonds_count))

        params = self.forward(x)
        upper_bounds = self.upper_bounds.to(device=x.device)
        lower_bounds = self.lower_bounds.to(device=x.device)

        return params.clip(lower_bounds, upper_bounds)


class HabitchNNL(L.LightningModule):
    """
    Neural network to predict PCSAFT parameters
    from (Habicht; Brandenbusch; Sadowski, 2023, 10.1016/j.fluid.2022.113657).
    """

    def __init__(self, config: ConfigDict):
        super().__init__()
        self.save_hyperparameters()

        self.config = config

        self.model = HabitchNN(3085)

    # pylint: disable=W0221
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model"""
        return self.model(x)

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

    def training_step(self, graphs, batch_idx) -> STEP_OUTPUT:  # pylint: disable=W0613
        """Training step of the model"""
        target: torch.Tensor = graphs.para

        ecfp, mw, atom_count, ring_count, rbonds_count = (
            graphs.ecfp,
            graphs.mw,
            graphs.atom_count,
            graphs.ring_count,
            graphs.rbond_count,
        )
        x = torch.hstack((ecfp, mw, atom_count, ring_count, rbonds_count))

        pred: torch.Tensor = self(x)
        ape = (pred - target) / target  # absolute percentage error
        zeros = torch.zeros_like(ape)
        loss = F.huber_loss(ape, zeros, delta=0.01)  # huber with ape
        loss_mape = mape(pred, target)
        self.log(
            "train_huber",
            loss,
            on_step=True,
            batch_size=target.shape[0],
            sync_dist=True,
        )
        self.log(
            "train_mape",
            loss_mape,
            on_step=True,
            batch_size=target.shape[0],
            sync_dist=True,
        )
        return loss

    def validation_step(  # pylint: disable=W0613,R0914
        self, graphs, batch_idx, dataloader_idx
    ) -> STEP_OUTPUT:
        metrics_dict = {}
        para_msigmae: torch.Tensor = (
            self.model.pred_with_bounds(graphs).squeeze().detach()
        )
        para_assoc = 10 ** (
            graphs.assoc * torch.tensor([-1.0, 1.0], device=para_msigmae.device)
        )
        pred_para = (
            torch.hstack([para_msigmae, para_assoc, graphs.munanb])
            .cpu()
            .to(torch.float64)
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

        self.log_dict(
            metrics_dict, on_step=False, on_epoch=True, batch_size=1, sync_dist=True
        )
        return metrics_dict

    def test_step(self, graphs, batch_idx, dataloader_idx) -> STEP_OUTPUT:
        return self.validation_step(graphs, batch_idx, dataloader_idx)


def get_conv(config: ConfigDict):  # pylint: disable=R0911,R0912
    """Returns the convolution layer."""
    aggregators = ["mean", "min", "max", "std"]
    scalers = ["identity", "amplification", "attenuation"]
    if config.conv == "PNA":  # 2020, https://doi.org/10.48550/arXiv.2004.05718
        return gnn.PNAConv(
            in_channels=config.hidden_dim,
            out_channels=config.hidden_dim,
            aggregators=aggregators,
            scalers=scalers,
            deg=torch.tensor(config.deg, dtype=torch.long),
            edge_dim=config.hidden_dim,
            towers=config.towers,
            pre_layers=config.pre_layers,
            post_layers=config.post_layers,
            divide_input=True,
        )

    if config.conv == "GCN":  # 2016-2017, https://doi.org/10.48550/arXiv.1609.02907
        return gnn.GCNConv(
            in_channels=config.hidden_dim,
            out_channels=config.hidden_dim,
            add_self_loops=config.add_self_loops,
        )

    if config.conv == "GAT":  # 2017-2018, https://doi.org/10.48550/arXiv.1710.10903
        assert (
            config.hidden_dim % config.heads == 0
        ), "hidden_dim must be divisible by heads"

        return gnn.GATConv(
            in_channels=config.hidden_dim,
            out_channels=config.hidden_dim // config.heads,
            heads=config.heads,
            concat=True,
            dropout=config.dropout,
            edge_dim=config.hidden_dim,
            add_self_loops=config.add_self_loops,
        )

    if config.conv == "GATv2":  # 2021-2022, https://doi.org/10.48550/arXiv.2105.14491
        assert (
            config.hidden_dim % config.heads == 0
        ), "hidden_dim must be divisible by heads"
        return gnn.GATv2Conv(
            in_channels=config.hidden_dim,
            out_channels=config.hidden_dim // config.heads,
            heads=config.heads,
            concat=True,
            dropout=config.dropout,
            edge_dim=config.hidden_dim,
            add_self_loops=config.add_self_loops,
        )

    if (
        config.conv == "Transformer"
    ):  # 2020-2021, https://doi.org/10.48550/arXiv.2009.03509
        assert (
            config.hidden_dim % config.heads == 0
        ), "hidden_dim must be divisible by heads"
        return gnn.TransformerConv(
            in_channels=config.hidden_dim,
            out_channels=config.hidden_dim // config.heads,
            heads=config.heads,
            concat=True,
            dropout=config.dropout,
            edge_dim=config.hidden_dim,
        )

    if config.conv == "SAGE":  # 2017-2018, https://doi.org/10.48550/arXiv.1706.02216
        return gnn.SAGEConv(
            in_channels=config.hidden_dim,
            out_channels=config.hidden_dim,
            aggr=aggregators,
        )

    if config.conv == "GIN":  # 2018-2019, https://doi.org/10.48550/arXiv.1810.00826
        return gnn.GINConv(
            nn=Sequential(
                Linear(config.hidden_dim, config.hidden_dim),
                ReLU(),
                Linear(config.hidden_dim, config.hidden_dim),
            ),
            train_eps=False,
        )

    if config.conv == "GINE":  # 2019-2020, https://doi.org/10.48550/arXiv.1905.12265
        return gnn.GINEConv(
            nn=Sequential(
                Linear(config.hidden_dim, config.hidden_dim),
                ReLU(),
                Linear(config.hidden_dim, config.hidden_dim),
            ),
            train_eps=False,
            edge_dim=config.hidden_dim,
        )

    if config.conv == "Edge":  # 2018-2019, https://doi.org/10.48550/arXiv.1801.07829
        return gnn.EdgeConv(
            nn=Sequential(
                Linear(2 * config.hidden_dim, config.hidden_dim),
                ReLU(),
                Linear(config.hidden_dim, config.hidden_dim),
            ),
            aggr="max",
        )

    if (
        config.conv == "GatedGraph"
    ):  # 2015-2017, https://doi.org/10.48550/arXiv.1511.05493
        return gnn.GatedGraphConv(
            out_channels=config.hidden_dim,
            num_layers=config.num_layers,
        )

    if config.conv == "Graph":  # 2018-2021, https://doi.org/10.48550/arXiv.1810.02244
        return gnn.GraphConv(
            in_channels=config.hidden_dim,
            out_channels=config.hidden_dim,
        )

    if config.conv == "ARMA":  # 2019-2021, https://doi.org/10.1109/TPAMI.2021.3054830
        return gnn.ARMAConv(
            in_channels=config.hidden_dim,
            out_channels=config.hidden_dim,
            num_stacks=config.num_stacks,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )

    if config.conv == "SG":  # 2019, https://doi.org/10.48550/arXiv.1902.07153
        return gnn.SGConv(
            in_channels=config.hidden_dim,
            out_channels=config.hidden_dim,
            add_self_loops=config.add_self_loops,
        )

    raise ValueError(f"Unsupported convolution: {config.conv}.")


def get_global_pool(config: ConfigDict):
    """Returns the global pooling layer."""
    if config.global_pool == "mean":
        return gnn.aggr.MeanAggregation()
    if config.global_pool == "max":
        return gnn.aggr.MaxAggregation()
    if config.global_pool == "add":
        return gnn.aggr.SumAggregation()
    raise ValueError(f"Unsupported global pooling: {config.global_pool}.")


def create_model(config: ConfigDict, deg: list):
    """Creates a model, as specified by the config."""
    config.deg = deg

    if config.model.lower() == "gnn":
        return GNNePCSAFTL(config)
    if config.model.lower() == "habitch":
        return HabitchNNL(config)
    raise ValueError(f"Unsupported model: {config.model}.")
