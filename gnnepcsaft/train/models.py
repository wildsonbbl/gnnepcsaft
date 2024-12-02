"""Module with all available Graph Neural Network models developed and used in the project"""

import dataclasses

import lightning as L
import ml_collections
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch.nn import BatchNorm1d, Dropout, Linear, ModuleList, ReLU, Sequential
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch_geometric.data import Data
from torch_geometric.nn import BatchNorm, PNAConv, global_add_pool
from torch_geometric.utils import add_self_loops
from torchmetrics.functional import mean_absolute_percentage_error as mape

from ..epcsaft import utils

# from typing import Any


pcsaft_den = utils.DenFromTensor.apply
pcsaft_vp = utils.VpFromTensor.apply
hloss = F.huber_loss


@dataclasses.dataclass
class PnaconvsParams:
    "Parameters for pna convolutions."
    propagation_depth: int
    pre_layers: int
    post_layers: int
    deg: torch.Tensor
    dropout: float = 0.0
    skip_connections: bool = False
    self_loops: bool = False


@dataclasses.dataclass
class ReadoutMLPParams:
    "Parameters for the MLP layers."
    num_mlp_layers: int
    num_para: int
    dropout: float = 0.0


class PNAPCSAFT(torch.nn.Module):
    """Graph neural network to predict ePCSAFT parameters"""

    def __init__(
        self,
        hidden_dim: int,
        pna_params: PnaconvsParams,
        mlp_params: ReadoutMLPParams,
    ):
        super().__init__()

        aggregators = ["mean", "min", "max", "std"]
        scalers = ["identity", "amplification", "attenuation"]
        self.pna_params = pna_params
        self.convs = ModuleList()
        self.batch_norms = ModuleList()

        self.node_embed = AtomEncoder(hidden_dim)
        self.edge_embed = BondEncoder(hidden_dim)

        for _ in range(pna_params.propagation_depth):
            conv = PNAConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                aggregators=aggregators,
                scalers=scalers,
                deg=pna_params.deg,
                edge_dim=hidden_dim,
                towers=2,
                pre_layers=pna_params.pre_layers,
                post_layers=pna_params.post_layers,
                divide_input=False,
            )
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(hidden_dim))

        self.mlp = Sequential()
        for _ in range(mlp_params.num_mlp_layers):
            self.mlp.append(Linear(hidden_dim, hidden_dim))
            self.mlp.append(BatchNorm1d(hidden_dim))
            self.mlp.append(ReLU())
            self.mlp.append(Dropout(p=mlp_params.dropout))

        self.mlp.append(
            Sequential(
                Linear(hidden_dim, hidden_dim // 2),
                BatchNorm1d(hidden_dim // 2),
                ReLU(),
                Dropout(p=mlp_params.dropout),
                Linear(hidden_dim // 2, hidden_dim // 4),
                BatchNorm1d(hidden_dim // 4),
                ReLU(),
                Dropout(p=mlp_params.dropout),
                Linear(hidden_dim // 4, mlp_params.num_para),
            )
        )

    def forward(
        self,
        data: Data,
    ):
        """Forward pass of the model"""

        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        if self.pna_params.self_loops:
            edge_index, edge_attr = add_self_loops(
                edge_index, edge_attr, 0, num_nodes=x.size(0)
            )
        x = self.node_embed(x)
        edge_attr = self.edge_embed(edge_attr)

        for conv, batch_norm in zip(self.convs, self.batch_norms):
            if self.pna_params.skip_connections:
                x_previous = x
            x = F.relu(batch_norm(conv(x, edge_index, edge_attr)))
            x = F.dropout(x, p=self.pna_params.dropout, training=self.training)
            if self.pna_params.skip_connections:
                x = x + x_previous

        x = global_add_pool(x, batch)
        x = self.mlp(x)
        return x


class PNApcsaftL(L.LightningModule):
    """Graph neural network to predict ePCSAFT parameters with pytorch lightning."""

    def __init__(
        self,
        pna_params: PnaconvsParams,
        mlp_params: ReadoutMLPParams,
        config: ml_collections.ConfigDict,
    ):
        super().__init__()
        self.config = config

        self.model = PNAPCSAFT(
            config.hidden_dim, pna_params=pna_params, mlp_params=mlp_params
        )

    # pylint: disable=W0221
    def forward(
        self,
        data: Data,
    ):
        """Forward pass of the model"""
        return self.model(data)

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
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
                nesterov=True,
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
        if self.config.dataset == "esper_assoc":
            target = graphs.assoc.view(-1, self.config.num_para)
        else:
            target = graphs.para.view(-1, self.config.num_para)
        pred = self(graphs)
        loss_mape = mape(pred, target)
        self.log(
            "train_mape",
            loss_mape,
            on_step=True,
            batch_size=target.shape[0],
            sync_dist=True,
        )
        return loss_mape

    def validation_step(self, graphs, batch_idx) -> STEP_OUTPUT:
        mape_den = 0.0
        huber_den = 0.0
        mape_vp = 0.0
        huber_vp = 0.0
        metrics_dict = {}

        pred_para = self(graphs).squeeze().to(torch.float64)
        if self.config.dataset == "esper_assoc":
            pred_para = 10 ** (
                pred_para * torch.tensor([-1.0, 1.0], device=pred_para.device)
            )
            pred_para = torch.hstack([graphs.para, pred_para, graphs.munanb])
        else:
            pred_para = torch.hstack([pred_para, graphs.assoc, graphs.munanb])
        datapoints = graphs.rho.to(torch.float64).view(-1, 5)
        if ~torch.all(datapoints == torch.zeros_like(datapoints)):
            pred = pcsaft_den(pred_para, datapoints)
            target = datapoints[:, -1].cpu()
            # pylint: disable = not-callable
            loss_mape = mape(pred, target)
            loss_huber = hloss(pred, target, reduction="mean")
            mape_den = loss_mape.item()
            huber_den = loss_huber.item()
            # self.log("mape_den", mape_den)
            metrics_dict.update(
                {
                    "mape_den": mape_den,
                    "huber_den": huber_den,
                }
            )

        datapoints = graphs.vp.to(torch.float64).view(-1, 5)
        if ~torch.all(datapoints == torch.zeros_like(datapoints)):
            pred = pcsaft_vp(pred_para, datapoints)
            target = datapoints[:, -1].cpu()
            result_filter = ~torch.isnan(pred)
            # pylint: disable = not-callable
            loss_mape = mape(pred[result_filter], target[result_filter])
            loss_huber = hloss(pred[result_filter], target[result_filter])
            if loss_mape.item() < 0.5:
                mape_vp = loss_mape.item()
                huber_vp = loss_huber.item()
                metrics_dict.update(
                    {
                        "mape_vp": mape_vp,
                        "huber_vp": huber_vp,
                    }
                )
        self.log_dict(metrics_dict, on_step=True, batch_size=1, sync_dist=True)
        return metrics_dict

    def test_step(self, graphs, batch_idx) -> STEP_OUTPUT:
        return self.validation_step(graphs, batch_idx)
