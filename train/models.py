"""Module with all available Graph Neural Network models developed and used in the project"""
import dataclasses

import torch
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch.nn import BatchNorm1d, Dropout, Linear, ModuleList, ReLU, Sequential
from torch_geometric.data import Data
from torch_geometric.nn import BatchNorm, PNAConv, global_add_pool
from torch_geometric.utils import add_self_loops


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
class RadoutMLPParams:
    "Parameters for the MLP layers."
    num_mlp_layers: int
    num_para: int


class PNAPCSAFT2(torch.nn.Module):
    """Graph neural network to predict ePCSAFT parameters"""

    def __init__(
        self,
        hidden_dim: int,
        pna_params: PnaconvsParams,
        mlp_params: RadoutMLPParams,
        dropout: float = 0.0,
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
            self.mlp.append(Dropout(p=dropout))

        self.ouput = Sequential(
            Linear(hidden_dim, hidden_dim // 2),
            BatchNorm1d(hidden_dim // 2),
            ReLU(),
            Dropout(p=dropout),
            Linear(hidden_dim // 2, hidden_dim // 4),
            BatchNorm1d(hidden_dim // 4),
            ReLU(),
            Dropout(p=dropout),
            Linear(hidden_dim // 4, mlp_params.num_para),
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
        x = self.ouput(x)
        return x
