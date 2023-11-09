import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList, ReLU, Sequential, BatchNorm1d
from torch_geometric.nn import PNAConv, global_add_pool, BatchNorm
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops


class PNAPCSAFT(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        propagation_depth: int,
        pre_layers: int,
        post_layers: int,
        num_mlp_layers: int,
        num_para: int,
        deg: torch.Tensor,
        dropout: float = 0.0,
        skip_connections: bool = False,
        add_self_loops: bool = False,
    ):
        super().__init__()

        aggregators = ["mean", "min", "max", "std"]
        scalers = ["identity", "amplification", "attenuation"]
        self.dropout = dropout
        self.num_mlp_layers = num_mlp_layers
        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        self.skip_connections = skip_connections
        self.add_self_loops = add_self_loops

        conv = PNAConv(
            in_channels=9,
            out_channels=hidden_dim,
            aggregators=aggregators,
            scalers=scalers,
            deg=deg,
            edge_dim=3,
            towers=4,
            pre_layers=pre_layers,
            post_layers=post_layers,
            divide_input=False,
        )
        self.convs.append(conv)
        self.batch_norms.append(BatchNorm(hidden_dim))

        for _ in range(propagation_depth - 1):
            conv = PNAConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                aggregators=aggregators,
                scalers=scalers,
                deg=deg,
                edge_dim=3,
                towers=4,
                pre_layers=pre_layers,
                post_layers=post_layers,
                divide_input=False,
            )
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(hidden_dim))

        self.mlp = Sequential(
            Linear(hidden_dim, hidden_dim),
            BatchNorm1d(hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            BatchNorm1d(hidden_dim),
            ReLU(),
        )
        self.ouput = Sequential(
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
        data: Data,
    ):
        x, edge_index, edge_attr, batch = (
            data.x.to(torch.float),
            data.edge_index,
            data.edge_attr.to(torch.float),
            data.batch,
        )

        if self.add_self_loops:
            edge_index, edge_attr = add_self_loops(
                edge_index, edge_attr, num_nodes=x.size(0)
            )

        x_scale = torch.tensor([[10.0, 10.0, 1000.0]], device=x.device)
        ground = torch.tensor([[1.0, 0.0, 0.0]], device=x.device)

        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index, edge_attr)))
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_add_pool(x, batch)
        for _ in range(self.num_mlp_layers - 1):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.mlp(x)
        x = self.ouput(x) * x_scale + ground
        return x


class PNAPCSAFT2(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        propagation_depth: int,
        pre_layers: int,
        post_layers: int,
        num_mlp_layers: int,
        num_para: int,
        deg: torch.Tensor,
        dropout: float = 0.0,
        skip_connections: bool = False,
        add_self_loops: bool = False,
    ):
        super().__init__()

        aggregators = ["mean", "min", "max", "std"]
        scalers = ["identity", "amplification", "attenuation"]
        self.dropout = dropout
        self.num_mlp_layers = num_mlp_layers
        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        self.skip_connections = skip_connections
        self.add_self_loops = add_self_loops

        self.node_embed = Linear(32, hidden_dim)

        self.edge_embed = Linear(11, hidden_dim)

        for _ in range(propagation_depth):
            conv = PNAConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                aggregators=aggregators,
                scalers=scalers,
                deg=deg,
                edge_dim=hidden_dim,
                towers=1,
                pre_layers=pre_layers,
                post_layers=post_layers,
                divide_input=False,
            )
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(hidden_dim))

        self.mlp = Sequential(
            Linear(hidden_dim, hidden_dim), BatchNorm1d(hidden_dim), ReLU()
        )
        self.ouput = Sequential(
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
        data: Data,
    ):
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        if self.add_self_loops:
            edge_index, edge_attr = add_self_loops(
                edge_index, edge_attr, num_nodes=x.size(0)
            )
        x = F.leaky_relu(self.node_embed(x))
        edge_attr = F.leaky_relu(self.edge_embed(edge_attr))

        for conv, batch_norm in zip(self.convs, self.batch_norms):
            if self.skip_connections:
                x_previous = x
            x = F.relu(batch_norm(conv(x, edge_index, edge_attr)))
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.skip_connections:
                x = x + x_previous

        x = global_add_pool(x, batch)
        for _ in range(self.num_mlp_layers):
            x = self.mlp(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.ouput(x)
        return x
