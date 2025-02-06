"Module to be used for RL training"

import multiprocessing as mp
import os

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch import nn, optim
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from ..data.graphdataset import ThermoMLDataset
from ..data.rdkit_util import assoc_number
from .models import GATPCSAFT, PCsaftL
from .utils import rhovp_data

# pylint: disable=missing-function-docstring, missing-class-docstring, invalid-name

file_dir = os.path.dirname(os.path.abspath(__file__))

workdir = os.getcwd()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define the Actor Network (GNN)
class GNNActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.msigmae: GATPCSAFT  # trained model
        self.assoc: GATPCSAFT  # trained model
        self.log_std = nn.Parameter(torch.ones(5))  # Learnable standard deviation

    def forward(self, _data) -> torch.Tensor:
        x, edge_index, edge_attr, batch = (
            _data.x,
            _data.edge_index,
            _data.edge_attr,
            _data.batch,
        )
        msigmae: torch.Tensor = self.msigmae(x, edge_index, edge_attr, batch)
        assoc: torch.Tensor = self.assoc(x, edge_index, edge_attr, batch)
        _mean = torch.hstack((msigmae, assoc))
        _std = torch.exp(self.log_std).expand_as(_mean)
        return _mean, _std


# Define the Critic Network
class Critic(nn.Module):
    def __init__(self, hidden_dim, propagation_depth, dropout, heads):
        super().__init__()
        self.state_head = GATPCSAFT(hidden_dim, propagation_depth, 5, dropout, heads)
        self.action_head = nn.Sequential(
            nn.Linear(5, 5),
            nn.ReLU(),
            nn.Linear(5, 5),
        )
        self.value_head = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1),
        )

    def forward(self, _state, _action) -> torch.Tensor:
        x, edge_index, edge_attr, batch = (
            _state.x,
            _state.edge_index,
            _state.edge_attr,
            _state.batch,
        )
        state_head = self.state_head(x, edge_index, edge_attr, batch)
        action_head = self.action_head(_action)
        _value = self.value_head(torch.hstack((state_head, action_head)))
        return _value


# PCSAFT parameters bounds
params_lower_bound = np.array([1.0, 1.9, 50.0, 1e-4, 200.0])
params_upper_bound = np.array([25.0, 4.5, 550.0, 0.9, 5000.0])

# Training parameters
num_episodes = 100_000
num_epochs = 10
batch_size = 512
actor_learning_rate = 1e-3
critic_learning_rate = 1e-2
clip_epsilon = 0.2
value_loss_coef = 0.5
entropy_coef = 0.01

# start wandb project logging
wandb.init(
    # Set the project where this run will be logged
    project="gnn-pc-saft",
    # Track hyperparameters and run metadata
    config={
        "num_episodes": num_episodes,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "actor_learning_rate": actor_learning_rate,
        "critic_learning_rate": critic_learning_rate,
        "clip_epsilon": clip_epsilon,
        "value_loss_coef": value_loss_coef,
        "entropy_coef": entropy_coef,
    },
    group="rl_training",
    tags=["rl_training", "train"],
    job_type="train",
)

# Initialize actor and critic
actor = GNNActor()
# pylint: disable=E1120
actor.msigmae = PCsaftL.load_from_checkpoint(
    checkpoint_path=os.path.join(
        workdir,
        "gnnepcsaft/train/checkpoints/esper_msigmae_7-epoch=72499-train_mape=0.0069.ckpt",
    )
).model
actor.assoc = PCsaftL.load_from_checkpoint(
    checkpoint_path=os.path.join(
        workdir,
        "gnnepcsaft/train/checkpoints/esper_assoc_8-epoch=149999-train_mape=0.0030.ckpt",
    )
).model
actor.to(device)
# pylint: enable=E1120
critic = Critic(hidden_dim=256, propagation_depth=8, dropout=0.0, heads=3)
critic.to(device)

actor_optimizer = optim.Adam(actor.parameters(), lr=actor_learning_rate)
critic_optimizer = optim.Adam(critic.parameters(), lr=critic_learning_rate)

tml = ThermoMLDataset(os.path.join(workdir, "gnnepcsaft/data/thermoml"))
tml_data = {}
for graph in tml:
    tml_data[graph.InChI] = (graph.rho, graph.vp)
tml_dataloader = DataLoader(tml, batch_size=batch_size, shuffle=True)


def compute_reward(args):  # pylint: disable=too-many-locals
    params, inchi = args
    na, nb = assoc_number(inchi)

    # Transform parameters
    params[-2] = 10 ** (-params[-2])
    params[-1] = 10 ** (params[-1])
    params_full = np.hstack((params, np.array([0.0, na, nb])))

    # Get states
    states_rho, states_vp = tml_data[inchi]
    rho_pred, vp_pred = rhovp_data(params_full, states_rho.numpy(), states_vp.numpy())

    # Get experimental data
    density_exp = states_rho[:, -1].numpy()
    pressure_exp = states_vp[:, -1].numpy()

    # Calculate errors
    density_error = (
        np.mean((rho_pred - density_exp) / density_exp)
        if len(density_exp) >= 1
        else 0.0
    )
    pressure_error = (
        np.mean((vp_pred - pressure_exp) / pressure_exp)
        if len(pressure_exp) >= 1
        else 0.0
    )

    reward = -(density_error**2 + pressure_error**2)

    # Check parameter bounds
    penalty = 0
    if not np.all(params_lower_bound <= params[:5]) or not np.all(
        params[:5] <= params_upper_bound
    ):
        lower_diff = params_lower_bound - params[:5]
        upper_diff = params[:5] - params_upper_bound
        total_diff = np.sum(np.maximum(0, lower_diff) + np.maximum(0, upper_diff))
        penalty = -total_diff  # Negative penalty
    reward += penalty

    return reward


# Training loop
for episode in tqdm(range(num_episodes)):
    rewards = []
    for graphs in tml_dataloader:
        # batch of graphs
        graphs.to(device)
        with torch.no_grad():
            # Actor forward pass
            mean, std = actor(graphs)

            distribution = torch.distributions.Normal(mean, std)
            actions = distribution.sample()

            # Store log probability and value for training

            old_log_probs = distribution.log_prob(actions).sum(dim=1).detach()
        for epoch in range(num_epochs):
            # Get state representation
            # state = graphs

            # Actor forward pass
            mean, std = actor(graphs)
            distribution = torch.distributions.Normal(mean, std)
            actions = distribution.sample()

            # Store log probability and value for training
            log_probs = distribution.log_prob(actions).sum(dim=1)
            ratio = torch.exp(log_probs - old_log_probs)

            # Critic forward pass
            values = torch.squeeze(critic(graphs, actions))

            # Compute predicted properties using PC-SAFT
            batch_pc_saft_params = actions.detach().cpu().numpy()
            # Prepare arguments for mapping in multiprocessing
            args_list = [
                (batch_pc_saft_params[idx], inchi)
                for idx, inchi in enumerate(graphs.InChI)
            ]

            # Use multiprocessing pool to calculate rewards with pc-saft parameters
            with mp.Pool(processes=mp.cpu_count()) as pool:
                rewards = pool.map(compute_reward, args_list)
            # Convert rewards to torch tensor
            rewards = torch.tensor(rewards, dtype=torch.float, device=device)
            rewards_norm = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

            # Calculate advantages
            advantages = rewards_norm - values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            surr1 = ratio * advantages
            surr2 = (
                torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
            )

            # Actor loss (PPO)
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic loss (Mean Squared Error)
            critic_loss = value_loss_coef * F.mse_loss(values, rewards_norm)

            # Add entropy bonus for exploration
            entropy_loss = -entropy_coef * distribution.entropy().sum(dim=1).mean()
            actor_loss += entropy_loss

            # Optimize
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            actor_optimizer.step()
            critic_optimizer.step()

    # Logging
    avg_reward = rewards.mean().item()
    avg_actor_loss = actor_loss.item()
    wandb.log({"avg_reward": avg_reward, "avg_actor_loss": avg_actor_loss})

print("Training completed.")
