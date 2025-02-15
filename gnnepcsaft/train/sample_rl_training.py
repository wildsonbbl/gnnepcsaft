"Module to be used for testing a RL training"

import copy
import math
import multiprocessing as mp
import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch import nn, optim
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from ..data.graphdataset import ThermoMLDataset
from ..data.rdkit_util import assoc_number
from .models import GNNePCSAFT, GNNePCSAFTL
from .utils import rhovp_data

# pylint: disable=missing-function-docstring, missing-class-docstring, invalid-name

file_dir = os.path.dirname(os.path.abspath(__file__))

workdir = os.getcwd()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state):
        self.buffer.append(
            (
                state.detach().cpu(),
                action.detach().cpu(),
                reward.detach().cpu(),
                next_state.detach().cpu(),
            )
        )

    def sample(self, _batch_size):
        batch = random.sample(self.buffer, _batch_size)
        return zip(*batch)

    def __len__(self):
        return len(self.buffer)


# Simple Gaussian noise
def add_exploration_noise(action, _noise_scale):
    noise = torch.randn_like(action) * _noise_scale
    return action + noise


# Define the Actor Network (GNN)
class GNNActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.msigmae: GNNePCSAFT  # trained model
        self.assoc: GNNePCSAFT  # trained model

    def forward(self, _data) -> torch.Tensor:
        x, edge_index, edge_attr, batch = (
            _data.x,
            _data.edge_index,
            _data.edge_attr,
            _data.batch,
        )
        msigmae: torch.Tensor = self.msigmae(x, edge_index, edge_attr, batch)
        assoc: torch.Tensor = self.assoc(x, edge_index, edge_attr, batch)
        _action = torch.hstack((msigmae, assoc))
        return _action


# Define the Critic Network
class Critic(nn.Module):
    def __init__(
        self,
        hidden_dim,
    ):
        super().__init__()
        # State processing
        self.state_head: GNNePCSAFT
        # Action processing
        self.action_head = nn.Sequential(
            nn.Linear(5, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4),
        )
        # Q-value output
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, _state, _action) -> torch.Tensor:
        x, edge_index, edge_attr, batch = (
            _state.x,
            _state.edge_index,
            _state.edge_attr,
            _state.batch,
        )
        state_features = self.state_head(x, edge_index, edge_attr, batch)
        action_features = self.action_head(_action)
        combined = torch.cat([state_features, action_features], dim=1)
        q_value = self.value_head(combined)
        return q_value


# PCSAFT parameters bounds
params_lower_bound = np.array([1.0, 1.9, 50.0, 1e-4, 200.0])
params_upper_bound = np.array([25.0, 4.5, 550.0, 0.9, 5000.0])

params_lower_bound_torch = torch.tensor(
    [1.0, 1.9, 50.0, -1 * math.log10(0.9), math.log10(200.0)],
    dtype=torch.float32,
    device=device,
)
params_upper_bound_torch = torch.tensor(
    [25.0, 4.5, 550.0, -1 * math.log10(0.0001), math.log10(5000.0)],
    dtype=torch.float32,
    device=device,
)

# Training parameters
num_episodes = 100_000
max_buffer_size = 100
batch_size = 10
discount_factor = 0.99  # Gamma
tau = 0.005  # Soft update parameter
actor_learning_rate = 1e-5
critic_learning_rate = 1e-4
noise_scale = 0.1  # For exploration

# start wandb project logging
wandb.init(
    # Set the project where this run will be logged
    project="gnn-pc-saft",
    # Track hyperparameters and run metadata
    config={
        "num_episodes": num_episodes,
        "batch_size": batch_size,
        "actor_learning_rate": actor_learning_rate,
        "critic_learning_rate": critic_learning_rate,
    },
    group="rl_training",
    tags=["rl_training", "train"],
    job_type="train",
)

# Initialize actor and critic
actor = GNNActor()
# pylint: disable=E1120
actor.msigmae = GNNePCSAFTL.load_from_checkpoint(
    checkpoint_path=os.path.join(
        workdir,
        "gnnepcsaft/train/checkpoints/esper_msigmae_7-epoch=72499-train_mape=0.0069.ckpt",
    )
).model
actor.assoc = GNNePCSAFTL.load_from_checkpoint(
    checkpoint_path=os.path.join(
        workdir,
        "gnnepcsaft/train/checkpoints/esper_assoc_8-epoch=149999-train_mape=0.0030.ckpt",
    )
).model
actor.to(device)
# pylint: enable=E1120
critic = Critic(hidden_dim=256)
critic.to(device)

# Initialize target networks
target_actor = GNNActor()
target_actor.msigmae = copy.deepcopy(actor.msigmae)
target_actor.assoc = copy.deepcopy(actor.assoc)
target_actor.to(device)

target_critic = Critic(hidden_dim=256)
target_critic.load_state_dict(critic.state_dict())
target_critic.to(device)

# Ensure target networks are in evaluation mode
target_actor.eval()
target_critic.eval()


actor_optimizer = optim.Adam(actor.parameters(), lr=actor_learning_rate)
critic_optimizer = optim.Adam(critic.parameters(), lr=critic_learning_rate)

# Experience replay buffer
replay_buffer = ReplayBuffer(max_size=max_buffer_size)

tml = ThermoMLDataset(os.path.join(workdir, "gnnepcsaft/data/thermoml"))
tml_data = {}
for graph in tml:
    tml_data[graph.InChI] = (graph.rho, graph.vp)
tml_dataloader = DataLoader(tml, batch_size=64, shuffle=True)


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
    # tml_dataloader deals with batching
    # actions = (batch_size, 5),
    # q_values = (batch_size, 1),
    # rewards = (batch_size)
    for graphs in tml_dataloader:
        # Move to device
        graphs.to(device)
        # Actor forward pass
        actions = actor(graphs)
        # Add exploration noise
        actions_noise: torch.Tensor = add_exploration_noise(actions, noise_scale)
        actions_noise = actions_noise.clamp(
            params_lower_bound_torch, params_upper_bound_torch
        )

        # Compute rewards
        batch_pc_saft_params = actions_noise.detach().cpu().numpy()
        args_list = [
            (batch_pc_saft_params[idx], inchi) for idx, inchi in enumerate(graphs.InChI)
        ]

        # Use multiprocessing pool to calculate rewards with pc-saft parameters
        with mp.Pool(processes=mp.cpu_count()) as pool:
            rewards = pool.map(compute_reward, args_list)
        # Convert rewards to torch tensor
        rewards = torch.tensor(rewards, dtype=torch.float, device=device)
        rewards_norm = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # For DDPG, next state is the same as current state in a stateless environment
        next_graphs = graphs.clone().to(device)
        # Push experience into replay buffer
        replay_buffer.push(graphs.clone(), actions_noise, rewards_norm, next_graphs)

        # Training step after enough samples are collected
        if len(replay_buffer) > batch_size:
            # Sample a mini-batch from the replay buffer
            (
                state_batch_list,
                action_batch_list,
                reward_batch_list,
                next_state_batch_list,
            ) = replay_buffer.sample(batch_size)

            for idx in range(batch_size):
                state_batch = state_batch_list[idx].to(device)
                action_batch = action_batch_list[idx].to(device)
                reward_batch = reward_batch_list[idx].to(device)
                next_state_batch = next_state_batch_list[idx].to(device)

                # Compute target Q-values
                with torch.no_grad():
                    next_actions = target_actor(next_state_batch)
                    next_q_values = target_critic(
                        next_state_batch, next_actions
                    ).squeeze()
                    target_q_values = reward_batch + discount_factor * next_q_values

                # Critic loss
                q_values = critic(state_batch, action_batch).squeeze()
                critic_loss = F.mse_loss(q_values, target_q_values)

                # Update critic
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                # Actor loss (maximize Q-values)
                actor_actions = actor(state_batch)
                actor_loss = -critic(state_batch, actor_actions).mean()

                # Update actor
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # Soft update target networks
                for target_param, param in zip(
                    target_critic.parameters(), critic.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1.0 - tau) * target_param.data
                    )
                for target_param, param in zip(
                    target_actor.parameters(), actor.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1.0 - tau) * target_param.data
                    )

            # Logging
            avg_reward = rewards.mean().item()
            avg_actor_loss = actor_loss.item()
            avg_critic_loss = critic_loss.item()
            metrics = {
                "avg_reward": avg_reward,
                "actor_loss": avg_actor_loss,
                "critic_loss": avg_critic_loss,
            }
            wandb.log(metrics)
            print(metrics)

print("Training completed.")
