# Copyright 2023 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Library file for executing training and evaluation on ogbg-molpcba."""

import os
import os.path as osp
from typing import Any, Dict, Iterable, Tuple, Optional

from absl import logging
from clu import checkpoint
from clu import metric_writers
from clu import metrics
from clu import parameter_overview
from clu import periodic_actions
import flax
import flax.core
import flax.linen as nn
from flax.training import train_state
import jax
import jax.numpy as jnp
import jraph
import ml_collections
import numpy as np
import optax

import models

import ml_pc_saft

from torch_geometric.loader import DataLoader
from graphdataset import ThermoMLDataset, ThermoMLjax, ParametersDataset
from jraphdataloading import get_batched_padded_graph_tuples as batchedjax

import wandb


def create_model(config: ml_collections.ConfigDict, deterministic: bool) -> nn.Module:
    """Creates a Flax model, as specified by the config."""
    if config.model == "GraphNet":
        return models.GraphNet(
            latent_size=config.latent_size,
            num_mlp_layers=config.num_mlp_layers,
            message_passing_steps=config.message_passing_steps,
            output_globals_size=config.num_para,
            dropout_rate=config.dropout_rate,
            skip_connections=config.skip_connections,
            layer_norm=config.layer_norm,
            use_edge_model=config.use_edge_model,
            deterministic=deterministic,
        )
    if config.model == "GraphConvNet":
        return models.GraphConvNet(
            latent_size=config.latent_size,
            num_mlp_layers=config.num_mlp_layers,
            message_passing_steps=config.message_passing_steps,
            output_globals_size=config.num_para,
            dropout_rate=config.dropout_rate,
            skip_connections=config.skip_connections,
            layer_norm=config.layer_norm,
            deterministic=deterministic,
        )
    raise ValueError(f"Unsupported model: {config.model}.")


def create_optimizer(config: ml_collections.ConfigDict) -> optax.GradientTransformation:
    """Creates an optimizer, as specified by the config."""
    if config.optimizer == "adam":
        return optax.adamw(learning_rate=config.learning_rate)
    if config.optimizer == "sgd":
        return optax.sgd(learning_rate=config.learning_rate, momentum=config.momentum)
    raise ValueError(f"Unsupported optimizer: {config.optimizer}.")


def add_prefix_to_keys(result: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """Adds a prefix to the keys of a dict, returning a new dict."""
    return {f"{prefix}_{key}": val for key, val in result.items()}


@flax.struct.dataclass
class EvalMetrics(metrics.Collection):
    loss: metrics.Average.from_output("loss")
    nan_number: metrics.Average.from_output("nan_number")
    errp: metrics.Average.from_output("errp")


@flax.struct.dataclass
class TrainMetrics(metrics.Collection):
    loss: metrics.Average.from_output("loss")
    nan_number: metrics.Average.from_output("nan_number")
    errp: metrics.Average.from_output("errp")


def replace_globals(graphs: jraph.GraphsTuple) -> jraph.GraphsTuple:
    """Replaces the globals attribute with a constant feature for each graph."""
    return graphs._replace(globals=jnp.ones([graphs.n_node.shape[0], 1]))


def get_predicted_para(
    state: train_state.TrainState,
    graphs: jraph.GraphsTuple,
    rngs: Optional[Dict[str, jnp.ndarray]],
) -> jnp.ndarray:
    """Get predicted logits from the network for input graphs."""
    pred_graphs = state.apply_fn(state.params, graphs, rngs=rngs)
    para = pred_graphs.globals
    return para

@jax.jit
def train_step(
    state: train_state.TrainState,
    graphs: jraph.GraphsTuple,
    rngs: Dict[str, jnp.ndarray],
) -> Tuple[train_state.TrainState, metrics.Collection]:
    """Performs one update step over the current batch of graphs."""

    def loss_fn(params, graphs: jraph.GraphsTuple):
        curr_state = state.replace(params=params)

        # Extract system state and experimental properties.
        sysstate = graphs.globals[:-1].reshape(-1,7)
        actual_prop = sysstate[:,-1]

        # Replace the global feature for graph prediction.
        graphs = replace_globals(graphs)

        # Compute predicted properties and resulting loss.
        pcsaft_params = get_predicted_para(curr_state, graphs, rngs)[:-1,:]
        pred_prop = ml_pc_saft.batch_pcsaft_layer(pcsaft_params, sysstate)
        loss = optax.log_cosh(pred_prop, actual_prop)
        mean_loss = jnp.nanmean(loss)

        return mean_loss, pred_prop

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, pred_prop), grads = grad_fn(state.params, graphs)
    state = state.apply_gradients(grads=grads)
    errp = jnp.nanmean((pred_prop / y) * 100.0)
    nan_number = jnp.sum(jnp.isnan(pred_prop))

    metrics_update = TrainMetrics.single_from_model_output(
        loss=loss,
        errp=errp,
        nan_number=nan_number,
    )
    return state, metrics_update

@jax.jit
def pre_train_step(
    state: train_state.TrainState,
    graphs: jraph.GraphsTuple,
    rngs: Dict[str, jnp.ndarray],
) -> Tuple[train_state.TrainState, metrics.Collection]:
    """Performs one update step over the current batch of graphs."""

    def loss_fn(params, graphs: jraph.GraphsTuple):
        curr_state = state.replace(params=params)

        # Extract fitted pcsaft parameters.
        actual_para = graphs.globals[:-1].reshape(-1,17)
        
        # Replace the global feature for graph prediction.
        graphs = replace_globals(graphs)

        # Compute predicted properties and resulting loss.
        pcsaft_params = get_predicted_para(curr_state, graphs, rngs)[:-1,:]
        loss = optax.log_cosh(pcsaft_params, actual_para)
        mean_loss = jnp.nanmean(loss)

        return mean_loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params, graphs)
    state = state.apply_gradients(grads=grads)

    metrics_update = TrainMetrics.single_from_model_output(
        loss=loss
    )
    return state, metrics_update


@jax.jit
def evaluate_step(
    state: train_state.TrainState,
    graphs: jraph.GraphsTuple,
    y: jnp.ndarray,
) -> metrics.Collection:
    """Computes metrics over a set of graphs."""

    # The target properties our model has to predict.
    sysstate = graphs.globals[:-1].reshape(-1,7)
    actual_prop = sysstate[:,-1]

    # Replace the global feature for graph prediction.
    graphs = replace_globals(graphs)

    # Get predicted properties.
    parameters = get_predicted_para(state, graphs, rngs=None)[:-1,:]
    pred_prop = ml_pc_saft.batch_pcsaft_layer(parameters, sysstate)

    # Compute the various metrics.
    loss = optax.log_cosh(pred_prop, actual_prop)
    loss = jnp.nanmean(loss)
    errp = jnp.nanmean((pred_prop / actual_prop) * 100.0)
    nan_number = jnp.sum(jnp.isnan(pred_prop))

    return EvalMetrics.single_from_model_output(
        loss=loss,
        errp=errp,
        nan_number=nan_number,
    )


def evaluate_model(
    state: train_state.TrainState,
    dataloaders: Dict[str, DataLoader],
    splits: Iterable[str],
) -> Dict[str, metrics.Collection]:
    """Evaluates the model on metrics over the specified splits."""

    # Loop over each split independently.
    eval_metrics = {}
    for split in splits:
        split_metrics = None

        # Loop over graphs.
        for graphs in dataloaders[split]:
            graphs, y = batchedjax(graphs)
            split_metrics_update = evaluate_step(state, graphs, y)

            # Update metrics.
            if split_metrics is None:
                split_metrics = split_metrics_update
            else:
                split_metrics = split_metrics.merge(split_metrics_update)
        eval_metrics[split] = split_metrics

    return eval_metrics  # pytype: disable=bad-return-type


def train_and_evaluate(
    config: ml_collections.ConfigDict, workdir: str
) -> train_state.TrainState:
    """Execute model training and evaluation loop.

    Args:
      config: Hyperparameter configuration for training and evaluation.
      workdir: Directory where the TensorBoard summaries are written to.

    Returns:
      The train state (which includes the `.params`).
    """
    # We only support single-host training.
    assert jax.process_count() == 1

    # Create writer for logs.
    writer = metric_writers.create_default_writer(workdir)
    writer.write_hparams(config.to_dict())

    wandb.login()
    run = wandb.init(
        # Set the project where this run will be logged
        project="gnn-pc-saft",
        # Track hyperparameters and run metadata
        config=config.to_dict(),
    )

    # Get datasets, organized by split.
    logging.info("Obtaining datasets.")

    if config.pre_train:
        path = osp.join('data','parameters')
        train_dataset = ParametersDataset(path)
        val_dataset = train_dataset
    else:
        path = osp.join("data", "thermoml", "train")
        train_dataset = ThermoMLDataset(path, Notebook=True, subset="train")
        path = osp.join("data", "thermoml", "val")
        val_dataset = ThermoMLDataset(path, Notebook=True, subset="val")

    train_dataset = ThermoMLjax(train_dataset) 
    val_dataset = ThermoMLjax(val_dataset)
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True
    )
    train_loader = iter(train_loader)
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, drop_last=True
    )
    val_loader = iter(val_loader)

    # Create and initialize the network.
    logging.info("Initializing network.")
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    init_graphs, _ = batchedjax(next(train_loader))
    init_graphs = replace_globals(init_graphs)
    init_net = create_model(config, deterministic=True)
    params = jax.jit(init_net.init)(init_rng, init_graphs)
    parameter_overview.log_parameter_overview(params)

    # Create the optimizer.
    tx = create_optimizer(config)

    # Create the training state.
    net = create_model(config, deterministic=False)
    state = train_state.TrainState.create(apply_fn=net.apply, params=params, tx=tx)

    # Set up checkpointing of the model.
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    ckpt = checkpoint.Checkpoint(checkpoint_dir, max_to_keep=2)
    state = ckpt.restore_or_initialize(state)
    initial_step = int(state.step) + 1

    # Create the evaluation state, corresponding to a deterministic model.
    eval_net = create_model(config, deterministic=True)
    eval_state = state.replace(apply_fn=eval_net.apply)

    # Hooks called periodically during training.
    report_progress = periodic_actions.ReportProgress(
        num_train_steps=config.num_train_steps, writer=writer
    )
    profiler = periodic_actions.Profile(num_profile_steps=5, logdir=workdir)
    hooks = [report_progress, profiler]

    # Begin training loop.
    logging.info("Starting training.")
    train_metrics = None
    for step in range(initial_step, config.num_train_steps + 1):
        # Split PRNG key, to ensure different 'randomness' for every step.
        rng, dropout_rng = jax.random.split(rng)

        # Perform one step of training.
        with jax.profiler.StepTraceAnnotation("train", step_num=step):
            graphs = batchedjax(next(train_loader))
            graphs = jax.device_put(graphs, jax.devices()[0])

            if config.pre_train:
                state, metrics_update = pre_train_step(
                    state, graphs, rngs={"dropout": dropout_rng}
                    )
            else:
                state, metrics_update = train_step(
                state, graphs, rngs={"dropout": dropout_rng}
            )

            # Update metrics.
            if train_metrics is None:
                train_metrics = metrics_update
            else:
                train_metrics = train_metrics.merge(metrics_update)

        # Quick indication that training is happening.
        logging.log_first_n(logging.INFO, "Finished training step %d.", 10, step)
        for hook in hooks:
            hook(step)

        # Log, if required.
        is_last_step = step == config.num_train_steps - 1
        if step % config.log_every_steps == 0 or is_last_step:
            wandb.log(add_prefix_to_keys(train_metrics.compute(), "train"), step=step)
            train_metrics = None

        # Evaluate on validation and test splits, if required.
        if step % config.eval_every_steps == 0 or is_last_step:
            eval_state = eval_state.replace(params=state.params)

            splits = ["validation"]
            dataloaders = {"validation": val_loader}
            with report_progress.timed("eval"):
                eval_metrics = evaluate_model(eval_state, dataloaders, splits=splits)
            for split in splits:
                wandb.log(
                    add_prefix_to_keys(eval_metrics[split].compute(), split), step=step
                )

        # Checkpoint model, if required.
        if step % config.checkpoint_every_steps == 0 or is_last_step:
            with report_progress.timed("checkpoint"):
                ckpt.save(state)
    wandb.finish()
    return state
