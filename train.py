import os
import os.path as osp
from typing import Any, Dict, Tuple, Optional

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
from flax.training import dynamic_scale as dynamic_scale_lib
import jax
import jax.numpy as jnp
import jraph
import ml_collections
import numpy as np
import optax

import models

from torch_geometric.loader import DataLoader


import wandb

import functools

from graphdataset import pureTMLDataset
from graph import from_InChI
from ml_pc_saft import batch_den, batch_VP
from jraphdataloading import get_padded_array, pad_graph_to_nearest_power_of_two as get_padded_graph
import pickle


def create_model(config: ml_collections.ConfigDict, deterministic: bool) -> nn.Module:
    """Creates a Flax model, as specified by the config."""
    platform = jax.local_devices()[0].platform
    if config.half_precision:
        if platform == 'tpu':
            model_dtype = jnp.bfloat16
        else:
            model_dtype = jnp.float16
    else:
        model_dtype = jnp.float32
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
            dtype=model_dtype
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
            dtype=model_dtype
        )
    raise ValueError(f"Unsupported model: {config.model}.")


def create_optimizer(
    config: ml_collections.ConfigDict, 
    lr_scheduler: optax.Schedule
) -> optax.GradientTransformation:
    """Creates an optimizer, as specified by the config."""
    if config.optimizer == "adam":
        return optax.adamw(learning_rate=config.learning_rate)
    if config.optimizer == "sgd":
        return optax.sgd(learning_rate=config.learning_rate, momentum=config.momentum)
    raise ValueError(f"Unsupported optimizer: {config.optimizer}.")


def create_learning_rate_fn(config: ml_collections.ConfigDict):
    """Creates learning rate schedule."""
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=config.learning_rate,
        transition_steps=config.warmup_steps,
    )
    cosine_steps = max(config.num_train_steps - config.warmup_steps, 1)
    cosine_fn = optax.cosine_decay_schedule(
        init_value=config.learning_rate, decay_steps=cosine_steps
    )
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn], boundaries=[config.warmup_steps]
    )
    return schedule_fn


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
    msle: metrics.Average.from_output("msle")
    nan_number: metrics.Average.from_output("nan_number")
    errp: metrics.Average.from_output("errp")
    lr: metrics.Average.from_output("lr")


@flax.struct.dataclass
class PreTrainMetrics(metrics.Collection):
    cosdist: metrics.Average.from_output("cosdist")
    msle: metrics.Average.from_output("msle")
    lr: metrics.Average.from_output("lr")


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


def train_step(
    state: train_state.TrainState,
    graphs: jraph.GraphsTuple,
    datapoints: jnp.ndarray,
    rngs: Dict[str, jnp.ndarray],
    learning_rate_fn: optax.Schedule,
) -> Tuple[train_state.TrainState, metrics.Collection]:
    """Performs one update step over the current batch of graphs."""

    def loss_fn(params, graphs: jraph.GraphsTuple, datapoints: jnp.ndarray):
        curr_state = state.replace(params=params)
        datapoints = datapoints.astype(jnp.float64)

        # Extract system state and experimental properties.
        actual_prop = datapoints[:, -1].astype(jnp.float32)

        # Replace the global feature for graph prediction.
        graphs = replace_globals(graphs)

        # Compute predicted properties and resulting loss.
        pcsaft_params = get_predicted_para(curr_state, graphs, rngs)[:-1, :]
        pcsaft_params = pcsaft_params.squeeze().astype(jnp.float64)
        pred_prop = batch_den(pcsaft_params, datapoints).astype(jnp.float32)
        loss = jnp.square(
            jnp.log(jnp.abs(actual_prop) + 1) - jnp.log(jnp.abs(pred_prop) + 1)
        )
        mean_loss = jnp.nanmean(loss)

        return mean_loss, (pred_prop, actual_prop)

    dynamic_scale = state.dynamic_scale
    if dynamic_scale:
        grad_fn = dynamic_scale.value_and_grad(loss_fn, has_aux=True)
        dynamic_scale, is_fin, aux, grads = grad_fn(state.params, graphs, datapoints)
        # dynamic loss takes care of averaging gradients across replicas
    else:
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        aux, grads = grad_fn(state.params, graphs, datapoints)
    
    (loss, (pred_prop, y)) = aux
    new_state = state.apply_gradients(grads=grads)
    errp = jnp.nanmean((pred_prop / y) * 100.0)
    nan_number = jnp.sum(jnp.isnan(pred_prop))

    lr = learning_rate_fn(state.step)

    if dynamic_scale:
    # if is_fin == False the gradients contain Inf/NaNs and optimizer state and
    # params should be restored (= skip this step).
        new_state = new_state.replace(
            opt_state=jax.tree_util.tree_map(
                functools.partial(jnp.where, is_fin),
                new_state.opt_state,
                state.opt_state),
            params=jax.tree_util.tree_map(
                functools.partial(jnp.where, is_fin),
                new_state.params,
                state.params),
            dynamic_scale=dynamic_scale)

    metrics_update = TrainMetrics.single_from_model_output(
        msle=loss,
        errp=errp,
        nan_number=nan_number,
        lr=lr,
    )
    return new_state, metrics_update


@functools.partial(jax.jit, static_argnums=3)
def pre_train_step(
    state: train_state.TrainState,
    graphs: jraph.GraphsTuple,
    rngs: Dict[str, jnp.ndarray],
    learning_rate_fn: optax.Schedule,
) -> Tuple[train_state.TrainState, metrics.Collection]:
    """Performs one update step over the current batch of graphs."""

    def loss_fn(params, graphs: jraph.GraphsTuple):
        curr_state = state.replace(params=params)

        # Extract fitted pcsaft parameters.
        actual_para = graphs.globals[:-1].reshape(-1, 17)

        # Replace the global feature for graph prediction.
        graphs = replace_globals(graphs)

        # Compute predicted properties and resulting loss.
        pcsaft_params = get_predicted_para(curr_state, graphs, rngs)[:-1, :]
        msle = jnp.square(
            jnp.log(jnp.abs(actual_para) + 1) - jnp.log(jnp.abs(pcsaft_params) + 1)
        )
        loss = optax.cosine_distance(pcsaft_params, actual_para)
        mean_loss = jnp.nanmean(loss)

        return mean_loss, msle

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, msle), grads = grad_fn(state.params, graphs)
    state = state.apply_gradients(grads=grads)

    lr = learning_rate_fn(state.step)

    metrics_update = PreTrainMetrics.single_from_model_output(
        cosdist=loss,
        msle=msle,
        lr=lr,
    )
    return state, metrics_update


@jax.jit
def evaluate_step(
    state: train_state.TrainState,
    graphs: jraph.GraphsTuple,
) -> metrics.Collection:
    """Computes metrics over a set of graphs."""

    # The target properties our model has to predict.
    sysstate = graphs.globals[:-1].reshape(-1, 7)
    actual_prop = sysstate[:, -1]

    # Replace the global feature for graph prediction.
    graphs = replace_globals(graphs)

    # Get predicted properties.
    parameters = get_predicted_para(state, graphs, rngs=None)[:-1, :]
    pred_prop = batch_den(parameters, sysstate)

    # Compute the various metrics.
    loss = jnp.square(
        jnp.log(jnp.abs(actual_prop) + 1) - jnp.log(jnp.abs(pred_prop) + 1)
    )
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
    dataloader: DataLoader,
) -> metrics.Collection:
    """Evaluates the model on metrics over the specified splits."""
    eval_metric = None
    # Loop over graphs.
    for graphs in dataloader:
        graphs = jax.tree_util.tree_map(np.asarray, graphs)
        eval_metric_update = evaluate_step(state, graphs)

        # Update metrics.
        if eval_metric is None:
            eval_metric = eval_metric_update
        else:
            eval_metric = eval_metric.merge(eval_metric_update)

    return eval_metric

class TrainState(train_state.TrainState):
  dynamic_scale: dynamic_scale_lib.DynamicScale

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
    
    platform = jax.local_devices()[0].platform

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

    
    path = './data/thermoml'
    train_dict = pureTMLDataset(path)

    if config.half_precision:
        input_dtype = jnp.int16
    else:
        input_dtype = jnp.int32

    # Create and initialize the network.
    logging.info("Initializing network.")
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    inchis = list(train_dict.keys())
    graph = from_InChI(inchis[0], dtype=input_dtype)
    init_graphs = get_padded_graph(graph)
    init_graphs = replace_globals(init_graphs)
    init_net = create_model(config, deterministic=True)
    params = jax.jit(init_net.init)(init_rng, init_graphs)
    parameter_overview.log_parameter_overview(params)

    # Create scheduler.
    sch = create_learning_rate_fn(config)

    # Create the optimizer.
    tx = create_optimizer(config, sch)

    # Create the training state.
    net = create_model(config, deterministic=False)
    dynamic_scale = None
    
    if config.half_precision and platform == 'gpu':
        dynamic_scale = dynamic_scale_lib.DynamicScale()
    else:
        dynamic_scale = None
    state = TrainState.create(apply_fn=net.apply, params=params, tx=tx, dynamic_scale=dynamic_scale)

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
    max_pad = config.max_pad
    train_metrics = None
    step = initial_step
    idxs = jnp.arange(len(train_dict))
    while step < config.num_train_steps + 1:
        rng, subkey = jax.random.split(rng)
        idxs = jax.random.permutation(subkey, idxs, independent=True)
        for idx in idxs:
            if 1 not in train_dict[inchis[idx]]:
                continue
            # Split PRNG key, to ensure different 'randomness' for every step.
            rng, dropout_rng = jax.random.split(rng)
            rng, subkey = jax.random.split(rng)

            # Perform one step of training.
            with jax.profiler.StepTraceAnnotation("train", step_num=step):
                graphs = get_padded_graph(from_InChI(inchis[idx], dtype = input_dtype))
                datapoints, _ = get_padded_array(train_dict[inchis[idx]][1], subkey, max_pad)
                
                state, metrics_update = train_step(
                    state,
                    graphs,
                    datapoints,
                    {"dropout": dropout_rng},
                    sch,
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
                wandb.log(
                    add_prefix_to_keys(train_metrics.compute(), "train"), step=step
                )
                train_metrics = None

            # Evaluate on validation and test splits, if required.
            #if step % config.eval_every_steps == 0 or (
            #    is_last_step & (not config.pre_train)
            #):
            #    eval_state = eval_state.replace(params=state.params)
            #
            #    with report_progress.timed("eval"):
            #        eval_metrics = evaluate_model(eval_state, val_loader)
            #        wandb.log(
            #            add_prefix_to_keys(eval_metrics.compute(), "val"), step=step
            #        )

            # Checkpoint model, if required.
            if step % config.checkpoint_every_steps == 0 or is_last_step:
                with report_progress.timed("checkpoint"):
                    ckpt.save(state)
            step += 1
    wandb.finish()
    return state
