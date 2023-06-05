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

import functools


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


def create_optimizer(config: ml_collections.ConfigDict, lr_scheduler: optax.Schedule) -> optax.GradientTransformation:
    """Creates an optimizer, as specified by the config."""
    if config.optimizer == "adam":
        return optax.adamw(learning_rate=config.learning_rate)
    if config.optimizer == "sgd":
        return optax.sgd(learning_rate=config.learning_rate, momentum=config.momentum)
    raise ValueError(f"Unsupported optimizer: {config.optimizer}.")


def create_learning_rate_fn(config: ml_collections.ConfigDict):
  """Creates learning rate schedule."""
  warmup_fn = optax.linear_schedule(
      init_value=0., end_value=config.learning_rate,
      transition_steps=config.warmup_steps)
  cosine_steps = max(config.num_train_steps - config.warmup_steps, 1)
  cosine_fn = optax.cosine_decay_schedule(
      init_value=config.learning_rate,
      decay_steps=cosine_steps)
  schedule_fn = optax.join_schedules(
      schedules=[warmup_fn, cosine_fn],
      boundaries=[config.warmup_steps])
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
    lr: metrics.Average.from_output('lr')

@flax.struct.dataclass
class PreTrainMetrics(metrics.Collection):
    cosdist: metrics.Average.from_output("cosdist")
    msle: metrics.Average.from_output("msle")
    lr: metrics.Average.from_output('lr')


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

@functools.partial(jax.jit, static_argnums=3)
def train_step(
    state: train_state.TrainState,
    graphs: jraph.GraphsTuple,
    rngs: Dict[str, jnp.ndarray],
    learning_rate_fn: optax.Schedule,
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
        loss = jnp.square(jnp.log(jnp.abs(actual_prop) + 1 ) - jnp.log(jnp.abs(pred_prop)+1))
        mean_loss = jnp.nanmean(loss)

        return mean_loss, (pred_prop, actual_prop)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (pred_prop, y)), grads = grad_fn(state.params, graphs)
    grads = jax.tree_util.tree_map(lambda x: jnp.nan_to_num(x), grads)
    state = state.apply_gradients(grads=grads)
    errp = jnp.nanmean((pred_prop / y) * 100.0)
    nan_number = jnp.sum(jnp.isnan(pred_prop))

    lr = learning_rate_fn(state.step)

    metrics_update = TrainMetrics.single_from_model_output(
        msle = loss,
        errp=errp,
        nan_number=nan_number,
        lr = lr,
    )
    return state, metrics_update

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
        actual_para = graphs.globals[:-1].reshape(-1,17)
        
        # Replace the global feature for graph prediction.
        graphs = replace_globals(graphs)

        # Compute predicted properties and resulting loss.
        pcsaft_params = get_predicted_para(curr_state, graphs, rngs)[:-1,:]
        msle = jnp.square(jnp.log(jnp.abs(actual_para) + 1 ) - jnp.log(jnp.abs(pcsaft_params)+1))
        loss = optax.cosine_distance(pcsaft_params, actual_para)
        mean_loss = jnp.nanmean(loss)

        return mean_loss, msle

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, msle), grads = grad_fn(state.params, graphs)
    state = state.apply_gradients(grads=grads)

    lr = learning_rate_fn(state.step)

    metrics_update = PreTrainMetrics.single_from_model_output(
        cosdist=loss,
        msle = msle,
        lr = lr,
    )
    return state, metrics_update


@jax.jit
def evaluate_step(
    state: train_state.TrainState,
    graphs: jraph.GraphsTuple,
) -> metrics.Collection:
    """Computes metrics over a set of graphs."""

    # The target properties our model has to predict.
    sysstate = graphs.globals[:-1].reshape(-1,7)
    actual_prop = sysstate[:,-1]

    # Replace the global feature for graph prediction.
    graphs = replace_globals(graphs)

    # Get predicted properties.
    parameters = get_predicted_para(state, graphs, rngs=None)[:-1,:]
    pred_prop = ml_pc_saft.batch_epcsaft_layer_test(parameters, sysstate)

    # Compute the various metrics.
    loss = jnp.square(jnp.log(jnp.abs(actual_prop) + 1 ) - jnp.log(jnp.abs(pred_prop)+1))
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
) ->  metrics.Collection:
    """Evaluates the model on metrics over the specified splits."""
    eval_metric = None
    # Loop over graphs.
    for graphs in dataloader:
        graphs = batchedjax(graphs)
        graphs = jax.tree_util.tree_map(np.asarray, graphs)
        eval_metric_update = evaluate_step(state, graphs)

        # Update metrics.
        if eval_metric is None:
            eval_metric = eval_metric_update
        else:
            eval_metric = eval_metric.merge(eval_metric_update)

    return eval_metric  


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
    
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, drop_last=True
    )
    

    # Create and initialize the network.
    logging.info("Initializing network.")
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    init_graphs = batchedjax(next(iter(train_loader)))
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
    step = initial_step
    while step < config.num_train_steps + 1:
        for graphs in train_loader:
                
            # Split PRNG key, to ensure different 'randomness' for every step.
            rng, dropout_rng = jax.random.split(rng)

            # Perform one step of training.
            with jax.profiler.StepTraceAnnotation("train", step_num=step):
                graphs = batchedjax(graphs)
                graphs = jax.tree_util.tree_map(np.asarray, graphs)
                

                if config.pre_train:
                    state, metrics_update = pre_train_step(
                        state, graphs, {"dropout": dropout_rng}, sch,
                        )
                else:
                    state, metrics_update = train_step(
                    state, graphs, {"dropout": dropout_rng}, sch,
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
            if step % config.eval_every_steps == 0 or (is_last_step & (not config.pre_train)):
                eval_state = eval_state.replace(params=state.params)

                with report_progress.timed("eval"):
                    eval_metrics = evaluate_model(eval_state, val_loader)
                    wandb.log(
                        add_prefix_to_keys(eval_metrics.compute(), 'val'), step=step
                    )

            # Checkpoint model, if required.
            if step % config.checkpoint_every_steps == 0 or is_last_step:
                with report_progress.timed("checkpoint"):
                    ckpt.save(state)
            step += 1
    wandb.finish()
    return state

from graphdataset import pureTMLDataset
from jaxopt import GaussNewton, LevenbergMarquardt
from ml_pc_saft import epcsaft_pure


def train():
    para_dict = {}
    data = pureTMLDataset("./data/thermoml/raw/pure.parquet") 

    maxnpoints = 0
    for datapoints in data:
        maxnpoints = max(maxnpoints, len(datapoints))
    
    def loss(parameters: jnp.ndarray, datapoints: list[tuple]) -> jnp.ndarray:
        
        ls = jnp.zeros(maxnpoints) 
        i = 0
        for (state, y) in datapoints:
            state = jnp.asarray(state)
            y = jnp.asarray(y)
            pred_y = epcsaft_pure(parameters, state)
            ls.at[i].set(jnp.log(jnp.abs(y) + 1 ) - jnp.log(jnp.abs(pred_y)+1))
            i +=1
        return ls

    solver = LevenbergMarquardt(loss, jit = True)

    for datapoints in data:
        (ids, _, _) = datapoints[0]
        statey = [(state, y) for _, state, y in datapoints]
        parameters = jnp.asarray([1.0, 1.0, 10.0, 0.1, 10.0, 1.0, 1.0])
        (params, state) = solver.run(parameters, statey)
        print(params, state)
        para_dict[ids[1]] = params
    return para_dict