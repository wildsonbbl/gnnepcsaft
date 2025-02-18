"""Module to config traning and model parameters"""

import ml_collections


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()
    config.job_type = "train"

    # Optimizer.
    config.optimizer = "adam"
    config.learning_rate = 1e-3
    config.change_opt = False
    config.change_sch = False
    config.weight_decay = 1e-2
    config.momentum = 0.9
    config.patience = 5
    config.warmup_steps = 100

    # Training hyperparameters.

    config.accelerator = "gpu"
    config.batch_size = 512
    config.num_train_steps = 100_000
    config.log_every_steps = 5000
    config.eval_every_steps = 10000
    config.checkpoint_every_steps = 10000
    config.dataset = "esper"
    config.checkpoint = ""

    # GNN hyperparameters.
    ## General
    config.model_name = "esper_msigmae_7"
    config.conv = "PNA"
    config.global_pool = "mean"
    config.propagation_depth = 7
    config.hidden_dim = 256
    config.dropout = 0.6
    config.add_self_loops = True
    config.num_para = 3
    ## PNA
    config.post_layers = 2
    config.pre_layers = 2
    config.towers = 2  # hidden_dim % towers == 0
    config.deg = []
    ## GatedGraphConv, ARMAConv
    config.num_layers = 2
    config.num_stacks = 2
    ## GAT, GATv2, TransformerConv
    config.heads = 4  # hidden_dim % heads == 0

    return config
