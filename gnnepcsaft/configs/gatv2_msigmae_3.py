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
    config.warmup_steps = 2

    # Training hyperparameters.

    config.accelerator = "auto"
    config.batch_size = 512
    config.num_train_steps = 200_000
    config.log_every_steps = 1000
    config.eval_every_steps = 2500
    config.dataset = "esper"
    config.checkpoint = "wildson/gnn-pc-saft/model-q4pwbnxs:v3"  # wandb artifact path
    config.resume_id = ""  # wandb run id
    config.model = "gnn"
    config.model_name = "gatv2_msigmae_1.2"
    config.linearity_penalty = False

    # GNN hyperparameters.
    ## General
    config.conv = "GATv2"
    config.global_pool = "add"
    config.propagation_depth = 3
    config.hidden_dim = 512
    config.dropout = 0.10
    config.add_self_loops = True
    config.num_para = 3
    ## GAT, GATv2, TransformerConv
    config.heads = 8  # hidden_dim % heads == 0
    config.deg = []

    return config
