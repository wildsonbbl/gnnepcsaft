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
    config.pad_size = 128
    config.num_train_steps = 150_000
    config.log_every_steps = 5_000
    config.eval_every_steps = 24_999
    config.checkpoint_every_steps = 25_000
    config.dataset = "esper"
    config.checkpoint = ""

    # GNN hyperparameters.
    config.model_name = "esper_msigmae_2"
    config.model = "PNAL"
    config.propagation_depth = 6
    config.hidden_dim = 64
    config.pre_layers = 2
    config.post_layers = 2
    config.num_mlp_layers = 2
    config.num_para = 3
    config.skip_connections = False
    config.add_self_loops = False
    config.dropout_rate = 0.0
    return config
