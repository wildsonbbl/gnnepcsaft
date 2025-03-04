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
    config.batch_size = 387
    config.num_train_steps = 450_000
    config.log_every_steps = 5000
    config.eval_every_steps = 9999
    config.checkpoint_every_steps = 10000
    config.dataset = "esper_assoc_only"
    config.checkpoint = "esper_assoc_8-epoch=274998-train_mape=0.0020.ckpt"

    # GNN hyperparameters.
    config.model_name = "esper_assoc_8"
    config.model = "GATL"
    config.propagation_depth = 7
    config.hidden_dim = 256
    config.post_layers = None
    config.pre_layers = None
    config.num_para = 2
    config.add_self_loops = True
    config.dropout_rate = 0.25
    config.heads = 3
    return config
