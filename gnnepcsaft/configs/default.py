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
    config.num_train_steps = 300_000
    config.log_every_steps = 5000
    config.eval_every_steps = 9999
    config.checkpoint_every_steps = 10000
    config.dataset = "esper"
    config.checkpoint = ""

    # GNN hyperparameters.
    config.model_name = "esper_msigmae_6"
    config.model = "GATL"
    config.propagation_depth = 6
    config.hidden_dim = 128
    config.post_layers = 1
    config.pre_layers = 3
    config.num_para = 3
    config.add_self_loops = True
    config.dropout_rate = 0.25
    config.heads = 2
    return config
