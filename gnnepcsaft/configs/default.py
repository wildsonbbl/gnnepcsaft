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
    config.warmup_steps = 1000

    # Training hyperparameters.

    config.accelerator = "gpu"
    config.batch_size = 512
    config.num_train_steps = 200_000
    config.log_every_steps = 1000
    config.eval_every_steps = 10000
    config.checkpoint_every_steps = 10001
    config.dataset = "esper"
    config.checkpoint = "esper_msigmae_3-epoch=23249-train_mape=0.0062.ckpt"

    # GNN hyperparameters.
    config.model_name = "esper_msigmae_3"
    config.model = "PNAL"
    config.propagation_depth = 5
    config.hidden_dim = 128
    config.pre_layers = 2
    config.post_layers = 1
    config.num_para = 3
    config.add_self_loops = True
    config.dropout_rate = 0.0
    return config
