"""Module to config traning and model parameters"""

import ml_collections


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()
    config.job_type = "train"

    # Optimizer.
    config.optimizer = "sgd"
    config.learning_rate = 1e-12
    config.change_opt = True
    config.change_sch = False
    config.weight_decay = 1e-2
    config.momentum = 0.9
    config.patience = 5
    config.warmup_steps = 100

    # Training hyperparameters.

    config.accelerator = "gpu"
    config.batch_size = 1842
    config.pad_size = 128
    config.num_train_steps = 1_000
    config.log_every_steps = 100
    config.eval_every_steps = 100
    config.checkpoint_every_steps = 100
    config.dataset = "esper"
    config.checkpoint = "esper_msigmae_2-epoch=37499-mape_den=0.0112.ckpt"

    # GNN hyperparameters.
    config.model_name = "esper_msigmae_2.1"
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
