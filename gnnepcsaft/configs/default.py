"""Module to config traning and model parameters"""
import ml_collections


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    # Optimizer.
    config.optimizer = "adam"
    config.learning_rate = 1.0e-3
    config.change_opt = False
    config.change_sch = False

    # Training hyperparameters.

    config.batch_size = 512
    config.pad_size = 128
    config.num_train_steps = 510_000
    config.warmup_steps = 100
    config.log_every_steps = 500
    config.eval_every_steps = 2500
    config.checkpoint_every_steps = 2500
    config.amp = False
    config.momentum = 0.9
    config.patience = 5
    config.weight_decay = 1e-3
    config.dataset = "ramirez"
    config.checkpoint = "last.ckpt"

    # GNN hyperparameters.
    config.model_name = "model6"
    config.model = "PNAL"
    config.propagation_depth = 2
    config.hidden_dim = 128
    config.pre_layers = 1
    config.post_layers = 3
    config.dropout_rate = 0.0
    config.num_mlp_layers = 1
    config.num_para = 3
    config.skip_connections = False
    config.add_self_loops = False
    return config
