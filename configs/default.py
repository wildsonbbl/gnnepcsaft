import ml_collections


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    # Optimizer.
    config.optimizer = "sgd"
    config.learning_rate = 1e-3

    # Training hyperparameters.
    
    config.batch_size = 128
    config.pad_size = 128
    config.num_train_steps = 6_000
    config.warmup_steps = 100
    config.log_every_steps = 100
    config.eval_every_steps = 2_000
    config.checkpoint_every_steps = 3_000
    config.half_precision = False
    config.momentum = 0.9
    config.patience = 1000
    config.weight_decay = 1e-2
    

    # GNN hyperparameters.
    config.model = "PNA"
    config.propagation_depth = 5
    config.hidden_dim = 256
    config.dropout_rate = 0.1
    config.num_mlp_layers = 1
    config.num_para = 3
    config.skip_connections = True
    config.pre_layers = 1
    config.post_layers = 1
    return config
