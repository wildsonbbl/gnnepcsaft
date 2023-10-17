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
    
    config.batch_size = 128
    config.pad_size = 128
    config.num_train_steps = 1_000_000
    config.warmup_steps = 100
    config.log_every_steps = 5_000
    config.eval_every_steps = 10_000
    config.checkpoint_every_steps = 10_000
    config.amp = True
    config.momentum = 0.9
    config.patience = 5
    config.weight_decay = 1e-10
    

    # GNN hyperparameters.
    config.model = "PNA"
    config.propagation_depth = 4
    config.hidden_dim = 128
    config.dropout_rate = 0.0
    config.num_mlp_layers = 2
    config.num_para = 3
    config.skip_connections = True
    config.pre_layers = 1
    config.post_layers = 3
    return config
