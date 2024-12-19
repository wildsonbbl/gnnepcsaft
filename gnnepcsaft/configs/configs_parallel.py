"""Module to get configs for model hyperparameters in parallel training"""


def get_configs() -> dict:
    """Get the hyperparameter configurations."""
    configs = [
        {
            "propagation_depth": 6,
            "hidden_dim": 64,
            "pre_layers": 1,
            "post_layers": 1,
            "add_self_loops": True,
        },
        {
            "propagation_depth": 6,
            "hidden_dim": 64,
            "pre_layers": 2,
            "post_layers": 2,
            "add_self_loops": True,
        },
        {
            "propagation_depth": 6,
            "hidden_dim": 64,
            "pre_layers": 1,
            "post_layers": 1,
            "add_self_loops": True,
        },
    ]
    return configs
