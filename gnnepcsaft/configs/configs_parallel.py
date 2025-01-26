"""Module to get configs for model hyperparameters in parallel training"""


def get_configs() -> dict:
    """Get the hyperparameter configurations."""
    configs = [
        {
            "propagation_depth": 7,
            "hidden_dim": 64,
            "post_layers": 3,
            "pre_layers": 2,
            "add_self_loops": True,
        },
        {
            "propagation_depth": 7,
            "hidden_dim": 64,
            "post_layers": 3,
            "pre_layers": 2,
            "add_self_loops": True,
        },
    ]
    return configs
