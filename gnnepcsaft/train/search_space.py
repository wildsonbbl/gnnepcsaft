"""Module for hyperparameter search space definition for tuning"""

from ray import tune


# Hyperparameter search space
def get_search_space():
    """Hyperparameter search space for tuning"""
    return {
        "propagation_depth": tune.choice([3, 4, 5, 6]),
        "hidden_dim": tune.choice([64, 128, 256]),
        "num_mlp_layers": tune.choice([0, 1, 2]),
        "pre_layers": tune.choice([1, 2]),
        "post_layers": tune.choice([1, 2]),
        "skip_connections": tune.choice([True, False]),
        "add_self_loops": tune.choice([True, False]),
    }
