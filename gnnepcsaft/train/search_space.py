"""Module for hyperparameter search space definition for tuning"""

from ray import tune


# Hyperparameter search space
def get_search_space() -> dict:
    """Hyperparameter search space for tuning"""
    return {
        "propagation_depth": tune.choice([3, 4, 5, 6, 7, 8]),
        "hidden_dim": tune.choice([64, 128, 256]),
        "post_layers": tune.choice([1, 2, 3]),
        "pre_layers": tune.choice([1, 2, 3]),
        "towers": tune.choice([1, 2, 3]),
        "conv": "PNA",
    }
