"""Module for hyperparameter search space definition for tuning"""

import numpy as np
from ray import tune


# Hyperparameter search space
def get_search_space():
    """Hyperparameter search space for tuning"""
    return {
        "propagation_depth": tune.sample_from(
            lambda spec: np.random.choice([3, 4, 5, 6, 7, 8]).item()
        ),
        "hidden_dim": tune.sample_from(
            lambda spec: np.random.choice([64, 128, 256]).item()
        ),
        "pre_layers": tune.sample_from(lambda spec: np.random.choice([1]).item()),
        "post_layers": tune.sample_from(lambda spec: np.random.choice([1]).item()),
        "heads": tune.sample_from(lambda spec: np.random.choice([1, 2, 3]).item()),
    }
