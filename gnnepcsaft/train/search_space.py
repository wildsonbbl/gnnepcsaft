"""Module for hyperparameter search space definition for tuning"""

import ConfigSpace as CS


# Hyperparameter search space
def get_search_space() -> CS.ConfigurationSpace:
    """Hyperparameter search space for tuning"""
    config_space = CS.ConfigurationSpace()
    config_space.add(
        CS.UniformIntegerHyperparameter("propagation_depth", lower=3, upper=8),
        CS.CategoricalHyperparameter("hidden_dim", [64, 128, 256]),
        CS.UniformIntegerHyperparameter("post_layers", lower=1, upper=3),
        CS.UniformIntegerHyperparameter("pre_layers", lower=1, upper=3),
        CS.CategoricalHyperparameter("towers", [1, 2, 4]),
        CS.Constant("conv", "PNA"),
    )
    return config_space


# {
#     "propagation_depth": tune.choice([3, 4, 5, 6, 7, 8]),
#     "hidden_dim": tune.choice([64, 128, 256]),
#     "heads": tune.choice([1, 2, 3]),
#     "conv": "GAT",
# }
