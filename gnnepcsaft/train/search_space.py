"""Module for hyperparameter search space definition for tuning"""

import ConfigSpace as CS


# Hyperparameter search space
def get_search_space() -> CS.ConfigurationSpace:
    """Hyperparameter search space for tuning"""
    search_space = CS.ConfigurationSpace()

    a = CS.Integer("propagation_depth", (3, 8))
    b = CS.Categorical("hidden_dim", [32, 64, 128, 256])
    c = CS.Constant("dropout", 0.0)
    d = CS.Constant("global_pool", "add")
    e = CS.Categorical("conv", ["PNA", "GATv2"])
    f = CS.Categorical("heads", [1, 2, 4, 8])
    g = CS.Integer("post_layers", (1, 3))
    h = CS.Integer("pre_layers", (1, 3))
    i = CS.Categorical("towers", [1, 2, 4])
    cond_i = CS.EqualsCondition(i, e, "PNA")
    cond_f = CS.EqualsCondition(f, e, "GATv2")
    search_space.add([a, b, c, d, e, f, g, h, i])
    search_space.add([cond_i, cond_f])

    return search_space
