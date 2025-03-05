"""Module for hyperparameter search space definition for tuning"""

import ConfigSpace as CS


# Hyperparameter search space
def get_search_space() -> CS.ConfigurationSpace:
    """Hyperparameter search space for tuning"""
    search_space = CS.ConfigurationSpace()

    a = CS.UniformIntegerHyperparameter("propagation_depth", 3, 8, 5)
    b = CS.CategoricalHyperparameter("hidden_dim", [32, 64, 128, 256], 128)
    c = CS.Constant("dropout", 0.0)
    d = CS.Constant("global_pool", "add")
    e = CS.CategoricalHyperparameter("conv", ["PNA", "GATv2"], "PNA")
    f = CS.CategoricalHyperparameter("heads", [1, 2, 4, 8], 2)
    g = CS.UniformIntegerHyperparameter("post_layers", 1, 4, 1)
    h = CS.UniformIntegerHyperparameter("pre_layers", 1, 4, 1)
    i = CS.CategoricalHyperparameter("towers", [1, 2, 4, 8], 2)
    cond_f = CS.EqualsCondition(f, e, "GATv2")
    cond_g = CS.EqualsCondition(g, e, "PNA")
    cond_h = CS.EqualsCondition(h, e, "PNA")
    cond_i = CS.EqualsCondition(i, e, "PNA")
    search_space.add([a, b, c, d, e, f, g, h, i])
    search_space.add([cond_f, cond_g, cond_h, cond_i])

    return search_space
