"""Module to get configs for model hyperparameters in parallel training"""

from typing import Any


def get_configs() -> list[dict[str, Any]]:
    """Get the hyperparameter configurations."""
    configs = [
        {
            "model_name": "esper_msigmae_5.5",
            "model": "gnn",
            "conv": "PNA",
            "dropout": 0.0,
            "propagation_depth": 6,
            "hidden_dim": 128,
            "post_layers": 1,
            "pre_layers": 3,
            "towers": 1,
        },
        {
            "model_name": "esper_msigmae_5.6",
            "model": "gnn",
            "conv": "PNA",
            "dropout": 0.1,
            "propagation_depth": 6,
            "hidden_dim": 128,
            "post_layers": 1,
            "pre_layers": 3,
            "towers": 1,
        },
        {
            "model_name": "esper_msigmae_5.7",
            "model": "gnn",
            "conv": "PNA",
            "dropout": 0.25,
            "propagation_depth": 6,
            "hidden_dim": 128,
            "post_layers": 1,
            "pre_layers": 3,
            "towers": 1,
        },
    ]
    return configs
