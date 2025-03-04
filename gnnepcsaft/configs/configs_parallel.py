"""Module to get configs for model hyperparameters in parallel training"""

from typing import Any


def get_configs() -> list[dict[str, Any]]:
    """Get the hyperparameter configurations."""
    configs = [
        {
            "model_name": "gnn_msigmae_3.1",
            "model": "gnn",
            "conv": "PNA",
            "dropout": 0.0,
            "propagation_depth": 6,
            "hidden_dim": 256,
            "post_layers": 3,
            "pre_layers": 2,
            "towers": 2,
        },
        {
            "model_name": "gnn_msigmae_3.2",
            "model": "gnn",
            "conv": "PNA",
            "dropout": 0.1,
            "propagation_depth": 6,
            "hidden_dim": 256,
            "post_layers": 3,
            "pre_layers": 2,
            "towers": 2,
        },
        {
            "model_name": "gnn_msigmae_3.3",
            "model": "gnn",
            "conv": "PNA",
            "dropout": 0.25,
            "propagation_depth": 6,
            "hidden_dim": 256,
            "post_layers": 3,
            "pre_layers": 2,
            "towers": 2,
        },
    ]
    return configs
