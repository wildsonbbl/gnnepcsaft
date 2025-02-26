"""Module to get configs for model hyperparameters in parallel training"""


def get_configs() -> list[dict]:
    """Get the hyperparameter configurations."""
    configs = [
        {
            "model_name": "gnn_msigmae_1",
            "model": "gnn",
            "conv": "GATv2",
            "propagation_depth": 7,
            "hidden_dim": 256,
            "heads": 2,
        },
        {
            "model_name": "gnn_msigmae_2",
            "model": "gnn",
            "conv": "PNA",
            "propagation_depth": 5,
            "hidden_dim": 256,
            "post_layers": 3,
            "pre_layers": 2,
            "towers": 2,
        },
    ]
    return configs
