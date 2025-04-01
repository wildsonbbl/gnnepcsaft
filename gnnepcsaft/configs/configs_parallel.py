"""Module to get configs for model hyperparameters in parallel training"""

from typing import Any


def get_configs() -> list[dict[str, Any]]:
    """Get the hyperparameter configurations."""
    configs = [
        {
            "model_name": "pna_msigmae_1.0",
            "model": "gnn",
            "conv": "PNA",
            "dropout": 0.0,
            "propagation_depth": 6,
            "hidden_dim": 256,
            "post_layers": 4,
            "pre_layers": 2,
            "towers": 1,
            "num_para": 3,
            "dataset": "esper",
            "batch_size": 512,
            "checkpoint": "wildson/gnn-pc-saft/model-gq9gcgmb:v3",
            "resume_id": "",
            "num_train_steps": 425_000,
        },
        {
            "model_name": "gatv2_msigmae_1.0",
            "model": "gnn",
            "conv": "GATv2",
            "dropout": 0.0,
            "propagation_depth": 3,
            "hidden_dim": 512,
            "heads": 8,
            "num_para": 3,
            "dataset": "esper",
            "batch_size": 512,
            "checkpoint": "wildson/gnn-pc-saft/model-54730h51:v0",
            "resume_id": "",
            "num_train_steps": 600_000,
        },
        {
            "model_name": "gatv2_assoc_1.0",
            "model": "gnn",
            "conv": "GATv2",
            "dropout": 0.0,
            "propagation_depth": 3,
            "hidden_dim": 512,
            "heads": 8,
            "num_para": 2,
            "dataset": "esper_assoc_only",
            "batch_size": 387 // 4 + 1,
            "checkpoint": "wildson/gnn-pc-saft/model-zme6255l:v1",
            "resume_id": "",
            "num_train_steps": 575_000,
        },
    ]
    return configs
