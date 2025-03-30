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
            "checkpoint": "wildson/gnn-pc-saft/model-gq9gcgmb:v2",
            "resume_id": "gq9gcgmb",
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
            "checkpoint": "wildson/gnn-pc-saft/model-hc2uvwd8:v2",
            "resume_id": "hc2uvwd8",
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
            "checkpoint": "wildson/gnn-pc-saft/model-vrvtxov3:v1",
            "resume_id": "vrvtxov3",
        },
    ]
    return configs
