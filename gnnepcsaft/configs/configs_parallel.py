"""Module to get configs for model hyperparameters in parallel training"""


def get_configs() -> list[dict]:
    """Get the hyperparameter configurations."""
    configs = [
        {
            "model_name": "esper_msigmae_7.1",
            "model": "GATL",
            "propagation_depth": 7,
            "hidden_dim": 256,
            "post_layers": None,
            "pre_layers": None,
            "heads": 3,
        },
        {
            "model_name": "esper_msigmae_5.3",
            "model": "PNAL",
            "propagation_depth": 6,
            "hidden_dim": 128,
            "post_layers": 1,
            "pre_layers": 3,
            "heads": None,
        },
    ]
    return configs
