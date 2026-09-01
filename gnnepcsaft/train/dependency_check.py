"""Dependency checking module for training scripts.

This module only uses standard library imports to ensure it can be imported
and run even if dev dependencies are not installed.
"""


def check_dev_dependencies():
    """Check if all required dev dependencies are installed for training.

    Raises:
        ImportError: If required dev dependencies are missing.
    """
    required_packages = {
        "pylint": "pylint",
        "pytest": "pytest",
        "ipywidgets": "ipywidgets",
        "matplotlib": "matplotlib",
        "lightning": "lightning",
        "torch": "torch",
        "torch_geometric": "torch_geometric",
        "torchmetrics": "torchmetrics",
        "wandb": "wandb",
        "absl": "absl-py",
        "jax": "jax",
        "seaborn": "seaborn",
        "ml_collections": "ml_collections",
        "ogb": "ogb",
        "polars": "polars",
        "scipy": "scipy",
        "ConfigSpace": "ConfigSpace",
        "hpbandster": "hpbandster",
        "ray": "ray[train,tune]",
        "teqp": "teqp",
        "PCSAFTsuperanc": "PCSAFTsuperanc",
        "xgboost": "xgboost",
        "onnx": "onnx",
        "onnxscript": "onnxscript",
        "onnxruntime": "onnxruntime",
        "plotly": "plotly",
        "nbformat": "nbformat",
        "matplotlib_venn": "matplotlib-venn",
    }

    missing_packages = []
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)

    if missing_packages:
        error_msg = (
            "⚠️  Missing required development dependencies for training!\n\n"
            f"Please install the missing packages:\n"
            f"  pip install {' '.join(missing_packages)}\n\n"
            f"Or install all dev dependencies with:\n"
            f"  pip install -e '.[dev]'\n"
        )
        raise ImportError(error_msg) from None
