# GNNPCSAFT Project

[![DOI](https://zenodo.org/badge/609414994.svg)](https://doi.org/10.5281/zenodo.17371237)

The project focuses on using Graph Neural Networks ([GNN](https://en.wikipedia.org/wiki/Graph_neural_network)) to estimate the pure-component parameters of the Equation of State [PC-SAFT](https://en.wikipedia.org/wiki/PC-SAFT).

Currently, the model takes into account the hard-chain, dispersive, and associative terms of PC-SAFT. Future work on polar and ionic terms is being studied.

Use cases of this package are demonstrated in Jupyter Notebooks:

- `compare.ipynb` ([Open in Colab](https://colab.research.google.com/github/wildsonbbl/gnnepcsaft/blob/main/compare.ipynb)): comparison of the performance of trained models
- `training.ipynb` ([Open in Colab](https://colab.research.google.com/github/wildsonbbl/gnnepcsaft/blob/main/training.ipynb)): notebook for model training
- `tuning.ipynb` ([Open in Colab](https://colab.research.google.com/github/wildsonbbl/gnnepcsaft/blob/main/tuning.ipynb)): notebook for hyperparameter tuning

Model checkpoints can be found at [Hugging Face](https://huggingface.co/wildsonbbl/gnnepcsaft).

Implementations with GNNPCSAFT:

- [GNNPCSAFT CLI](https://github.com/wildsonbbl/gnnepcsaftcli)
- [GNNPCSAFT APP](https://github.com/wildsonbbl/gnnpcsaftapp)
- [GNNPCSAFT MCP](https://github.com/wildsonbbl/gnnepcsaft_mcp_server)
- [GNNPCSAFT Webapp](https://github.com/wildsonbbl/gnnepcsaftwebapp)
- [GNNPCSAFT Chat](https://github.com/wildsonbbl/gnnpcsaftchat)
