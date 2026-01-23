# GNNPCSAFT Project

[![DOI](https://zenodo.org/badge/609414994.svg)](https://doi.org/10.5281/zenodo.17371237)

The project focuses on using Graph Neural Networks ([GNN](https://en.wikipedia.org/wiki/Graph_neural_network)) to estimate the pure-component parameters of the Equation of State [PC-SAFT](https://en.wikipedia.org/wiki/PC-SAFT).

Currently, the model takes into account the hard-chain, dispersive, and associative terms of PC-SAFT. Future work on polar and ionic terms is being studied.

Code is being developed mainly in Pytorch ([PyG](https://pytorch-geometric.readthedocs.io/en/latest/index.html#)).

You can find a model deployed in a Desktop App at [SourceForge](https://sourceforge.net/projects/gnnpcsaft/).

A CLI to use a model can be found at [GNNPCSAFT CLI](https://github.com/wildsonbbl/gnnepcsaftcli) and installed with [pipx](https://github.com/pypa/pipx):

```bash
pipx install gnnepcsaftcli
```

Model checkpoints can be found at [Hugging Face](https://huggingface.co/wildsonbbl/gnnepcsaft).

Use cases of this package are demonstrated in Jupyter Notebooks:

- `compare.ipynb` ([Open in Colab](https://colab.research.google.com/github/wildsonbbl/gnnepcsaft/blob/main/compare.ipynb)): comparison of the performance of trained models
- `training.ipynb` ([Open in Colab](https://colab.research.google.com/github/wildsonbbl/gnnepcsaft/blob/main/training.ipynb)): notebook for model training
- `tuning.ipynb` ([Open in Colab](https://colab.research.google.com/github/wildsonbbl/gnnepcsaft/blob/main/tuning.ipynb)): notebook for hyperparameter tuning

---

Work in progress.
