# GNNePCSAFT Project

Project focused in the use of graph neural networks to estimate the pure-component parameters of the Equation of State [ePC-SAFT](https://en.wikipedia.org/wiki/PC-SAFT).

The motivation of this work is to be able to use a robust Equation of State, ePC-SAFT, without prior need of experimental data. Equations of State are important to calculate thermodynamic properties, and are pre-requisite in process simulators.

Currently, the model takes in account only the hard-chain, dispersive and assoc terms of ePC-SAFT. Future work on polar and ionic terms are being studied.

Code is being developed mainly in Pytorch (PYG).

You can find the model deployed at [GNNePCSAFT Webapp](https://gnnepcsaft.wildsonbbl.com/).

A CLI to use the model can be found at [GNNePCSAFT CLI](https://github.com/wildsonbbl/gnnepcsaftcli) and installed with `pipx`:

```bash
pipx install gnnepcsaftcli
```

Checkpoints can be found at [Hugging Face](https://huggingface.co/wildsonbbl/gnnepcsaft).

Use cases of this package are demonstrated in Jupyter Notebooks:

- `compare.ipynb` ([Open in Colab](https://colab.research.google.com/github/wildsonbbl/gnnepcsaft/blob/main/compare.ipynb)): comparison of the performance between two or more trained models
- `demo.ipynb` ([Open in Colab](https://colab.research.google.com/github/wildsonbbl/gnnepcsaft/blob/main/demo.ipynb)): pt-br demonstration of models capabilities
- `training.ipynb` ([Open in Colab](https://colab.research.google.com/github/wildsonbbl/gnnepcsaft/blob/main/training.ipynb)): notebook for model training
- `tuning.ipynb` ([Open in Colab](https://colab.research.google.com/github/wildsonbbl/gnnepcsaft/blob/main/tuning.ipynb)): notebook for hyperparameter tuning

---

Work in progess.
