## GNNePCSAFT

Project focused in the use of graph neural networks to estimate the pure-component parameters of the Equation of State ePC-SAFT.

The motivation of this work is to be able to use a robust Equation of State, ePC-SAFT, without prior need of experimental data. Equations of State are important to calculate thermodynamic properties, and are pre-requisite in process simulators.

Currently, the model takes in account only the hard-chain and dispersive terms of ePC-SAFT. Future work on associative, polar and ionic terms are being studied.

Code is being developed mainly in Pytorch (PYG) and secondarily in JAX (JRAPH).

You can find the model deployed at [GNNePCSAFT app](https://gnnepcsaft.wildsonbbl.com/).

Use cases of this package are demonstrated in Jupyter Notebooks:

- `compare.ipynb`: comparison of the performance between two or more trained models
- `demo.ipynb`: pt-br demonstration of the models capabilities
- `evalmodels.ipynb`: code to evaluate all saved models in `train/checkpoints` folder at once
- `evalref.ipynb`: code to evaluate perfomance of reference parameter data on experimental data
- `moleculargraphs.ipynb`: code to build all datasets used
- `training.ipynb`:  Code for model training
- `tuning.ipynb`: code for hyperparameter tuning

--------------
Work in progess.


