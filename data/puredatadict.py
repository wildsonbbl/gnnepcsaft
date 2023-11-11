"""Module to be used for ThermoML Archive dataset building"""
import pickle

import polars as pl
from rdkit import Chem, RDLogger

# pylint: disable = no-name-in-module
from rdkit.Chem.rdMolDescriptors import CalcExactMolWt

RDLogger.DisableLog("rdApp.*")


def puretmldataset(root: str) -> dict:
    """
    Dataset creator with ThemoML Archive experimental density
    and vapor pressure data.

    PARAMETERS
    ----------
    root (str) – Path where the pure.parquet file is.
    """

    pure = pl.read_parquet(root)

    puredatadict = {}
    for row in pure.iter_rows():
        inchi = row[1]
        tp = row[-2]
        ids = row[:2]
        state = row[2:-1]
        y = row[-1]
        if tp == 1:
            mol_weight = mw(inchi)
            if mol_weight == 0:
                continue
            y = y * 1000.0 / mol_weight

        if inchi in puredatadict:
            if tp in puredatadict[inchi]:
                puredatadict[inchi][tp].append((ids, state, y))
            else:
                puredatadict[inchi][tp] = [(ids, state, y)]
        else:
            puredatadict[inchi] = {tp: [(ids, state, y)]}
    with open("./data/thermoml/raw/pure.pkl", "wb") as file:
        # A new file will be created
        pickle.dump(puredatadict, file)
    return puredatadict


def mw(inchi: str) -> float:
    "Calcultes molecular weight."
    try:
        mol = Chem.MolFromInchi(inchi, removeHs=False, sanitize=False)
        mol_weight = CalcExactMolWt(mol)
    except (TypeError, ValueError):
        mol_weight = 0

    return mol_weight


if __name__ == "__main__":
    puretmldataset("./data/thermoml/raw/pure.parquet")
