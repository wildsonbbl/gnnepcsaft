import polars as pl
import pickle
import os

from rdkit import Chem, RDLogger
from rdkit.Chem.rdMolDescriptors import CalcExactMolWt

RDLogger.DisableLog("rdApp.*")


def pureTMLDataset(root: str) -> dict:
    """
    Dataset creator/manipulator.

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
                # print(f'error: {ids[0]} with mw = 0')
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
    try:
        mol = Chem.MolFromInchi(inchi, removeHs=False)
        mol_weight = CalcExactMolWt(mol)
    except:
        mol_weight = 0

    return mol_weight


if ~os.path.exists("./data/thermoml/raw/pure.pkl"):
    pureTMLDataset("./data/thermoml/raw/pure.parquet")
