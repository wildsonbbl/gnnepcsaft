import pickle

import polars as pl
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
    ramirez = pl.read_parquet("./data/ramirez2022/raw/data.parquet")

    critic = {}
    for row in ramirez.iter_rows():
        inchi = row[-1]
        tc = row[1]
        pc = row[2] * 100000
        critic[inchi] = (tc, pc)
        
    puredatadict = {}
    for row in pure.iter_rows():
        inchi = row[1]
        if check_if_elementary(inchi):
            continue
        if inchi in critic:
            tc, pc = critic[inchi]
            tk, ppa = row[2:4]
            if (tk > tc * 0.9) or (ppa > pc * 0.9):
                continue
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
        mol = Chem.MolFromInchi(inchi, removeHs=False, sanitize=False)
        mol_weight = CalcExactMolWt(mol)
    except:
        mol_weight = 0

    return mol_weight


def check_if_elementary(inchi: str) -> bool:
    mol = Chem.MolFromInchi(inchi, removeHs=False, sanitize=False)
    mol = Chem.AddHs(mol)
    atoms = []
    for atom in mol.GetAtoms():
        atoms += [atom.GetAtomicNum()]
    unique_atoms = set(atoms)

    return len(unique_atoms) == 1


if __name__ == "__main__":
    pureTMLDataset("./data/thermoml/raw/pure.parquet")
