import polars as pl
import pickle
import os

from rdkit import Chem, RDLogger
from rdkit.Chem.rdMolDescriptors import CalcExactMolWt

RDLogger.DisableLog("rdApp.*")

from openbabel.pybel import readstring

from delfta.calculator import DelftaCalculator


def pureTMLDataset(root: str) -> dict:
    """
    Dataset creator/manipulator.

    PARAMETERS
    ----------
    root (str) – Path where the pure.parquet file is.
    """

    pure = pl.read_parquet(root)
    calc = DelftaCalculator(
        tasks="dipole", delta=False, xtbopt=True, verbose=False, progress=False
    )
    dipole_from_inchi = {}
    list_error = []
    print('delfta firt try with xtbopt=True')
    for inchi in pure["inchi1"].unique().to_list():
        try:
            mol = readstring("inchi", inchi)
            dipole = calc.predict(mol)["dipole"].item()
            dipole_from_inchi[inchi] = dipole
        except:
            list_error.append(inchi)
    calc = DelftaCalculator(
        tasks="dipole", delta=False, xtbopt=False, verbose=False, progress=False
    )
    print('delfta firt try with xtbopt=False')
    for inchi in list_error:
        try:
            mol = readstring("inchi", inchi)
            dipole = calc.predict(mol)["dipole"].item()
            dipole_from_inchi[inchi] = dipole
        except:
            dipole_from_inchi[inchi] = 0.0
            print(f"error at {inchi}")

    puredatadict = {}
    for row in pure.iter_rows():
        inchi = row[1]
        dipole = dipole_from_inchi[inchi]
        tp = row[-2]
        ids = row[:2]
        state = row[2:-1] + (dipole,)
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
