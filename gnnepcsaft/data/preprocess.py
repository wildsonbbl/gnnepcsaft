"""Module to be used for ThermoML Archive dataset building"""

import os.path as osp
import pickle
from urllib.parse import quote
from urllib.request import HTTPError, urlopen

import polars as pl
from absl import app, flags, logging
from rdkit import Chem, RDLogger

# pylint: disable = no-name-in-module
from rdkit.Chem.rdMolDescriptors import CalcExactMolWt

RDLogger.DisableLog("rdApp.*")


def puretmldataset(root: str, save_dir: str) -> dict:
    """
    Preprocess ThermoML Archive pure component
    density and vapor pressure experimental data
    and saves into pkl file used in data loading.

    PARAMETERS
    ----------
    root (str) – Path where the ThermoML data file is.

    save_dir (str) - Directory to save output file.
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
    file_path = osp.join(save_dir, "thermoml/raw/pure.pkl")
    with open(file_path, "wb") as file:
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


def ramirez(root: str, save_dir: str):
    """
    Preprocess `Ramírez-Vélez et al. (2022, doi: 10.1002/aic.17722)` data
    and saves into dataframe file used in data loading.

    PARAMETERS
    ----------
    root (str) – Path where the ramirez data file is.

    save_dir (str) - Directory to save output file.
    """
    data = pl.read_csv(root, has_header=True, separator=";")

    inchis = data["name"].map_elements(to_inchi).rename("inchi")
    data = data.with_columns(inchis)
    file_path = osp.join(save_dir, "ramirez2022/raw/data.parquet")
    data.write_parquet(file_path)


def to_inchi(ids: str) -> str:
    "Tries to return InChI from input compound name with pubchem or cactus api."
    try:
        url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
            + quote(ids)
            + "/property/InChI/TXT"
        )
        with urlopen(url) as ans:
            ans = ans.read().decode("utf8").rstrip()
            ans = ans.split("\n")[0]

    except (TypeError, ValueError, HTTPError):
        try:
            url = (
                "http://cactus.nci.nih.gov/chemical/structure/" + quote(ids) + "/inchi"
            )
            with urlopen(url) as ans:
                ans = ans.read().decode("utf8").rstrip()
                ans = ans.split("\n")[0]
        except (TypeError, ValueError, HTTPError):
            print("not ok:", url)
            ans = None
    return ans


FLAGS = flags.FLAGS

flags.DEFINE_string("root", osp.dirname(__file__), "Path to data directory.")
flags.DEFINE_string("save_dir", osp.dirname(__file__), "Path to output directory")


def main(argv):
    """Execution from command line"""
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    logging.info("Calling preprocess!")

    puretmldataset(osp.join(FLAGS.root, "thermoml/raw/pure.parquet"), FLAGS.save_dir)
    logging.info("Pure TML dataset preprocessed!")
    ramirez(osp.join(FLAGS.root, "ramirez2022/raw/RAMIREZ2022.csv"), FLAGS.save_dir)
    logging.info("Ramírez-Vélez dataset preprocessed!")


if __name__ == "__main__":
    app.run(main)
