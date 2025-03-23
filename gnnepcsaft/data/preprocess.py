"""Module to be used for ThermoML Archive dataset building"""

import os.path as osp
from urllib.parse import quote
from urllib.request import HTTPError, urlopen

import polars as pl
from absl import app, flags, logging


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
            print("not ok:", ids)
            ans = ""
    return ans


FLAGS = flags.FLAGS

flags.DEFINE_string("root", osp.dirname(__file__), "Path to data directory.")
flags.DEFINE_string("save_dir", osp.dirname(__file__), "Path to output directory")


def main(argv):
    """Execution from command line"""
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    logging.info("Calling preprocess!")
    ramirez(osp.join(FLAGS.root, "ramirez2022/raw/RAMIREZ2022.csv"), FLAGS.save_dir)
    logging.info("Ramírez-Vélez dataset preprocessed!")


if __name__ == "__main__":
    app.run(main)
