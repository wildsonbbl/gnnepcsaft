import rdkit.Chem as Chem
from rdkit.Chem.Fragments import (
    fr_COO,
    fr_Al_OH,
    fr_Ar_OH,
    fr_ketone,
    fr_NH0,
    fr_NH1,
    fr_NH2,
    fr_ester,
    fr_ether,
    fr_halogen,
    fr_amide,
    fr_unbrch_alkane,
    fr_aldehyde,
    fr_benzene,
    fr_epoxide,
    fr_sulfide,
    fr_SH,
    fr_nitrile,
    fr_phenol,
    fr_phos_acid,
)

from urllib.request import urlopen
from urllib.parse import quote
from time import sleep
import numpy as np


def complexity(ids: str) -> str:
    try:
        url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchi/property/complexity/TXT?inchi="
            + quote(ids)
        )
        ans = urlopen(url).read().decode("utf8").rstrip()
        ans = ans.split("\n")[0]
        if ans:
            ans = float(ans)
    except:
        print("not ok:", url)
        ans = None

    # sleep(0.1)
    return ans


def get_family_groups(InChI: str) -> set[str]:
    try:
        mol = Chem.MolFromInchi(InChI, sanitize=True, treatWarningAsError=True)
    except:
        mol = Chem.MolFromInchi(InChI, sanitize=False)

    if mol is None:
        mol = Chem.MolFromInchi(InChI, sanitize=False)

    frs_1 = [
        fr_COO,
        fr_ketone,
        fr_ester,
        fr_ether,
        fr_aldehyde,
        fr_NH0,
        fr_NH1,
        fr_NH2,
        fr_amide,
        fr_halogen,
        fr_phenol,
        fr_epoxide,
        fr_sulfide,
        fr_SH,
        fr_nitrile,
        fr_phos_acid,
    ]
    frs_str1 = [
        "carb acid",
        "ketone/aldehyde",
        "ester/ether/epoxide",
        "ester/ether/epoxide",
        "ketone/aldehyde",
        "amine/amide/nitrile",
        "amine/amide/nitrile",
        "amine/amide/nitrile",
        "amine/amide/nitrile",
        "halogen",
        "phenol",
        "ester/ether/epoxide",
        "thiol/thiolether",
        "thiol/thiolether",
        "amine/amide/nitrile",
        "phosphoric acid",
    ]
    frs_2 = [
        fr_Al_OH,
        fr_Ar_OH,
        fr_benzene,
    ]
    frs_str2 = ["hydroxyl", "hydroxyl", "benzene"]

    frags = [fr(mol) for fr in frs_1]
    frs_str_match = []
    for i, frag in enumerate(frags):
        if frag > 0:
            frs_str_match.append(frs_str1[i])

    if len(frs_str_match) == 0:
        frags = [fr(mol) for fr in frs_2]
        frs_str_match = []
        for i, frag in enumerate(frags):
            if frag > 0:
                frs_str_match.append(frs_str2[i])
    if (len(frs_str_match) == 0) & (fr_unbrch_alkane(mol) > 0):
        frs_str_match.append("unbranched alkane")
    if len(frs_str_match) == 0:
        frs_str_match.append("other")

    return set(frs_str_match)
