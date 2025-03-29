"""Module for calculating molecule complexity and family group."""

from urllib.parse import quote
from urllib.request import HTTPError, urlopen

from rdkit import Chem

# pyright: reportAttributeAccessIssue=false
# pylint: disable = no-name-in-module
from rdkit.Chem.Fragments import (
    fr_Al_OH,
    fr_aldehyde,
    fr_amide,
    fr_Ar_OH,
    fr_benzene,
    fr_COO,
    fr_epoxide,
    fr_ester,
    fr_ether,
    fr_halogen,
    fr_ketone,
    fr_NH0,
    fr_NH1,
    fr_NH2,
    fr_nitrile,
    fr_phenol,
    fr_phos_acid,
    fr_SH,
    fr_sulfide,
    fr_unbrch_alkane,
)


def complexity(ids: str) -> float:
    """Complexity as implemented by `PubChem`."""
    try:
        url = (
            "https://pubchem.ncbi.nlm.nih.gov/"
            + "rest/pug/compound/inchi/property/complexity/TXT?inchi="
            + quote(ids)
        )
        with urlopen(url) as ans:
            ans = ans.read().decode("utf8").rstrip()
            ans = ans.split("\n")[0]
            if ans:
                ans = float(ans)
    except (TypeError, ValueError, HTTPError):
        print("not ok:", ids)
        ans = float("inf")

    # sleep(0.1)
    return ans


def get_family_groups(inchi: str) -> list[str]:
    """Find a family groups for a molecule."""
    mol = Chem.MolFromInchi(inchi, sanitize=True)

    if mol is None:
        mol = Chem.MolFromInchi(inchi, sanitize=False)

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

    return list(set(frs_str_match))
