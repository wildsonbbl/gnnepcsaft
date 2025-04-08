"Module for helper functions with rdkit"

from typing import List, Tuple

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

# pylint: disable = no-name-in-module
from rdkit.Chem.Fragments import fr_COO2  # type: ignore
from rdkit.Chem.Fragments import fr_Imine, fr_isocyan, fr_isothiocyan  # type: ignore
from rdkit.Chem.rdMolDescriptors import (
    CalcExactMolWt,
    CalcNumHBA,
    CalcNumHBD,
    CalcNumRings,
    CalcNumRotatableBonds,
)

RDLogger.DisableLog("rdApp.*")  # type: ignore


# pylint: disable = invalid-name
def inchitosmiles(
    InChI: str, with_hydrogen: bool = False, kekulize: bool = False
) -> str:
    """Transform InChI to a SMILES.

    Args:
        InChI (str): InChI
        with_hydrogen (bool): Add hydrogens to the molecule
        kekulize (bool): Kekulize the molecule
    """
    mol = Chem.MolFromInchi(InChI)
    if mol is None:
        raise ValueError("InChI is not valid")

    # pylint: disable = no-member
    if with_hydrogen:
        mol = Chem.AddHs(mol)
    if kekulize:
        Chem.Kekulize(mol)

    smiles = Chem.MolToSmiles(mol)
    return smiles


def smilestoinchi(
    smiles: str, with_hydrogen: bool = False, kekulize: bool = False
) -> str:
    """Transform SMILES to InChI.

    Args:
        smiles (str): SMILES
        with_hydrogen (bool): Add hydrogens to the molecule
        kekulize (bool): Kekulize the molecule
    """
    # pylint: disable = no-member
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("SMILES is not valid")

    if with_hydrogen:
        mol = Chem.AddHs(mol)
    if kekulize:
        Chem.Kekulize(mol)

    inchi = Chem.MolToInchi(mol)
    return inchi  # type: ignore


def assoc_number(inchi: str) -> Tuple[int, int]:
    """Calculates the number of H-bond acceptors/donors

    Args:
        inchi (str): InChI
    """

    exceptions = (
        "InChI=1S/H2O/h1H2",
        "InChI=1S/Cl2/c1-2",
        "InChI=1S/F2/c1-2",
    )  # From Esper et al. (2023) 10.1021/acs.iecr.3c02255
    if inchi in exceptions:
        return 1, 1
    mol = Chem.MolFromInchi(inchi, removeHs=False)
    if mol is None:
        raise ValueError("InChI is not valid")
    mol = Chem.AddHs(mol)
    na = CalcNumHBA(mol)
    nb = CalcNumHBD(mol)
    if na > 0:
        n_coo = fr_COO2(mol)
        n_Imine = fr_Imine(mol)
        n_isocyanates = fr_isocyan(mol)
        n_isothiocyanates = fr_isothiocyan(mol)
        n_priamide = len(
            mol.GetSubstructMatches(
                Chem.AddHs(
                    Chem.MolFromInchi("InChI=1S/CH3NO/c2-1-3/h1H,(H2,2,3)"),
                    onlyOnAtoms=[1, 2],
                )
            )
        )
        n_sulfuro = len(
            mol.GetSubstructMatches(
                Chem.AddHs(Chem.MolFromSmiles("S(=O)(=O)O"), onlyOnAtoms=[3])
            )
        )

        na -= (
            n_coo + n_priamide + n_Imine - n_isocyanates - n_isothiocyanates + n_sulfuro
        )

    return na, nb


def mw(inchi: str) -> float:
    """Calcultes molecular weight.

    Args:
        inchi (str): InChI
    """
    try:
        mol = Chem.MolFromInchi(inchi, removeHs=False, sanitize=False)
        mol_weight = CalcExactMolWt(mol)
    except (TypeError, ValueError):
        mol_weight = 0

    return mol_weight


def ECFP(smiles: str, radius: int = 3, nBits: int = 3072) -> np.ndarray:
    """Calculates ECFP fingerprints.

    Args:
        smiles (str): SMILES
        radius (int): Radius
        nBits (int): Number of bits
    """
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)  # type: ignore
        fp = np.array([fp], dtype=np.int8)
    except (TypeError, ValueError):
        fp = np.zeros((1, nBits), dtype=np.int8)
    return fp


def ring_count(smiles: str) -> int:
    """Calculates the number of rings.

    Args:
        smiles (str): SMILES
    """
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        _ring_count = CalcNumRings(mol)
    except (TypeError, ValueError):
        _ring_count = 0
    return _ring_count


def rbond_count(smiles: str) -> int:
    """Calculates the number of rotatable bonds.

    Args:
        smiles (str): SMILES
    """
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        _rbond_count = CalcNumRotatableBonds(mol)
    except (TypeError, ValueError):
        _rbond_count = 0
    return _rbond_count


def atom_count(smiles: str) -> List[int]:
    """Count the number of each atom in a molecule.

    Args:
        smiles (str): SMILES
    """
    atom_count_list = [0] * 119
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        for atom in mol.GetAtoms():
            atom_count_list[atom.GetAtomicNum()] += 1
    except (TypeError, ValueError):
        pass
    return atom_count_list
