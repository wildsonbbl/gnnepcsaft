"""Module for molecular graph design."""

import torch
from ogb.utils.mol import smiles2graph
from rdkit import Chem, RDLogger

# pylint: disable = no-name-in-module
from rdkit.Chem.Fragments import fr_COO2, fr_Imine, fr_isocyan, fr_isothiocyan
from rdkit.Chem.rdMolDescriptors import CalcNumHBA, CalcNumHBD
from torch_geometric.data import Data


# pylint: disable = invalid-name
def from_InChI(
    InChI: str,
    with_hydrogen: bool = False,
    kekulize: bool = False,
) -> Data:
    r"""Converts a InChI string to a :class:`torch_geometric.data.Data`
    instance.

    Args:
        InChI (str): The InChI string.
        with_hydrogen (bool, optional): If set to :obj:`True`, will store
            hydrogens in the molecule graph. (default: :obj:`False`)
        kekulize (bool, optional): If set to :obj:`True`, converts aromatic
            bonds to single/double bonds. (default: :obj:`False`)
    """

    RDLogger.DisableLog("rdApp.*")

    smiles = inchitosmiles(InChI, with_hydrogen, kekulize)
    graph = smiles2graph(smiles)

    x = graph["node_feat"]
    edge_attr = graph["edge_feat"]
    edge_index = graph["edge_index"]

    return Data(
        x=torch.from_numpy(x),
        edge_attr=torch.from_numpy(edge_attr),
        edge_index=torch.from_numpy(edge_index),
        InChI=InChI,
        smiles=smiles,
    )


def inchitosmiles(InChI, with_hydrogen, kekulize):
    "Transform InChI to a SMILES."
    mol = Chem.MolFromInchi(InChI)

    # pylint: disable = no-member
    if with_hydrogen:
        mol = Chem.AddHs(mol)
    if kekulize:
        Chem.Kekulize(mol)

    smiles = Chem.MolToSmiles(mol)
    return smiles


def smilestoinchi(smiles, with_hydrogen=False, kekulize=False):
    "Transform SMILES to InChI."
    # pylint: disable = no-member
    mol = Chem.MolFromSmiles(smiles)

    if with_hydrogen:
        mol = Chem.AddHs(mol)
    if kekulize:
        Chem.Kekulize(mol)

    inchi = Chem.MolToInchi(mol)
    return inchi


def from_smiles(smiles: str) -> Data:
    r"""Converts a smile string to a :class:`torch_geometric.data.Data`
    instance.

    Args:
        smile (str): The smile string.
    """

    RDLogger.DisableLog("rdApp.*")

    graph = smiles2graph(smiles)

    x = graph["node_feat"]
    edge_attr = graph["edge_feat"]
    edge_index = graph["edge_index"]

    return Data(
        x=torch.from_numpy(x),
        edge_attr=torch.from_numpy(edge_attr),
        edge_index=torch.from_numpy(edge_index),
        smiles=smiles,
    )


def assoc_number(inchi: str):
    "Calculates the number of H-bond acceptors/donors"

    exceptions = (
        "InChI=1S/H2O/h1H2",
        "InChI=1S/Cl2/c1-2",
        "InChI=1S/F2/c1-2",
    )  # From Esper et al. (2023) 10.1021/acs.iecr.3c02255
    if inchi in exceptions:
        return 1, 1
    mol = Chem.MolFromInchi(inchi, removeHs=False)
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
