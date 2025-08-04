"""Module for molecular graph design."""

import torch
from rdkit import RDLogger
from torch_geometric.data import Data

from .ogb_utils import smiles2graph  # from ogb.utils.mol import smiles2graph
from .rdkit_util import *  # pylint: disable = unused-wildcard-import, wildcard-import


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

    RDLogger.DisableLog("rdApp.*")  # type: ignore

    smiles = inchitosmiles(InChI, with_hydrogen, kekulize)
    graph = smiles2graph(smiles)

    x = graph["node_feat"]
    edge_attr = graph["edge_feat"]
    edge_index = graph["edge_index"]

    SELECTED_ATOMS = (6, 7, 8, 9, 14, 15, 16, 17, 35, 53)
    g_atom_count = atom_count(smiles)
    g_atom_count = [g_atom_count[i] for i in SELECTED_ATOMS]

    return Data(
        x=torch.from_numpy(x),
        edge_attr=torch.from_numpy(edge_attr),
        edge_index=torch.from_numpy(edge_index),
        InChI=InChI,
        smiles=smiles,
        ecfp=torch.from_numpy(ECFP(smiles, nBits=2**14)),
        mw=torch.as_tensor([[mw(InChI)]]),
        ring_count=torch.as_tensor([[ring_count(smiles)]]),
        rbond_count=torch.as_tensor([[rbond_count(smiles)]]),
        atom_count=torch.as_tensor([g_atom_count]),
    )


def from_smiles(smiles: str) -> Data:
    r"""Converts a smile string to a :class:`torch_geometric.data.Data`
    instance.

    Args:
        smile (str): The smile string.
    """

    inchi = smilestoinchi(smiles)
    return from_InChI(inchi)
