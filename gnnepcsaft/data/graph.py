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
