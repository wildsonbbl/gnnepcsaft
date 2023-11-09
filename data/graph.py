import torch
from torch_geometric.data import Data
from deepchem.feat import MolGraphConvFeaturizer, GraphData


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
    from rdkit import Chem, RDLogger
    from torch_geometric.data import Data

    RDLogger.DisableLog("rdApp.*")

    mol = Chem.MolFromInchi(InChI)
    featurizer = MolGraphConvFeaturizer(True, True)

    if with_hydrogen:
        mol = Chem.AddHs(mol)
    if kekulize:
        Chem.Kekulize(mol)

    graph: GraphData = featurizer.featurize(mol)[0]
    x = graph.node_features
    edge_attr = graph.edge_features
    edge_index = graph.edge_index

    return Data(
        x=torch.from_numpy(x).float(),
        edge_attr=torch.from_numpy(edge_attr).float(),
        edge_index=torch.from_numpy(edge_index).long(),
        InChI=InChI,
    )


def from_smile(
    smile: str,
    with_hydrogen: bool = False,
    kekulize: bool = False,
) -> Data:
    r"""Converts a smile string to a :class:`torch_geometric.data.Data`
    instance.

    Args:
        smile (str): The smile string.
        with_hydrogen (bool, optional): If set to :obj:`True`, will store
            hydrogens in the molecule graph. (default: :obj:`False`)
        kekulize (bool, optional): If set to :obj:`True`, converts aromatic
            bonds to single/double bonds. (default: :obj:`False`)
    """
    from rdkit import Chem, RDLogger
    from torch_geometric.data import Data

    RDLogger.DisableLog("rdApp.*")

    mol = Chem.MolFromSmiles(smile)
    featurizer = MolGraphConvFeaturizer(True, True)

    if with_hydrogen:
        mol = Chem.AddHs(mol)
    if kekulize:
        Chem.Kekulize(mol)

    graph: GraphData = featurizer.featurize(mol)[0]
    x = graph.node_features
    edge_attr = graph.edge_features
    edge_index = graph.edge_index

    return Data(
        x=torch.from_numpy(x).float(),
        edge_attr=torch.from_numpy(edge_attr).float(),
        edge_index=torch.from_numpy(edge_index).long(),
        smile=smile,
    )
