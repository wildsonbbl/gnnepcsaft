import numpy as np
from rdkit import Chem
import deepchem as dc
import deepchem.feat as ft
import torch_geometric as pyg
from torch_geometric.data import Data
import os.path as osp
import os
import torch
from torch_geometric.data import Dataset, download_url
import polars as pl
from tqdm.notebook import tqdm as ntqdm
from tqdm import tqdm


def BinaryGraph(InChI1: str, InChI2: str):
    """
    Make one graph out of 2 InChI keys for a binary system

    Parameters
    ------------
    InChI1: str
        InChI value of molecule
    InChI2: str 
        InChI value of molecule

    Return
    ------------
    Graph: deepchem.feat.GraphData
        Graph of binary system
    """

    mol1 = Chem.MolFromInchi(InChI1, removeHs=False)
    mol2 = Chem.MolFromInchi(InChI2, removeHs=False)

    mol1 = Chem.AddHs(mol1)
    mol2 = Chem.AddHs(mol2)

    featurizer = ft.MolGraphConvFeaturizer(use_edges=True)

    graph = featurizer.featurize((mol1, mol2))

    node_features = np.concatenate((graph[0].node_features,
                                    graph[1].node_features))
    edge_features = np.concatenate((graph[0].edge_features,
                                    graph[1].edge_features))
    edge_index = np.concatenate((graph[0].edge_index,
                                 graph[1].edge_index + graph[0].num_nodes),
                                axis=1)

    return ft.GraphData(node_features, edge_index, edge_features)


class ThermoMLDataset(Dataset):
    """
    Molecular Graph dataset creator/manipulator.

    PARAMETERS
    ----------
    root (str, optional) – Root directory where the dataset should be saved. (optional: None).

    transform (callable, optional) – A function/transform that takes in an Data object and 
    returns a transformed version. The data object will be transformed
    before every access. (default: None).

    pre_transform (callable, optional) – A function/transform that takes in
    an Data object and returns a transformed version. The data object will be
    transformed before being saved to disk. (default: None).

    pre_filter (callable, optional) – A function that takes in an Data object 
    and returns a boolean value, indicating whether the data object should be 
    included in the final dataset. (default: None).

    log (bool, optional) – Whether to print any console output while downloading 
    and processing the dataset. (default: True).

    Notebook (bool, optional) - Whether to use tqdm.notebook progress bar while processing data.

    VP (bool, optional) - Whether is being used for vapor pressure data.

    nrow (int, optional) - How many rows of raw data to process. None is for all data.
    """

    def __init__(self, root, transform=None, pre_transform=None,
                 pre_filter=None, log=True, Notebook=True, VP=False, nrow: int = None):
        self.Notebook = Notebook
        self.VP = VP
        self.nrow = nrow
        super().__init__(root, transform, pre_transform, pre_filter, log)

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        try:
            files = os.listdir(self.processed_dir)
        except:
            os.mkdir(self.processed_dir)
            files = os.listdir(self.processed_dir)
        return files

    def download(self):
        return print('no url to download from')

    def process(self):
        idx = 0
        cols = ['TK', 'PkPa',
                'mlc1', 'mlc2', 'm1', 'm2']
        if self.VP:
            cols.remove("PkPa")

        progressbar = [ntqdm if self.Notebook else tqdm][0]

        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            self.dataframe = pl.read_parquet(raw_path)
            total = int([self.dataframe.shape[0] if self.nrow == None else self.nrow][0])

            for datapoint in progressbar(self.dataframe.iter_rows(named=True),
                                         desc='data points',
                                         total=total):
                if idx >= self.nrow:
                    break
                y = [datapoint[col] for col in cols]

                graph = BinaryGraph(datapoint['inchi1'], datapoint['inchi2'])
                graph = graph.to_pyg_graph()
                graph.y = torch.tensor(y, dtype=torch.float64)
                graph.c1 = datapoint['c1']
                graph.c2 = datapoint['c2']
                graph.phase = datapoint['phase']
                graph.type = datapoint['type']

                torch.save(graph, osp.join(
                    self.processed_dir, f'data_{idx}.pt'))
                

                idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data
