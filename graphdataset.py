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
import glob


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

    """

    def __init__(self, root, transform=None, pre_transform=None,
                 pre_filter=None, log=True, Notebook=True):
        self.Notebook = Notebook
        
        super().__init__(root, transform, pre_transform, pre_filter, log)

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        try:
            path = osp.join(self.processed_dir,'data*.pt')
            files = [file.split('/')[-1] for file in glob.glob(path)]
        except:
            os.mkdir(self.processed_dir)
            path = osp.join(self.processed_dir,'data*.pt')
            files = [file.split('/')[-1] for file in glob.glob(path)]
        return files

    def download(self):
        return print('no url to download from')

    def process(self, nrow: int = None):
        idx = 0
        cols = ['mlc1', 'mlc2', 'TK', 'PPa',
                'phase', 'type', 'm']

        progressbar = [ntqdm if self.Notebook else tqdm][0]

        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            if nrow != None:
                self.dataframe = pl.read_parquet(raw_path, n_rows = nrow)
            else:
                self.dataframe = pl.read_parquet(raw_path)

            total = self.dataframe.shape[0]
            
            for datapoint in progressbar(self.dataframe.iter_rows(named=True),
                                         desc='data points',
                                         total=total):
                
                y = [datapoint[col] for col in cols]

                graph = BinaryGraph(datapoint['inchi1'], datapoint['inchi2'])
                graph = graph.to_pyg_graph()
                graph.y = torch.tensor(y, dtype=torch.float64)
                graph.c1 = datapoint['c1']
                graph.c2 = datapoint['c2']

                torch.save(graph, osp.join(
                    self.processed_dir, f'data_{idx}.pt'))
                

                idx += 1
        print('Done!')

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data
