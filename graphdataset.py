from torch_geometric.data import Data
import torch
from torch_geometric.data import InMemoryDataset
import polars as pl
from tqdm.notebook import tqdm as ntqdm
from tqdm import tqdm
from graph import from_InChI


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

    graph1 = from_InChI(InChI1)
    graph2 = from_InChI(InChI2)

    x = torch.concatenate((graph1.x,
                                    graph2.x))
    edge_attr = torch.concatenate((graph1.edge_attr,
                                    graph2.edge_attr))
    edge_index = torch.concatenate((graph1.edge_index,
                                 graph2.edge_index + graph1.num_nodes),
                                axis=1)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


class ThermoMLDataset(InMemoryDataset):

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
                 pre_filter=None, Notebook=True, subset='train', nrow: int = None):
        self.Notebook = Notebook
        self.nrow = nrow
        if subset in ['train', 'test','val']:
            self.subset = subset
        else:
            raise ValueError('subset should be either train or test')
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.subset + '.parquet']

    @property
    def processed_file_names(self):
        return 'tmldata.pt'

    def download(self):
        return print('no url to download from')

    def process(self):
        nrow = self.nrow
        cols = ['mlc1', 'mlc2', 'TK', 'PPa',
                'phase', 'type', 'm']

        progressbar = [ntqdm if self.Notebook else tqdm][0]
        datalist = []
        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            if nrow != None:
                dataframe = pl.read_parquet(raw_path, n_rows = nrow)
            else:
                dataframe = pl.read_parquet(raw_path)

            total = dataframe.shape[0]
            
            for datapoint in progressbar(dataframe.iter_rows(named=True),
                                         desc='data points',
                                         total=total):
                
                y = [datapoint[col] for col in cols]

                try:
                        
                    graph = BinaryGraph(datapoint['inchi1'], datapoint['inchi2'])
                    graph.y = torch.tensor(y, dtype=torch.float)
                    graph.c1 = datapoint['c1']
                    graph.c2 = datapoint['c2']

                    datalist.append(graph)
                except:
                    continue
                

        dataframe = []
        torch.save(self.collate(datalist), self.processed_paths[0])
        datalist= []
        print('Done!')
