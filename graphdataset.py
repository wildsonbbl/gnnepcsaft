from torch_geometric.data import Data, Dataset
import torch
from torch_geometric.data import InMemoryDataset
import polars as pl
from tqdm.notebook import tqdm as ntqdm
from tqdm import tqdm
from graph import from_InChI


def BinaryGraph(InChI1: str, InChI2: str) -> Data:

    """
    Make one graph out of 2 InChI keys for a binary system

    Parameters
    ------------
    InChI1: :class:`str`
        InChI value of molecule
    InChI2: :class:`str`
        InChI value of molecule

    Return
    ------------
    Graph: :class:`torch_geometric.data.Data`
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

class ThermoMLjax(Dataset):
    def __init__(self, dataset):
      """ Initializes the data reader by loading in data. """
      self.dataset = dataset
    
    def __len__(self):
      return len(self.dataset)

    def __getitem__(self, idx):
      sample = self.dataset[idx]
      
      data = Data(node_attr = sample.x,
                  num_nodes = sample.num_nodes,
                  n_node = sample.num_nodes,
                  edges = sample.edge_attr,
                  n_edge = sample.num_edges,
                  senders = sample.edge_index[0],
                  receivers = sample.edge_index[1],
                  globals = sample.y,
                  )
      return data

    def len(self):
      return self.__len__(self)
    
    def get(self, idx):
      return self.__getitem__(self, idx)

class ParametersDataset(InMemoryDataset):

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
                 pre_filter=None, Notebook=True):
        self.Notebook = Notebook
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['kij.parquet', 'pure.parquet']

    @property
    def processed_file_names(self):
        return 'paradata.pt'

    def download(self):
        return print('no url to download from')

    def process(self):
        

        progressbar = [ntqdm if self.Notebook else tqdm][0]
        datalist = []
        kij = pl.read_parquet(self.raw_paths[0])
        pure = pl.read_parquet(self.raw_paths[1])
        paradict = {}
        for row in pure.iter_rows():
            paradict[row[1].rstrip()]=row[2:]
        
        for row in progressbar(pure.iter_rows(), desc='datapoint', total=pure.shape[0]):
            inchi1 = row[1]
            inchi2 = row[1]
            try:
                graph = BinaryGraph(inchi1, inchi2)
                graph.y = torch.tensor((row[2:] + row[2:] + (0,0,0)), dtype = torch.float)
                graph.c1 = row[0]
                graph.c2 = row[0]
                datalist.append(graph)
            except:
                continue 

        for row in progressbar(kij.iter_rows(), desc='datapoint', total=kij.shape[0]):
            inchi1 = row[-1]
            inchi2 = row[-2]
            k_ij = row[5]
            para1 = paradict[inchi1]
            para2 = paradict[inchi2]
            try:
                graph = BinaryGraph(inchi1, inchi2)
                graph.y = torch.tensor((para1 + para2 + (k_ij,0,0)), dtype=torch.float)
                graph.c1 = row[4]
                graph.c2 = row[3]
                datalist.append(graph)
            except:
                continue 

        
        torch.save(self.collate(datalist), self.processed_paths[0])
        print('Done!')