from torch_geometric.data import Data
import torch
from torch_geometric.data import InMemoryDataset
import polars as pl
from graph import from_InChI
import pickle

from rdkit import Chem, RDLogger
from rdkit.Chem.rdMolDescriptors import CalcExactMolWt

RDLogger.DisableLog("rdApp.*")



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

    x = torch.concatenate((graph1.x, graph2.x))
    edge_attr = torch.concatenate((graph1.edge_attr, graph2.edge_attr))
    edge_index = torch.concatenate(
        (graph1.edge_index, graph2.edge_index + graph1.num_nodes), dim=1
    )

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

    """

    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        subset="train",
        dtype=torch.float64,
        graph_dtype=torch.int32
    ):
        self.dtype = dtype
        self.graph_dtype = graph_dtype

        if subset in ["train", "val", "test"]:
            self.subset = subset
        else:
            raise ValueError("subset should be either train, val or test")
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["pure.pkl"]

    @property
    def processed_file_names(self):
        return [self.subset + "_pure.pt"]

    def download(self):
        return print("no url to download from")

    def process(self):
        datalist = []
        print("### Loading dictionary of data ###")
        with open(self.raw_paths[0], 'rb') as f:
                data_dict = pickle.load(f)
        print(
            f"### Done!\n Whole dataset size = {len(data_dict)} ###"
            f"\n### Starting to make graphs ###"
        )

        for inchi in data_dict:
            try:
                if self.subset == 'test':
                    if (3 in data_dict[inchi]) & (1 not in data_dict[inchi]):
                        graph = from_InChI(inchi, dtype = self.graph_dtype)
                        states = [
                            torch.concatenate(
                                [
                                    torch.tensor(state, dtype=self.dtype),
                                    torch.tensor([y], dtype=self.dtype),
                                ]
                            )[None, ...]
                            for _, state, y in data_dict[inchi][3]
                        ]

                        states = torch.concatenate(states, 0)
                        graph.states = states

                        datalist.append(graph)
                elif self.subset == 'val':
                    if (3 in data_dict[inchi]) & (1 in data_dict[inchi]):
                        graph = from_InChI(inchi, dtype = self.graph_dtype)
                        states = [
                            torch.concatenate(
                                [
                                    torch.tensor(state, dtype=self.dtype),
                                    torch.tensor([y], dtype=self.dtype),
                                ]
                            )[None, ...]
                            for _, state, y in data_dict[inchi][3]
                        ]

                        states = torch.concatenate(states, 0)
                        graph.states = states

                        datalist.append(graph)
                else:
                    if (1 in data_dict[inchi]):
                        graph = from_InChI(inchi, dtype = self.graph_dtype)
                        states = [
                            torch.concatenate(
                                [
                                    torch.tensor(state, dtype=self.dtype),
                                    torch.tensor([y], dtype=self.dtype),
                                ]
                            )[None, ...]
                            for _, state, y in data_dict[inchi][1]
                        ]

                        states = torch.concatenate(states, 0)
                        graph.states = states

                        datalist.append(graph)
            except:
                continue

        torch.save(self.collate(datalist), self.processed_paths[0])
        print("### Done! ###")


def get_padded_array(
    states: torch.Tensor, max_pad: int = 2**10
) -> torch.Tensor:
    indexes = torch.randperm(states.shape[0])
    states = states[indexes]
    pad_size = max_pad

    states = states.repeat(pad_size // states.shape[0] + 1, 1)
    return states[:pad_size, :]
