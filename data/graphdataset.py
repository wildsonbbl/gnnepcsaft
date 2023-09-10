from torch_geometric.data import Data
from torch.utils.data import Dataset as ds
import torch
from torch_geometric.data import InMemoryDataset
from data.graph import from_InChI
import pickle
import polars as pl


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

    x = torch.cat((graph1.x, graph2.x))
    edge_attr = torch.cat((graph1.edge_attr, graph2.edge_attr))
    edge_index = torch.cat(
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
        dtype=torch.float64,
        graph_dtype=torch.long,
    ):
        self.dtype = dtype
        self.graph_dtype = graph_dtype

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["pure.pkl"]

    @property
    def processed_file_names(self):
        return ["tml_graph_data.pt"]

    def download(self):
        return print("no url to download from")

    def process(self):
        datalist = []

        with open(self.raw_paths[0], "rb") as f:
            data_dict = pickle.load(f)

        for inchi in data_dict:
            graph = from_InChI(inchi, with_hydrogen=False)
            if graph.x.shape[0] <= 2:
                graph = from_InChI(inchi, with_hydrogen=True)
            if (3 in data_dict[inchi]) & (1 not in data_dict[inchi]):
                states = [
                    torch.cat(
                        [
                            torch.tensor(state, dtype=self.dtype),
                            torch.tensor([y], dtype=self.dtype),
                        ]
                    )[None, ...]
                    for _, state, y in data_dict[inchi][3]
                ]

                states = torch.cat(states, 0)

                graph.vp = states
                graph.rho = torch.zeros((1, 5))

                datalist.append(graph)
            elif (3 in data_dict[inchi]) & (1 in data_dict[inchi]):
                vp = [
                    torch.cat(
                        [
                            torch.tensor(state, dtype=self.dtype),
                            torch.tensor([y], dtype=self.dtype),
                        ]
                    )[None, ...]
                    for _, state, y in data_dict[inchi][3]
                ]

                vp = torch.cat(vp, 0)
                rho = [
                    torch.cat(
                        [
                            torch.tensor(state, dtype=self.dtype),
                            torch.tensor([y], dtype=self.dtype),
                        ]
                    )[None, ...]
                    for _, state, y in data_dict[inchi][1]
                ]

                rho = torch.cat(rho, 0)

                graph.vp = vp
                graph.rho = rho

                datalist.append(graph)
            elif 1 in data_dict[inchi]:
                rho = [
                    torch.cat(
                        [
                            torch.tensor(state, dtype=self.dtype),
                            torch.tensor([y], dtype=self.dtype),
                        ]
                    )[None, ...]
                    for _, state, y in data_dict[inchi][1]
                ]

                rho = torch.cat(rho, 0)

                graph.vp = torch.zeros((1, 5))
                graph.rho = rho
                datalist.append(graph)

        torch.save(self.collate(datalist), self.processed_paths[0])


class ThermoML_padded(ds):
    def __init__(self, dataset: ThermoMLDataset, pad_size: int = 32):
        """Initializes the data reader by loading in data."""
        self.dataset = dataset
        self.pad = pad_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        vp = sample.vp
        n = vp.shape[0]
        pad = _nearest_bigger_power_of_two(n)
        if pad > self.pad:
            pad = self.pad
        vp = get_padded_array(vp, pad)

        rho = sample.rho
        n = rho.shape[0]
        pad = _nearest_bigger_power_of_two(n)
        if pad > self.pad:
            pad = self.pad
        rho = get_padded_array(rho, pad)

        data = Data(
            x=sample.x,
            edge_attr=sample.edge_attr,
            edge_index=sample.edge_index,
            rho=rho,
            vp=vp,
            InChI=sample.InChI,
        )
        return data

    def len(self):
        return self.__len__(self)

    def get(self, idx):
        return self.__getitem__(self, idx)


def get_padded_array(states: torch.Tensor, pad_size: int = 2**10) -> torch.Tensor:
    indexes = torch.randperm(states.shape[0])
    states = states[indexes]
    states = states.repeat(pad_size // states.shape[0] + 1, 1)
    return states[:pad_size, :]


def _nearest_bigger_power_of_two(x: int) -> int:
    """Computes the nearest power of two greater than x for padding."""
    y = 2
    while y < x:
        y *= 2
    return y


class ramirez(InMemoryDataset):

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

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["data.parquet"]

    @property
    def processed_file_names(self):
        return ["ra_graph_data.pt"]

    def download(self):
        return print("no url to download from")

    def process(self):
        datalist = []
        data = pl.read_parquet(self.raw_paths[0])

        for row in data.iter_rows():
            inchi = row[-1]
            para = row[3:6]
            graph = from_InChI(inchi, with_hydrogen=False)
            if graph.x.shape[0] <= 2:
                graph = from_InChI(inchi, with_hydrogen=True)

            graph.para = torch.tensor(para)
            graph.critic = torch.tensor(row[1:3])
            datalist.append(graph)

        torch.save(self.collate(datalist), self.processed_paths[0])


class ThermoMLpara(InMemoryDataset):

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

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["para3_fitted.pkl", "para_ramirez.parquet"]

    @property
    def processed_file_names(self):
        return ["tml_para_graph_data.pt"]

    def download(self):
        return print("no url to download from")

    def process(self):
        datalist = []
        inchis = []
        
        radata = pl.read_parquet(self.raw_paths[1])
        print(f"ramirez dataset size: {radata.shape[0]}")
        with open(self.raw_paths[0], "rb") as file:
            fitted = pickle.load(file)
        print(f"thermoml dataset size: {len(fitted)}")

        for inchi in fitted:
            para, mden, mvp = fitted[inchi] 
            if (mden > 3 / 100) or (mvp > 5/100):
                continue
            graph = from_InChI(inchi, with_hydrogen=False)
            if graph.x.shape[0] <= 2:
                graph = from_InChI(inchi, with_hydrogen=True)

            graph.para = torch.tensor(para)
            datalist.append(graph)
            inchis.append(inchi)

        for row in radata.iter_rows():
            inchi = row[-1]
            if inchi in inchis:
                continue
            para = row[3:6]
            graph = from_InChI(inchi, with_hydrogen=False)
            if graph.x.shape[0] <= 2:
                graph = from_InChI(inchi, with_hydrogen=True)

            graph.para = torch.tensor(para)
            datalist.append(graph)

        torch.save(self.collate(datalist), self.processed_paths[0])
