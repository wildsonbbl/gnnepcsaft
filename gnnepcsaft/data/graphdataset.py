"""Module for molecular graph dataset building."""

import polars as pl
import torch
from torch_geometric.data import InMemoryDataset

from .graph import from_InChI
from .rdkit_util import mw


class ThermoMLDataset(InMemoryDataset):
    """
    Molecular Graph dataset creator/manipulator with `ThermoML Archive` experimental data.

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
    ):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return ["pure.parquet"]

    @property
    def processed_file_names(self):
        return ["tml_graph_data.pt"]

    def download(self):
        raise ValueError(
            f"No url to download from. Provide data at {self.raw_paths[0]}."
        )

    def process(self):
        datalist = []

        with open(self.raw_paths[0], "rb") as f:
            data = pl.read_parquet(f)
        inchis = data.unique("inchi1")["inchi1"].to_list()

        for inchi in inchis:
            try:
                graph = from_InChI(inchi)
            except (TypeError, ValueError) as e:
                print(f"Error for InChI:\n {inchi}", e, sep="\n\n", end="\n\n")
                continue

            graph.vp = (
                data.filter(pl.col("inchi1") == inchi, pl.col("tp") == 3)
                .select("TK", "PPa", "phase", "tp", "m")
                .to_numpy()
            )
            rho = (
                data.filter(pl.col("inchi1") == inchi, pl.col("tp") == 1)
                .select("TK", "PPa", "phase", "tp", "m")
                .to_numpy()
            )

            rho[:, -1] *= 1000 / mw(inchi)  # convert to mol/ m³
            graph.rho = rho

            datalist.append(graph)

        torch.save(self.collate(datalist), self.processed_paths[0])


class Ramirez(InMemoryDataset):
    """
    Molecular Graph dataset creator/manipulator with `ePC-SAFT` parameters from
    `Ramírez-Vélez et al. (2022, doi: 10.1002/aic.17722)`.

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
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return ["data.parquet"]

    @property
    def processed_file_names(self):
        return ["ra_graph_data.pt"]

    def download(self):
        raise ValueError(
            f"No url to download from. Provide data at {self.raw_paths[0]}."
        )

    def process(self):
        datalist = []
        data = pl.read_parquet(self.raw_paths[0])

        for row in data.iter_rows():
            inchi = row[-1]
            para = row[3:6]
            try:
                graph = from_InChI(inchi)
            except (TypeError, ValueError) as e:
                print(f"Error for InChI:\n {inchi}", e, sep="\n\n", end="\n\n")
                continue

            graph.para = torch.tensor(para)
            graph.critic = torch.tensor(row[1:3])
            datalist.append(graph)

        torch.save(self.collate(datalist), self.processed_paths[0])


class Esper(InMemoryDataset):
    """
    Molecular Graph dataset creator/manipulator with `ePC-SAFT` parameters from
    `Esper et al. (2023, doi: 10.1021/acs.iecr.3c02255)`.

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
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return ["SI_pcp-saft_parameters.csv"]

    @property
    def processed_file_names(self):
        return ["es_graph_data.pt"]

    def download(self):
        raise ValueError(
            f"No url to download from. Provide data at {self.raw_paths[0]}."
        )

    def process(self):
        datalist = []
        dtype = torch.float32
        data = pl.read_csv(self.raw_paths[0], separator="	")

        for row in data.iter_rows():

            inchi = row[2]
            para = [value if value else 0.0001 for value in row[8:11]]
            assoc = row[12:14] if all(row[12:14]) else [0.0001, 200]
            munanb = [value if value else 0.0 for value in row[11:12] + row[14:16]]
            try:
                graph = from_InChI(inchi)
            except (TypeError, ValueError) as e:
                print(f"Error for InChI:\n {inchi}", e, sep="\n\n", end="\n\n")
                continue

            graph.para = torch.tensor([para], dtype=dtype)
            graph.assoc = torch.tensor([assoc], dtype=dtype).log10().abs()
            graph.munanb = torch.tensor([munanb], dtype=dtype)
            datalist.append(graph)

        torch.save(self.collate(datalist), self.processed_paths[0])
