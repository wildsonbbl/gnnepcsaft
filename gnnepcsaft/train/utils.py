"""Model with important functions to help model training"""

import multiprocessing as mp
import os.path as osp
import time
from tempfile import TemporaryDirectory
from typing import Any, List, Union

import numpy as np
import torch
import torch_geometric.transforms as T
import xgboost as xgb
from absl import logging
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from ray import tune
from ray.tune.experiment.trial import Trial
from sklearn.ensemble import RandomForestRegressor
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import degree
from xgboost import Booster

from ..data.graph import assoc_number
from ..data.graphdataset import Esper, Ramirez, ThermoMLDataset
from ..epcsaft.utils import pure_den_feos, pure_vp_feos

# PCSAFT parameters bounds
params_lower_bound = np.array([1.0, 1.9, 50.0, 1e-4, 200.0, 0, 0, 0])
params_upper_bound = np.array([25.0, 4.5, 550.0, 0.9, 5000.0, np.inf, np.inf, np.inf])
params_mean_msigmae = torch.tensor([4.0534, 3.6834, 266.8723])
params_std_msigmae = torch.tensor([2.0430, 0.3976, 57.7073])
params_mean_assoc = torch.tensor([3.6416, 2.5086])
params_std_assoc = torch.tensor([0.8802, 0.4257])


def calc_deg(dataset: str, workdir: str) -> List:
    """Calculates deg for `PNA` conv."""
    if dataset == "ramirez":
        path = osp.join(workdir, "data/ramirez2022")
        train_dataset = Ramirez(path)
    elif dataset in ("esper", "esper_assoc", "esper_assoc_only"):
        path = osp.join(workdir, "data/esper2023")
        train_dataset = Esper(path)
    else:
        raise ValueError(
            f"dataset is either ramirez or esper, got >>> {dataset} <<< instead"
        )
    # Compute the maximum in-degree in the training data.
    max_degree = -1
    for data in train_dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, int(d.max()))

    # Compute the in-degree histogram tensor
    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in train_dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())
    return deg.tolist()


def rhovp_data(parameters: List, rho: np.ndarray, vp: np.ndarray):
    """Calculates density and vapor pressure with ePC-SAFT"""

    den_array = rho_single((parameters, rho))
    vp_array = vp_single((parameters, vp))

    return den_array, vp_array


# pylint: disable=R0903
class TransformParameters(BaseTransform):
    "To add parameters to test dataset."

    def __init__(self, para_data: dict) -> None:
        super().__init__()
        self.para_data = para_data

    def forward(self, data: Any) -> Any:
        if data.InChI in self.para_data:
            data.para, data.assoc, data.munanb = self.para_data[data.InChI]
        else:
            data.para, data.assoc = (
                torch.zeros(1, 3),
                torch.zeros(1, 2),
            )
            data.munanb = torch.tensor(
                [(0,) + assoc_number(data.InChI)], dtype=torch.float32
            )
        return data


def build_test_dataset(
    workdir: str,
    train_dataset: Union[Esper, Ramirez],
    transform: Union[None, BaseTransform] = None,
) -> tuple[ThermoMLDataset, ThermoMLDataset]:
    "Builds test dataset."

    para_data = {}
    if isinstance(train_dataset, (Esper,)):
        for graph in train_dataset:
            para_data[graph.InChI] = (
                graph.para,
                graph.assoc,
                graph.munanb,
            )
    if transform:
        transform = T.Compose([TransformParameters(para_data), transform])
    else:
        transform = TransformParameters(para_data)
    tml_dataset = ThermoMLDataset(
        osp.join(workdir, "data/thermoml"), transform=transform
    )

    val_msigmae_idx: List[int] = []
    train_idx: List[int] = []
    # separate test and val dataset
    for idx, graph in enumerate(tml_dataset):
        if graph.InChI not in para_data and graph.munanb[0, -1] == 0:
            val_msigmae_idx.append(idx)
        if graph.InChI in para_data:
            train_idx.append(idx)
    val_msigmae_dataset = tml_dataset[val_msigmae_idx]
    train_val_dataset = tml_dataset[train_idx]
    return val_msigmae_dataset, train_val_dataset  # type: ignore


def build_train_dataset(workdir, dataset, transform=None) -> Union[Esper, Ramirez]:
    "Builds train dataset."
    if dataset == "ramirez":
        path = osp.join(workdir, "data/ramirez2022")
        return Ramirez(path, transform=transform)
    if dataset == "esper":
        path = osp.join(workdir, "data/esper2023")
        return Esper(path, transform=transform)
    if dataset == "esper_assoc":
        path = osp.join(workdir, "data/esper2023")
        train_dataset = Esper(path, transform=transform)
        assoc_idx = []
        non_assoc_idx = []
        for i, graph in enumerate(train_dataset):
            if all(graph.munanb[0, 1:] > 0):
                assoc_idx.append(i)
            if all(graph.munanb[0, 1:] == 0):
                non_assoc_idx.append(i)
        dataset_idxs = assoc_idx * 4 + non_assoc_idx
        return train_dataset[dataset_idxs]  # type: ignore
    if dataset == "esper_assoc_only":
        path = osp.join(workdir, "data/esper2023")
        train_dataset = Esper(path, transform=transform)
        as_idx = []
        for i, graph in enumerate(train_dataset):
            if all(graph.munanb[0, 1:] > 0):
                as_idx.append(i)
        return train_dataset[as_idx]  # type: ignore
    raise ValueError(
        f"dataset is either ramirez, esper, esper_assoc \
              or esper_assoc_only, got >>> {dataset} <<< instead"
    )


class EpochTimer(Callback):
    "Elapsed time counter."

    start_time: float = time.time()

    def on_train_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        end_time = time.time()

        elapsed_time = end_time - self.start_time
        logging.log_first_n(
            logging.INFO, "Elapsed time %.4f min.", 20, elapsed_time / 60
        )


class CustomRayTrainReportCallback(Callback):
    "Lightning Callback for Ray Tune reporting."

    def on_validation_end(self, trainer, pl_module):

        with TemporaryDirectory() as tmpdir:
            # Fetch metrics
            metrics = trainer.callback_metrics
            metrics = {k: v.item() for k, v in metrics.items()}

            # Add customized metrics
            metrics["epoch"] = trainer.current_epoch
            metrics["step"] = trainer.global_step

            trial_id = tune.get_context().get_trial_id()
            # Save model checkpoint file to tmpdir
            ckpt_path = osp.join(tmpdir, f"{trial_id}.ckpt")
            trainer.save_checkpoint(ckpt_path, weights_only=False)
            checkpoint = tune.Checkpoint.from_directory(tmpdir)

            # Report to train session
            tune.report(metrics=metrics, checkpoint=checkpoint)


class TrialTerminationReporter(tune.JupyterNotebookReporter):
    """Reporter for Ray to report only when trial is terminated"""

    def __init__(self):
        super().__init__()
        self.num_terminated = 0

    def should_report(self, trials, done=False):
        """
        Reports only on trial termination events.
        It does so by tracking increase in number of trials terminated.
        """
        old_num_terminated = self.num_terminated
        self.num_terminated = len([t for t in trials if t.status == Trial.TERMINATED])
        return self.num_terminated > old_num_terminated


class CustomStopper(tune.Stopper):
    """Custom ray tune experiment/trial stopper"""

    def __init__(self, max_iter: int):
        self.should_stop = False
        self.max_iter = max_iter

    def __call__(self, trial_id, result):
        return result["training_iteration"] >= self.max_iter

    def stop_all(self):
        return False


def rho_single(args: tuple[List, np.ndarray]) -> np.ndarray:
    """Calculates density with ePC-SAFT for a single pararameter"""
    parameters, states = args
    rho_for_all_states = []

    for state in states:
        try:
            rho_for_state = pure_den_feos(parameters, state)
        except (AssertionError, RuntimeError):
            rho_for_state = 0.0
        rho_for_all_states += [rho_for_state]
    return np.asarray(rho_for_all_states)


def rho_batch(
    parameters_batch: List[List[Any]], states_batch: List[np.ndarray]
) -> List[np.ndarray]:
    """
    Calculates density with ePC-SAFT
    for a batch of parameters
    using nested multiprocessing
    """
    args_list = [
        (para, states)
        for para, states in zip(parameters_batch, states_batch)
        if states.shape[0] > 0
    ]
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=ctx.cpu_count() // 2) as pool:
        den = pool.map(rho_single, args_list)
    return den


def vp_single(args: tuple[List, np.ndarray]) -> np.ndarray:
    """Calculates vapor pressure with ePC-SAFT for a single parameter"""
    parameters, states = args
    vp_for_all_states = []
    for state in states:
        try:
            vp_for_state = pure_vp_feos(parameters, state)
        except (AssertionError, RuntimeError):
            vp_for_state = 0.0
        vp_for_all_states += [vp_for_state]
    return np.asarray(vp_for_all_states)


def vp_batch(
    parameters_batch: List[List[Any]], states_batch: List[np.ndarray]
) -> List[np.ndarray]:
    """
    Calculates vapor pressure with ePC-SAFT
    for a batch of parameters
    using nested multiprocessing
    """
    args_list = [
        (para, states)
        for para, states in zip(parameters_batch, states_batch)
        if states.shape[0] > 0
    ]
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=ctx.cpu_count() // 2) as pool:
        vp = pool.map(vp_single, args_list)
    return vp


def rf_xgb_evaluation(
    graphs: Data, model: Union[RandomForestRegressor, Booster]
) -> tuple[float, float]:
    """Evaluation function for RF and XGBoost models"""
    if isinstance(model, Booster):
        x = xgb.DMatrix(
            torch.hstack(
                (
                    graphs.ecfp,
                    graphs.mw,
                    graphs.atom_count,
                    graphs.ring_count,
                    graphs.rbond_count,
                )
            ).numpy()
        )
        pred_msigmae = model.predict(x)
    else:
        x = torch.hstack(
            (
                graphs.ecfp,
                graphs.mw,
                graphs.atom_count,
                graphs.ring_count,
                graphs.rbond_count,
            )
        ).numpy()
        pred_msigmae = model.predict(x)

    assert isinstance(pred_msigmae, np.ndarray)
    assert pred_msigmae.shape == graphs.para.numpy().shape
    para_assoc = 10 ** (graphs.assoc.numpy() * np.array([-1.0, 1.0]))
    pred_params = np.hstack([pred_msigmae, para_assoc, graphs.munanb.numpy()])
    pred_params.clip(params_lower_bound, params_upper_bound, out=pred_params)
    assert pred_params.shape == (len(graphs.rho), 8)
    assert isinstance(graphs.rho[0], np.ndarray)
    assert isinstance(graphs.vp[0], np.ndarray)
    assert isinstance(graphs.rho, List)
    assert isinstance(graphs.vp, List)
    pred_rho = rho_batch(pred_params.tolist(), graphs.rho)
    pred_vp = vp_batch(pred_params.tolist(), graphs.vp)
    assert isinstance(pred_rho[0], np.ndarray)
    assert isinstance(pred_vp[0], np.ndarray)
    assert isinstance(pred_rho, List)
    assert isinstance(pred_vp, List)
    rho = [rho[:, -1] for rho in graphs.rho if rho.shape[0] > 0]
    vp = [vp[:, -1] for vp in graphs.vp if vp.shape[0] > 0]
    mape_den = []
    for pred, exp in zip(pred_rho, rho):
        assert pred.shape == exp.shape
        mape_den += [np.mean(np.abs(pred - exp) / exp).item()]
    mape_den = np.asarray(mape_den).mean().item()
    mape_vp = []
    for pred, exp in zip(pred_vp, vp):
        assert pred.shape == exp.shape
        mape_vp += [np.mean(np.abs(pred - exp) / exp).item()]
    mape_vp = np.asarray(mape_vp).mean().item()
    return mape_den, mape_vp
