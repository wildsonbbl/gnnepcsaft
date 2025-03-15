"""Model with important functions to help model training"""

import multiprocessing as mp
import os.path as osp
import time
from tempfile import TemporaryDirectory
from typing import Any

import ml_collections
import numpy as np
import torch
import torch_geometric.transforms as T
from absl import logging
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from ray import tune
from ray.tune.experiment.trial import Trial
from torch.utils.data import ConcatDataset
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import degree

from ..data.graph import assoc_number
from ..data.graphdataset import Esper, Ramirez, ThermoMLDataset
from ..epcsaft.utils import pure_den_feos, pure_vp_feos


def calc_deg(dataset: str, workdir: str) -> list:
    """Calculates deg for `PNAPCSAFT` model."""
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


def create_optimizer(config: ml_collections.ConfigDict, params):
    """Creates an optimizer, as specified by the config."""
    if config.optimizer == "adam":
        return torch.optim.AdamW(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            amsgrad=True,
            eps=1e-5,
        )
    if config.optimizer == "sgd":
        return torch.optim.SGD(
            params,
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            nesterov=True,
        )
    raise ValueError(f"Unsupported optimizer: {config.optimizer}.")


def savemodel(model, optimizer, scaler, path, step):
    """To checkpoint model during training."""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "step": step,
        },
        path,
    )


def rhovp_data(parameters: np.ndarray, rho: np.ndarray, vp: np.ndarray):
    """Calculates density and vapor pressure with ePC-SAFT"""

    den_array = rho_single((parameters, rho))
    vp_array = vp_sigle((parameters, vp))

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


class LogAssoc(BaseTransform):
    "Log10 assoc for training."

    def __init__(self, workdir: str) -> None:
        super().__init__()
        path = osp.join(workdir, "data/esper2023")
        train_dataset = Esper(path)
        assoc = {}
        msigmae = {}
        for graph in train_dataset:
            assoc[graph.InChI] = torch.abs(torch.log10(graph.assoc))
            msigmae[graph.InChI] = torch.abs(torch.log10(graph.para))
        self.assoc = assoc
        self.msigmae = msigmae

    def forward(self, data: Any) -> Any:
        data.assoc = self.assoc[data.InChI]
        data.para = self.msigmae[data.InChI]
        return data


def build_test_dataset(workdir, train_dataset, transform=None):
    "Builds test dataset."

    para_data = {}
    if isinstance(train_dataset, (Esper, ConcatDataset)):
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
    val_assoc_idx = []
    val_idx = []
    train_idx = []
    # separate test and val dataset
    for idx, graph in enumerate(tml_dataset):
        if graph.InChI not in para_data and graph.munanb[0, -1] == 0:
            val_idx.append(idx)
        if graph.InChI in para_data and graph.munanb[0, -1] > 0:
            val_assoc_idx.append(idx)
        if graph.InChI in para_data and graph.munanb[0, -1] == 0:
            train_idx.append(idx)
    val_assoc_dataset = tml_dataset[val_assoc_idx]
    val_dataset = tml_dataset[val_idx]
    train_val_dataset = tml_dataset[train_idx]
    return val_dataset, train_val_dataset, val_assoc_dataset


def build_train_dataset(workdir, dataset, transform=None):
    "Builds train dataset."
    if dataset == "ramirez":
        path = osp.join(workdir, "data/ramirez2022")
        train_dataset = Ramirez(path, transform=transform)
    elif dataset == "esper":
        path = osp.join(workdir, "data/esper2023")
        train_dataset = Esper(path, transform=transform)
    elif dataset == "esper_assoc":
        path = osp.join(workdir, "data/esper2023")
        train_dataset = Esper(path, transform=transform)
        as_idx = []
        non_as_idx = []
        for i, graph in enumerate(train_dataset):
            if all(graph.munanb[1:] > 0):
                as_idx.append(i)
            if all(graph.munanb[1:] == 0):
                non_as_idx.append(i)
        train_dataset = ConcatDataset(
            [train_dataset[as_idx]] * 4 + [train_dataset[non_as_idx]]
        )
    elif dataset == "esper_assoc_only":
        path = osp.join(workdir, "data/esper2023")
        train_dataset = Esper(path, transform=transform)
        as_idx = []
        for i, graph in enumerate(train_dataset):
            if all(graph.munanb[1:] > 0):
                as_idx.append(i)
        train_dataset = train_dataset[as_idx]
    else:
        raise ValueError(
            f"dataset is either ramirez, esper, esper_assoc \
              or esper_assoc_only, got >>> {dataset} <<< instead"
        )

    return train_dataset


def input_artifacts(workdir: str, dataset: str, model="last_checkpoint"):
    "Creates input wandb artifacts"
    # pylint: disable=C0415
    import wandb

    if dataset == "ramirez":
        ramirez_path = workdir + "/data/ramirez2022"
        ramirez_art = wandb.Artifact(name="ramirez", type="dataset")
        ramirez_art.add_dir(local_path=ramirez_path, name="ramirez2022")
        wandb.use_artifact(ramirez_art)
    if dataset == "thermoml":
        thermoml_path = workdir + "/data/thermoml"
        thermoml_art = wandb.Artifact(name="thermoml", type="dataset")
        thermoml_art.add_dir(local_path=thermoml_path, name="thermoml")
        wandb.use_artifact(thermoml_art)
    model_path = workdir + f"/train/checkpoints/{model}.pth"
    model_art = wandb.Artifact(name="model", type="model")
    if osp.exists(model_path):
        model_art.add_file(local_path=model_path, name="last_checkpoint.pth")
        wandb.use_artifact(model_art)


def output_artifacts(workdir: str):
    "Creates output wandb artifacts"
    # pylint: disable=C0415
    import wandb

    model_path = workdir + "/train/checkpoints/last_checkpoint.pth"
    model_art = wandb.Artifact(name="model", type="model")
    if osp.exists(model_path):
        model_art.add_file(local_path=model_path, name="last_checkpoint.pth")
        wandb.log_artifact(model_art)


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


# taking vp data off for performance boost
# pylint: disable=R0903
class VpOff(BaseTransform):
    "take vp data off thermoml dataset"

    def forward(self, data: Any) -> Any:

        data.vp = torch.tensor([])
        return data


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

            checkpoint = None
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


def rho_single(args):
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


def rho_batch(parameters_batch: list, states_batch: list):
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


def vp_sigle(args):
    """Calculates vapor pressure with ePC-SAFT for a single pararameter"""
    parameters, states = args
    vp_for_all_states = []
    for state in states:
        try:
            vp_for_state = pure_vp_feos(parameters, state)
        except (AssertionError, RuntimeError):
            vp_for_state = 0.0
        vp_for_all_states += [vp_for_state]
    return np.asarray(vp_for_all_states)


def vp_batch(parameters_batch: list, states_batch: list):
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
        vp = pool.map(vp_sigle, args_list)
    return vp
