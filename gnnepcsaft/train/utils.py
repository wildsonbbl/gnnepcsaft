"""Model with important functions to help model training"""
import os.path as osp

import ml_collections
import numpy as np
import torch
from pcsaft import (  # pylint: disable = no-name-in-module
    SolutionError,
    flashTQ,
    pcsaft_den,
)
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree

from ..data.graphdataset import Ramirez, ThermoMLDataset, ThermoMLpara
from . import models


def calc_deg(dataset: str, workdir: str) -> torch.Tensor:
    """Calculates deg for `PNAPCSAFT` model."""
    if dataset == "ramirez":
        path = osp.join(workdir, "data/ramirez2022")
        train_dataset = Ramirez(path)
    elif dataset == "thermoml":
        path = osp.join(workdir, "data/thermoml")
        train_dataset = ThermoMLpara(path)
    else:
        raise ValueError(
            f"dataset is either ramirez or thermoml, got >>> {dataset} <<< instead"
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
    return deg


def create_model(
    config: ml_collections.ConfigDict, deg: torch.Tensor
) -> torch.nn.Module:
    """Creates a model, as specified by the config."""

    if config.model == "PNA":
        pna_params = models.PnaconvsParams(
            propagation_depth=config.propagation_depth,
            pre_layers=config.pre_layers,
            post_layers=config.post_layers,
            deg=deg,
            skip_connections=config.skip_connections,
            self_loops=config.add_self_loops,
        )
        mlp_params = models.ReadoutMLPParams(
            num_mlp_layers=config.num_mlp_layers,
            num_para=config.num_para,
            dropout=config.dropout_rate,
        )
        return models.PNAPCSAFT(
            hidden_dim=config.hidden_dim,
            pna_params=pna_params,
            mlp_params=mlp_params,
        )
    raise ValueError(f"Unsupported model: {config.model}.")


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


def mape(parameters: np.ndarray, rho: np.ndarray, vp: np.ndarray, mean: bool = True):
    """
    Calculates mean absolute percentage error
    of ePC-SAFT predicted density and vapor pressurre
    relative to experimental data.

    """
    parameters = np.abs(parameters)
    m = parameters[0]
    s = parameters[1]
    e = parameters[2]
    pred_mape = 0.0

    if ~np.all(rho == np.zeros_like(rho)):
        pred_mape = []
        for state in rho:
            x = np.asarray([1.0])
            t = state[0]
            p = state[1]
            phase = "liq" if state[2] == 1 else "vap"
            params = {"m": m, "s": s, "e": e}
            den = pcsaft_den(t, p, x, params, phase=phase)
            pred_mape += [np.abs((state[-1] - den) / state[-1])]

    den = np.asarray(pred_mape)
    if mean:
        den = den.mean()

    pred_mape = 0.0
    if ~np.all(vp == np.zeros_like(vp)):
        pred_mape = []
        for state in vp:
            x = np.asarray([1.0])
            t = state[0]
            p = state[1]
            phase = "liq" if state[2] == 1 else "vap"
            params = {"m": m, "s": s, "e": e}
            try:
                vp, _, _ = flashTQ(t, 0, x, params, p)
                pred_mape += [np.abs((state[-1] - vp) / state[-1])]
            except SolutionError:
                pass

    vp = np.asarray(pred_mape)
    if mean:
        vp = vp.mean()

    return den, vp


def rhovp_data(parameters: np.ndarray, rho: np.ndarray, vp: np.ndarray):
    """Calculates density and vapor pressure with ePC-SAFT"""
    parameters = np.abs(parameters)
    m = parameters[0]
    s = parameters[1]
    e = parameters[2]
    den = []

    if ~np.all(rho == np.zeros_like(rho)):
        for state in rho:
            x = np.asarray([1.0])
            t = state[0]
            p = state[1]
            phase = ["liq" if state[2] == 1 else "vap"][0]
            params = {"m": m, "s": s, "e": e}
            den += [pcsaft_den(t, p, x, params, phase=phase)]

    den = np.asarray(den)

    vpl = []
    if ~np.all(vp == np.zeros_like(vp)):
        for state in vp:
            x = np.asarray([1.0])
            t = state[0]
            p = state[1]
            phase = ["liq" if state[2] == 1 else "vap"][0]
            params = {"m": m, "s": s, "e": e}
            try:
                vp, _, _ = flashTQ(t, 0, x, params, p)
                vpl += [vp]
            except SolutionError:
                vpl += [np.nan]

    vp = np.asarray(vpl)

    return den, vp


def create_schedulers(config, optimizer):
    "Creates lr schedulers."

    class Noop:
        """Dummy noop scheduler"""

        def step(self, *args, **kwargs):
            """Scheduler step"""

        def __getattr__(self, _):
            return self.step

    if config.change_sch:
        scheduler = Noop()
        scheduler2 = ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=config.patience,
            verbose=True,
            cooldown=config.patience,
            min_lr=1e-15,
            eps=1e-15,
        )
    else:
        scheduler = CosineAnnealingWarmRestarts(optimizer, config.warmup_steps)
        scheduler2 = Noop()
    return scheduler, scheduler2


def load_checkpoint(config, workdir, model, optimizer, scaler):
    "Loads saved model checkpoints."
    ckp_path = osp.join(workdir, "train/checkpoints/last_checkpoint.pth")
    initial_step = 1
    if osp.exists(ckp_path):
        checkpoint = torch.load(ckp_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        if not config.change_opt:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        step = checkpoint["step"]
        initial_step = int(step) + 1
        del checkpoint
    return ckp_path, initial_step


def build_datasets_loaders(config, workdir, dataset):
    "Builds train and test dataset loader."
    train_dataset = build_train_dataset(workdir, dataset)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    test_loader, para_data = build_test_dataset(workdir, train_dataset)
    return train_loader, test_loader, para_data


def build_test_dataset(workdir, train_dataset):
    "Builds test dataset."
    test_loader = ThermoMLDataset(osp.join(workdir, "data/thermoml"))

    para_data = {}
    for graph in train_dataset:
        inchi, para = graph.InChI, graph.para.view(-1, 3)
        para_data[inchi] = para
    return test_loader, para_data


def build_train_dataset(workdir, dataset):
    "Builds train dataset."
    if dataset == "ramirez":
        path = osp.join(workdir, "data/ramirez2022")
        train_dataset = Ramirez(path)
    elif dataset == "thermoml":
        path = osp.join(workdir, "data/thermoml")
        train_dataset = ThermoMLpara(path)
    else:
        raise ValueError(
            f"dataset is either ramirez or thermoml, got >>> {dataset} <<< instead"
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
        model_art.add_file(local_path=model_path)
        wandb.use_artifact(model_art)


def output_artifacts(workdir: str):
    "Creates output wandb artifacts"
    # pylint: disable=C0415
    import wandb

    model_path = workdir + "/train/checkpoints/last_checkpoint.pth"
    model_art = wandb.Artifact(name="model", type="model")
    if osp.exists(model_path):
        model_art.add_file(local_path=model_path)
        wandb.log_artifact(model_art)
