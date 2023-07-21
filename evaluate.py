import os.path as osp
from absl import app
from absl import flags
from ml_collections import config_flags

from absl import logging
import ml_collections

import models

import torch
from torch.nn import HuberLoss
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch_geometric.loader import DataLoader
from torchmetrics import MeanAbsolutePercentageError

import ml_pc_saft
import jax

import wandb

from graphdataset import ThermoMLDataset, ThermoML_padded, ramirez
import pickle

deg = torch.tensor([78, 5572, 8525, 2569, 602, 1, 2])


def create_model(config: ml_collections.ConfigDict) -> torch.nn.Module:
    """Creates a Flax model, as specified by the config."""
    platform = jax.local_devices()[0].platform
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = torch.float64
    if config.model == "PNA":
        return models.PNA(
            hidden_dim=config.hidden_dim,
            propagation_depth=config.propagation_depth,
            pre_layers=config.pre_layers,
            post_layers=config.post_layers,
            num_mlp_layers=config.num_mlp_layers,
            num_para=config.num_para,
            deg=deg,
            layer_norm=config.layer_norm,
            dtype=model_dtype,
            device=device,
        )
    raise ValueError(f"Unsupported model: {config.model}.")


def evaluate(config: ml_collections.ConfigDict, workdir: str):
    """Execute model training and evaluation loop.

    Args:
      config: Hyperparameter configuration for training and evaluation.
      workdir: Directory where the TensorBoard summaries are written to.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    platform = jax.local_devices()[0].platform
    # Create writer for logs.
    wandb.login()
    run = wandb.init(
        # Set the project where this run will be logged
        project="gnn-pc-saft",
        # Track hyperparameters and run metadata
        config=config.to_dict(),
    )

    # Get datasets, organized by split.
    logging.info("Obtaining datasets.")
    model_dtype = torch.float64

    path = osp.join(workdir, "data/ramirez2022")
    dataset = ramirez(path)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    if osp.exists("./data/thermoml/processed/parameters.pkl"):
        parameters = pickle.load(open("./data/thermoml/processed/parameters.pkl", "rb"))
        print(f"inchis saved: {len(parameters.keys())}")
    else:
        print("missing parameters.pkl")
    
    randn = torch.rand(128)
    temperature = 283.15 * (1.0 - randn) + 423.15 * randn
    pressure = 50662.5 * (1.0 - randn) + 506625 * randn
    phase = fntype = torch.ones(128)
    datapoints = torch.stack([temperature, pressure, phase, fntype], dim=1)
    datapoints = datapoints.to(device, torch.float64)

    # Create and initialize the network.
    logging.info("Initializing network.")
    model = create_model(config).to(device, model_dtype)
    pcsaft_den = ml_pc_saft.PCSAFT_den.apply
    pcsaft_vp = ml_pc_saft.PCSAFT_vp.apply
    mape = MeanAbsolutePercentageError().to(device)

    # Set up checkpointing of the model.
    ckp_path = osp.join(workdir, "training/last_checkpoint.pth")
    checkpoint = torch.load(ckp_path)
    model.load_state_dict(checkpoint["model_state_dict"])

    # test fn
    @torch.no_grad()
    def test_den(test='test'):
        model.eval()
        total_loss = []
        for graphs in loader:
            if test == "test":
                if (graphs.InChI[0] in parameters):
                    continue
            else:
                if (graphs.InChI[0] not in parameters):
                    continue
            ref_para = graphs.para
            ref_zeros = torch.zeros(4)
            ref_para = torch.cat([ref_para, ref_zeros], 0)
            ref_para = ref_para.to(device)
            graphs = graphs.to(device)
            target_para = model(graphs).squeeze()
            pred_y = pcsaft_den(target_para, datapoints)
            y = pcsaft_den(ref_para, datapoints)
            loss = mape(pred_y, y)
            wandb.log(
                {
                    "mape_den": loss.item(),
                },
            )
            total_loss += [loss.item()]

        return torch.tensor(total_loss).nanmean().item()

    # Evaluate on validation or test, if required.
    val_mape_den = test_den("val")
    test_mape_den = test_den("test")

    wandb.log(
        {
            "val_mape_den": val_mape_den,
            "test_mape_den": test_mape_den
        }
    )

    wandb.finish()


FLAGS = flags.FLAGS

flags.DEFINE_string("workdir", None, "Directory to store model data.")
config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=True,
)


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    logging.info("JAX host: %d / %d", jax.process_index(), jax.process_count())
    logging.info("JAX local devices: %r", jax.local_devices())

    logging.info("Calling evaluate")

    evaluate(FLAGS.config, FLAGS.workdir)


if __name__ == "__main__":
    flags.mark_flags_as_required(["config", "workdir"])
    app.run(main)
