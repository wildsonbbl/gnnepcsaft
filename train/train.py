import os.path as osp, os, time
from absl import app
from absl import flags
from ml_collections import config_flags

from absl import logging
import ml_collections

from train import models

import torch
from torch.nn import HuberLoss
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from torchmetrics import MeanAbsolutePercentageError

from epcsaft import epcsaft_cython

import wandb

from data.graphdataset import ThermoMLDataset, ramirez, ThermoMLpara

from train.model_deg import calc_deg
 

def create_model(
    config: ml_collections.ConfigDict, deg: torch.Tensor
) -> torch.nn.Module:
    """Creates a model, as specified by the config."""

    if config.model == "PNA":
        return models.PNAPCSAFT(
            hidden_dim=config.hidden_dim,
            propagation_depth=config.propagation_depth,
            pre_layers=config.pre_layers,
            post_layers=config.post_layers,
            num_mlp_layers=config.num_mlp_layers,
            num_para=config.num_para,
            deg=deg,
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


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str, dataset: str):
    """Execute model training and evaluation loop.

    Args:
      config: Hyperparameter configuration for training and evaluation.
      workdir: Working Directory.
    """
    class Noop(object):
        def step(*args, **kwargs): pass 
        def __getattr__(self, _): return self.step

    deg = calc_deg(dataset, workdir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = config.amp
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

    if dataset == "ramirez":
        path = osp.join(workdir, "data/ramirez2022")
        train_dataset = ramirez(path)
    elif dataset == "thermoml":
        path = osp.join(workdir, "data/thermoml")
        train_dataset = ThermoMLpara(path)
    else:
        ValueError(
            f"dataset is either ramirez or thermoml, got >>> {dataset} <<< instead"
        )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    test_loader = ThermoMLDataset(osp.join(workdir, "data/thermoml"))

    para_data = {}
    for graph in train_dataset:
        inchi, para = graph.InChI, graph.para.view(-1, 3)
        para_data[inchi] = para

    # Create and initialize the network.
    logging.info("Initializing network.")
    model = create_model(config, deg).to(device)
    pcsaft_den = epcsaft_cython.PCSAFT_den.apply
    pcsaft_vp = epcsaft_cython.PCSAFT_vp.apply
    HLoss = HuberLoss("mean").to(device)
    mape = MeanAbsolutePercentageError().to(device)

    # Create the optimizer.
    optimizer = create_optimizer(config, model.parameters())
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Set up checkpointing of the model.
    if dataset == "ramirez":
        ckp_path = osp.join(workdir, "train/checkpoints/ra_last_checkpoint.pth")
    else:
        ckp_path = osp.join(workdir, "train/checkpoints/tml_last_checkpoint.pth")
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

    # Scheduler
    if config.change_sch:
        scheduler = Noop()
        scheduler2 = ReduceLROnPlateau(optimizer, mode='min', patience = config.patience, verbose=True)
    else:
        scheduler = CosineAnnealingWarmRestarts(optimizer, config.warmup_steps)
        scheduler2 = Noop()

    @torch.no_grad()
    def test(test="test"):
        model.eval()
        total_mape_den = []
        total_huber_den = []
        total_mape_vp = []
        total_huber_vp = []
        for graphs in test_loader:
            if test == "test":
                if graphs.InChI in para_data:
                    continue
            if test == "val":
                if graphs.InChI not in para_data:
                    continue
            graphs.x = graphs.x.to(torch.float)
            graphs.edge_attr = graphs.edge_attr.to(torch.float)
            graphs.edge_index = graphs.edge_index.to(torch.int64)
            graphs = graphs.to(device)
            pred_para = model(graphs).squeeze().to("cpu", torch.float64)

            datapoints = graphs.rho.to("cpu", torch.float64).view(-1, 5)
            if ~torch.all(datapoints == torch.zeros_like(datapoints)):
                pred = pcsaft_den(pred_para, datapoints)
                target = datapoints[:, -1]
                loss_mape = mape(pred, target)
                loss_huber = HLoss(pred, target)
                total_mape_den += [loss_mape.item()]
                total_huber_den += [loss_huber.item()]

            datapoints = graphs.vp.to("cpu", torch.float64).view(-1, 5)
            if ~torch.all(datapoints == torch.zeros_like(datapoints)):
                pred = pcsaft_vp(pred_para, datapoints)
                target = datapoints[:, -1]
                result_filter = ~torch.isnan(pred)
                loss_mape = mape(pred[result_filter], target[result_filter])
                loss_huber = HLoss(pred[result_filter], target[result_filter])
                total_mape_vp += [loss_mape.item()]
                total_huber_vp += [loss_huber.item()]

        return (
            torch.tensor(total_mape_den).nanmean().item(),
            torch.tensor(total_huber_den).nanmean().item(),
            torch.tensor(total_mape_vp).nanmean().item(),
            torch.tensor(total_huber_vp).nanmean().item(),
        )

    # Begin training loop.
    logging.info("Starting training.")
    step = initial_step
    total_loss_mape = []
    total_loss_huber = []
    lr = []
    start_time = time.time()

    model.train()
    while step < config.num_train_steps + 1:
        for graphs in train_loader:
            target = graphs.para.to(device).view(-1, 3)
            graphs.x = graphs.x.to(torch.float)
            graphs.edge_attr = graphs.edge_attr.to(torch.float)
            graphs.edge_index = graphs.edge_index.to(torch.int64)
            graphs = graphs.to(device)
            optimizer.zero_grad()
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=use_amp
            ):
                pred = model(graphs)
                loss_mape = mape(pred, target)
                loss_huber = HLoss(pred, target)
            scaler.scale(loss_mape).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss_mape += [loss_mape.item()]
            total_loss_huber += [loss_huber.item()]
            if config.chang_sch:
                lr += [config.learning_rate]
            else:
                lr += scheduler.get_last_lr()
            scheduler.step(step)

            # Quick indication that training is happening.
            logging.log_first_n(logging.INFO, "Finished training step %d.", 10, step)

            # Log, if required.
            is_last_step = step == config.num_train_steps
            if step % config.log_every_steps == 0 or is_last_step:
                end_time = time.time()
                elapsed_time = end_time - start_time
                start_time = time.time()
                logging.log_first_n(
                    logging.INFO, "Elapsed time %.4f min.", 20, elapsed_time / 60
                )
                wandb.log(
                    {
                        "train_mape": torch.tensor(total_loss_mape).mean().item(),
                        "train_huber": torch.tensor(total_loss_huber).mean().item(),
                        "train_lr": torch.tensor(lr).mean().item(),
                    },
                    step=step,
                )
                total_loss_mape = []
                total_loss_huber = []
                lr = []

            # Checkpoint model, if required.
            if step % config.checkpoint_every_steps == 0 or is_last_step:
                savemodel(model, optimizer, scaler, ckp_path, step)

            # Evaluate on validation.
            if step % config.eval_every_steps == 0 or is_last_step:
                mape_den, huber_den, mape_vp, huber_vp = test(test="val")
                scheduler2.step(mape_den)
                wandb.log(
                    {
                        "mape_den": mape_den,
                        "huber_den": huber_den,
                        "mape_vp": mape_vp,
                        "huber_vp": huber_vp,
                    },
                    step=step,
                )
                model.train()

            step += 1
            if step >= config.num_train_steps + 1:
                wandb.finish()
                break


def savemodel(model, optimizer, scaler, path, step):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "step": step,
        },
        path,
    )


FLAGS = flags.FLAGS

flags.DEFINE_string("workdir", None, "Working Directory.")
flags.DEFINE_string("dataset", None, "Dataset to train model on")
config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=True,
)


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    logging.info("Calling train and evaluate")

    train_and_evaluate(FLAGS.config, FLAGS.workdir, FLAGS.dataset)


if __name__ == "__main__":
    flags.mark_flags_as_required(["config", "workdir", "dataset"])
    app.run(main)
