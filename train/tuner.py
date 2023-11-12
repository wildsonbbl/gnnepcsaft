"""Module to be used for hyperparameter tuning"""
import os
import os.path as osp
import tempfile
from functools import partial

import ml_collections
import ray
import torch
from absl import app, flags, logging
from ml_collections import config_flags
from ray import train, tune
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.train import Checkpoint
from ray.tune import JupyterNotebookReporter
from ray.tune.experiment.trial import Trial
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.bohb import TuneBOHB
from torch.nn import HuberLoss
from torchmetrics import MeanAbsolutePercentageError

from epcsaft import epcsaft_cython

from .utils import (
    build_datasets_loaders,
    calc_deg,
    create_model,
    create_optimizer,
    create_schedulers,
    savemodel,
)

os.environ["WANDB_SILENT"] = "true"
os.environ["RAY_AIR_NEW_OUTPUT"] = "0"


class TrialTerminationReporter(JupyterNotebookReporter):
    """Reporter for ray to report only when trial is terminated"""

    def __init__(self):
        super().__init__()
        self.num_terminated = 0

    def should_report(self, trials, done=False):
        """Reports only on trial termination events."""
        old_num_terminated = self.num_terminated
        self.num_terminated = len([t for t in trials if t.status == Trial.TERMINATED])
        return self.num_terminated > old_num_terminated


class CustomStopper(tune.Stopper):
    """Custom experiment/trial stopper"""

    def __init__(self, max_iter: int):
        self.should_stop = False
        self.max_iter = max_iter

    def __call__(self, trial_id, result):
        if not self.should_stop and result["train_mape"] != result["train_mape"]:
            self.should_stop = True
        return self.should_stop or result["training_iteration"] >= self.max_iter

    def stop_all(self):
        return False


def train_and_evaluate(
    config_tuner: dict, config: ml_collections.ConfigDict, workdir: str, dataset: str
):
    """Execute model training and evaluation loop.

    Args:
      config: Hyperparameter configuration for training and evaluation.
      workdir: Working Directory.
      dataset: dataset name (ramirez or thermoml)
    """

    deg = calc_deg(dataset, workdir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = config.amp

    config.propagation_depth = config_tuner["propagation_depth"]
    config.hidden_dim = config_tuner["hidden_dim"]
    config.num_mlp_layers = config_tuner["num_mlp_layers"]
    config.pre_layers = config_tuner["pre_layers"]
    config.post_layers = config_tuner["post_layers"]
    config.skip_connections = config_tuner["skip_connections"]
    config.add_self_loops = config_tuner["add_self_loops"]

    # Get datasets, organized by split.
    logging.info("Obtaining datasets.")
    train_loader, test_loader, para_data = build_datasets_loaders(
        config, workdir, dataset
    )

    # Create and initialize the network.
    logging.info("Initializing network.")
    model = create_model(config, deg)
    model.to(device)
    # pylint: disable=no-member
    pcsaft_den = epcsaft_cython.DenFromTensor.apply
    hloss = HuberLoss("mean")
    hloss.to(device)
    mape = MeanAbsolutePercentageError()
    mape.to(device)

    # Create the optimizer.
    optimizer = create_optimizer(config, model.parameters())
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Set up checkpointing of the model.
    step = load_checkpoint(config, model, optimizer, scaler)

    # Scheduler
    scheduler, scheduler2 = create_schedulers(config, optimizer)

    @torch.no_grad()
    def test(test="test"):
        model.eval()
        total_mape_den = []
        total_huber_den = []
        for graphs in test_loader:
            if test == "test":
                if graphs.InChI in para_data:
                    continue
            if test == "val":
                if graphs.InChI not in para_data:
                    continue
            graphs = graphs.to(device)
            pred_para = model(graphs).squeeze().to("cpu", torch.float64)

            datapoints = graphs.rho.to("cpu", torch.float64).view(-1, 5)
            if ~torch.all(datapoints == torch.zeros_like(datapoints)):
                pred = pcsaft_den(pred_para, datapoints)
                target = datapoints[:, -1]
                # pylint: disable = not-callable
                loss_mape = mape(pred, target)
                loss_huber = hloss(pred, target)
                total_mape_den += [loss_mape.item()]
                total_huber_den += [loss_huber.item()]

        return (
            torch.tensor(total_mape_den).nanmean().item(),
            torch.tensor(total_huber_den).nanmean().item(),
        )

    # Begin training loop.
    logging.info("Starting training.")
    total_loss_mape = []
    total_loss_huber = []
    lr = []

    def train_step(graphs):
        target = graphs.para.to(device).view(-1, 3)
        graphs = graphs.to(device)
        optimizer.zero_grad()
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=use_amp
        ):
            pred = model(graphs)
            loss_mape = mape(pred, target)
            loss_huber = hloss(pred, target)
        scaler.scale(loss_mape).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss_mape.append(loss_mape.item())
        total_loss_huber.append(loss_huber.item())
        if config.change_sch:
            lr.append(optimizer.param_groups[0]["lr"])
        else:
            lr.append(scheduler.get_last_lr()[0])
        scheduler.step(step)
        return pred

    def train_logging():
        mape_den, huber_den = test(test="val")
        with tempfile.TemporaryDirectory() as tempdir:
            savemodel(
                model,
                optimizer,
                scaler,
                osp.join(tempdir, "checkpoint.pt"),
                step,
            )

            train.report(
                {
                    "train_mape": torch.tensor(total_loss_mape).mean().item(),
                    "train_huber": torch.tensor(total_loss_huber).mean().item(),
                    "train_lr": torch.tensor(lr).mean().item(),
                    "mape_den": mape_den,
                    "huber_den": huber_den,
                },
                checkpoint=Checkpoint.from_directory(tempdir),
            )
        scheduler2.step(torch.tensor(total_loss_huber).mean())

    model.train()
    while step < config.num_train_steps + 1:
        for graphs in train_loader:
            _ = train_step(graphs)

            # Quick indication that training is happening.
            logging.log_first_n(logging.INFO, "Finished training step %d.", 10, step)

            # Log, if required.
            if step % config.log_every_steps == 0:
                train_logging()
                total_loss_mape = []
                total_loss_huber = []
                lr = []
                model.train()

            step += 1
            if step > config.num_train_steps:
                break


def load_checkpoint(config, model, optimizer, scaler):
    "Loads saved model checkpoints."
    initial_step = 1
    checkpoint = train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            checkpoint = torch.load(osp.join(checkpoint_dir, "checkpoint.pt"))
            model.load_state_dict(checkpoint["model_state_dict"])
            if not config.change_opt:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
            step = checkpoint["step"]
            initial_step = int(step) + 1
            del checkpoint
    return initial_step


FLAGS = flags.FLAGS

flags.DEFINE_string("workdir", None, "Working Directory.")
flags.DEFINE_string("dataset", None, "Dataset to train model on")
flags.DEFINE_list("tags", [], "Dataset to train model on")
flags.DEFINE_string(
    "restoredir", None, "Directory path to restore previous tuning results"
)
flags.DEFINE_string("resumedir", None, "Directory path to resume unfinished tuning")
flags.DEFINE_integer("verbose", 0, "Ray tune verbose")
flags.DEFINE_integer("num_cpu", 1, "Number of CPU threads for trial")
flags.DEFINE_integer("num_gpus", 1, "Number of GPUs for trial")
flags.DEFINE_integer("num_init_gpus", 1, "Number of GPUs to be initialized")
flags.DEFINE_integer("num_samples", 100, "Number of trials")
flags.DEFINE_boolean("get_result", False, "Whether to show results or continue tuning")
flags.DEFINE_float(
    "time_budget_s",
    3600,
    "Global time budget in seconds after which all trials are stopped",
)
config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=True,
)


def main(argv):
    """Execution from command line"""
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    logging.info("Calling tuner")

    ptrain = partial(
        train_and_evaluate,
        config=FLAGS.config,
        workdir=FLAGS.workdir,
        dataset=FLAGS.dataset,
    )

    config = FLAGS.config

    search_space = {
        "propagation_depth": tune.choice([2, 3, 4, 5, 6, 7]),
        "hidden_dim": tune.choice([32, 64, 128, 256, 512]),
        "num_mlp_layers": tune.choice([1, 2, 3]),
        "pre_layers": tune.choice([1, 2, 3]),
        "post_layers": tune.choice([1, 2, 3]),
        "skip_connections": tune.choice([True, False]),
        "add_self_loops": tune.choice([True, False]),
    }
    max_t = config.num_train_steps // config.log_every_steps - 1

    search_alg = TuneBOHB(metric="mape_den", mode="min", seed=77)
    if FLAGS.restoredir:
        search_alg.restore_from_dir(FLAGS.restoredir)
        search_space = None
    search_alg = ConcurrencyLimiter(search_alg, 4)
    scheduler = HyperBandForBOHB(
        metric="mape_den",
        mode="min",
        max_t=max_t,
        stop_last_trials=True,
    )
    reporter = TrialTerminationReporter()
    stopper = CustomStopper(max_t)

    ray.init(num_gpus=FLAGS.num_init_gpus)
    resources = {"cpu": FLAGS.num_cpu, "gpu": FLAGS.num_gpus}
    trainable = tune.with_resources(tune.with_parameters(ptrain), resources=resources)

    if FLAGS.resumedir:
        tuner = tune.Tuner.restore(
            FLAGS.resumedir,
            trainable,
            resume_unfinished=True,
            resume_errored=False,
            restart_errored=False,
        )
    else:
        tuner = tune.Tuner(
            trainable,
            param_space=search_space,
            tune_config=tune.TuneConfig(
                search_alg=search_alg,
                scheduler=scheduler,
                num_samples=FLAGS.num_samples,
                time_budget_s=FLAGS.time_budget_s,
                reuse_actors=True,
            ),
            run_config=train.RunConfig(
                name="gnnpcsaft",
                storage_path=None,
                callbacks=[
                    WandbLoggerCallback(
                        "gnn-pc-saft",
                        FLAGS.dataset,
                        tags=["tuning", FLAGS.dataset] + FLAGS.tags,
                    )
                ],
                verbose=FLAGS.verbose,
                checkpoint_config=train.CheckpointConfig(
                    num_to_keep=1, checkpoint_at_end=False
                ),
                progress_reporter=reporter,
                log_to_file=True,
                stop=stopper,
            ),
        )

    if FLAGS.get_result:
        result = tuner.get_results()
    else:
        result = tuner.fit()

    best_trial = result.get_best_result(
        metric="mape_den",
        mode="min",
    )
    best_trials = result.get_dataframe("mape_den", "min").sort_values("mape_den")
    best_trials = best_trials[
        [
            "mape_den",
            "train_mape",
            "trial_id",
            "training_iteration",
        ]
    ]
    print(f"\nBest trial config:\n {best_trial.config}")
    print(f"\nBest trial final metrics:\n {best_trial.metrics}")
    print(f"\nBest trials final metrics:\n {best_trials.head(10)}")


if __name__ == "__main__":
    flags.mark_flags_as_required(["config", "workdir", "dataset"])
    app.run(main)
