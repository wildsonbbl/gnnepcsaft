import os.path as osp, tempfile, os

os.environ["WANDB_SILENT"] = "true"
os.environ["RAY_AIR_NEW_OUTPUT"]="0"
from absl import app
from absl import flags
from ml_collections import config_flags

from absl import logging
import ml_collections

import torch
from torch.nn import HuberLoss
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from torchmetrics import MeanAbsolutePercentageError

from epcsaft import epcsaft_cython

from data.graphdataset import ThermoMLDataset, ramirez, ThermoMLpara

from train.utils import calc_deg, create_optimizer, create_model, savemodel

from ray import tune, train
import ray
from ray.train import Checkpoint
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
from ray.tune.search import ConcurrencyLimiter
from ray.tune import JupyterNotebookReporter
from ray.tune.experiment.trial import Trial
from functools import partial


class TrialTerminationReporter(JupyterNotebookReporter):
    def __init__(self):
        super(TrialTerminationReporter, self).__init__()
        self.num_terminated = 0

    def should_report(self, trials, done=False):
        """Reports only on trial termination events."""
        old_num_terminated = self.num_terminated
        self.num_terminated = len([t for t in trials if t.status == Trial.TERMINATED])
        return self.num_terminated > old_num_terminated


class CustomStopper(tune.Stopper):
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
    """

    class Noop(object):
        def step(*args, **kwargs):
            pass

        def __getattr__(self, _):
            return self.step

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

    # Scheduler
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

        return (
            torch.tensor(total_mape_den).nanmean().item(),
            torch.tensor(total_huber_den).nanmean().item(),
        )

    # Begin training loop.
    logging.info("Starting training.")
    step = initial_step
    total_loss_mape = []
    total_loss_huber = []
    lr = []

    model.train()
    while step < config.num_train_steps + 1:
        for graphs in train_loader:
            target = graphs.para.to(device).view(-1, 3)
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
            if config.change_sch:
                lr += [optimizer.param_groups[0]["lr"]]
            else:
                lr += scheduler.get_last_lr()
            scheduler.step(step)

            # Quick indication that training is happening.
            logging.log_first_n(logging.INFO, "Finished training step %d.", 10, step)

            # Log, if required.
            is_last_step = step == config.num_train_steps
            if step % config.log_every_steps == 0:
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
                total_loss_mape = []
                total_loss_huber = []
                lr = []
                model.train()

            step += 1
            if step >= config.num_train_steps + 1:
                break


FLAGS = flags.FLAGS

flags.DEFINE_string("workdir", None, "Working Directory.")
flags.DEFINE_string("dataset", None, "Dataset to train model on")
flags.DEFINE_string("restoredir", None, "Directory path to restore previous tuning results")
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
                        "gnn-pc-saft", FLAGS.dataset, tags=["tuning", FLAGS.dataset]
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
