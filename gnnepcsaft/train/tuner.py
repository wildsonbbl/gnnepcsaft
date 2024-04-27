"""Module to be used for hyperparameter tuning"""

import os
import os.path as osp
from functools import partial
from tempfile import TemporaryDirectory
from typing import Any

import lightning as L
import ml_collections
import torch

# import ray
from absl import app, flags, logging
from lightning.pytorch.callbacks import Callback
from ml_collections import config_flags
from ray import train, tune
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.train import Checkpoint
from ray.train.lightning import RayDDPStrategy, RayLightningEnvironment, prepare_trainer
from ray.train.torch import TorchTrainer
from ray.tune import JupyterNotebookReporter
from ray.tune.experiment.trial import Trial
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.bohb import TuneBOHB
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import BaseTransform

from .utils import build_test_dataset, build_train_dataset, calc_deg, create_model

os.environ["WANDB_SILENT"] = "true"
# os.environ["WANDB_MODE"] = "offline"
# os.environ["RAY_AIR_NEW_OUTPUT"] = "0"


# taking vp data off for performance boost
# pylint: disable=R0903
class VpOff(BaseTransform):
    "take vp data off thermoml dataset"

    def forward(self, data: Any) -> Any:

        data.vp = torch.zeros(1, 5)
        return data


class CustomRayTrainReportCallback(Callback):
    "Custom ray tuner checkpoint."

    def on_validation_end(self, trainer, pl_module):

        with TemporaryDirectory() as tmpdir:
            # Fetch metrics
            metrics = trainer.callback_metrics
            metrics = {k: v.item() for k, v in metrics.items()}

            # Add customized metrics
            metrics["epoch"] = trainer.current_epoch
            metrics["step"] = trainer.global_step

            checkpoint = None
            global_rank = train.get_context().get_world_rank()
            if global_rank == 0:
                # Save model checkpoint file to tmpdir
                ckpt_path = os.path.join(tmpdir, "ckpt.pt")
                trainer.save_checkpoint(ckpt_path, weights_only=False)

                checkpoint = Checkpoint.from_directory(tmpdir)

            # Report to train session
            train.report(metrics=metrics, checkpoint=checkpoint)


class TrialTerminationReporter(JupyterNotebookReporter):
    """Reporter for ray to report only when trial is terminated"""

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
    """Custom experiment/trial stopper"""

    def __init__(self, max_iter: int):
        self.should_stop = False
        self.max_iter = max_iter

    def __call__(self, trial_id, result):
        return result["training_iteration"] >= self.max_iter

    def stop_all(self):
        return False


# pylint: disable=R0914
# pylint: disable=R0915
def tune_training(
    config_tuner: dict, config: ml_collections.ConfigDict, workdir: str, dataset: str
):
    """Execute model training and evaluation loop.

    Args:
      config: Hyperparameter configuration for training and evaluation.
      workdir: Working Directory.
      dataset: dataset name (ramirez or thermoml)
    """
    # selected hyperparameters to test
    config.propagation_depth = config_tuner["propagation_depth"]
    config.hidden_dim = config_tuner["hidden_dim"]
    config.num_mlp_layers = config_tuner["num_mlp_layers"]
    config.pre_layers = config_tuner["pre_layers"]
    config.post_layers = config_tuner["post_layers"]
    config.skip_connections = config_tuner["skip_connections"]
    config.add_self_loops = config_tuner["add_self_loops"]

    # Dataset building
    train_dataset = build_train_dataset(workdir, dataset)
    tml_dataset, para_data = build_test_dataset(
        workdir, train_dataset, transform=VpOff()
    )
    test_idx = []
    val_idx = []
    # separate test and val dataset
    for idx, graph in enumerate(tml_dataset):
        if graph.InChI in para_data:
            val_idx.append(idx)
        else:
            test_idx.append(idx)
    # test_dataset = tml_dataset[test_idx]
    val_dataset = tml_dataset[val_idx]

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
    )

    # creating model from config
    deg = calc_deg(dataset, workdir)
    model = create_model(config, deg)

    # creating Lighting trainer function
    trainer = L.Trainer(
        devices="auto",
        accelerator="auto",
        strategy=RayDDPStrategy(),
        max_steps=config.num_train_steps,
        val_check_interval=config.log_every_steps,
        check_val_every_n_epoch=None,
        num_sanity_val_steps=0,
        callbacks=[CustomRayTrainReportCallback()],
        plugins=[RayLightningEnvironment()],
        enable_progress_bar=False,
    )
    trainer: L.Trainer = prepare_trainer(trainer)

    checkpoint: Checkpoint = train.get_checkpoint()
    ckpt_path = None
    if checkpoint:
        with checkpoint.as_directory() as ckpt_dir:
            ckpt_path = osp.join(ckpt_dir, "ckpt.pt")

    # training run
    trainer.fit(
        model,
        train_loader,
        val_dataset,
        ckpt_path=ckpt_path,
    )


FLAGS = flags.FLAGS

flags.DEFINE_string("workdir", None, "Working Directory.")
flags.DEFINE_string("dataset", None, "Dataset to train model on")
flags.DEFINE_list("tags", [], "wandb tags")
flags.DEFINE_string(
    "restoredir",
    None,
    "Directory path to restore the state of a searcher from previous tuning results",
)
flags.DEFINE_string("resumedir", None, "Directory path to resume unfinished tuning")
flags.DEFINE_integer("verbose", 0, "Ray tune verbose")
flags.DEFINE_float("num_cpu", 1.0, "Fraction of CPU threads per trial")
flags.DEFINE_float("num_gpus", 1.0, "Fraction of GPUs per trial")
flags.DEFINE_float("num_cpu_trainer", 1.0, "Fraction of CPUs for trainer resources")
flags.DEFINE_integer(
    "max_concurrent", 4, "Maximum concurrent samples from the underlying searcher."
)
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

    logging.info("Calling tuner!")

    ptrain = partial(
        tune_training,
        config=FLAGS.config,
        workdir=FLAGS.workdir,
        dataset=FLAGS.dataset,
    )

    config = FLAGS.config
    # Hyperparameter search space
    search_space = {
        "propagation_depth": tune.choice([3, 4, 5, 6]),
        "hidden_dim": tune.choice([64, 128, 256]),
        "num_mlp_layers": tune.choice([0, 1, 2]),
        "pre_layers": tune.choice([1, 2]),
        "post_layers": tune.choice([1, 2]),
        "skip_connections": tune.choice([True, False]),
        "add_self_loops": tune.choice([True, False]),
    }
    max_t = config.num_train_steps // config.log_every_steps - 1
    # BOHB search algorithm
    search_alg = TuneBOHB(metric="mape_den", mode="min", seed=77)
    if FLAGS.restoredir:
        search_alg.restore_from_dir(FLAGS.restoredir)
        search_space = None
    search_alg = ConcurrencyLimiter(search_alg, max_concurrent=FLAGS.max_concurrent)
    # Early stopping scheduler for BOHB
    scheduler = HyperBandForBOHB(
        metric="mape_den",
        mode="min",
        max_t=max_t,
        stop_last_trials=True,
    )
    # reporter = TrialTerminationReporter()
    # stopper = CustomStopper(max_t)

    # ray.init(num_gpus=FLAGS.num_init_gpus)
    resources_per_worker = {"CPU": FLAGS.num_cpu, "GPU": FLAGS.num_gpus}
    trainer_resources = {"CPU": FLAGS.num_cpu_trainer}
    scaling_config = train.ScalingConfig(
        num_workers=1,
        use_gpu=True,
        resources_per_worker=resources_per_worker,
        trainer_resources=trainer_resources,
    )
    run_config = train.RunConfig(
        name="gnnpcsaft",
        storage_path=None,
        verbose=FLAGS.verbose,
        checkpoint_config=train.CheckpointConfig(
            num_to_keep=1,
        ),
        progress_reporter=None,
        log_to_file=True,
        stop=None,
        callbacks=[
            WandbLoggerCallback(
                "gnn-pc-saft",
                FLAGS.dataset,
                tags=["tuning", FLAGS.dataset] + FLAGS.tags,
            )
        ],
    )

    trainable = TorchTrainer(
        ptrain, scaling_config=scaling_config, run_config=run_config
    )

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
            param_space={"train_loop_config": search_space},
            tune_config=tune.TuneConfig(
                search_alg=search_alg,
                scheduler=scheduler,
                num_samples=FLAGS.num_samples,
                time_budget_s=FLAGS.time_budget_s,
                reuse_actors=True,
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
