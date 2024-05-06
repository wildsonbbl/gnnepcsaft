"""Module to be used for hyperparameter tuning"""

import os
from functools import partial

import ml_collections
import torch

# import ray
from absl import app, flags, logging
from ray import train, tune
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.train.torch import TorchTrainer
from ray.tune.experiment.trial import Trial
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.bohb import TuneBOHB

from .train import ltrain_and_evaluate

os.environ["WANDB_SILENT"] = "true"
# os.environ["WANDB_MODE"] = "offline"
# os.environ["RAY_AIR_NEW_OUTPUT"] = "0"


class TrialTerminationReporter(tune.JupyterNotebookReporter):
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

    ltrain_and_evaluate(config, workdir, dataset)


FLAGS = flags.FLAGS

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


def main(argv):
    """Execution from command line"""
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    logging.info("Calling tuner!")
    torch.set_float32_matmul_precision("medium")

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
    max_t = config.num_train_steps // config.eval_every_steps
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
    scaling_config = train.ScalingConfig(
        num_workers=1,
        use_gpu=True,
        resources_per_worker={"CPU": FLAGS.num_cpu, "GPU": FLAGS.num_gpus},
        trainer_resources={"CPU": FLAGS.num_cpu_trainer},
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
