"""Module to be used for hyperparameter tuning"""

import os
from functools import partial

import ml_collections
import torch
from absl import app, flags, logging
from ray import tune
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune.experiment.trial import Trial
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.bohb import TuneBOHB

from .search_space import get_search_space
from .train import training_updated

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


FLAGS = flags.FLAGS

flags.DEFINE_string(
    "restoredir",
    None,
    "Directory path to restore the state of a searcher from previous tuning results",
)
flags.DEFINE_string("resumedir", None, "Directory path to resume unfinished tuning")
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

    config: ml_collections.ConfigDict = FLAGS.config
    # Hyperparameter search space
    search_space = get_search_space()
    max_t = config.num_train_steps // config.eval_every_steps
    # BOHB search algorithm
    search_alg = TuneBOHB(
        space=search_space,
        metric="mape_den/dataloader_idx_0",
        mode="min",
        seed=77,
        max_concurrent=FLAGS.max_concurrent,
    )
    search_alg = ConcurrencyLimiter(search_alg, max_concurrent=FLAGS.max_concurrent)
    if FLAGS.restoredir:
        search_alg.restore_from_dir(FLAGS.restoredir)
        search_space = None
    # Early stopping scheduler for BOHB
    scheduler = HyperBandForBOHB(
        time_attr="training_iteration",
        metric="mape_den/dataloader_idx_0",
        mode="min",
        max_t=max_t,
        stop_last_trials=True,
    )
    # reporter = TrialTerminationReporter()
    # stopper = CustomStopper(max_t)

    trainable = tune.with_resources(
        partial(
            training_updated,
            config=FLAGS.config,
            workdir=FLAGS.workdir,
        ),
        resources={
            "CPU": config.num_cpu,
            "GPU": config.num_gpus,
        },
    )

    if FLAGS.resumedir:
        tuner = tune.Tuner.restore(
            FLAGS.resumedir,
            trainable,
            resume_unfinished=True,
            resume_errored=False,
            restart_errored=True,
        )
    else:
        tuner = tune.Tuner(
            trainable,
            tune_config=tune.TuneConfig(
                search_alg=search_alg,
                scheduler=scheduler,
                num_samples=FLAGS.num_samples,
                time_budget_s=FLAGS.time_budget_s,
                reuse_actors=False,
            ),
            run_config=tune.RunConfig(
                name="gnnpcsaft",
                storage_path=None,
                checkpoint_config=tune.CheckpointConfig(
                    num_to_keep=1,
                ),
                progress_reporter=None,
                log_to_file=False,
                stop=None,
                callbacks=(
                    [
                        WandbLoggerCallback(
                            "gnn-pc-saft",
                            config.dataset,
                            tags=["tuning", config.dataset] + FLAGS.tags,
                        )
                    ]
                ),
            ),
        )

    if FLAGS.get_result:
        tuner.get_results()
    else:
        tuner.fit()


if __name__ == "__main__":
    flags.mark_flags_as_required(["config", "workdir"])
    app.run(main)
