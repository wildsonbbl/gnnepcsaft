"""Module to be used for hyperparameter tuning"""

import os
from functools import partial

import torch
from absl import app, flags, logging
from ray import tune
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB

from ..configs.search_space import get_search_space
from .train import training_updated

os.environ["WANDB_SILENT"] = "true"
# os.environ["WANDB_MODE"] = "offline"
# os.environ["RAY_AIR_NEW_OUTPUT"] = "0"

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

    config: dict = FLAGS.config
    # Hyperparameter search space
    search_space = get_search_space()
    max_t = config["num_train_steps"] // config["eval_every_steps"]
    # BOHB search algorithm
    search_alg = TuneBOHB(
        space=search_space,
        metric="mape_den/dataloader_idx_1",
        mode="min",
        points_to_evaluate=[
            {
                "conv": "PNA",
                "global_pool": "add",
                "propagation_depth": 6,
                "hidden_dim": 256,
                "post_layers": 4,
                "pre_layers": 2,
                "towers": 1,
                "dropout": 0.0,
            }
        ],
        seed=77,
    )
    if FLAGS.restoredir:
        search_alg.restore_from_dir(FLAGS.restoredir)
    # Early stopping scheduler for BOHB
    scheduler = HyperBandForBOHB(
        time_attr="training_iteration",
        metric="mape_den/dataloader_idx_1",
        mode="min",
        max_t=max_t,
        stop_last_trials=True,
    )

    trainable = tune.with_resources(
        partial(
            training_updated,
            config=FLAGS.config,
            workdir=FLAGS.workdir,
        ),
        resources={
            "CPU": FLAGS.num_cpu,
            "GPU": FLAGS.num_gpus,
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
                trial_name_creator=lambda trial: trial.trial_id,
                trial_dirname_creator=lambda trial: trial.trial_id,
            ),
            run_config=tune.RunConfig(
                name="gnnpcsaft",
                storage_path=os.path.join(FLAGS.workdir, "ray_results"),
                checkpoint_config=tune.CheckpointConfig(
                    num_to_keep=1,
                ),
                verbose=0,
                progress_reporter=None,
                log_to_file=False,
                callbacks=(
                    [
                        WandbLoggerCallback(
                            "gnn-pc-saft",
                            config["dataset"],
                            tags=["tuning", config["dataset"]] + FLAGS.tags,
                        )
                    ]
                ),
            ),
        )

    tuner.fit()


if __name__ == "__main__":
    flags.mark_flags_as_required(["config", "workdir"])
    app.run(main)
