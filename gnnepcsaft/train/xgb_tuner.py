"""Module to tune xgb"""

import ConfigSpace as CS
from absl import app, flags, logging
from ray import tune
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB

from .xgb_training import training

FLAGS = flags.FLAGS
flags.DEFINE_float("num_cpu", 1.0, "Fraction of CPU threads per trial for ray")
flags.DEFINE_float("num_gpus", 1.0, "Fraction of GPUs per trial for ray")
flags.DEFINE_integer(
    "max_concurrent", 2, "Maximum concurrent samples from the underlying searcher."
)


def main(argv):
    """Execution from command line"""
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    logging.info("Calling tuner!")
    # Hyperparameter search space
    search_space = CS.ConfigurationSpace()
    search_space.add(
        CS.UniformFloatHyperparameter("eta", lower=1e-4, upper=1e-1, log=True),
        CS.UniformFloatHyperparameter("lambda", lower=1e-6, upper=1e-1, log=True),
        CS.UniformFloatHyperparameter("alpha", lower=1e-6, upper=1e-1, log=True),
        CS.Constant("num_boost_round", 20000),
    )

    workdir = FLAGS.workdir
    # BOHB search algorithm
    search_alg = TuneBOHB(
        space=search_space,
        metric="mape_den",
        mode="min",
        seed=77,
    )

    # Early stopping scheduler for BOHB
    scheduler = HyperBandForBOHB(
        metric="mape_den",
        mode="min",
        max_t=1,
        stop_last_trials=True,
    )

    # tune run config with wandb logger
    run_config = tune.RunConfig(
        name="gnnpcsaft",
        storage_path=None,
        callbacks=(
            [
                WandbLoggerCallback(
                    "gnn-pc-saft",
                    "xgb",
                    tags=["tuning", "xgb", "esper"],
                )
            ]
        ),
    )

    def tune_xgb(config: dict):
        """trainable fun for the xgb model"""
        mape_den, mape_vp = training(workdir, config)

        tune.report({"mape_den": mape_den, "mape_vp": mape_vp, "done": True})

    # Run the tuner
    tuner = tune.Tuner(
        tune.with_resources(
            tune_xgb, resources={"cpu": FLAGS.num_cpu, "gpu": FLAGS.num_gpus}
        ),
        tune_config=tune.TuneConfig(
            num_samples=60, search_alg=search_alg, scheduler=scheduler
        ),
        run_config=run_config,
    )
    result = tuner.fit()
    best_trial = result.get_best_result(
        metric="mape_den",
        mode="min",
    )

    print(f"\nBest trial config:\n {best_trial.config}")
    print(f"\nBest trial final metrics:\n {best_trial.metrics}")


if __name__ == "__main__":
    flags.mark_flags_as_required(["workdir"])
    app.run(main)
