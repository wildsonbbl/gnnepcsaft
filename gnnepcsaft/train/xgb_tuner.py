"""Module to tune xgb"""

from absl import app, flags, logging
from ray import train, tune

from .xgb_training import training

FLAGS = flags.FLAGS


def main(argv):
    """Execution from command line"""
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    logging.info("Calling tuner!")
    # Define the search space
    search_config = {
        "eta": tune.loguniform(1e-4, 1e-1),
        "lambda": tune.loguniform(1e-6, 1e-1),
        "alpha": tune.loguniform(1e-6, 1e-1),
        "num_boost_round": 100,
    }

    workdir = FLAGS.workdir

    def tune_xgb(config: dict):
        """Tune the xgb model"""
        mape_den, mape_vp = training(workdir, config)

        train.report({"mape_den": mape_den, "mape_vp": mape_vp, "done": True})

    # Run the tuner
    tuner = tune.Tuner(
        tune.with_resources(tune_xgb, resources={"cpu": 2, "gpu": 0.5}),
        tune_config=tune.TuneConfig(num_samples=2),
        param_space=search_config,
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
