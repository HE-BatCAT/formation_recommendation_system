import argparse
import logging
import subprocess
import sys

from inputs import ProjectConfig, setup_logging
from surrogate.estimators import ESTIMATORS, create_estimator

setup_logging()
logger = logging.getLogger(__name__)


class TrainingCommand:
    def run(self):
        arguments = self._arguments()
        models = list(ESTIMATORS) if "all" in arguments.model else arguments.model
        if len(models) == 1:
            self._train(models[0])
            return
        # XGBoost (Homebrew libomp) and torch (bundled libomp) cannot share a
        # process on macOS: torch hangs or XGBoost segfaults. Train each model
        # in its own interpreter.
        failed = []
        for model in models:
            code = subprocess.call(
                [sys.executable, "-m", "surrogate.train_models", "--model", model]
            )
            if code != 0:
                failed.append(model)
        if failed:
            print(f"Training failed for: {', '.join(failed)} (see formation_system.log)")
            sys.exit(1)

    def _train(self, model):
        logger.info(f"===== Training surrogate model: {model} =====")
        print(f"Training {model} (cross-validation + final model) ...", flush=True)
        estimator = create_estimator(model, ProjectConfig())
        try:
            summary = estimator.train()
        except NotImplementedError as error:
            logger.warning(f"Skipping {model}: {error}")
            print(f"  skipped {model}: {error}")
            return
        test = summary[summary["split"] == "test"]
        columns = ["target", "mae_mean", "mae_std", "rmse_mean", "rmse_std", "r2_mean", "r2_std"]
        print(f"Test metrics across {estimator.split['cv_folds']} folds (mean and std):")
        print(test[columns].to_string(index=False, float_format=lambda v: f"{v:.4g}"))
        print(f"  artifacts: {estimator.models_dir}", flush=True)

    def _arguments(self):
        parser = argparse.ArgumentParser(description="Train surrogate models")
        parser.add_argument(
            "--model",
            nargs="+",
            default=["xgboost"],
            choices=[*ESTIMATORS, "all"],
            help="Model(s) to train (default: xgboost).",
        )
        return parser.parse_args()


if __name__ == "__main__":
    TrainingCommand().run()
