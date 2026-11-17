"""What every surrogate shares: paths, dataset loading and result files.

Splitting and training are left to each model so it can use its own tooling
(numpy for XGBoost, ``torch.utils.data`` for the PyTorch models). All of them
read the same ``[split]`` ratios and run protocol-level K-fold
cross-validation, then refit a final model on every protocol.

Each model owns ``models_dir/<name>/`` and ``plots_dir/<name>/``. Besides its
model file(s) it writes:

    cv_predictions.csv  out-of-fold test prediction for every protocol
    metadata.json       final-model description (read when loading the model)

Per-fold metrics and the mean/std summary across folds go to the log.
"""

import json
import logging

import pandas as pd

from surrogate.evaluation import plot_trajectories, summarize_cv
from surrogate.features import TrajectoryFeaturizer
from surrogate.targets import TARGETS

logger = logging.getLogger(__name__)


class SurrogateEstimator:
    def __init__(self, project_config, name, models_dir, plots_dir):
        self.project_config = project_config
        self.name = name
        self.config = project_config.surrogate.data
        self.split = self.config["split"]
        self.params = project_config.model_config(name)
        self.featurizer = TrajectoryFeaturizer(project_config.surrogate)
        self.feature_columns = self.featurizer.feature_columns()
        self.models_dir = models_dir
        self.plots_dir = plots_dir
        self.plotting = self.config["plots"]["enabled"]

    # ------------------------------------------------------- to implement
    def train(self):
        """Cross-validate, fit the final model, save; return ``save_results(...)``."""
        raise NotImplementedError

    def predict(self, frame):
        """DataFrame with one column per target, aligned with ``frame``'s rows."""
        raise NotImplementedError

    def load_final_model(self):
        """Restore the fitted final model from ``self.models_dir``."""
        raise NotImplementedError

    # ------------------------------------------------------------- shared
    def load_dataset(self):
        path = self.project_config.resolve_path(self.config["paths"]["dataset"])
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {path}. Run surrogate.create_dataset first."
            )
        data = pd.read_csv(path).dropna(subset=self.feature_columns + list(TARGETS))
        logger.info(
            f"[{self.name}] {len(data)} rows, {data['protocol_key'].nunique()} protocols"
        )
        return data.reset_index(drop=True)

    def save_results(self, metric_rows, prediction_rows, metadata):
        self.models_dir.mkdir(parents=True, exist_ok=True)
        cv_metrics = pd.DataFrame(metric_rows)
        cv_predictions = pd.DataFrame(prediction_rows)
        summary = summarize_cv(cv_metrics)
        self._log_table("cross-validation metrics per fold", self._by_split_target_fold(cv_metrics))
        self._log_table("cross-validation summary (mean and std across folds)", summary)

        cv_predictions.to_csv(self.models_dir / "cv_predictions.csv", index=False)
        if self.plotting:
            self.plots_dir.mkdir(parents=True, exist_ok=True)
            plot_trajectories(
                cv_predictions, self.plots_dir, self.name, self.config["plots"]["max_test_protocols"]
            )
        metadata = {
            "model_name": self.name,
            "feature_columns": self.feature_columns,
            "target_columns": list(TARGETS),
            "split": self.split,
            **metadata,
        }
        with (self.models_dir / "metadata.json").open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2, default=str)
        return summary

    def _log_table(self, title, table):
        text = table.to_string(index=False, float_format=lambda value: f"{value:.5g}")
        logger.info(f"[{self.name}] {title}:\n{text}")

    @staticmethod
    def _by_split_target_fold(cv_metrics):
        table = cv_metrics.drop(columns="model")
        table["split"] = pd.Categorical(table["split"], ["train", "validation", "test"])
        table["target"] = pd.Categorical(table["target"], list(TARGETS))
        return table.sort_values(["split", "target", "fold"])
