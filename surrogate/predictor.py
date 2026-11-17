import json

import joblib
import pandas as pd

from surrogate.estimators import model_dir


class SurrogatePredictor:
    """XGBoost surrogate used by the MCO, loaded from ``models_dir/xgboost/``."""

    def __init__(self, surrogate_config, project_config):
        self.config = surrogate_config.data
        self.models_dir = model_dir(project_config, "xgboost")
        self.metadata_path = self.models_dir / "metadata.json"
        self._models = None
        self._metadata = None

    def available(self):
        return self.metadata_path.exists()

    def predict(self, formation_features, cycle_numbers):
        self._load()
        feature_columns = self._metadata["feature_columns"]
        formation_columns = [
            column for column in feature_columns if column != "ageing_cycle"
        ]
        missing_columns = [
            column for column in formation_columns if column not in formation_features
        ]
        if missing_columns:
            preview = ", ".join(missing_columns[:5])
            raise ValueError(
                "Formation featurization is missing "
                f"{len(missing_columns)} trained features (for example: {preview})."
            )

        rows = []
        for cycle in cycle_numbers:
            rows.append({**formation_features, "ageing_cycle": cycle})
        frame = pd.DataFrame(rows).reindex(columns=feature_columns)
        predictions = pd.DataFrame({"ageing_cycle": cycle_numbers})
        for target, model in self._models.items():
            # XGBoost pipelines are fitted on numpy arrays, not DataFrames.
            predictions[target] = model.predict(frame.to_numpy(dtype=float))
        return predictions

    def _load(self):
        if self._models is not None:
            return
        if not self.available():
            raise FileNotFoundError(f"Surrogate metadata not found: {self.metadata_path}")
        with self.metadata_path.open("r", encoding="utf-8") as file:
            self._metadata = json.load(file)
        self._models = {
            target: joblib.load(self.models_dir / filename)
            for target, filename in self._metadata["model_files"].items()
        }
