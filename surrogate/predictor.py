"""Trained-surrogate access for the MCO and protocol validation."""

import json

import pandas as pd

from surrogate.estimators import ESTIMATORS, create_estimator, model_dir


class SurrogatePredictor:
    """A trained surrogate (xgboost, gru or custom) loaded once and reused for every prediction.

    The model is chosen in ``inputs/mco.toml`` (``[recommendation] surrogate_model``)
    and loaded from ``models_dir/<model>/`` the first time ``predict`` is called.
    A model counts as trained when its ``metadata.json`` exists.
    """

    def __init__(self, project_config, model_name):
        if model_name not in ESTIMATORS:
            raise ValueError(
                f"Unknown surrogate_model '{model_name}'; choose one of {', '.join(ESTIMATORS)}"
            )
        self.project_config = project_config
        self.model_name = model_name
        self.metadata_path = model_dir(project_config, model_name) / "metadata.json"
        self._estimator = None  # loaded lazily by _load()

    def available(self):
        return self.metadata_path.exists()

    def require_trained(self):
        """Raise with the training command if the selected model has not been trained."""
        if not self.available():
            raise FileNotFoundError(
                f"Surrogate model '{self.model_name}' is not trained "
                f"({self.metadata_path} not found). Train it first:\n"
                f"    python -m surrogate.train_models --model {self.model_name}"
            )

    def predict(self, formation_features, cycle_numbers):
        """``DataFrame`` with ``ageing_cycle`` and one column per target, in physical units.

        ``formation_features`` is the dict from ``TrajectoryFeaturizer.extract``;
        one row is predicted per requested cycle.
        """
        estimator = self._load()
        frame = pd.DataFrame(
            [{**formation_features, "ageing_cycle": cycle} for cycle in cycle_numbers]
        )
        predictions = estimator.predict(frame).reset_index(drop=True)
        predictions.insert(0, "ageing_cycle", list(cycle_numbers))
        return predictions

    def _load(self):
        """Create the estimator, check its features match training, restore the final model."""
        if self._estimator is not None:
            return self._estimator
        self.require_trained()
        estimator = create_estimator(self.model_name, self.project_config)
        with self.metadata_path.open("r", encoding="utf-8") as file:
            trained_columns = json.load(file)["feature_columns"]
        # Guards against e.g. trajectory_feature_count changing after training.
        if trained_columns != estimator.feature_columns:
            raise ValueError(
                f"Surrogate model '{self.model_name}' was trained on different features than "
                f"the current featurizer produces ({len(trained_columns)} vs "
                f"{len(estimator.feature_columns)} columns). Retrain it:\n"
                f"    python -m surrogate.train_models --model {self.model_name}"
            )
        estimator.load_final_model()
        self._estimator = estimator
        return estimator
