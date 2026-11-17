import json

import joblib
import pandas as pd


class SurrogatePredictor:
    """Serves the trained ageing surrogate named by ``paths.models_dir`` in ``inputs/surrogate.toml``.

    The backend is read from ``metadata.json``: the XGBoost models of the original system, or a neural
    model (AMS-CeNN, GRU, MLP) trained with ``python -m surrogate.train_neural``. Both return the same
    frame, so the optimizer does not need to know which one is loaded.
    """

    def __init__(self, surrogate_config, project_config, models_dir=None):
        self.config = surrogate_config.data
        self.models_dir = project_config.resolve_path(
            models_dir if models_dir is not None else self.config["paths"]["models_dir"]
        )
        self.metadata_path = self.models_dir / "metadata.json"
        self._models = None
        self._metadata = None
        self._neural = None

    def available(self):
        return self.metadata_path.exists()

    def predict(self, formation_features, cycle_numbers):
        frame = self._frame(formation_features, cycle_numbers)
        if self._neural is not None:
            return self._neural.to_dataframe(self._neural.predict_frame(frame), cycle_numbers)
        predictions = pd.DataFrame({"ageing_cycle": cycle_numbers})
        for target, model in self._models.items():
            predictions[target] = model.predict(frame)
        return predictions

    def gradient(self, formation_features, cycle_numbers):
        """Exact derivatives of the targets with respect to the formation features (neural models only).

        Returns ``(array, feature_names)``; the array has shape (cycles, targets, features).
        """
        frame = self._frame(formation_features, cycle_numbers)
        if self._neural is None:
            raise NotImplementedError(
                "The XGBoost surrogate is piecewise constant and has no usable gradient; "
                "point paths.models_dir at a neural model."
            )
        names = [c for c in self._metadata["feature_columns"] if c != "ageing_cycle"]
        return self._neural.jacobian(frame), names

    def backend(self):
        self._load()
        return self._metadata.get("backend", "xgboost")

    def _frame(self, formation_features, cycle_numbers):
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
        return pd.DataFrame(rows).reindex(columns=feature_columns)

    def _load(self):
        if self._metadata is not None:
            return
        if not self.available():
            raise FileNotFoundError(f"Surrogate metadata not found: {self.metadata_path}")
        with self.metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        if metadata.get("backend") == "torch":
            from surrogate.neural.predictor import NeuralEnsemble

            self._neural = NeuralEnsemble(self.models_dir, metadata)
            self._models = {}
        else:
            self._models = {
                target: joblib.load(self.models_dir / filename)
                for target, filename in metadata["model_files"].items()
            }
        self._metadata = metadata
