"""Single registry of surrogate models.

Each entry says where the estimator class lives. Every model keeps its files in
its own folder: ``models_dir/<name>/`` and ``plots_dir/<name>/``, and reads its
hyper-parameters from ``inputs/models/<name>.toml``. Classes are imported
lazily so training the GRU never imports XGBoost and vice versa (they cannot
share a process on macOS).

To add a model: write a ``SurrogateEstimator`` (or ``TorchEstimator``)
subclass, add an entry here and create ``inputs/models/<name>.toml``.
"""

import importlib
from typing import NamedTuple


class ModelEntry(NamedTuple):
    module: str      # import path of the estimator module
    class_name: str  # estimator class inside that module


ESTIMATORS = {
    "xgboost": ModelEntry("surrogate.estimators.xgboost_estimator", "XGBoostEstimator"),
    "gru": ModelEntry("surrogate.estimators.gru_estimator", "GRUEstimator"),
    "custom": ModelEntry("surrogate.estimators.custom_estimator", "CustomEstimator"),
}


def model_dir(project_config, name):
    """``surrogate/models/<name>/``: model files, metadata.json, cv_predictions.csv."""
    paths = project_config.surrogate.data["paths"]
    return project_config.resolve_path(paths["models_dir"]) / name


def plots_dir(project_config, name):
    """``surrogate/plots/<name>/``: trajectory plots and learning curves."""
    paths = project_config.surrogate.data["paths"]
    return project_config.resolve_path(paths["plots_dir"]) / name


def create_estimator(name, project_config):
    """Import and construct the estimator registered as ``name``."""
    entry = ESTIMATORS[name]
    estimator_class = getattr(importlib.import_module(entry.module), entry.class_name)
    return estimator_class(
        project_config, name, model_dir(project_config, name), plots_dir(project_config, name)
    )


__all__ = ["ESTIMATORS", "create_estimator", "model_dir", "plots_dir"]
