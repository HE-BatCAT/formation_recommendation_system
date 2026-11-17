"""XGBoost surrogate on numpy arrays.

One ``SimpleImputer -> StandardScaler -> XGBRegressor`` pipeline per target.
Cross-validation is a protocol-level K-fold done with numpy index arrays, so
rows of one protocol (its ageing cycles) never straddle train and test.
"""

import json
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from surrogate.estimators.base import SurrogateEstimator
from surrogate.evaluation import score_split
from surrogate.targets import TARGETS

logger = logging.getLogger(__name__)


class XGBoostEstimator(SurrogateEstimator):
    def train(self):
        data = self.load_dataset()
        x = data[self.feature_columns].to_numpy(dtype=float)
        y = data[list(TARGETS)].to_numpy(dtype=float)
        protocols = data["protocol_key"].to_numpy()
        cycles = data["ageing_cycle"].to_numpy()
        logger.info(f"Training XGBoost with configuration: {self.params}")

        metric_rows, prediction_rows = [], []
        for fold, splits in enumerate(self._folds(protocols), start=1):
            models = self._fit(x[splits["train"]], y[splits["train"]])
            for split, rows in splits.items():
                metrics, predictions = score_split(
                    self.name, fold, split, protocols[rows], cycles[rows],
                    y[rows], self._predict(models, x[rows]),
                )
                metric_rows += metrics
                if split == "test":
                    prediction_rows += predictions
            logger.info(f"[xgboost] fold {fold} done")

        # Final model for prediction/MCO: every protocol (XGBoost uses no validation set).
        self.models = self._fit(x, y)
        model_files = self._save_models()
        return self.save_results(
            metric_rows,
            prediction_rows,
            {
                "model_type": "XGBRegressor",
                "model_files": model_files,
                "params": self.params,
            },
        )

    def _folds(self, protocols):
        """Protocol-level K-fold with numpy; yields row indices per split."""
        rng = np.random.default_rng(self.split["random_state"])
        shuffled = rng.permutation(np.unique(protocols))
        for test in np.array_split(shuffled, self.split["cv_folds"]):
            rest = rng.permutation(np.setdiff1d(shuffled, test))
            n_validation = round(len(rest) * self.split["validation_fraction_of_train"])
            groups = {
                "train": rest[n_validation:],
                "validation": rest[:n_validation],
                "test": test,
            }
            yield {
                split: np.flatnonzero(np.isin(protocols, group_protocols))
                for split, group_protocols in groups.items()
            }

    def _fit(self, x, y):
        from xgboost import XGBRegressor

        models = []
        for column in range(y.shape[1]):
            model = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("regressor", XGBRegressor(**self.params)),
                ]
            )
            models.append(model.fit(x, y[:, column]))
        return models

    @staticmethod
    def _predict(models, x):
        return np.column_stack([model.predict(x) for model in models])

    def predict(self, frame):
        x = frame[self.feature_columns].to_numpy(dtype=float)
        return pd.DataFrame(
            self._predict(self.models, x), index=frame.index, columns=list(TARGETS)
        )

    def _save_models(self):
        self.models_dir.mkdir(parents=True, exist_ok=True)
        model_files = {}
        for target, model in zip(TARGETS, self.models):
            model_files[target] = f"{target}_xgboost.joblib"
            joblib.dump(model, self.models_dir / model_files[target])
        logger.info(f"Saved XGBoost models to {self.models_dir}")
        return model_files

    def load_final_model(self):
        with (self.models_dir / "metadata.json").open("r", encoding="utf-8") as file:
            model_files = json.load(file)["model_files"]
        self.models = [joblib.load(self.models_dir / model_files[target]) for target in TARGETS]
