"""Metrics, cross-validation summaries and plots shared by every surrogate.

Only numpy/pandas/matplotlib are used here, so XGBoost and the PyTorch models
report results in exactly the same format.
"""

import math

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from surrogate.targets import TARGETS

matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS = ("mae", "rmse", "r2", "mape_pct")


def regression_metrics(actual, predicted):
    """MAE, RMSE, R^2 and MAPE (%) of one target; R^2 is NaN for a single row."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return {
        "mae": mean_absolute_error(actual, predicted),
        "rmse": math.sqrt(mean_squared_error(actual, predicted)),
        "r2": r2_score(actual, predicted) if len(actual) > 1 else np.nan,
        # Scale-free, so errors are comparable across targets. Inflated for
        # targets with values near zero (e.g. plating at early cycles).
        "mape_pct": float(
            np.mean(np.abs(predicted - actual) / np.maximum(np.abs(actual), 1e-12)) * 100
        ),
    }


def score_split(model, fold, split, protocol_keys, cycles, actual, predicted):
    """Metric rows and prediction rows for one split of one fold.

    ``actual`` and ``predicted`` are ``(rows, len(TARGETS))`` arrays aligned with
    ``protocol_keys`` and ``cycles``.
    """
    metric_rows, prediction_rows = [], []
    for column, target in enumerate(TARGETS):
        metric_rows.append(
            {
                "model": model,
                "fold": fold,
                "split": split,
                "target": target,
                "rows": len(protocol_keys),
                "protocols": len(set(protocol_keys)),
                **regression_metrics(actual[:, column], predicted[:, column]),
            }
        )
        prediction_rows.extend(
            {
                "model": model,
                "fold": fold,
                "target": target,
                "protocol_key": key,
                "ageing_cycle": int(cycle),
                "actual": float(truth),
                "predicted": float(guess),
            }
            for key, cycle, truth, guess in zip(
                protocol_keys, cycles, actual[:, column], predicted[:, column]
            )
        )
    return metric_rows, prediction_rows


def summarize_cv(cv_metrics):
    """Mean of each metric across folds, per split and target."""
    summary = cv_metrics.groupby(["split", "target"])[list(METRICS)].mean().reset_index()
    summary["split"] = pd.Categorical(summary["split"], ["train", "validation", "test"])
    summary["target"] = pd.Categorical(summary["target"], list(TARGETS))
    return summary.sort_values(["split", "target"]).reset_index(drop=True)


def plot_trajectories(predictions, plots_dir, model, max_protocols):
    """Out-of-fold (test) predictions vs PyBaMM for a few protocols, one figure per target."""
    for target in TARGETS:
        subset = predictions[predictions["target"] == target]
        shown = subset["protocol_key"].drop_duplicates().iloc[:max_protocols]
        figure, axis = plt.subplots(figsize=(11, 6))
        for color_index, protocol in enumerate(shown):
            values = subset[subset["protocol_key"] == protocol].sort_values("ageing_cycle")
            color = plt.get_cmap("tab10")(color_index)
            axis.plot(values["ageing_cycle"], values["actual"], "-o", color=color,
                      markersize=5, linewidth=1.6, label=f"{protocol} - PyBaMM")
            axis.plot(values["ageing_cycle"], values["predicted"], "--x", color=color,
                      markersize=6, linewidth=1.6, label=f"{protocol} - {model}")
        axis.set_xlabel("Ageing cycle")
        axis.set_ylabel(target)
        axis.set_title(f"{target}: out-of-fold predictions ({model})")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
        figure.tight_layout()
        figure.savefig(plots_dir / f"{target}_trajectory_test.png", dpi=300)
        plt.close(figure)
