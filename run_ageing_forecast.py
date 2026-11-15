"""Example: forecast ageing cycles 65 to 500 of one candidate from its first 64 measured cycles.

    python run_ageing_forecast.py
    python run_ageing_forecast.py --models ageing_forecast/models_gru --candidate "x1=0.5_x2=0.5_x3=0.1_x4=1"

The candidate here comes from the training data, so the error printed is not a test error; the held-out
accuracy of each model is in its metadata.json under "cross_validation".
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ageing_forecast import AgeingForecaster

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=None, help="model folder; default: the selected model")
    parser.add_argument("--candidate", default="x1=0.5_x2=0.5_x3=0.1_x4=1")
    parser.add_argument("--measured", type=int, default=64)
    args = parser.parse_args()

    folder = args.models
    if folder is None:
        selection = json.loads((ROOT / "ageing_forecast" / "cv" / "selection.json").read_text())
        folder = ROOT / "ageing_forecast" / f"models_{selection['best_overall'].replace('-', '_')}"
    forecaster = AgeingForecaster(folder)
    truth = pd.read_csv(ROOT / "ageing_forecast" / "data" / f"{args.candidate}.csv").sort_values("cycle")
    forecast = forecaster.forecast(truth.iloc[:args.measured])

    names = [c for c in forecast.columns if c != "cycle"]
    actual = truth.iloc[args.measured:][names].to_numpy(float)
    span = np.ptp(truth[names].to_numpy(float), axis=0)
    error = np.sqrt((((forecast[names].to_numpy(float) - actual) / span) ** 2).mean(0)) * 100
    print(f"model {Path(folder).name}, candidate {args.candidate}, {args.measured} measured cycles")
    print(forecast.iloc[[0, len(forecast) // 2, -1]].to_string(index=False))
    print("nRMSE in percent of each indicator's range: "
          + ", ".join(f"{n} {e:.2f}" for n, e in zip(names, error)))


if __name__ == "__main__":
    main()
