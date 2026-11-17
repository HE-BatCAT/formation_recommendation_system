"""List, verify and score every trained surrogate in surrogate/models*.

    python -m surrogate.compare_models                        # verify the saved models
    python -m surrogate.compare_models --dataset points.csv   # score them on new simulated points

The dataset must have the columns of surrogate/dataset.csv (protocol_key, the formation features,
ageing_cycle and the four targets), for example the output of surrogate.create_dataset for new protocols.
Errors are given as nRMSE in percent of each target's range in the training dataset.
"""
import argparse
import json

import numpy as np
import pandas as pd

from inputs import ProjectConfig
from surrogate.predictor import SurrogatePredictor

TARGETS = ["capacity", "plating", "sei", "resistance"]


def model_folders(config):
    root = config.project_root / "surrogate"
    return sorted(p for p in root.glob("models*") if (p / "metadata.json").exists())


def verify(folder, config, training):
    """Reload a neural model and compare with the outputs stored when it was trained."""
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    check = metadata.get("reference_check")
    if not check:
        return "no reference stored"
    rows = training[training["protocol_key"] == check["protocol_key"]]
    if rows.empty:
        return "reference protocol not in dataset"
    features = {c: float(rows.iloc[0][c]) for c in metadata["feature_columns"] if c != "ageing_cycle"}
    predictor = SurrogatePredictor(config.surrogate, config, models_dir=str(folder))
    got = predictor.predict(features, check["ageing_cycles"])[TARGETS].to_numpy(float)
    difference = float(np.abs(got - np.array(check["expected"])).max())
    return f"ok (max difference {difference:.1e})" if difference < 1e-4 else f"MISMATCH {difference:.3g}"


def score(folder, config, data, span):
    predictor = SurrogatePredictor(config.surrogate, config, models_dir=str(folder))
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    features = [c for c in metadata["feature_columns"] if c != "ageing_cycle"]
    errors, per_protocol = [], {}
    for key, rows in data.groupby("protocol_key", sort=False):
        rows = rows.sort_values("ageing_cycle")
        formation = {c: float(rows.iloc[0][c]) for c in features}
        predicted = predictor.predict(formation, rows["ageing_cycle"].tolist())[TARGETS].to_numpy(float)
        error = predicted - rows[TARGETS].to_numpy(float)
        errors.append(error)
        per_protocol[key] = float(np.mean(np.sqrt((error ** 2).mean(0)) / span * 100))
    error = np.concatenate(errors)
    nrmse = np.sqrt((error ** 2).mean(0)) / span * 100
    worst = max(per_protocol, key=per_protocol.get)
    return dict(
        model=folder.name, backend=predictor.backend(), nrmse_mean=float(nrmse.mean()),
        **{f"nrmse_{t}": float(v) for t, v in zip(TARGETS, nrmse)},
        **{f"mae_{t}": float(v) for t, v in zip(TARGETS, np.abs(error).mean(0))},
        worst_protocol_nrmse=per_protocol[worst], worst_protocol=worst,
    ), per_protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=None, help="CSV with new points, in dataset.csv format")
    parser.add_argument("--out", default=None, help="write the per-model table to this CSV")
    args = parser.parse_args()

    config = ProjectConfig()
    training = pd.read_csv(config.resolve_path(config.surrogate.data["paths"]["dataset"]))
    span = np.ptp(training[TARGETS].to_numpy(float), axis=0)
    folders = model_folders(config)
    if not folders:
        raise SystemExit("No trained models found in surrogate/models*")

    print("Saved models")
    for folder in folders:
        metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("backend") == "torch":
            try:
                note = (f"{metadata['arm']}, certified contraction: {metadata['certified_contraction']}, "
                        f"reload check: {verify(folder, config, training)}")
            except Exception as error:
                note = f"{metadata['arm']}, reload failed ({type(error).__name__}: {error})"
        else:
            note = metadata.get("model_type", "xgboost")
        print(f"  {folder.name:40} {note}")

    if args.dataset is None:
        return
    data = pd.read_csv(args.dataset)
    known = set(training["protocol_key"])
    seen = sum(key in known for key in data["protocol_key"].unique())
    print(f"\n{data['protocol_key'].nunique()} protocols in {args.dataset}; "
          f"{seen} of them are also in the training dataset")
    rows = []
    for folder in folders:
        try:
            row, _ = score(folder, config, data, span)
            rows.append(row)
        except Exception as error:  # a model that cannot be loaded should not hide the others
            print(f"  {folder.name}: failed ({type(error).__name__}: {error})")
    table = pd.DataFrame(rows).sort_values("nrmse_mean")
    print("\n" + table.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    if args.out:
        table.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
