"""Train and select the ageing-trajectory forecaster.

    python -m ageing_forecast.train cv                  # grouped cross-validation over candidates, resumable
    python -m ageing_forecast.train cv --arms gru       # a subset
    python -m ageing_forecast.train summarize           # table and selection.json
    python -m ageing_forecast.train final --arms gru    # train on every candidate and save
    python -m ageing_forecast.train final --selected

Data: one CSV per formation candidate in ageing_forecast/data, 500 ageing cycles of the four indicators,
simulated with this repository's PyBaMM pipeline. A candidate is never in its own training set.
Settings are in inputs/ageing_forecast.toml.
"""
import argparse
import glob
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ageing_forecast.forecaster import AgeingForecaster, block_rollout, build_forecaster
from inputs.config import ProjectConfig, TomlConfig


class ForecastTrainer:
    def __init__(self, project_config=None):
        self.project = project_config or ProjectConfig()
        config = TomlConfig(self.project.project_root / "inputs" / "ageing_forecast.toml").data
        self.settings = config["training"]
        self.indicators = config["data"]["indicators"]
        self.cycles = config["data"]["aging_cycles"]
        self.lookback, self.block = config["model"]["lookback"], config["model"]["block"]
        self.data_dir = self.project.resolve_path(config["paths"]["data_dir"])
        self.cv_dir = self.project.resolve_path(config["paths"]["cv_dir"])
        self.models_root = self.project.resolve_path(config["paths"]["models_dir"])
        self._trajectories = None

    # ------------------------------------------------------------------ data
    def trajectories(self):
        if self._trajectories is None:
            found = {}
            for path in sorted(glob.glob(str(self.data_dir / "*.csv"))):
                frame = pd.read_csv(path).sort_values("cycle")
                values = frame[self.indicators].to_numpy(np.float64)
                if len(values) == self.cycles and np.isfinite(values).all():
                    found[Path(path).stem] = values
            self._trajectories = found
        return self._trajectories

    def windows(self, series):
        size = self.lookback + self.block
        stacked = np.stack([series[i:i + size] for i in range(len(series) - size + 1)]).astype(np.float32)
        return stacked[:, :self.lookback], stacked[:, self.lookback:]

    def _fit(self, arm, names, seed, custom=None):
        import torch

        from surrogate.neural.models import count_parameters, fit, training_overrides

        optimizer = custom.get("training", {}) if custom else training_overrides(arm)

        data = self.trajectories()
        pooled = np.concatenate([data[n] for n in names])
        mu, sd = pooled.mean(0), pooled.std(0) + 1e-8
        pairs = [self.windows((data[n] - mu) / sd) for n in names]
        x = np.concatenate([p[0] for p in pairs])
        y = np.concatenate([p[1] for p in pairs])
        np.random.seed(seed)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = build_forecaster(arm, self.lookback, self.block, len(self.indicators), custom=custom)
        model, epochs, _ = fit(
            model, x, y, None, None, epochs=self.settings["epochs"], patience=0,
            batch_size=optimizer.get("batch_size", self.settings["batch_size"]),
            learning_rate=optimizer.get("learning_rate", self.settings["learning_rate"]),
            weight_decay=optimizer.get("weight_decay", 0.0),
            seed=seed, device=device,
        )
        return model, mu, sd, dict(epochs=epochs, parameters=count_parameters(model), windows=len(x)), device

    # ------------------------------------------------------------------ cross-validation
    def cross_validate(self, arms=None, seeds=None):
        data = self.trajectories()
        names = sorted(data)
        arms = list(arms or self.settings["arms"])
        seeds = list(seeds or self.settings["seeds"])
        steps = self.cycles - self.lookback
        print(f"{len(names)} candidates, lookback {self.lookback}, block {self.block}, "
              f"{self.settings['folds']} grouped folds, seeds {seeds}", flush=True)
        for arm in arms:
            for seed in seeds:
                order = np.random.default_rng(seed + 7919).permutation(len(names))
                for fold, chunk in enumerate(np.array_split(order, self.settings["folds"])):
                    path = self.cv_dir / arm / f"seed{seed}_fold{fold}.json"
                    if path.exists():
                        continue
                    test = [names[i] for i in chunk]
                    train = [n for n in names if n not in set(test)]
                    started = time.perf_counter()
                    model, mu, sd, info, device = self._fit(arm, train, seed)
                    window = np.stack([(data[n][:self.lookback] - mu) / sd for n in test])
                    rolled = block_rollout(model, window, steps, self.block, device) * sd + mu
                    candidates = []
                    for name, predicted in zip(test, rolled):
                        truth = data[name]
                        span = np.ptp(truth, axis=0)
                        span = np.where(span == 0, 1.0, span)
                        finite = bool(np.isfinite(predicted).all())
                        error = (np.sqrt((((predicted - truth[self.lookback:]) / span) ** 2).mean(0)) * 100
                                 if finite else np.full(len(self.indicators), np.nan))
                        candidates.append(dict(
                            candidate=name, diverged=not finite, nrmse=error.tolist(),
                            final_actual=truth[-1].tolist(),
                            final_predicted=predicted[-1].tolist() if finite else None,
                        ))
                    record = dict(arm=arm, seed=seed, fold=fold, seconds=time.perf_counter() - started,
                                  **info, candidates=candidates)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = path.with_suffix(f".tmp{os.getpid()}")
                    temporary.write_text(json.dumps(record), encoding="utf-8")
                    temporary.replace(path)
                    typical = np.nanmedian([np.mean(c["nrmse"]) for c in candidates])
                    print(f"  {arm:18} seed {seed} fold {fold}: median nRMSE {typical:6.2f} %  "
                          f"{record['seconds']:.0f} s", flush=True)

    # ------------------------------------------------------------------ hyperparameter search
    FAMILIES = {
        "cenn-certified": dict(arm="ams-cenn-forecast-tuned", kind="cenn", variant="C1-BoundedTau"),
        "gru": dict(arm="gru-forecast-tuned", kind="gru", variant=None),
    }

    @staticmethod
    def sample(family, trial):
        rng = np.random.default_rng([sum(map(ord, "forecast" + family)), trial])
        pick = lambda options: options[int(rng.integers(len(options)))]
        training = dict(
            learning_rate=float(10 ** rng.uniform(np.log10(3e-4), np.log10(3e-3))),
            batch_size=pick([128, 256, 512]),
            weight_decay=pick([0.0, 1e-5, 1e-4]),
        )
        if family == "gru":
            overrides = dict(hidden=pick([64, 128, 256]), layers=pick([1, 2, 3]),
                             dropout=pick([0.0, 0.1, 0.2, 0.3]))
        else:
            overrides = dict(
                linear_skip=pick([True, True, False]),
                multiscale_mode=pick(["none", "parallel_ensemble"]),
                head_type=pick(["linear", "mlp"]),
                hidden_dim=pick([32, 64, 128]),
                num_layers=pick([2, 4, 6]),
                K=pick([2, 4, 8]),
                neighborhood=pick([3, 5, 7]),
                dropout=pick([0.0, 0.1, 0.25]),
                spectral_rho=pick([0.9, 0.99]),
                alpha_min=pick([0.0, 0.25, 0.5]),
            )
        return dict(overrides=overrides, training=training)

    def _rollout_errors(self, model, mu, sd, names, device):
        """Mean-over-indicator nRMSE of each candidate, cycles lookback + 1 to the end; inf if diverged."""
        data = self.trajectories()
        window = np.stack([(data[n][:self.lookback] - mu) / sd for n in names])
        rolled = block_rollout(model, window, self.cycles - self.lookback, self.block, device) * sd + mu
        out = []
        for name, predicted in zip(names, rolled):
            span = np.ptp(data[name], axis=0)
            span = np.where(span == 0, 1.0, span)
            if not np.isfinite(predicted).all():
                out.append(float("inf"))
                continue
            out.append(float(np.mean(
                np.sqrt((((predicted - data[name][self.lookback:]) / span) ** 2).mean(0)) * 100)))
        return out

    def tune(self, family, first, last):
        """Random search. Fold f is the test set, fold f + 1 the validation set a trial is selected on."""
        base = self.FAMILIES[family]
        names = sorted(self.trajectories())
        tuning = self.settings["tuning"]
        order = np.random.default_rng(tuning["seed"] + 7919).permutation(len(names))
        chunks = np.array_split(order, self.settings["folds"])
        for trial in range(first, last):
            path = self.cv_dir / "tuning" / family / f"trial{trial:03d}.json"
            if path.exists():
                continue
            drawn = self.sample(family, trial)
            custom = dict(kind=base["kind"], variant=base["variant"], **drawn)
            started, folds, failed = time.perf_counter(), [], None
            try:
                for fold in range(tuning["folds"]):
                    test = [names[i] for i in chunks[fold]]
                    validation = [names[i] for i in chunks[(fold + 1) % len(chunks)]]
                    train = [n for n in names if n not in set(test) | set(validation)]
                    model, mu, sd, info, device = self._fit(None, train, tuning["seed"], custom=custom)
                    val = self._rollout_errors(model, mu, sd, validation, device)
                    tst = self._rollout_errors(model, mu, sd, test, device)
                    folds.append(dict(fold=fold, validation_median=float(np.median(val)),
                                      validation_worst=float(np.max(val)), test_median=float(np.median(tst)),
                                      test_worst=float(np.max(tst)), parameters=info["parameters"]))
            except Exception as error:
                failed = f"{type(error).__name__}: {error}"
            done = folds and not failed
            record = dict(
                family=family, trial=trial, **drawn, folds=folds, failed=failed,
                # selection score: typical and worst validation candidate, equally weighted
                validation=float(np.mean([0.5 * f["validation_median"] + 0.5 * f["validation_worst"]
                                          for f in folds])) if done else None,
                validation_median=float(np.mean([f["validation_median"] for f in folds])) if done else None,
                validation_worst=float(np.max([f["validation_worst"] for f in folds])) if done else None,
                test_median=float(np.mean([f["test_median"] for f in folds])) if done else None,
                test_worst=float(np.max([f["test_worst"] for f in folds])) if done else None,
                seconds=time.perf_counter() - started)
            if record["validation"] is not None and not np.isfinite(record["validation"]):
                record.update(validation=None, failed="diverged rollout")
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(f".tmp{os.getpid()}")
            temporary.write_text(json.dumps(record), encoding="utf-8")
            temporary.replace(path)
            print(f"  tune {family} trial {trial:3d}: validation median {record['validation_median']} "
                  f"worst {record['validation_worst']}  {record['seconds']:.0f} s  {record['failed'] or ''}",
                  flush=True)

    def tune_summarize(self, family):
        from surrogate.neural.models import certified_spec, register_tuned

        base = self.FAMILIES[family]
        records = [json.loads(p.read_text(encoding="utf-8"))
                   for p in sorted((self.cv_dir / "tuning" / family).glob("trial*.json"))]
        valid = [r for r in records if r["validation"] is not None]
        if not valid:
            raise RuntimeError(f"No finished trials for {family}")
        keys = ("validation", "validation_median", "validation_worst", "test_median", "test_worst", "seconds")
        table = pd.DataFrame([dict(trial=r["trial"], **{k: r[k] for k in keys}, **r["overrides"],
                                   **r["training"]) for r in valid]).sort_values("validation")
        table.to_csv(self.cv_dir / "tuning" / f"{family}_trials.csv", index=False)
        best = min(valid, key=lambda r: r["validation"])
        certified = base["kind"] == "cenn" and certified_spec(base["variant"], best["overrides"])
        register_tuned(base["arm"], dict(
            kind=base["kind"], variant=base["variant"], overrides=best["overrides"],
            training=best["training"],
            description=f"ageing forecaster, random search, best of {len(valid)} trials by validation "
                        f"error (trial {best['trial']}); certified contraction: {certified}",
        ))
        print(table.head(10).to_string(index=False, float_format=lambda v: f"{v:.4g}"))
        print(f"registered {base['arm']} from trial {best['trial']} "
              f"({len(valid)} valid trials, {len(records) - len(valid)} failed or diverged)", flush=True)
        return base["arm"]

    def summarize(self):
        from scipy.stats import spearmanr

        from surrogate.neural.models import architecture, describe, is_certified, training_overrides

        rows = []
        for folder in sorted(p for p in self.cv_dir.glob("*") if p.is_dir() and p.name != "tuning"):
            by_seed = {}
            for path in folder.glob("seed*_fold*.json"):
                record = json.loads(path.read_text(encoding="utf-8"))
                by_seed.setdefault(record["seed"], []).append(record)
            complete = {s: r for s, r in by_seed.items() if len(r) == self.settings["folds"]}
            if not complete:
                continue
            per_seed = []
            for records in complete.values():
                cands = [c for r in records for c in r["candidates"]]
                ok = [c for c in cands if not c["diverged"]]
                mean_error = np.array([np.mean(c["nrmse"]) for c in ok])
                by_indicator = np.array([c["nrmse"] for c in ok])
                per_seed.append(dict(
                    median=float(np.median(mean_error)), worst=float(np.max(mean_error)),
                    indicator=np.median(by_indicator, axis=0),
                    diverged=len(cands) - len(ok),
                    rank=float(spearmanr([c["final_actual"][0] for c in ok],
                                         [c["final_predicted"][0] for c in ok]).correlation),
                ))
            first = next(iter(complete.values()))[0]
            rows.append(dict(
                arm=folder.name, certified=is_certified(folder.name), seeds=len(complete),
                nrmse_median=float(np.mean([s["median"] for s in per_seed])),
                nrmse_worst_candidate=float(np.max([s["worst"] for s in per_seed])),
                **{f"nrmse_{name}": float(np.mean([s["indicator"][j] for s in per_seed]))
                   for j, name in enumerate(self.indicators)},
                rank_final_capacity=float(np.mean([s["rank"] for s in per_seed])),
                diverged=int(np.sum([s["diverged"] for s in per_seed])),
                parameters=first["parameters"],
                seconds_per_fit=float(np.mean([r["seconds"] for rs in complete.values() for r in rs])),
                description=describe(folder.name),
            ))
        if not rows:
            raise RuntimeError(f"No complete cross-validation results in {self.cv_dir}")
        table = pd.DataFrame(rows).sort_values(["diverged", "nrmse_median"]).reset_index(drop=True)
        table.to_csv(self.cv_dir / "summary.csv", index=False)
        cellular = table[table["arm"].str.contains("cenn")]
        selection = dict(
            criterion="no diverged rollouts, then lowest median nRMSE over held-out candidates for "
                      f"cycles {self.lookback + 1} to {self.cycles}, averaged over seeds",
            best_overall=table.iloc[0]["arm"],
            best_cellular=None if cellular.empty else cellular.iloc[0]["arm"],
            best_certified=None if table[table["certified"]].empty else table[table["certified"]].iloc[0]["arm"],
        )
        (self.cv_dir / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
        print(table.drop(columns=["description"]).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
        print(json.dumps(selection, indent=2), flush=True)
        return table, selection

    # ------------------------------------------------------------------ final models
    def train_final(self, arm, seeds=None):
        import torch

        from surrogate.neural.models import architecture, describe, is_certified, training_overrides

        seeds = list(seeds or self.settings["seeds"])
        data = self.trajectories()
        names = sorted(data)
        out = Path(f"{self.models_root}_{arm.replace('-', '_')}")
        out.mkdir(parents=True, exist_ok=True)
        print(f"final {arm}: all {len(names)} candidates, seeds {seeds} -> {out}", flush=True)
        model_files, mu, sd, info = {}, None, None, {}
        for seed in seeds:
            model, mu, sd, info, _ = self._fit(arm, names, seed)
            filename = f"member_seed{seed}.pt"
            torch.save({k: v.cpu() for k, v in model.state_dict().items()}, out / filename)
            model_files[f"seed{seed}"] = filename
            print(f"  seed {seed}: {info['epochs']} epochs on {info['windows']} windows", flush=True)
        summary = self.cv_dir / "summary.csv"
        cross_validation = None
        if summary.exists():
            match = pd.read_csv(summary)
            match = match[match["arm"] == arm]
            cross_validation = None if match.empty else json.loads(match.iloc[0].to_json())
        metadata = {
            "backend": "torch", "arm": arm, "description": describe(arm),
            "certified_contraction": is_certified(arm),
            "indicators": self.indicators, "aging_cycles": self.cycles,
            "lookback": self.lookback, "block": self.block,
            "model_files": model_files,
            "normalization": {"mu": mu.tolist(), "sd": sd.tolist()},
            "ensemble": "mean of the members; one member per seed",
            "final_fit_rule": f"trained on all {len(names)} candidates for a fixed "
                              f"{self.settings['epochs']} epochs, the budget used in cross-validation",
            "architecture": architecture(arm),
            "training": {**{k: self.settings[k] for k in ("epochs", "batch_size", "learning_rate")},
                         **training_overrides(arm)},
            "parameters": info.get("parameters"),
            "cross_validation": cross_validation,
            "torch_version": torch.__version__,
        }
        (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        probe = AgeingForecaster(out).forecast(data[names[0]][:self.lookback])
        metadata["reference_check"] = {
            "candidate": names[0], "measured_cycles": self.lookback,
            "expected_final": probe[self.indicators].iloc[-1].astype(float).tolist(),
        }
        (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["cv", "summarize", "final", "tune", "tune-summarize"])
    parser.add_argument("--family", choices=sorted(ForecastTrainer.FAMILIES), default="cenn-certified")
    parser.add_argument("--trials", nargs=2, type=int, default=[0, 20], metavar=("FIRST", "LAST"))
    parser.add_argument("--arms", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--selected", action="store_true")
    args = parser.parse_args()
    trainer = ForecastTrainer()
    if args.command == "cv":
        trainer.cross_validate(args.arms, args.seeds)
    elif args.command == "summarize":
        trainer.summarize()
    elif args.command == "tune":
        trainer.tune(args.family, *args.trials)
    elif args.command == "tune-summarize":
        trainer.tune_summarize(args.family)
    else:
        arms = list(args.arms or [])
        if args.selected:
            selection = json.loads((trainer.cv_dir / "selection.json").read_text(encoding="utf-8"))
            for key in ("best_overall", "best_cellular", "best_certified"):
                if selection.get(key) and selection[key] not in arms:
                    arms.append(selection[key])
        if not arms:
            parser.error("final needs --arms or --selected")
        for arm in arms:
            trainer.train_final(arm, args.seeds)


if __name__ == "__main__":
    main()
