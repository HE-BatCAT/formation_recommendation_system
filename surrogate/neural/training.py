"""Cross-validation, model selection and final training for the neural surrogates.

Cross-validation is grouped by protocol: all 15 ageing rows of a protocol sit in the same fold, so every
score is for protocols the model has not seen. All arms, XGBoost included, see the same folds.

Final models are trained on every protocol except a small validation set used for early stopping, once
per seed, and saved as an ensemble.
"""
import json
import logging
import os
import time

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TARGETS = ("capacity", "plating", "sei", "resistance")
SIGNALS = ("voltage", "current", "q_signed")


def say(message):
    logger.info(message)
    print(message, flush=True)


class NeuralTrainer:
    def __init__(self, surrogate_config, project_config, featurizer):
        self.config = surrogate_config.data
        self.settings = self.config["neural"]
        self.project_config = project_config
        self.featurizer = featurizer
        self.count = featurizer.feature_count
        self.cycle_scale = float(self.config["data"]["aging_cycles"])
        self.dataset_path = project_config.resolve_path(self.config["paths"]["dataset"])
        self.cv_dir = project_config.resolve_path(self.settings["cv_dir"])
        self._data = None

    # ------------------------------------------------------------------ data
    def data(self):
        if self._data is None:
            data = pd.read_csv(self.dataset_path)
            needed = self.featurizer.feature_columns() + list(TARGETS)
            self._data = data.dropna(subset=needed).reset_index(drop=True)
        return self._data

    def arrays(self, frame):
        """(rows, count, signals + 1) inputs with the scaled ageing cycle as a constant channel; targets."""
        traces = np.stack(
            [frame[[f"{s}_{i:03d}" for i in range(self.count)]].to_numpy(np.float32) for s in SIGNALS],
            axis=-1,
        )
        cycle = np.repeat(
            (frame["ageing_cycle"].to_numpy(np.float32) / self.cycle_scale)[:, None, None],
            self.count, axis=1,
        )
        return np.concatenate([traces, cycle], axis=-1), frame[list(TARGETS)].to_numpy(np.float32)

    @staticmethod
    def statistics(x, y):
        flat = x.reshape(-1, x.shape[-1])
        return dict(mu_x=flat.mean(0), sd_x=flat.std(0) + 1e-8, mu_y=y.mean(0), sd_y=y.std(0) + 1e-8)

    def folds(self, seed):
        """Same grouped folds as the September benchmark: test fold f, validation fold f + 1."""
        data = self.data()
        protocols = np.array(sorted(data["protocol_key"].unique()))
        count = self.settings["folds"]
        chunks = np.array_split(np.random.default_rng(seed + 7919).permutation(len(protocols)), count)
        for fold in range(count):
            test = set(protocols[chunks[fold]])
            validation = set(protocols[chunks[(fold + 1) % count]])
            keys = data["protocol_key"]
            yield (fold, data[~keys.isin(test | validation)], data[keys.isin(validation)],
                   data[keys.isin(test)])

    # ------------------------------------------------------------------ fitting
    def _device(self):
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def _fit_neural(self, arm, train, validation, seed, custom=None, budget=None):
        """arm: a registered arm. custom: {"kind", "variant", "overrides", "training"} for a search trial."""
        import torch

        from .models import build, build_custom, count_parameters, fit, training_overrides

        x_train, y_train = self.arrays(train)
        x_val, y_val = self.arrays(validation)
        stats = self.statistics(x_train, y_train)
        scale_x = lambda x: ((x - stats["mu_x"]) / stats["sd_x"]).astype(np.float32)
        scale_y = lambda y: ((y - stats["mu_y"]) / stats["sd_y"]).astype(np.float32)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if custom is None:
            model = build(arm, x_train.shape[-1], len(TARGETS), self.count)
            optimizer = training_overrides(arm)
        else:
            model = build_custom(custom["kind"], custom.get("variant"), custom.get("overrides", {}),
                                 x_train.shape[-1], len(TARGETS), self.count)
            optimizer = custom.get("training", {})
        budget = budget or self.settings
        model, epochs, best_epoch = fit(
            model, scale_x(x_train), scale_y(y_train), scale_x(x_val), scale_y(y_val),
            epochs=budget["epochs"], patience=budget["patience"],
            batch_size=optimizer.get("batch_size", self.settings["batch_size"]),
            learning_rate=optimizer.get("learning_rate", self.settings["learning_rate"]),
            weight_decay=optimizer.get("weight_decay", 0.0),
            seed=seed, device=self._device(),
        )
        info = dict(epochs=epochs, best_epoch=best_epoch, parameters=count_parameters(model))
        return model, stats, info

    def _predict_neural(self, model, stats, frame):
        from .models import predict

        x, _ = self.arrays(frame)
        x = ((x - stats["mu_x"]) / stats["sd_x"]).astype(np.float32)
        return predict(model, x, self._device()) * stats["sd_y"] + stats["mu_y"]

    def _fit_predict_xgboost(self, train, validation, test):
        """The original pipeline and hyperparameters; trained on train + validation, as it has no early stop."""
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from xgboost import XGBRegressor

        features = self.featurizer.feature_columns()
        pool = pd.concat([train, validation])
        predicted = np.zeros((len(test), len(TARGETS)), np.float32)
        for j, target in enumerate(TARGETS):
            model = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("regressor", XGBRegressor(**self.config["xgboost"])),
            ])
            model.fit(pool[features], pool[target])
            predicted[:, j] = model.predict(test[features])
        return predicted

    # ------------------------------------------------------------------ cross-validation
    def cross_validate(self, arms=None, seeds=None):
        arms = list(arms or self.settings["arms"])
        seeds = list(seeds or self.settings["seeds"])
        say(f"cross-validation: {len(arms)} arms, seeds {seeds}, {self.settings['folds']} grouped folds, "
            f"device {self._device()}")
        for arm in arms:
            for seed in (seeds[:1] if arm == "xgboost" else seeds):
                for fold, train, validation, test in self.folds(seed):
                    path = self.cv_dir / arm / f"seed{seed}_fold{fold}.json"
                    if path.exists():
                        continue
                    started = time.perf_counter()
                    if arm == "xgboost":
                        predicted, info = self._fit_predict_xgboost(train, validation, test), {}
                    else:
                        model, stats, info = self._fit_neural(arm, train, validation, seed)
                        predicted = self._predict_neural(model, stats, test)
                    record = dict(
                        arm=arm, seed=seed, fold=fold, seconds=time.perf_counter() - started, **info,
                        train_protocols=int(train["protocol_key"].nunique()),
                        protocol_key=test["protocol_key"].tolist(),
                        ageing_cycle=test["ageing_cycle"].tolist(),
                        actual=test[list(TARGETS)].to_numpy(float).tolist(),
                        predicted=np.asarray(predicted, float).tolist(),
                    )
                    self._write_json(path, record)
                    error = self._score(record)["nrmse_mean"]
                    say(f"  {arm:28} seed {seed} fold {fold}: nRMSE {error:6.2f} %  "
                        f"epochs {info.get('epochs', '-')}  {record['seconds']:.0f} s")

    @staticmethod
    def _write_json(path, record):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f".tmp{os.getpid()}")
        temporary.write_text(json.dumps(record), encoding="utf-8")
        temporary.replace(path)

    def _target_range(self):
        return np.ptp(self.data()[list(TARGETS)].to_numpy(float), axis=0)

    def _score(self, record):
        """nRMSE in percent of each target's range over the whole dataset, plus per-protocol detail."""
        from scipy.stats import spearmanr

        span = self._target_range()
        actual, predicted = np.array(record["actual"]), np.array(record["predicted"])
        keys, cycles = np.array(record["protocol_key"]), np.array(record["ageing_cycle"])
        error = predicted - actual
        nrmse = np.sqrt((error ** 2).mean(0)) / span * 100
        total = ((actual - actual.mean(0)) ** 2).sum(0)
        r2 = 1 - (error ** 2).sum(0) / np.where(total == 0, np.nan, total)
        per_protocol = [
            float(np.mean(np.sqrt((error[keys == key] ** 2).mean(0)) / span * 100))
            for key in np.unique(keys)
        ]
        last = cycles == cycles.max()
        rank = (float(spearmanr(actual[last, 0], predicted[last, 0]).correlation)
                if last.sum() > 2 else float("nan"))
        return dict(
            nrmse_mean=float(nrmse.mean()), **{f"nrmse_{t}": float(v) for t, v in zip(TARGETS, nrmse)},
            r2_mean=float(np.nanmean(r2)), protocol_median=float(np.median(per_protocol)),
            protocol_worst=float(np.max(per_protocol)), rank_capacity=rank,
        )

    # ------------------------------------------------------------------ hyperparameter search
    FAMILIES = {
        "cenn-certified": dict(arm="ams-cenn-tuned", kind="cenn", variant="C1-BoundedTau"),
        "cenn-relaxed": dict(arm="ams-cenn-relaxed-tuned", kind="cenn", variant="ABL-GateParam-Unbounded"),
        "gru": dict(arm="gru-tuned", kind="gru", variant=None),
    }

    @staticmethod
    def sample(family, trial):
        """Trial `trial` of a family, reproducible from its index alone."""
        rng = np.random.default_rng([sum(map(ord, family)), trial])
        pick = lambda options: options[int(rng.integers(len(options)))]
        training = dict(
            learning_rate=float(10 ** rng.uniform(np.log10(3e-4), np.log10(3e-3))),
            batch_size=pick([32, 64, 128]),
            weight_decay=pick([0.0, 1e-5, 1e-4]),
        )
        if family == "gru":
            overrides = dict(hidden=pick([64, 128, 256]), layers=pick([1, 2, 3]),
                             dropout=pick([0.0, 0.1, 0.2, 0.3]))
        else:
            overrides = dict(
                linear_skip=pick([False, False, True]),
                multiscale_mode=pick(["none", "parallel_ensemble"]),
                head_type=pick(["linear", "mlp"]),
                hidden_dim=pick([32, 64, 128]),
                num_layers=pick([2, 4, 6]),
                K=pick([2, 4, 8]),
                neighborhood=pick([3, 5, 7]),
                dropout=pick([0.0, 0.1, 0.25]),
                spectral_rho=pick([0.9, 0.99]),
            )
            # certified keeps the gate bounded below one; the lower bound is free to move
            overrides["alpha_min"] = pick([0.0, 0.25, 0.5]) if family == "cenn-certified" else 0.0
        return dict(overrides=overrides, training=training)

    def tune(self, family, first, last):
        """Random search. Each trial is scored on the validation folds; test folds are only recorded."""
        base = self.FAMILIES[family]
        tuning = self.settings["tuning"]
        splits = [s for s in self.folds(tuning["seed"]) if s[0] < tuning["folds"]]
        for trial in range(first, last):
            path = self.cv_dir / "tuning" / family / f"trial{trial:03d}.json"
            if path.exists():
                continue
            drawn = self.sample(family, trial)
            custom = dict(kind=base["kind"], variant=base["variant"], **drawn)
            started = time.perf_counter()
            folds = []
            try:
                for fold, train, validation, test in splits:
                    model, stats, info = self._fit_neural(None, train, validation, tuning["seed"],
                                                          custom=custom, budget=tuning)
                    score = lambda frame: self._score(dict(
                        protocol_key=frame["protocol_key"].tolist(),
                        ageing_cycle=frame["ageing_cycle"].tolist(),
                        actual=frame[list(TARGETS)].to_numpy(float).tolist(),
                        predicted=self._predict_neural(model, stats, frame).tolist()))["nrmse_mean"]
                    folds.append(dict(fold=fold, validation=score(validation), test=score(test), **info))
                failed = None
            except Exception as error:  # an invalid combination should not stop the search
                failed = f"{type(error).__name__}: {error}"
            record = dict(family=family, trial=trial, **drawn, folds=folds, failed=failed,
                          validation=float(np.mean([f["validation"] for f in folds])) if folds and not failed else None,
                          test=float(np.mean([f["test"] for f in folds])) if folds and not failed else None,
                          seconds=time.perf_counter() - started)
            self._write_json(path, record)
            say(f"  tune {family} trial {trial:3d}: validation "
                f"{record['validation'] if record['validation'] is not None else float('nan'):6.2f} %  "
                f"test {record['test'] if record['test'] is not None else float('nan'):6.2f} %  "
                f"{record['seconds']:.0f} s  {failed or ''}")

    def tune_summarize(self, family):
        """Pick the trial with the lowest validation error and register it as an arm."""
        from .models import certified_spec, register_tuned

        base = self.FAMILIES[family]
        records = [json.loads(p.read_text(encoding="utf-8"))
                   for p in sorted((self.cv_dir / "tuning" / family).glob("trial*.json"))]
        valid = [r for r in records if r["validation"] is not None]
        if not valid:
            raise RuntimeError(f"No finished trials for {family}")
        table = pd.DataFrame([
            dict(trial=r["trial"], validation=r["validation"], test=r["test"], seconds=r["seconds"],
                 **r["overrides"], **r["training"]) for r in valid
        ]).sort_values("validation").reset_index(drop=True)
        table.to_csv(self.cv_dir / "tuning" / f"{family}_trials.csv", index=False)
        best = min(valid, key=lambda r: r["validation"])
        certified = base["kind"] == "cenn" and certified_spec(base["variant"], best["overrides"])
        register_tuned(base["arm"], dict(
            kind=base["kind"], variant=base["variant"], overrides=best["overrides"],
            training=best["training"],
            description=f"random search, best of {len(valid)} trials by validation error "
                        f"(trial {best['trial']}); certified contraction: {certified}",
        ))
        say("\n" + table.head(10).to_string(index=False, float_format=lambda v: f"{v:.4g}"))
        say(f"\nregistered {base['arm']} from trial {best['trial']} "
            f"({len(valid)} valid trials, {len(records) - len(valid)} failed)")
        return base["arm"]

    # ------------------------------------------------------------------ selection
    def summarize(self):
        from .models import describe, is_cellular, is_certified

        rows = []
        for folder in sorted(p for p in self.cv_dir.glob("*") if p.is_dir() and p.name != "tuning"):
            arm = folder.name
            by_seed = {}
            for path in folder.glob("seed*_fold*.json"):
                record = json.loads(path.read_text(encoding="utf-8"))
                by_seed.setdefault(record["seed"], []).append(record)
            complete = {s: r for s, r in by_seed.items() if len(r) == self.settings["folds"]}
            if not complete:
                continue
            scores = []
            for records in complete.values():
                pooled = {k: sum((r[k] for r in records), []) for k in
                          ("protocol_key", "ageing_cycle", "actual", "predicted")}
                scores.append(self._score(pooled))
            first = next(iter(complete.values()))
            mean = lambda key: float(np.mean([s[key] for s in scores]))
            rows.append(dict(
                arm=arm,
                family="xgboost" if arm == "xgboost" else ("cellular" if is_cellular(arm) else "reference"),
                certified=bool(arm != "xgboost" and is_certified(arm)),
                seeds=len(complete),
                nrmse_mean=mean("nrmse_mean"),
                nrmse_seed_min=float(np.min([s["nrmse_mean"] for s in scores])),
                nrmse_seed_max=float(np.max([s["nrmse_mean"] for s in scores])),
                **{f"nrmse_{t}": mean(f"nrmse_{t}") for t in TARGETS},
                r2_mean=mean("r2_mean"), protocol_median=mean("protocol_median"),
                protocol_worst=mean("protocol_worst"), rank_capacity=mean("rank_capacity"),
                parameters=first[0].get("parameters"),
                epochs=float(np.mean([r.get("epochs", np.nan) for rs in complete.values() for r in rs])),
                seconds_per_fit=float(np.mean([r["seconds"] for rs in complete.values() for r in rs])),
                description="original XGBoost pipeline" if arm == "xgboost" else describe(arm),
            ))
        if not rows:
            raise RuntimeError(f"No complete cross-validation results in {self.cv_dir}")
        table = pd.DataFrame(rows).sort_values("nrmse_mean").reset_index(drop=True)
        table.to_csv(self.cv_dir / "summary.csv", index=False)

        def best(mask):
            subset = table[mask]
            return None if subset.empty else subset.iloc[0]["arm"]

        selection = dict(
            criterion="lowest mean nRMSE over targets under protocol-grouped cross-validation, "
                      "averaged over seeds",
            best_overall=best(table["family"] != "xgboost"),
            best_cellular=best(table["family"] == "cellular"),
            best_certified=best(table["certified"]),
            best_reference=best(table["family"] == "reference"),
        )
        (self.cv_dir / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
        shown = table[["arm", "family", "certified", "seeds", "nrmse_mean", "nrmse_seed_min",
                       "nrmse_seed_max", "protocol_worst", "rank_capacity", "parameters", "epochs"]]
        say("\n" + shown.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
        say("\n" + json.dumps(selection, indent=2))
        return table, selection

    # ------------------------------------------------------------------ final models
    def models_dir(self, arm):
        return self.project_config.resolve_path(f"surrogate/models_{arm.replace('-', '_')}")

    def train_final(self, arm, seeds=None):
        import torch

        from .models import describe, is_certified, training_overrides
        from .predictor import NeuralEnsemble

        seeds = list(seeds or self.settings["seeds"])
        data = self.data()
        protocols = np.array(sorted(data["protocol_key"].unique()))
        order = np.random.default_rng(self.settings["final_split_random_state"]).permutation(len(protocols))
        held = set(protocols[order[:self.settings["final_validation_protocols"]]])
        train = data[~data["protocol_key"].isin(held)]
        validation = data[data["protocol_key"].isin(held)]
        out = self.models_dir(arm)
        out.mkdir(parents=True, exist_ok=True)
        say(f"final {arm}: {train['protocol_key'].nunique()} training protocols, "
            f"{len(held)} for early stopping, seeds {seeds} -> {out}")

        model_files, members, stats = {}, [], None
        for seed in seeds:
            started = time.perf_counter()
            model, stats, info = self._fit_neural(arm, train, validation, seed)
            filename = f"member_seed{seed}.pt"
            torch.save({k: v.cpu() for k, v in model.state_dict().items()}, out / filename)
            model_files[f"seed{seed}"] = filename
            validation_error = self._score(dict(
                protocol_key=validation["protocol_key"].tolist(),
                ageing_cycle=validation["ageing_cycle"].tolist(),
                actual=validation[list(TARGETS)].to_numpy(float).tolist(),
                predicted=self._predict_neural(model, stats, validation).tolist(),
            ))["nrmse_mean"]
            members.append(dict(seed=seed, **info, seconds=time.perf_counter() - started,
                                validation_nrmse=validation_error))
            say(f"  seed {seed}: best epoch {info['best_epoch']}/{info['epochs']}, "
                f"validation nRMSE {validation_error:.2f} %")

        summary_path = self.cv_dir / "summary.csv"
        cross_validation = None
        if summary_path.exists():
            table = pd.read_csv(summary_path)
            match = table[table["arm"] == arm]
            if not match.empty:
                cross_validation = json.loads(match.iloc[0].to_json())

        features = self.featurizer.feature_columns()
        from .models import architecture

        metadata = {
            "backend": "torch",
            "architecture": architecture(arm),
            "model_type": "AMS-CeNN" if arm.startswith(("ams-cenn", "cenn")) else arm.upper(),
            "arm": arm,
            "description": describe(arm),
            "certified_contraction": is_certified(arm),
            "input_type": "formation_voltage_current_charge_trajectory",
            "trajectory_feature_count": self.count,
            "signals": list(SIGNALS),
            "cycle_scale": self.cycle_scale,
            "feature_columns": features,
            "target_columns": list(TARGETS),
            "model_files": model_files,
            "normalization": {k: np.asarray(v, float).tolist() for k, v in stats.items()},
            "ensemble": "mean of the members; one member per seed",
            "final_fit_rule": (
                f"trained on {train['protocol_key'].nunique()} protocols with early stopping on "
                f"{len(held)} held-out protocols (random state "
                f"{self.settings['final_split_random_state']}); the reported accuracy is the "
                "cross-validation below, not this validation set"
            ),
            "validation_protocols": sorted(held),
            "members": members,
            "training": {**{k: self.settings[k] for k in ("epochs", "patience", "batch_size", "learning_rate")},
                         **training_overrides(arm)},
            "cross_validation": cross_validation,
            "dataset": str(self.dataset_path.name),
            "rows": len(data),
            "train_rows": len(train),
            "validation_rows": len(validation),
            "torch_version": torch.__version__,
        }
        # A fixed input and the output it must give, so the saved model can be verified after shipping.
        ensemble = NeuralEnsemble(out, metadata)
        probe = data[data["protocol_key"] == protocols[0]].sort_values("ageing_cycle")
        cycles = [int(probe["ageing_cycle"].iloc[0]), int(probe["ageing_cycle"].iloc[-1])]
        frame = probe[probe["ageing_cycle"].isin(cycles)][features]
        metadata["reference_check"] = {
            "protocol_key": str(protocols[0]),
            "ageing_cycles": cycles,
            "expected": ensemble.predict_frame(frame).astype(float).tolist(),
        }
        with (out / "metadata.json").open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)
        say(f"  saved {len(model_files)} members and metadata.json")
        return out
