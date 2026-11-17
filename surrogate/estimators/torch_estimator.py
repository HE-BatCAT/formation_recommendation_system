"""Training loop shared by the PyTorch surrogates (GRU, custom).

A subclass only implements ``build_network()``: an ``nn.Module`` mapping
standardised ``sequence (batch, time, channels)`` and ``cycles (batch, M)`` to
standardised targets ``(batch, M, n_targets)``. This class handles the rest:

* ``FormationDataset`` + protocol-level K-fold via ``torch_data``;
* per-fold standardisation fitted on training protocols only;
* AdamW, input-noise augmentation, gradient clipping and early stopping on
  the validation protocols (best epoch restored);
* a final model refitted on all protocols (minus validation for early stopping).
"""

import copy
import logging
import math

import matplotlib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from surrogate.estimators.base import SurrogateEstimator
from surrogate.estimators.torch_data import (
    FormationDataset,
    formation_sequences,
    protocol_folds,
    split_off,
)
from surrogate.evaluation import score_split
from surrogate.targets import TARGETS

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

#: Training defaults for every torch model; overridden by the estimator's
#: DEFAULTS, then by inputs/models/<name>.toml.
DEFAULT_TRAINING = {
    "sequence_stride": 1,
    "input_noise_std": 0.0,
    "learning_rate": 1e-3,
    "weight_decay": 0.0,
    "batch_size": 16,
    "max_epochs": 1000,
    "patience": 100,
    "min_delta": 1e-4,
    "grad_clip_norm": 1.0,
    "seed": 0,
    "device": "cpu",
}


class Scaling(nn.Module):
    """Standardisation of inputs, log-cycle and targets, kept as buffers so it is saved."""

    def __init__(self, n_channels, n_targets):
        super().__init__()
        self.register_buffer("x_mean", torch.zeros(n_channels))
        self.register_buffer("x_std", torch.ones(n_channels))
        self.register_buffer("cycle_mean", torch.zeros(()))
        self.register_buffer("cycle_std", torch.ones(()))
        self.register_buffer("y_mean", torch.zeros(n_targets))
        self.register_buffer("y_std", torch.ones(n_targets))

    @torch.no_grad()
    def fit(self, sequences, cycles, targets):
        """Fit on training protocols only; NaN targets are ignored."""
        self.x_mean.copy_(sequences.mean(dim=(0, 1)))
        self.x_std.copy_(sequences.std(dim=(0, 1)).clamp_min(1e-8))
        log_cycles = torch.log1p(cycles)
        self.cycle_mean.copy_(log_cycles.mean())
        self.cycle_std.copy_(log_cycles.std().nan_to_num(1.0).clamp_min(1e-8))
        y = targets.reshape(-1, targets.shape[-1])
        mean = y.nanmean(dim=0)
        self.y_mean.copy_(mean)
        self.y_std.copy_((y - mean).pow(2).nanmean(dim=0).sqrt().clamp_min(1e-8))

    def inputs(self, sequence, cycles):
        return (
            (sequence - self.x_mean) / self.x_std,
            (torch.log1p(cycles) - self.cycle_mean) / self.cycle_std,
        )

    def targets(self, y):
        return (y - self.y_mean) / self.y_std

    def inverse(self, y):
        return y * self.y_std + self.y_mean


class ScaledModel(nn.Module):
    """Raw inputs in, standardised targets out; adds input noise while training."""

    def __init__(self, network, n_channels, n_targets, input_noise_std):
        super().__init__()
        self.scaling = Scaling(n_channels, n_targets)
        self.network = network
        self.input_noise_std = input_noise_std

    def forward(self, sequence, cycles):
        sequence, cycles = self.scaling.inputs(sequence, cycles)
        if self.training and self.input_noise_std > 0:
            sequence = sequence + self.input_noise_std * torch.randn_like(sequence)
        return self.network(sequence, cycles)


class TorchEstimator(SurrogateEstimator):
    #: Network hyper-parameter defaults; ``inputs/models/<name>.toml`` overrides them.
    DEFAULTS = {}

    def __init__(self, project_config, name, models_dir, plots_dir):
        super().__init__(project_config, name, models_dir, plots_dir)
        # Precedence (last wins): training defaults < model DEFAULTS < TOML file.
        self.params = {**DEFAULT_TRAINING, **self.DEFAULTS, **self.params}
        self.device = torch.device(self.params["device"])
        self.model, self.history = None, None

    def build_network(self):
        """Return the ``nn.Module``; implemented by each torch model."""
        raise NotImplementedError

    # ------------------------------------------------------------ training
    def train(self):
        """K-fold CV (fit + score per fold), then the final model; returns the CV summary."""
        dataset = FormationDataset(
            self.load_dataset(), self.featurizer, TARGETS, self.params["sequence_stride"]
        )
        logger.info(f"[{self.name}] {len(dataset)} protocols, params {self.params}")

        metric_rows, prediction_rows = [], []
        folds = protocol_folds(
            len(dataset),
            self.split["cv_folds"],
            self.split["validation_fraction_of_train"],
            self.split["random_state"],
        )
        for fold, splits in enumerate(folds, start=1):
            model, _ = self._fit(dataset, splits["train"], splits["validation"])
            for split, indices in splits.items():
                metrics, predictions = self._score(dataset, indices, model, fold, split)
                metric_rows += metrics
                if split == "test":
                    prediction_rows += predictions
            logger.info(f"[{self.name}] fold {fold} done")

        # Final model for prediction: all protocols, validation kept for early stopping.
        generator = torch.Generator().manual_seed(self.split["random_state"])
        train, validation = split_off(
            torch.arange(len(dataset)), self.split["validation_fraction_of_train"], generator
        )
        self.model, self.history = self._fit(dataset, train, validation)
        self._save_model()
        if self.plotting:
            self._plot_learning_curve()
        return self.save_results(
            metric_rows,
            prediction_rows,
            {
                "model_type": type(self).__name__,
                "model_file": self._model_file().name,
                "params": self.params,
                "weights": sum(p.numel() for p in self.model.parameters()),
                "final_best_epoch": self.history["best_epoch"],
            },
        )

    def _new_model(self):
        """Fresh network wrapped with (still unfitted) standardisation."""
        model = ScaledModel(
            self.build_network(),
            len(self.featurizer.CHANNELS),
            len(TARGETS),
            self.params["input_noise_std"],
        )
        return model.to(self.device)

    def _fit(self, dataset, train, validation):
        """Train on ``train`` protocols, early-stop on ``validation``; returns (model, history).

        The weights of the best validation epoch are restored at the end.
        """
        p = self.params
        torch.manual_seed(p["seed"])
        model = self._new_model()
        model.scaling.fit(dataset.sequences[train], dataset.cycles, dataset.targets[train])
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=p["learning_rate"], weight_decay=p["weight_decay"]
        )
        loader = DataLoader(
            Subset(dataset, train.tolist()),
            batch_size=p["batch_size"],
            shuffle=True,
            generator=torch.Generator().manual_seed(p["seed"]),
        )

        best_loss, best_state, best_epoch, waited = math.inf, None, 0, 0
        history = {"train": [], "validation": []}
        for epoch in range(1, p["max_epochs"] + 1):
            model.train()
            total = 0.0
            for sequence, targets in loader:
                optimizer.zero_grad()
                loss = self._loss(model, sequence, dataset.cycles, targets)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), p["grad_clip_norm"])
                optimizer.step()
                total += loss.item() * len(sequence)
            history["train"].append(total / len(train))

            if len(validation):
                monitor = self._evaluation_loss(model, dataset, validation)
            else:
                monitor = history["train"][-1]
            history["validation"].append(monitor)
            if monitor < best_loss - p["min_delta"]:
                best_loss, best_epoch, waited = monitor, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                waited += 1
                if waited >= p["patience"]:
                    break

        model.load_state_dict(best_state)
        model.eval()
        history["best_epoch"] = best_epoch
        logger.info(
            f"[{self.name}] stopped at epoch {epoch}, best epoch {best_epoch}, "
            f"validation loss {best_loss:.4g}"
        )
        return model, history

    def _loss(self, model, sequence, cycles, targets):
        """MSE on standardised targets, ignoring cycles a protocol has no value for."""
        cycles = cycles.expand(len(sequence), -1).to(self.device)
        predicted = model(sequence.to(self.device), cycles)
        expected = model.scaling.targets(targets.to(self.device))
        observed = ~torch.isnan(expected)
        return (predicted - expected)[observed].pow(2).mean()

    @torch.no_grad()
    def _evaluation_loss(self, model, dataset, indices):
        model.eval()
        return self._loss(
            model, dataset.sequences[indices], dataset.cycles, dataset.targets[indices]
        ).item()

    @torch.no_grad()
    def _predict(self, model, sequence, cycles):
        """Prediction in physical units: ``(batch, M, targets)``."""
        model.eval()
        output = model(sequence.to(self.device), cycles.to(self.device))
        return model.scaling.inverse(output).cpu()

    def _score(self, dataset, indices, model, fold, split):
        """Metric and prediction rows for ``indices``; (protocol, cycle) pairs with NaN are skipped."""
        cycles = dataset.cycles.expand(len(indices), -1)
        predicted = self._predict(model, dataset.sequences[indices], cycles).numpy()
        actual = dataset.targets[indices].numpy()
        observed = ~np.isnan(actual).any(axis=-1)  # (protocols, cycles) with all targets
        keys = np.asarray(dataset.protocol_keys)[indices.numpy()]
        return score_split(
            self.name,
            fold,
            split,
            np.broadcast_to(keys[:, None], observed.shape)[observed],
            cycles.numpy()[observed],
            actual[observed],
            predicted[observed],
        )

    # ---------------------------------------------------------- prediction
    def predict(self, frame):
        """One prediction per row of ``frame`` at that row's ``ageing_cycle``."""
        sequence = torch.from_numpy(
            formation_sequences(frame, self.featurizer, self.params["sequence_stride"])
        )
        cycles = torch.tensor(frame["ageing_cycle"].to_numpy(np.float32)).unsqueeze(1)
        predicted = self._predict(self.model, sequence, cycles)[:, 0]
        return pd.DataFrame(predicted.numpy(), index=frame.index, columns=list(TARGETS))

    # --------------------------------------------------------- persistence
    def _model_file(self):
        """``models_dir/<name>.pt``: params + state dict (network and scaling buffers)."""
        return self.models_dir / f"{self.name}.pt"

    def _save_model(self):
        self.models_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"params": self.params, "state": self.model.state_dict()}, self._model_file())
        logger.info(f"Saved {self.name} model to {self._model_file()}")

    def load_final_model(self):
        """Rebuild the network from the saved params (not the current TOML) and load its weights."""
        saved = torch.load(self._model_file(), map_location="cpu", weights_only=True)
        self.params = saved["params"]
        self.device = torch.device("cpu")
        self.model = self._new_model()
        self.model.load_state_dict(saved["state"])
        self.model.eval()

    def _plot_learning_curve(self):
        """Final model's train/validation loss per epoch -> plots_dir/learning_curve.png."""
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        figure, axis = plt.subplots(figsize=(9, 5))
        axis.plot(self.history["train"], linewidth=1.2, label="train")
        axis.plot(self.history["validation"], linestyle="--", linewidth=1.4, label="validation")
        axis.axvline(self.history["best_epoch"] - 1, color="grey", linestyle=":",
                     label="restored best epoch")
        axis.set_yscale("log")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("MSE (standardised targets)")
        axis.set_title(f"{self.name} final-model learning curve")
        axis.grid(True, alpha=0.3)
        axis.legend()
        figure.tight_layout()
        figure.savefig(self.plots_dir / "learning_curve.png", dpi=200)
        plt.close(figure)
