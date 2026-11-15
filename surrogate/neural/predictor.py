"""Inference for a saved neural surrogate: an ensemble of identically built members.

The saved directory holds ``metadata.json`` (architecture, normalization, feature order) and one
``state_dict`` file per member. Predictions are the member mean, in physical units.
"""
import numpy as np
import pandas as pd


class NeuralEnsemble:
    def __init__(self, models_dir, metadata):
        import torch

        from .models import build_saved

        self.torch = torch
        self.metadata = metadata
        self.count = int(metadata["trajectory_feature_count"])
        self.signals = list(metadata["signals"])
        self.targets = list(metadata["target_columns"])
        self.cycle_scale = float(metadata["cycle_scale"])
        norm = metadata["normalization"]
        self.mu_x = torch.tensor(norm["mu_x"], dtype=torch.float32)
        self.sd_x = torch.tensor(norm["sd_x"], dtype=torch.float32)
        self.mu_y = torch.tensor(norm["mu_y"], dtype=torch.float32)
        self.sd_y = torch.tensor(norm["sd_y"], dtype=torch.float32)
        self.members = []
        for filename in metadata["model_files"].values():
            model = build_saved(metadata, len(self.signals) + 1, len(self.targets), self.count)
            state = torch.load(models_dir / filename, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            self.members.append(model.eval())

    # ------------------------------------------------------------------ inputs
    def _traces(self, frame):
        """(rows, count, signals) raw formation traces from a frame holding the feature columns."""
        blocks = [
            frame[[f"{signal}_{index:03d}" for index in range(self.count)]].to_numpy(np.float32)
            for signal in self.signals
        ]
        return self.torch.from_numpy(np.stack(blocks, axis=-1))

    def _network_input(self, traces, cycles):
        cycle = (cycles / self.cycle_scale).reshape(-1, 1, 1).expand(-1, self.count, 1)
        return (self.torch.cat([traces, cycle], dim=-1) - self.mu_x) / self.sd_x

    def _cycles(self, frame):
        return self.torch.tensor(frame["ageing_cycle"].to_numpy(np.float32))

    # ------------------------------------------------------------------ outputs
    def member_predictions(self, frame):
        """(members, rows, targets) in physical units."""
        x = self._network_input(self._traces(frame), self._cycles(frame))
        with self.torch.no_grad():
            out = self.torch.stack([m(x) for m in self.members])
        return (out * self.sd_y + self.mu_y).numpy()

    def predict_frame(self, frame):
        """(rows, targets) ensemble mean in physical units."""
        return self.member_predictions(frame).mean(axis=0)

    def spread_frame(self, frame):
        """(rows, targets) standard deviation across members; a rough disagreement signal."""
        return self.member_predictions(frame).std(axis=0)

    def jacobian(self, frame):
        """Exact derivative of every target with respect to every formation feature.

        Returns (rows, targets, signals * count); the last axis follows the feature column order
        (all voltage samples, then current, then q_signed), in physical units per feature unit.
        """
        traces = self._traces(frame).clone().requires_grad_(True)
        x = self._network_input(traces, self._cycles(frame))
        out = self.torch.stack([m(x) for m in self.members]).mean(dim=0) * self.sd_y + self.mu_y
        rows = []
        for j in range(len(self.targets)):
            (grad,) = self.torch.autograd.grad(out[:, j].sum(), traces, retain_graph=True)
            rows.append(grad.permute(0, 2, 1).reshape(len(frame), -1))
        return self.torch.stack(rows, dim=1).numpy()

    def to_dataframe(self, values, cycle_numbers):
        predictions = pd.DataFrame({"ageing_cycle": list(cycle_numbers)})
        for j, target in enumerate(self.targets):
            predictions[target] = values[:, j].astype(float)
        return predictions
