"""Forecast the rest of an ageing trajectory from its first measured cycles.

The model reads the last ``lookback`` cycles of the four indicators and returns the next ``block`` cycles
in one pass. Longer horizons are reached by sliding the window over its own output (block rollout):
from 64 measured cycles, cycle 500 is reached in seven passes.
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def build_forecaster(arm, lookback, block, indicators, custom=None):
    """A registry model with ``block * indicators`` outputs, reshaped to (batch, block, indicators).

    custom: {"kind", "variant", "overrides"} builds an unregistered model (a search trial) instead.
    """
    import torch.nn as nn

    from surrogate.neural.models import build, build_custom

    class Forecaster(nn.Module):
        def __init__(self):
            super().__init__()
            if custom is None:
                self.core = build(arm, indicators, block * indicators, lookback)
            else:
                self.core = build_custom(custom["kind"], custom.get("variant"), custom.get("overrides", {}),
                                         indicators, block * indicators, lookback)

        def forward(self, x):
            return self.core(x).reshape(x.size(0), block, indicators)

    return Forecaster()


def block_rollout(model, window, steps, block, device="cpu", chunk=512):
    """window (batch, lookback, indicators), standardized -> (batch, steps, indicators)."""
    import torch

    out = []
    with torch.no_grad():
        for start in range(0, len(window), chunk):
            x = torch.as_tensor(np.asarray(window[start:start + chunk], np.float32), device=device)
            parts = []
            for _ in range(math.ceil(steps / block)):
                predicted = model(x)
                parts.append(predicted)
                x = torch.cat([x[:, block:, :], predicted], dim=1)[:, -x.size(1):, :]
            out.append(torch.cat(parts, dim=1)[:, :steps].cpu().numpy())
    return np.concatenate(out, axis=0)


class AgeingForecaster:
    def __init__(self, models_dir):
        self.models_dir = Path(models_dir)
        self.metadata_path = self.models_dir / "metadata.json"
        self._members = None
        self._metadata = None

    def available(self):
        return self.metadata_path.exists()

    def forecast(self, measured, until_cycle=None):
        """measured: the indicators of consecutive ageing cycles starting at cycle 1, as a DataFrame with
        the indicator columns or an array (cycles, indicators). At least ``lookback`` cycles are needed.

        Returns a DataFrame with one row per forecast cycle, from the first unmeasured cycle to
        ``until_cycle``.
        """
        self._load()
        meta = self._metadata
        indicators, lookback, block = meta["indicators"], meta["lookback"], meta["block"]
        values = (measured[indicators].to_numpy(float) if isinstance(measured, pd.DataFrame)
                  else np.asarray(measured, float))
        if values.ndim != 2 or values.shape[1] != len(indicators):
            raise ValueError(f"Expected (cycles, {len(indicators)}) values for {indicators}")
        if len(values) < lookback:
            raise ValueError(f"The forecaster needs at least {lookback} measured cycles, got {len(values)}")
        until_cycle = int(until_cycle or meta["aging_cycles"])
        steps = until_cycle - len(values)
        if steps <= 0:
            raise ValueError("until_cycle must lie beyond the measured cycles")
        mu, sd = np.array(meta["normalization"]["mu"]), np.array(meta["normalization"]["sd"])
        window = ((values[-lookback:] - mu) / sd)[None]
        rolled = np.stack([block_rollout(m, window, steps, block)[0] for m in self._members])
        forecast = rolled.mean(axis=0) * sd + mu
        frame = pd.DataFrame({"cycle": np.arange(len(values) + 1, until_cycle + 1)})
        for j, name in enumerate(indicators):
            frame[name] = forecast[:, j]
        return frame

    def _load(self):
        if self._members is not None:
            return
        if not self.available():
            raise FileNotFoundError(f"Forecaster metadata not found: {self.metadata_path}")
        import torch

        with self.metadata_path.open("r", encoding="utf-8") as file:
            meta = json.load(file)
        members = []
        for filename in meta["model_files"].values():
            model = build_forecaster(meta["arm"], meta["lookback"], meta["block"], len(meta["indicators"]),
                                     custom=meta.get("architecture"))
            model.load_state_dict(torch.load(self.models_dir / filename, map_location="cpu",
                                             weights_only=True))
            members.append(model.eval())
        self._members, self._metadata = members, meta
