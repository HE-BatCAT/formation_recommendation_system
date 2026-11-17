"""Custom PyTorch surrogate: PatchAttention, a channel-mixed patch Transformer.

PatchTST-style, except that voltage and current share one token per patch
(PatchTST keeps one token stream per channel). Full description: ``Model.md``.

Pipeline for one protocol (shapes for the tuned configuration in
``inputs/models/custom.toml``: ``sequence_stride = 1``, ``patch_len = 8``,
``d_model = 32``):

1. Optional input reduction by ``sequence_stride`` (average pooling, done in
   ``torch_data.formation_sequences``). A stride of 2 equals a one-level Haar
   approximation up to a sqrt(2) factor that standardisation absorbs, with
   the detail coefficients discarded. Stride 1 (no reduction) currently
   works best: 256 time steps.
2. Non-overlapping patches of ``patch_len`` steps: 256 / 8 = 32 patches.
   No padding: the length must be divisible by ``patch_len``.
3. Voltage and current of a patch form one token
   (8 x 2 = 16 values -> ``Linear(16, d_model)``) plus a learnable positional
   embedding.
4. ``n_blocks`` (2) pre-norm Transformer encoder blocks (``n_heads`` = 2)
   over the patch tokens, then a final LayerNorm.
5. Flattened tokens (32 x 32 = 1024) -> one linear layer that predicts every
   target at every training ageing cycle at once:
   ``(batch, n_cycles, n_targets)``.

Because the head predicts the fixed training cycle grid
(``training_ageing_mesh``), the network ignores the cycle values it is given.
The grid is stored in ``self.params["cycle_grid"]`` (so it is saved with the
model), and ``predict`` interpolates the grid predictions in log-cycle space
for any other cycle requested.

Run with:  python -m surrogate.train_models --model custom
"""

import logging

import numpy as np
import pandas as pd
import torch
from torch import nn

from surrogate.estimators.torch_data import formation_sequences
from surrogate.estimators.torch_estimator import TorchEstimator
from surrogate.targets import TARGETS

logger = logging.getLogger(__name__)


class PatchAttention(nn.Module):
    """Channel-mixed patch Transformer predicting all cycles in one step.

    Input ``(B, time_steps, n_channels)`` (standardised) ->
    output ``(B, n_cycles, n_targets)`` (standardised targets).
    """

    def __init__(
        self, time_steps, n_channels, patch_len, d_model, n_heads, ffn_size,
        n_blocks, dropout, n_cycles, n_targets,
    ):
        super().__init__()
        if time_steps % patch_len:
            raise ValueError(
                f"{time_steps} time steps are not divisible into patches of {patch_len}; "
                "change sequence_stride or patch_len (no padding is applied)"
            )
        self.patch_len = patch_len
        self.n_patches = time_steps // patch_len
        self.n_cycles, self.n_targets = n_cycles, n_targets

        # Patch embedding: one token = patch_len steps of every channel, flattened.
        self.embedding = nn.Linear(patch_len * n_channels, d_model)
        # Learnable positional embedding, one vector per patch position.
        self.position = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        nn.init.trunc_normal_(self.position, std=0.02)
        self.embedding_dropout = nn.Dropout(dropout)

        # Each block: x + MHA(LN(x)), then x + FFN(LN(x)) (pre-norm, GELU FFN).
        block = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm
        )
        # Pre-norm leaves the residual stream un-normalised, hence the final LayerNorm.
        # Nested tensors are a padding-mask optimisation; unused (and unsupported) here.
        self.encoder = nn.TransformerEncoder(
            block, num_layers=n_blocks, norm=nn.LayerNorm(d_model), enable_nested_tensor=False
        )
        # Flatten-head (as in PatchTST): every patch position keeps its own weights,
        # no pooling over tokens. Output = all cycles x all targets in one step.
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(dropout),
            nn.Linear(self.n_patches * d_model, n_cycles * n_targets),
        )

    def forward(self, sequence, cycles):
        """``sequence (B, time, channels)``, ``cycles (B, n_cycles)`` -> ``(B, n_cycles, targets)``.

        ``cycles`` must be the training cycle grid; only its length is checked.
        """
        if cycles.shape[1] != self.n_cycles:
            raise ValueError(
                f"network predicts a fixed grid of {self.n_cycles} cycles, got {cycles.shape[1]}"
            )
        batch, time_steps, channels = sequence.shape
        # (B, P, patch_len * channels): each token holds both channels of one patch
        # (time-major, so values alternate voltage, current, voltage, ...).
        patches = sequence.reshape(batch, self.n_patches, self.patch_len * channels)
        tokens = self.embedding_dropout(self.embedding(patches) + self.position)
        tokens = self.encoder(tokens)                                # (B, P, d_model)
        return self.head(tokens).reshape(batch, self.n_cycles, self.n_targets)


class CustomEstimator(TorchEstimator):
    """Trains ``PatchAttention`` with the shared ``TorchEstimator`` loop.

    Adds the cycle grid to the parameters (it sizes the head) and a ``predict``
    that interpolates between grid cycles.
    """

    #: Used only for keys missing from inputs/models/custom.toml (tuned values).
    DEFAULTS = {
        "sequence_stride": 1,  # no reduction; 2 = one-level Haar approximation (average pool)
        "patch_len": 8,
        "d_model": 32,
        "n_heads": 2,
        "ffn_size": 64,
        "n_blocks": 2,
        "dropout": 0.05,
    }

    def train(self):
        # The head size depends on the ageing-cycle grid; keep it in params so it
        # is saved with the model and restored by load_final_model().
        cycles = np.sort(self.load_dataset()["ageing_cycle"].unique())
        self.params["cycle_grid"] = [float(cycle) for cycle in cycles]
        return super().train()

    def build_network(self):
        p = self.params
        return PatchAttention(
            # Sequence length after pooling, the same as formation_sequences produces.
            time_steps=self.featurizer.feature_count // p["sequence_stride"],
            n_channels=len(self.featurizer.CHANNELS),
            patch_len=p["patch_len"],
            d_model=p["d_model"],
            n_heads=p["n_heads"],
            ffn_size=p["ffn_size"],
            n_blocks=p["n_blocks"],
            dropout=p["dropout"],
            n_cycles=len(p["cycle_grid"]),
            n_targets=len(TARGETS),
        )

    def predict(self, frame):
        """Predict the whole cycle grid, then interpolate (in log-cycle) at each row's cycle.

        Cycles on the grid are returned exactly; cycles between grid points are
        linearly interpolated in log(1 + cycle); cycles outside the grid are
        clamped to the nearest end (with a warning).
        """
        grid = torch.tensor(self.params["cycle_grid"], dtype=torch.float32)
        sequence = torch.from_numpy(
            formation_sequences(frame, self.featurizer, self.params["sequence_stride"])
        )
        # (rows, n_cycles, targets) in physical units.
        on_grid = self._predict(self.model, sequence, grid.expand(len(sequence), -1)).numpy()

        requested = frame["ageing_cycle"].to_numpy(np.float64)
        log_grid, log_requested = np.log1p(grid.numpy()), np.log1p(requested)
        outside = (log_requested < log_grid[0]) | (log_requested > log_grid[-1])
        if outside.any():
            logger.warning(
                f"[{self.name}] {outside.sum()} requested cycles outside the trained grid "
                f"[{grid[0]:.0f}, {grid[-1]:.0f}]; clamped to the nearest end"
            )
        predicted = np.stack(
            [
                [np.interp(log_requested[row], log_grid, on_grid[row, :, target])
                 for target in range(on_grid.shape[-1])]
                for row in range(len(frame))
            ]
        )
        return pd.DataFrame(predicted, index=frame.index, columns=list(TARGETS))
