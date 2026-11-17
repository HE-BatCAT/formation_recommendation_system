"""GRU surrogate over the formation voltage / current trajectory.

A small GRU encodes the formation sequence once per protocol; the encoding is
paired with each ageing cycle and an MLP head predicts all four targets.

With only ~128 protocols the network is kept small (32 hidden units,
``num_layers`` from inputs/models/gru.toml) and the sequence is average-pooled
256 -> 64 steps (``sequence_stride = 4``).
Regularisation (dropout, weight decay, input noise, early stopping) and
protocol-level cross-validation come from ``TorchEstimator``.
"""

import torch
from torch import nn

from surrogate.estimators.torch_estimator import TorchEstimator
from surrogate.targets import TARGETS


class FormationGRU(nn.Module):
    """GRU encoder + MLP head; the cycle is an input, so any cycle can be predicted."""

    def __init__(self, n_channels, hidden_size, num_layers, head_hidden_size, dropout, n_targets):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # last hidden state + mean over time + ageing-cycle scalar
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_size + 1, head_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_size, n_targets),
        )

    def forward(self, sequence, cycles):
        """``sequence (B, time, channels)``, ``cycles (B, M)`` -> ``(B, M, targets)``."""
        outputs, hidden = self.gru(sequence)
        # Last layer's final hidden state + mean of its outputs over time.
        encoding = torch.cat([hidden[-1], outputs.mean(dim=1)], dim=-1)
        # Same protocol encoding for every requested cycle; the head adds the cycle.
        encoding = encoding.unsqueeze(1).expand(-1, cycles.shape[1], -1)
        return self.head(torch.cat([encoding, cycles.unsqueeze(-1)], dim=-1))


class GRUEstimator(TorchEstimator):
    """Trains ``FormationGRU`` with the shared ``TorchEstimator`` loop."""

    #: Used only for keys missing from inputs/models/gru.toml.
    DEFAULTS = {
        "sequence_stride": 4,
        "hidden_size": 32,
        "num_layers": 1,
        "head_hidden_size": 32,
        "dropout": 0.2,
        "input_noise_std": 0.02,
        "weight_decay": 1e-3,
        "max_epochs": 50,
        "patience": 10,
    }

    def build_network(self):
        p = self.params
        return FormationGRU(
            n_channels=len(self.featurizer.CHANNELS),
            hidden_size=p["hidden_size"],
            num_layers=p["num_layers"],
            head_hidden_size=p["head_hidden_size"],
            dropout=p["dropout"],
            n_targets=len(TARGETS),
        )
