"""PyTorch data pipeline for the GRU and custom models.

``FormationDataset`` has one item per formation protocol, so any split by item
index (``torch.utils.data.Subset``) is automatically a protocol-level split:
a protocol's ageing cycles can never leak between train, validation and test.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


def formation_sequences(frame, featurizer, stride=1):
    """Flat feature columns -> ``(rows, time, channels)`` array, average-pooled by ``stride``."""
    sequence = np.stack(
        [
            frame[featurizer.channel_columns(channel)].to_numpy(dtype=np.float32)
            for channel in featurizer.CHANNELS
        ],
        axis=-1,
    )
    if stride > 1:
        rows, length, channels = sequence.shape
        length -= length % stride
        sequence = sequence[:, :length].reshape(rows, length // stride, stride, channels).mean(axis=2)
    return sequence


class FormationDataset(Dataset):
    """Item ``i`` = (sequence ``[time, channels]``, targets ``[cycles, targets]``) of protocol ``i``.

    ``self.cycles`` lists the ageing cycles shared by all items. A target is
    NaN where a protocol has no value at that cycle; losses must mask it.
    """

    def __init__(self, frame, featurizer, targets, stride=1):
        first_rows = frame.drop_duplicates("protocol_key")
        self.protocol_keys = first_rows["protocol_key"].tolist()
        cycles = np.sort(frame["ageing_cycle"].unique())
        self.cycles = torch.tensor(cycles, dtype=torch.float32)
        self.sequences = torch.from_numpy(formation_sequences(first_rows, featurizer, stride))

        values = np.full((len(self.protocol_keys), len(cycles), len(targets)), np.nan, np.float32)
        protocol_index = frame["protocol_key"].map(
            {key: index for index, key in enumerate(self.protocol_keys)}
        ).to_numpy()
        cycle_index = np.searchsorted(cycles, frame["ageing_cycle"])
        values[protocol_index, cycle_index] = frame[list(targets)].to_numpy(dtype=np.float32)
        self.targets = torch.from_numpy(values)

    def __len__(self):
        return len(self.protocol_keys)

    def __getitem__(self, index):
        return self.sequences[index], self.targets[index]


def split_off(indices, fraction, generator):
    """Shuffle ``indices`` and split off ``fraction`` of them: ``(kept, held_out)``."""
    shuffled = indices[torch.randperm(len(indices), generator=generator)]
    n_held_out = round(len(shuffled) * fraction)
    return shuffled[n_held_out:], shuffled[:n_held_out]


def protocol_folds(n_protocols, n_folds, validation_fraction, seed):
    """Protocol-level K-fold with torch; yields ``{"train", "validation", "test"}`` index tensors."""
    generator = torch.Generator().manual_seed(seed)
    folds = torch.tensor_split(torch.randperm(n_protocols, generator=generator), n_folds)
    for k, test in enumerate(folds):
        rest = torch.cat([fold for i, fold in enumerate(folds) if i != k])
        train, validation = split_off(rest, validation_fraction, generator)
        yield {"train": train, "validation": validation, "test": test}
