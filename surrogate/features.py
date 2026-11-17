import math

import numpy as np


class TrajectoryFeaturizer:
    #: Formation signals, in the order they appear in the feature vector.
    CHANNELS = ("voltage", "current")

    def __init__(self, surrogate_config):
        config = surrogate_config.data
        self.feature_count = config["data"]["trajectory_feature_count"]
        self.signals = config["signals"]

    def extract(self, formation_solution):
        time_s = self._solution_variable(formation_solution, self.signals["time"])
        voltage = self._solution_variable(formation_solution, self.signals["voltage"])
        current = self._solution_variable(formation_solution, self.signals["current"])
        time_s, voltage, current = self._unique_time(time_s, voltage, current)
        time_h = (time_s - time_s[0]) / 3600

        signals = dict(zip(self.CHANNELS, (voltage, current)))
        features = {}
        for name in self.CHANNELS:
            uniform = self._uniform_time(time_h, signals[name], self.feature_count)
            features.update(zip(self.channel_columns(name), map(float, uniform)))
        return features

    def channel_columns(self, name):
        """Feature columns of one signal, e.g. ``voltage_000`` ... ``voltage_255``."""
        return [f"{name}_{index:03d}" for index in range(self.feature_count)]

    def feature_columns(self):
        columns = [column for name in self.CHANNELS for column in self.channel_columns(name)]
        return columns + ["ageing_cycle"]

    def _solution_variable(self, solution, name):
        return np.asarray(solution[name].entries, dtype=float).reshape(-1)

    def _unique_time(self, time_s, *series):
        unique_time, indices = np.unique(time_s, return_index=True)
        return [unique_time] + [values[indices] for values in series]

    def _uniform_time(self, time_h, values, count):
        if len(time_h) == 1 or math.isclose(time_h[-1], time_h[0]):
            return np.full(count, values[-1])
        target = np.linspace(time_h[0], time_h[-1], count)
        return np.interp(target, time_h, values)
