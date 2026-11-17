"""Formation trajectory -> fixed-length surrogate input features.

Each formation signal (voltage, current) is resampled at
``trajectory_feature_count`` (256) evenly spaced times between the start and
end of formation, by linear interpolation. Formations of different duration
therefore all give the same number of features; the duration itself is not a
feature (it is an MCO objective instead).

Feature columns: ``voltage_000 ... voltage_255, current_000 ... current_255``
(+ ``ageing_cycle`` for row-wise models such as XGBoost).
"""

import math

import numpy as np


class TrajectoryFeaturizer:
    #: Formation signals, in the order they appear in the feature vector.
    CHANNELS = ("voltage", "current")

    def __init__(self, surrogate_config):
        config = surrogate_config.data
        self.feature_count = config["data"]["trajectory_feature_count"]
        self.signals = config["signals"]  # channel -> PyBaMM variable name

    def extract(self, formation_solution):
        """``{column: value}`` features of one formation solution."""
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
        """All model input columns: every channel's samples, then ``ageing_cycle``."""
        columns = [column for name in self.CHANNELS for column in self.channel_columns(name)]
        return columns + ["ageing_cycle"]

    def _solution_variable(self, solution, name):
        return np.asarray(solution[name].entries, dtype=float).reshape(-1)

    def _unique_time(self, time_s, *series):
        """Drop repeated time stamps (PyBaMM repeats them at step boundaries)."""
        unique_time, indices = np.unique(time_s, return_index=True)
        return [unique_time] + [values[indices] for values in series]

    def _uniform_time(self, time_h, values, count):
        """``count`` samples evenly spaced in time (linear interpolation, no averaging)."""
        if len(time_h) == 1 or math.isclose(time_h[-1], time_h[0]):
            return np.full(count, values[-1])
        target = np.linspace(time_h[0], time_h[-1], count)
        return np.interp(target, time_h, values)
