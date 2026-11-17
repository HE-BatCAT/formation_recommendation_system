import numpy as np


class SolutionAnalyzer:
    """Extracts per-cycle ageing metrics and diagnostics from a PyBaMM solution."""

    def __init__(self, simulation_config):
        config = simulation_config.data
        self.output_specs = config["outputs"]
        self.diagnostic_specs = config["diagnostics"]
        self.output_by_key = {spec["key"]: spec for spec in self.output_specs}

    # ------------------------------------------------------------------ cycles

    @staticmethod
    def experiment_cycles(solution):
        """Cycles with nonzero duration (drops empty/initialisation cycles)."""
        cycles = [
            cycle for cycle in solution.cycles
            if len(time := cycle["Time [s]"].entries) > 1 and time[-1] > time[0]
        ]
        if not cycles:
            raise ValueError("Solution contains no nonzero-duration experiment cycle")
        return cycles

    # ------------------------------------------------------------ single values

    @staticmethod
    def variable_end(cycle_or_step, variable_name):
        """Last value of ``variable_name``, or NaN if the model did not output it."""
        try:
            return float(cycle_or_step[variable_name].entries[-1])
        except KeyError:
            return np.nan

    @staticmethod
    def discharge_capacity(cycle):
        """Ah delivered in the cycle's last step (assumed to be the discharge)."""
        capacity = cycle.steps[-1]["Discharge capacity [A.h]"].entries
        return float(capacity[-1] - capacity[0])

    def metric_value(self, cycle, spec):
        if spec["aggregation"] == "discharge_step":
            return self.discharge_capacity(cycle)
        return self.variable_end(cycle, spec["variable"])

    # ------------------------------------------------------------------ series

    def metric_series(self, solution, key):
        """One value of metric ``key`` per experiment cycle."""
        spec = self.output_by_key[key]
        return [self.metric_value(cycle, spec) for cycle in self.experiment_cycles(solution)]

    def metrics_at_cycles(self, solution, keys, cycles):
        """``{cycle: {key: value}}`` for 1-based ageing cycles present in ``solution``.

        Only the requested cycles are evaluated, not the full trajectory.
        """
        experiment = self.experiment_cycles(solution)
        specs = [(key, self.output_by_key[key]) for key in keys]
        return {
            n: {key: self.metric_value(experiment[n - 1], spec) for key, spec in specs}
            for n in cycles
            if 1 <= n <= len(experiment)
        }

    def first_and_last_metrics(self, solution):
        """``{key: (first, last)}`` using the first and last finite value of each metric."""
        cycles = self.experiment_cycles(solution)
        values = {}
        for spec in self.output_specs:
            first = self._first_finite(spec, cycles)
            if first is None:
                raise ValueError(f"No finite values for {spec['variable']}")
            values[spec["key"]] = (first, self._first_finite(spec, reversed(cycles)))
        return values

    def _first_finite(self, spec, cycles):
        """First finite metric value scanning ``cycles`` in order (stops early)."""
        finite = (v for v in (self.metric_value(c, spec) for c in cycles) if np.isfinite(v))
        return next(finite, None)

    # ----------------------------------------------------------- check-ups

    def measured_capacity(self, check_solution):
        return self.discharge_capacity(self.experiment_cycles(check_solution)[-1])

    def measured_resistance(self, check_solution):
        last_step = self.experiment_cycles(check_solution)[-1].steps[-1]
        return self.variable_end(last_step, self.output_by_key["resistance"]["variable"])

    # ------------------------------------------------------------------ tables

    @staticmethod
    def _format_table(title, rows, label_width, value_width, value_format):
        lines = [
            title,
            f"{'Metric':<{label_width}} {'Start':>{value_width}} {'Final':>{value_width}}",
            f"{'-' * label_width} {'-' * value_width} {'-' * value_width}",
        ]
        lines += [
            f"{label:<{label_width}} {start:>{value_width}{value_format}} {final:>{value_width}{value_format}}"
            for label, start, final in rows
        ]
        return "\n".join(lines)

    def results_table(self, solution, cycle_count):
        metrics = self.first_and_last_metrics(solution)
        rows = [(spec["label"], *metrics[spec["key"]]) for spec in self.output_specs]
        return self._format_table(
            f"Ageing results after {cycle_count} cycles:", rows, 50, 12, ".3f"
        )

    def diagnostics_table(self, solution):
        cycles = self.experiment_cycles(solution)
        first, last = cycles[0], cycles[-1]
        rows, warnings = [], []
        for spec in self.diagnostic_specs:
            label, variable = spec["label"], spec["variable"]
            start, final = self.variable_end(first, variable), self.variable_end(last, variable)
            rows.append((label, start, final))
            low, high = spec["physical_range"]
            if np.isfinite(final) and not low <= final <= high:
                warnings.append(f"{label} outside [{low}, {high}]: {final:.6g}")

        table = self._format_table("Physical-state diagnostics:", rows, 48, 14, ".6g")
        if warnings:
            table += "\n\nPhysical-state warnings:\n" + "\n".join(f"- {w}" for w in warnings)
        return table
