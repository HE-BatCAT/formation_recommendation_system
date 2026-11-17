"""Pre-step -> formation -> ageing, shared by every script that simulates a protocol.

Used by dataset creation, the MCO, single simulations and protocol validation,
so the order of experiments and the state hand-over live in one place.
"""


class FormationPipeline:
    """Runs the fixed pre-step once, then formation and ageing from the right states."""

    def __init__(self, experiment_factory, simulator):
        self.experiments = experiment_factory
        self.simulator = simulator
        self._pre_step = None  # cached pre-step solution

    def pre_step(self):
        """Run the fixed pre-step once and reuse it for every formation."""
        if self._pre_step is None:
            self._pre_step = self.simulator.run_solver("pre-step", self.experiments.pre_step())
        return self._pre_step

    def formation(self, candidate, label=None):
        """Formation of ``candidate`` starting from the end of the pre-step."""
        experiment = self.experiments.formation(candidate)
        return self._solve(experiment, self.pre_step().last_state, label)

    def aging(self, formation, cycles, label=None):
        """``cycles`` ageing cycles starting from the end of ``formation``."""
        experiment = self.experiments.aging(cycles)
        return self._solve(experiment, formation.last_state, label)

    @staticmethod
    def formation_time_h(formation):
        """Duration of a formation solution in hours (an MCO objective)."""
        return (formation.t[-1] - formation.t[0]) / 3600

    def _solve(self, experiment, last_state, label):
        """With a label, log the step and its runtime; otherwise solve quietly."""
        if label:
            return self.simulator.run_solver(label, experiment, last_state)
        return self.simulator.solve(experiment, last_state)
