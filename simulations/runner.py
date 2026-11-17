"""PyBaMM model/solver construction and solving of one experiment."""

import time

import pybamm

import logging

logger = logging.getLogger(__name__)


class BatterySimulator:
    """DFN model + IDAKLU solver configured from ``simulation.toml``.

    It only solves experiments; which experiments to run is decided by
    ``ExperimentFactory`` / ``FormationPipeline`` and results are read by
    ``SolutionAnalyzer``.
    """

    def __init__(self, simulation_config, parameter_values):
        self.config = simulation_config.data
        self.parameter_values = parameter_values

    def build_model(self):
        """A fresh DFN with the degradation sub-models in ``[model.options]``."""
        return pybamm.lithium_ion.DFN(options=self.config["model"]["options"])

    def build_solver(self):
        solver = self.config["solver"]
        return pybamm.IDAKLUSolver(
            rtol=solver["relative_tolerance"],
            atol=solver["absolute_tolerance"],
        )

    def solve(self, experiment, last_state=None):
        """Solve ``experiment``; ``last_state`` continues from a previous solution's end state."""
        simulation = pybamm.Simulation(
            self.build_model(),
            experiment=experiment,
            parameter_values=self.parameter_values,
            solver=self.build_solver(),
        )
        return simulation.solve(starting_solution=last_state)

    def run_solver(self, label, experiment, last_state=None):
        """``solve`` with the step name and its runtime written to the log."""
        logger.info(f"------Running {label}------")
        started = time.perf_counter()
        solution = self.solve(experiment, last_state)
        elapsed = time.perf_counter() - started
        logger.info(f"{label.capitalize()} elapsed in {elapsed:.2f} seconds\n")
        return solution
