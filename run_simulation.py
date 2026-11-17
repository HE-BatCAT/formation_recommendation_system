import logging

from inputs import ProjectConfig, setup_logging
from simulations import (
    BatterySimulator,
    ExperimentFactory,
    FormationPipeline,
    ParameterLoader,
    ResultPlotter,
    SolutionAnalyzer,
)

setup_logging()
logger = logging.getLogger(__name__)


class IndividualSimulationRunner:
    def __init__(self):
        self.config = ProjectConfig()

    def run(self):
        parameter_values = ParameterLoader(self.config.parameters_path).load()
        experiments = ExperimentFactory(self.config.simulation)
        simulator = BatterySimulator(self.config.simulation, parameter_values)
        pipeline = FormationPipeline(experiments, simulator)
        analyzer = SolutionAnalyzer(self.config.simulation)
        run_config = self.config.simulation.section("run")
        candidate = experiments.default_formation_candidate()

        logger.info(f"Formation choice: {candidate}\n")
        
        formation = pipeline.formation(candidate, label="formation")
        formation_time_h = pipeline.formation_time_h(formation)
        logger.info(f"Formation simulation time: {formation_time_h:.2f} hours\n")

        capacity_check = experiments.capacity_check()
        formation_check = simulator.run_solver(
            label="post-formation C/20 check",
            experiment=capacity_check,
            last_state=formation.last_state,
        )

        aging_cycles = run_config["aging_cycles"]
        aging = pipeline.aging(formation, aging_cycles, label="aging")

        aging_check = simulator.run_solver(
            label="post-aging C/20 check",
            experiment=capacity_check,
            last_state=aging.last_state,
        )

        self._print_checkup_table(analyzer, formation_check, aging_check)

        plots_dir = self.config.resolve_path(run_config["plots_dir"])
        saved_plots = ResultPlotter(analyzer, plots_dir).plot_outputs(aging)
        logger.info("\nSaved plots:")
        for path in saved_plots:
            logger.info(path)

    def _print_checkup_table(self, analyzer, formation_check, aging_check):
        formation_capacity = analyzer.measured_capacity(formation_check)
        aging_capacity = analyzer.measured_capacity(aging_check)
        formation_resistance = analyzer.measured_resistance(formation_check)
        aging_resistance = analyzer.measured_resistance(aging_check)
        logger.info(
            "C/20 diagnostic checks\n"
            f"{'State':<24} {'Capacity [A.h]':>16} {'ECM [Ohm]':>14}\n"
            f"{'-' * 24} {'-' * 16} {'-' * 14}\n"
            f"{'After formation':<24} "
            f"{formation_capacity:>16.3f} {formation_resistance:>14.6f}\n"
            f"{'After aging':<24} "
            f"{aging_capacity:>16.3f} {aging_resistance:>14.6f}\n"
            f"{'Change':<24} "
            f"{aging_capacity - formation_capacity:>16.3f} "
            f"{aging_resistance - formation_resistance:>14.6f}\n"
        )


if __name__ == "__main__":
    try:
        IndividualSimulationRunner().run()
    except Exception as e:
        logger.exception(f"An error occurred: {e}")
        