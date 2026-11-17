"""Entry point: rank formation protocols with the MCO.

    python run_recommendation.py

Settings: inputs/mco.toml (grid, weights, mode, surrogate_model).
Results: mco/results/{all,top,failed}_candidates.csv; log: formation_system.log.
"""

import logging

from inputs import ProjectConfig, setup_logging
from mco import FormationOptimizer
from simulations import (
    BatterySimulator,
    ExperimentFactory,
    FormationPipeline,
    ParameterLoader,
    SolutionAnalyzer,
)
from surrogate import SurrogatePredictor, TrajectoryFeaturizer

setup_logging()
logger = logging.getLogger(__name__)

class RecommendationRunner:
    """Wires configuration, simulation, surrogate and MCO classes together."""

    def __init__(self):
        self.config = ProjectConfig()

    def run(self):
        """Run the MCO; returns the top-ranked candidates."""
        parameter_values = ParameterLoader(self.config.parameters_path).load()
        pipeline = FormationPipeline(
            ExperimentFactory(self.config.simulation),
            BatterySimulator(self.config.simulation, parameter_values),
        )
        analyzer = SolutionAnalyzer(self.config.simulation)
        recommendation = self.config.mco.section("recommendation")
        predictor = SurrogatePredictor(self.config, recommendation["surrogate_model"])
        featurizer = TrajectoryFeaturizer(self.config.surrogate)
        results_dir = self.config.resolve_path(recommendation["results_dir"])

        optimizer = FormationOptimizer(
            self.config.mco,
            pipeline,
            analyzer,
            results_dir,
            surrogate_predictor=predictor,
            trajectory_featurizer=featurizer,
        )
        return optimizer.run()


if __name__ == "__main__":
    RecommendationRunner().run()
