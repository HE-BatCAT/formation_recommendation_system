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
    def __init__(self):
        self.config = ProjectConfig()

    def run(self):
        parameter_values = ParameterLoader(self.config.parameters_path).load()
        pipeline = FormationPipeline(
            ExperimentFactory(self.config.simulation),
            BatterySimulator(self.config.simulation, parameter_values),
        )
        analyzer = SolutionAnalyzer(self.config.simulation)
        predictor = SurrogatePredictor(self.config.surrogate, self.config)
        featurizer = TrajectoryFeaturizer(self.config.surrogate)
        results_dir = self.config.resolve_path(
            self.config.mco.section("recommendation")["results_dir"]
        )

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
