"""Entry point: simulate the surrogate grid with PyBaMM and write the dataset CSV.

    python -m surrogate.create_dataset                         # skip if the CSV exists
    python -m surrogate.create_dataset --force                 # regenerate
    python -m surrogate.create_dataset --force --limit-candidates 2   # quick test
"""

import argparse
import logging

from inputs import ProjectConfig, setup_logging
from simulations import (
    BatterySimulator,
    ExperimentFactory,
    FormationPipeline,
    ParameterLoader,
    SolutionAnalyzer,
)
from surrogate.dataset import SurrogateDatasetBuilder
from surrogate.features import TrajectoryFeaturizer

setup_logging()
logger = logging.getLogger(__name__)

class DatasetCommand:
    def run(self):
        arguments = self._arguments()
        config = ProjectConfig()
        # Wire the simulation layer exactly as the MCO does, so features match.
        parameters = ParameterLoader(config.parameters_path).load()
        pipeline = FormationPipeline(
            ExperimentFactory(config.simulation),
            BatterySimulator(config.simulation, parameters),
        )
        analyzer = SolutionAnalyzer(config.simulation)
        featurizer = TrajectoryFeaturizer(config.surrogate)
        builder = SurrogateDatasetBuilder(
            config.surrogate,
            config,
            pipeline,
            analyzer,
            featurizer,
        )

        logging.info(f"Creating surrogate dataset at: {builder.dataset_path}")
        builder.create(force=arguments.force, limit_candidates=arguments.limit_candidates)

    def _arguments(self):
        parser = argparse.ArgumentParser(description="Create the surrogate dataset")
        parser.add_argument("--force", action="store_true", help="overwrite an existing dataset")
        parser.add_argument("--limit-candidates", type=int,
                            help="simulate only the first N grid protocols")
        return parser.parse_args()


if __name__ == "__main__":
    DatasetCommand().run()
