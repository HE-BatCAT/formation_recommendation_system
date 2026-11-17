"""Check a trained surrogate against PyBaMM for one formation protocol.

    python -m surrogate.validate_protocol --x1 0.1 --x2 0.5 --x3 30 --x4 3
    python -m surrogate.validate_protocol --x1 0.1 --x2 0.5 --x3 30 --x4 3 --model gru
    python -m surrogate.validate_protocol --x1 0.1 --x2 0.5 --x3 30 --x4 3 --cycles 1 100 500

Tests ``--model`` (xgboost, gru or custom); without it, the model selected in
``inputs/mco.toml`` (``[recommendation] surrogate_model``).
PyBaMM runs the pre-step, formation and ageing up to the largest requested
cycle; the surrogate predicts from the same formation. Per-target MAE and MSE,
plus the combined MAE and MSE over all four targets (raw units, averaged), go
to ``formation_system.log``.
"""

import argparse
import logging

import numpy as np
import pandas as pd

from inputs import ProjectConfig, setup_logging
from simulations import (
    BatterySimulator,
    ExperimentFactory,
    FormationCandidate,
    FormationPipeline,
    ParameterLoader,
    SolutionAnalyzer,
)
from surrogate.estimators import ESTIMATORS
from surrogate.features import TrajectoryFeaturizer
from surrogate.predictor import SurrogatePredictor
from surrogate.targets import TARGETS

setup_logging()
logger = logging.getLogger(__name__)


class ProtocolValidation:
    def __init__(self, arguments):
        self.config = ProjectConfig()
        self.candidate = FormationCandidate(arguments.x1, arguments.x2, arguments.x3, arguments.x4)
        self.training_mesh = self.config.surrogate.data["data"]["training_ageing_mesh"]
        self.cycles = sorted(arguments.cycles or self.training_mesh)
        model_name = arguments.model or self.config.mco.section("recommendation")["surrogate_model"]
        self.predictor = SurrogatePredictor(self.config, model_name)

    def run(self):
        self.predictor.require_trained()  # fail before the long PyBaMM run
        logger.info(
            f"===== Validating surrogate '{self.predictor.model_name}' on {self.candidate} ====="
        )
        self._warn_unseen_cycles()

        formation, actual = self._simulate()
        features = TrajectoryFeaturizer(self.config.surrogate).extract(formation)
        predicted = self.predictor.predict(features, self.cycles).set_index("ageing_cycle")

        comparison = self._comparison(actual, predicted)
        errors = self._errors(comparison)
        self._log(comparison, errors)
        return errors

    def _simulate(self):
        parameters = ParameterLoader(self.config.parameters_path).load()
        pipeline = FormationPipeline(
            ExperimentFactory(self.config.simulation),
            BatterySimulator(self.config.simulation, parameters),
        )
        formation = pipeline.formation(self.candidate, label="formation")
        aging = pipeline.aging(formation, max(self.cycles), label="aging")
        by_cycle = SolutionAnalyzer(self.config.simulation).metrics_at_cycles(
            aging, TARGETS, self.cycles
        )
        actual = pd.DataFrame.from_dict(by_cycle, orient="index").reindex(
            index=self.cycles, columns=list(TARGETS)
        )
        missing = actual.index[actual.isna().any(axis=1)].tolist()
        if missing:
            logger.warning(f"PyBaMM has no values at cycles {missing}; they are left out.")
        return formation, actual.dropna()

    def _comparison(self, actual, predicted):
        """Long table: one row per (target, cycle) with PyBaMM, surrogate and error."""
        rows = [
            {
                "target": target,
                "ageing_cycle": cycle,
                "pybamm": actual.loc[cycle, target],
                "surrogate": predicted.loc[cycle, target],
            }
            for target in TARGETS
            for cycle in actual.index
        ]
        table = pd.DataFrame(rows)
        table["error"] = table["surrogate"] - table["pybamm"]
        return table

    @staticmethod
    def _errors(comparison):
        """Per-target MAE/MSE over the cycles, then the combined row (raw units, averaged)."""
        grouped = comparison.groupby("target", sort=False)["error"]
        errors = pd.DataFrame({
            "mae": grouped.apply(lambda e: np.mean(np.abs(e))),
            "mse": grouped.apply(lambda e: np.mean(e**2)),
        })
        errors.loc["combined"] = errors.mean()
        return errors.reset_index().rename(columns={"index": "target"})

    def _warn_unseen_cycles(self):
        unseen = [cycle for cycle in self.cycles if cycle not in self.training_mesh]
        if unseen:
            logger.warning(
                f"Cycles {unseen} are not in training_ageing_mesh {self.training_mesh}; "
                "the surrogate never saw targets there."
            )

    def _log(self, comparison, errors):
        number = lambda value: f"{value:.6g}"
        logger.info(
            f"[{self.predictor.model_name}] surrogate vs PyBaMM per cycle:\n"
            + comparison.to_string(index=False, float_format=number)
        )
        logger.info(
            f"[{self.predictor.model_name}] errors over cycles {self.cycles} "
            "(combined = mean of the 4 targets, raw units):\n"
            + errors.to_string(index=False, float_format=number)
        )
        combined = errors.set_index("target").loc["combined"]
        print(
            f"{self.predictor.model_name} on {self.candidate}: combined MAE={combined['mae']:.6g}, "
            f"combined MSE={combined['mse']:.6g} (details in formation_system.log)"
        )


def main():
    parser = argparse.ArgumentParser(description="Compare a trained surrogate with PyBaMM")
    parser.add_argument("--x1", type=float, required=True, help="charge C-rate")
    parser.add_argument("--x2", type=float, required=True, help="discharge C-rate")
    parser.add_argument("--x3", type=float, required=True, help="rest minutes")
    parser.add_argument("--x4", type=int, required=True, help="number of formation cycles")
    parser.add_argument("--model", choices=list(ESTIMATORS),
                        help="surrogate to test (default: surrogate_model in mco.toml)")
    parser.add_argument("--cycles", nargs="+", type=int,
                        help="ageing cycles to compare (default: training_ageing_mesh)")
    try:
        ProtocolValidation(parser.parse_args()).run()
    except Exception as error:
        logger.exception(f"Validation failed: {error}")
        raise SystemExit(f"Validation failed: {error}")


if __name__ == "__main__":
    main()
