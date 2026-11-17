"""Formation recommendation by multi-criteria optimisation (MCO).

For every candidate in ``[grid]`` of ``mco.toml``: simulate formation with
PyBaMM, then get ageing metrics at the first and last cycle either from a
trained surrogate (fast) or from full PyBaMM ageing (slow). Each objective is
min-max normalised across candidates into a 0-1 penalty; the weighted sum is
the score, and lower is better.
"""

import csv
import itertools
import logging
import math
import time
from dataclasses import asdict

from simulations import FormationCandidate
from surrogate.targets import TARGETS

logger = logging.getLogger(__name__)


class CandidateGrid:
    """Cartesian product of the ``[grid]`` lists in ``mco.toml``."""

    def __init__(self, grid_config):
        self.grid = grid_config

    def candidates(self):
        """Yield one ``FormationCandidate`` per combination of grid values."""
        keys = list(self.grid)
        for values in itertools.product(*(self.grid[key] for key in keys)):
            yield FormationCandidate(**dict(zip(keys, values)))


class CandidateScorer:
    """Weighted sum of min-max normalised penalties (``[weights]`` in ``mco.toml``)."""

    def __init__(self, weights):
        self.weights = weights

    def rank(self, results):
        """Add a ``score`` to every result and return them sorted best (lowest) first.

        Normalisation is relative to the evaluated candidates, so scores are only
        comparable within one run.
        """
        # Every candidate's value of each objective, for the min-max range.
        objective_values = {
            "formation_time_h": [row["formation_time_h"] for row in results],
            "final_capacity": [row["capacity_final"] for row in results],
            "capacity_fade": [row["capacity_fade"] for row in results],
            "plating_growth": [row["plating_growth"] for row in results],
            "sei_growth": [row["sei_growth"] for row in results],
            "resistance_growth": [row["resistance_growth"] for row in results],
        }
        for result in results:
            penalties = {
                "formation_time_h": self._lower(
                    result["formation_time_h"], objective_values["formation_time_h"]
                ),
                "final_capacity": self._higher(
                    result["capacity_final"], objective_values["final_capacity"]
                ),
                "capacity_fade": self._lower(
                    result["capacity_fade"], objective_values["capacity_fade"]
                ),
                "plating_growth": self._lower(
                    result["plating_growth"], objective_values["plating_growth"]
                ),
                "sei_growth": self._lower(
                    result["sei_growth"], objective_values["sei_growth"]
                ),
                "resistance_growth": self._lower(
                    result["resistance_growth"], objective_values["resistance_growth"]
                ),
            }
            result["score"] = sum(
                self.weights[key] * penalties[key]
                for key in self.weights
            )
        return sorted(results, key=lambda row: row["score"])

    def _lower(self, value, values):
        """Penalty for lower-is-better: 0 at the minimum, 1 at the maximum."""
        minimum, maximum = min(values), max(values)
        return 0.0 if math.isclose(minimum, maximum) else (value - minimum) / (maximum - minimum)

    def _higher(self, value, values):
        """Penalty for higher-is-better: 0 at the maximum, 1 at the minimum."""
        minimum, maximum = min(values), max(values)
        return 0.0 if math.isclose(minimum, maximum) else (maximum - value) / (maximum - minimum)


class ResultRepository:
    """Writes result rows as CSV files into ``results_dir``."""

    def __init__(self, results_dir):
        self.results_dir = results_dir

    def save(self, filename, rows):
        """Write ``rows`` (list of dicts) to ``results_dir/filename``; nothing if empty."""
        if not rows:
            return
        self.results_dir.mkdir(parents=True, exist_ok=True)
        path = self.results_dir / filename
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


class FormationOptimizer:
    """Evaluates, scores and ranks every MCO candidate; saves the results."""

    def __init__(
        self,
        mco_config,
        pipeline,
        analyzer,
        results_dir,
        surrogate_predictor=None,
        trajectory_featurizer=None,
    ):
        self.config = mco_config.data
        self.pipeline = pipeline
        self.analyzer = analyzer
        self.grid = CandidateGrid(self.config["grid"])
        self.scorer = CandidateScorer(self.config["weights"])
        self.repository = ResultRepository(results_dir)
        self.surrogate_predictor = surrogate_predictor
        self.trajectory_featurizer = trajectory_featurizer

    def run(self):
        """Evaluate all candidates and return the ``top_n`` ranked results.

        A failing candidate is logged and skipped; only if all fail is an error raised.
        """
        recommendation = self.config["recommendation"]
        candidates = list(self.grid.candidates())
        use_surrogate = self._use_surrogate(recommendation["mode"])
        mode = (
            f"surrogate ({self.surrogate_predictor.model_name})" if use_surrogate else "PyBaMM ageing"
        )
        logger.info(f"Evaluating {len(candidates)} candidates using {mode}")

        self.pipeline.pre_step()
        results = []
        failures = []
        for index, candidate in enumerate(candidates, start=1):
            logger.info(f"Candidate {index}/{len(candidates)}: {candidate}")
            started = time.perf_counter()
            try:
                result = self._evaluate(candidate, recommendation, use_surrogate)
                result["runtime_s"] = time.perf_counter() - started
                results.append(result)
                logger.info(
                    f"  formation={result['formation_time_h']:.2f} h, "
                    f"final capacity={result['capacity_final']:.6f} A.h"
                )
            except Exception as error:
                failures.append({**asdict(candidate), "error": str(error)})
                logger.exception(f"  failed: {error}")

        if not results:
            self.repository.save("failed_candidates.csv", failures)
            raise RuntimeError("All recommendation candidates failed")

        ranked = self.scorer.rank(results)
        top_n = recommendation["top_n"]
        self.repository.save("all_candidates.csv", ranked)
        self.repository.save("top_candidates.csv", ranked[:top_n])
        self.repository.save("failed_candidates.csv", failures)
        self._print_top(ranked[:top_n])
        return ranked[:top_n]

    def _use_surrogate(self, mode):
        """Resolve ``[recommendation] mode`` (auto / surrogate / pybamm) to a bool."""
        if mode == "pybamm" or self.surrogate_predictor is None:
            return False
        if mode == "surrogate":
            self.surrogate_predictor.require_trained()  # raises with the training command
            return True
        # auto: use the surrogate if trained, otherwise fall back to PyBaMM ageing
        if self.surrogate_predictor.available():
            return True
        logger.warning(
            f"Surrogate '{self.surrogate_predictor.model_name}' is not trained; "
            f"falling back to PyBaMM ageing. Train it with: "
            f"python -m surrogate.train_models --model {self.surrogate_predictor.model_name}"
        )
        return False

    def _evaluate(self, candidate, recommendation, use_surrogate):
        """Formation (always PyBaMM) + ageing metrics (surrogate or PyBaMM) for one candidate."""
        formation_solution = self.pipeline.formation(candidate)
        result = asdict(candidate)
        result["formation_time_h"] = self.pipeline.formation_time_h(formation_solution)

        if use_surrogate:
            features = self.trajectory_featurizer.extract(formation_solution)
            predictions = self.surrogate_predictor.predict(
                features,
                recommendation["prediction_cycles"],
            )
            metrics = self._metrics_from_predictions(predictions)
        else:
            aging_solution = self.pipeline.aging(
                formation_solution, recommendation["aging_cycles"]
            )
            metrics = self._metrics_from_solution(aging_solution)

        result.update(metrics)
        return result

    def _metrics_from_solution(self, solution):
        return self._metrics_from_pairs(self.analyzer.first_and_last_metrics(solution))

    def _metrics_from_predictions(self, predictions):
        """(start, final) = first and last of ``prediction_cycles``."""
        pairs = {
            key: (float(predictions[key].iloc[0]), float(predictions[key].iloc[-1]))
            for key in TARGETS
        }
        return self._metrics_from_pairs(pairs)

    def _metrics_from_pairs(self, pairs):
        """``<key>_start/_final/_delta`` plus ``capacity_fade`` / ``<key>_growth`` = |final - start|."""
        result = {}
        for key in TARGETS:
            start, final = pairs[key]
            result[f"{key}_start"] = start
            result[f"{key}_final"] = final
            result[f"{key}_delta"] = final - start
            result[f"{key}_growth" if key != "capacity" else "capacity_fade"] = abs(final - start)
        return result

    def _print_top(self, rows):
        """Log the top recommendations as a table (to formation_system.log)."""
        logger.info("\nRecommended formation protocols")
        logger.info(
            f"{'Rank':>4} {'Score':>8} {'Charge C':>9} {'Discharge C':>12} "
            f"{'Rest':>8} {'Cycles':>7} {'Form h':>9} {'Final cap':>10}"
        )
        logger.info("-" * 86)
        for rank, row in enumerate(rows, start=1):
            logger.info(
                f"{rank:>4} {row['score']:>8.3f} {row['x1_charge_rate']:>9.3g} "
                f"{row['x2_discharge_rate']:>12.3g} {row['x3_rest_minutes']:>8.1f} "
                f"{row['x4_num_cycles']:>7} {row['formation_time_h']:>9.2f} "
                f"{row['capacity_final']:>10.3f}"
            )
