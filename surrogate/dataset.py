"""Build the surrogate training dataset with PyBaMM.

For every protocol in ``[grid]`` of ``surrogate.toml``: formation (from the
shared pre-step state) -> formation features -> ageing -> targets at each
``training_ageing_mesh`` cycle. The CSV has one row per (protocol, ageing
cycle); all rows of a protocol share the same formation features.
"""

import csv
import itertools
import time
from dataclasses import asdict

import numpy as np
import pandas as pd

from simulations import FormationCandidate
from surrogate.targets import TARGETS

import logging

logger = logging.getLogger(__name__)

class SurrogateDatasetBuilder:
    def __init__(
        self,
        surrogate_config,
        project_config,
        pipeline,
        analyzer,
        featurizer,
    ):
        self.config = surrogate_config.data
        self.project_config = project_config
        self.pipeline = pipeline
        self.analyzer = analyzer
        self.featurizer = featurizer
        self.dataset_path = project_config.resolve_path(self.config["paths"]["dataset"])
        self.failed_path = project_config.resolve_path(
            self.config["paths"]["failed_candidates"]
        )

    def create(self, force=False, limit_candidates=None):
        """Simulate the grid and write the CSV; skipped if it exists unless ``force``.

        ``limit_candidates`` keeps only the first N protocols (quick pipeline test).
        """
        if self.dataset_path.exists() and not force:
            logger.info(f"Dataset already exists; skipping creation: {self.dataset_path}")
            return self.dataset_path

        candidates = list(self._candidates())
        if limit_candidates is not None:
            candidates = candidates[:limit_candidates]

        self.pipeline.pre_step()
        aging_cycles = self.config["data"]["aging_cycles"]
        mesh = self.config["data"]["training_ageing_mesh"]
        rows = []
        failures = []

        for index, candidate in enumerate(candidates, start=1):
            logging.info(f"Dataset candidate {index}/{len(candidates)}: {candidate}")
            started = time.perf_counter()
            try:
                formation = self.pipeline.formation(candidate)
                aging = self.pipeline.aging(formation, aging_cycles)
                features = self.featurizer.extract(formation)
                by_cycle = self.analyzer.metrics_at_cycles(aging, TARGETS, mesh)
                for cycle_number, targets in by_cycle.items():
                    # Skip cycles where any target is NaN/inf (e.g. variable not output).
                    if not all(np.isfinite(value) for value in targets.values()):
                        continue
                    rows.append(
                        {
                            "protocol_key": self._protocol_key(candidate),
                            **asdict(candidate),
                            **features,
                            "ageing_cycle": cycle_number,
                            **targets,
                        }
                    )
                logger.info(f"  completed in {time.perf_counter() - started:.2f} s")
            except Exception as error:
                # A solver failure loses only this protocol; it is logged to failed_candidates.csv.
                failures.append({**asdict(candidate), "error": str(error)})
                logger.exception(f"  failed: {error}")

        self.dataset_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(self.dataset_path, index=False)
        self._write_failures(failures)
        logger.info(f"Saved {len(rows)} rows to {self.dataset_path}")
        return self.dataset_path

    def _candidates(self):
        """Cartesian product of the ``[grid]`` lists as ``FormationCandidate``s."""
        grid = self.config["grid"]
        keys = list(grid)
        for values in itertools.product(*(grid[key] for key in keys)):
            yield FormationCandidate(**dict(zip(keys, values)))

    def _protocol_key(self, candidate):
        """Unique protocol id; splits group rows by it so cycles never leak across splits."""
        return (
            f"x1={candidate.x1_charge_rate}|x2={candidate.x2_discharge_rate}|"
            f"x3={candidate.x3_rest_minutes}|x4={candidate.x4_num_cycles}"
        )

    def _write_failures(self, rows):
        if not rows:
            return
        self.failed_path.parent.mkdir(parents=True, exist_ok=True)
        with self.failed_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
