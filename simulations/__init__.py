"""PyBaMM simulation layer: parameters, experiments, solving and result extraction.

    ParameterLoader     inputs/parameters.toml -> pybamm.ParameterValues
    ExperimentFactory   builds pre-step / formation / ageing / check-up experiments
    FormationCandidate  one formation protocol (x1..x4)
    BatterySimulator    DFN model + IDAKLU solver, runs one experiment
    FormationPipeline   pre-step -> formation -> ageing with state hand-over
    SolutionAnalyzer    per-cycle ageing metrics from a solution
    ResultPlotter       plots of every configured [[outputs]] metric
"""

from simulations.analysis import SolutionAnalyzer
from simulations.experiments import ExperimentFactory, FormationCandidate
from simulations.parameters import ParameterLoader
from simulations.pipeline import FormationPipeline
from simulations.plotting import ResultPlotter
from simulations.runner import BatterySimulator

__all__ = [
    "BatterySimulator",
    "ExperimentFactory",
    "FormationCandidate",
    "FormationPipeline",
    "ParameterLoader",
    "ResultPlotter",
    "SolutionAnalyzer",
]
