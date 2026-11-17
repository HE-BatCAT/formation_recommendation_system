"""Formation protocols and the PyBaMM experiments built from ``simulation.toml``."""

from dataclasses import dataclass

import pybamm


@dataclass(frozen=True)
class FormationCandidate:
    """One formation protocol; the four decision variables of the MCO and the dataset grid."""

    x1_charge_rate: float     # formation charge C-rate
    x2_discharge_rate: float  # formation discharge C-rate
    x3_rest_minutes: float    # rest between charge and discharge [min]
    x4_num_cycles: int        # number of formation cycles


class ExperimentFactory:
    """Builds every ``pybamm.Experiment`` the project runs, from ``simulation.toml``."""

    def __init__(self, simulation_config):
        self.config = simulation_config.data

    def pre_step(self):
        """Fixed conditioning before formation (``[experiments] pre_step``), one cycle."""
        return pybamm.Experiment([tuple(self.config["experiments"]["pre_step"])])

    def default_formation_candidate(self):
        """Protocol from ``[formation] default_*``; used by ``run_simulation.py``."""
        formation = self.config["formation"]
        return FormationCandidate(
            x1_charge_rate=formation["default_charge_rate"],
            x2_discharge_rate=formation["default_discharge_rate"],
            x3_rest_minutes=formation["default_rest_minutes"],
            x4_num_cycles=formation["default_cycles"],
        )

    def formation(self, candidate=None):
        """Charge at x1 C -> rest x3 min -> discharge at x2 C, repeated x4 times."""
        formation = self.config["formation"]
        candidate = candidate or self.default_formation_candidate()
        cycle = (
            f"Charge at {candidate.x1_charge_rate}C until "
            f"{formation['charge_voltage']}V",
            f"Rest for {candidate.x3_rest_minutes} minutes",
            f"Discharge at {candidate.x2_discharge_rate}C until "
            f"{formation['discharge_voltage']}V",
        )
        return pybamm.Experiment([cycle] * candidate.x4_num_cycles)

    def aging(self, cycles):
        """The ``[experiments] aging`` cycle repeated ``cycles`` times."""
        cycle = tuple(self.config["experiments"]["aging"])
        return pybamm.Experiment([cycle] * cycles)

    def capacity_check(self):
        """Slow ``[capacity_check]`` charge/discharge for capacity and resistance check-ups."""
        check = self.config["capacity_check"]
        cycle = (
            f"Charge at {check['c_rate']}C until {check['upper_voltage']}V",
            f"Discharge at {check['c_rate']}C until {check['lower_voltage']}V",
        )
        return pybamm.Experiment([cycle])
