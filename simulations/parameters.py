"""Load ``inputs/parameters.toml`` into ``pybamm.ParameterValues``."""

import importlib
import tomllib

import pybamm


class ParameterLoader:
    """Converts typed ``[[parameter]]`` TOML entries into PyBaMM parameter values.

    Each entry is ``{name, type, value}``; later entries with the same name
    override earlier ones (a dict keeps the last value).
    """

    def __init__(self, parameter_path):
        self.parameter_path = parameter_path

    def load(self):
        with self.parameter_path.open("rb") as file:
            config = tomllib.load(file)

        parameters = {}
        for entry in config.get("parameter", []):
            parameters[entry["name"]] = self._convert(entry)
        return pybamm.ParameterValues(parameters)

    def _convert(self, entry):
        """Typed TOML value -> Python value; ``function`` is an import ``module:name``."""
        value_type = entry["type"].strip().lower()
        raw_value = entry["value"]
        if value_type == "function":
            # e.g. "pybamm.input.parameters.lithium_ion.Mohtat2020:graphite_ocp_PeymanMPM"
            module_name, function_name = raw_value.split(":", maxsplit=1)
            return getattr(importlib.import_module(module_name), function_name)
        if value_type == "list":
            return raw_value
        if value_type == "int":
            return int(raw_value)
        if value_type == "float":
            return float(raw_value)
        if value_type == "str":
            return raw_value
        raise ValueError(f"Unsupported parameter type '{value_type}'")
