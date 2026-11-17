"""Access to the project's TOML configuration files under ``inputs/``.

Every entry-point script builds one ``ProjectConfig`` and hands its parts to
the classes that need them, so no class reads a TOML file on its own:

    simulation.toml    -> ProjectConfig.simulation   (PyBaMM model, experiments, outputs)
    mco.toml           -> ProjectConfig.mco          (recommendation grid and weights)
    surrogate.toml     -> ProjectConfig.surrogate    (dataset, features, splits, plots)
    parameters.toml    -> ProjectConfig.parameters_path (read by ParameterLoader)
    models/<name>.toml -> ProjectConfig.model_config(name) (surrogate hyper-parameters)
"""

import tomllib
from pathlib import Path


class TomlConfig:
    """One parsed TOML file; ``data`` is the full nested dict."""

    def __init__(self, path):
        self.path = Path(path)
        with self.path.open("rb") as file:
            self.data = tomllib.load(file)

    def section(self, name):
        """Top-level table ``[name]`` as a dict (``KeyError`` if missing)."""
        return self.data[name]


class ProjectConfig:
    """All project configuration, with paths resolved from the project root."""

    def __init__(self, project_root=None):
        # Default root: the folder that contains ``inputs/`` (parent of this file's folder).
        self.project_root = (
            Path(project_root)
            if project_root is not None
            else Path(__file__).resolve().parents[1]
        )
        inputs_dir = self.project_root / "inputs"
        self.models_config_dir = inputs_dir / "models"
        self.simulation = TomlConfig(inputs_dir / "simulation.toml")
        self.mco = TomlConfig(inputs_dir / "mco.toml")
        self.surrogate = TomlConfig(inputs_dir / "surrogate.toml")
        self.parameters_path = inputs_dir / "parameters.toml"

    def model_config(self, name):
        """Hyper-parameters from ``inputs/models/<name>.toml``."""
        return TomlConfig(self.models_config_dir / f"{name}.toml").data

    def resolve_path(self, value):
        """Absolute paths are kept; relative ones are taken from the project root."""
        path = Path(value)
        return path if path.is_absolute() else self.project_root / path
