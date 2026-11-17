"""Configuration (``ProjectConfig``) and logging set-up shared by all entry points."""

from inputs.config import ProjectConfig
from inputs.logs import setup_logging

__all__ = ["ProjectConfig", "setup_logging"]
