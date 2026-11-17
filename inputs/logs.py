"""Logging set-up: every entry point appends to one log file in the working directory."""

import logging


def setup_logging(filename="formation_system.log"):
    """Append INFO-level logs from every entry-point script to one file.

    Training metrics, simulation timings and recommendations all go here; the
    console only shows short progress messages.
    """
    logging.basicConfig(
        filename=filename,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        filemode="a",
    )
