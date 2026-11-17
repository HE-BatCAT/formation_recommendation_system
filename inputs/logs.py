import logging


def setup_logging(filename="formation_system.log"):
    """Append INFO-level logs from every entry-point script to one file."""
    logging.basicConfig(
        filename=filename,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        filemode="a",
    )
