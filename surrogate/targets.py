"""Ageing metrics predicted by the surrogates and ranked by the MCO.

Each key must match an ``[[outputs]]`` key in ``inputs/simulation.toml``.
The order is the column order of every target array and prediction table.
"""

TARGETS = ("capacity", "plating", "sei", "resistance")
