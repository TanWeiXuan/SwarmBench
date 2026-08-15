from __future__ import annotations

from pathlib import Path

from .assignment import AssignmentController
from .defend import DefendController
from .greedy_value import GreedyValueController
from .potential_field import PotentialFieldController
from .rush import RushController

BASELINE_NAMES = ("rush", "defend", "greedy_value", "assignment", "potential_field")


def baseline_path(name: str) -> Path:
    if name not in BASELINE_NAMES:
        raise ValueError(f"unknown baseline {name!r}; choose from {', '.join(BASELINE_NAMES)}")
    return Path(__file__).with_name(f"{name}.py")


__all__ = [
    "AssignmentController",
    "BASELINE_NAMES",
    "DefendController",
    "GreedyValueController",
    "PotentialFieldController",
    "RushController",
    "baseline_path",
]

