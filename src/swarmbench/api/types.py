"""Small public value types shared by controllers and the engine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Team(str, Enum):
    A = "A"
    B = "B"

    @property
    def opponent(self) -> "Team":
        return Team.B if self is Team.A else Team.A


class DroneType(str, Enum):
    FAST = "FAST"
    SLOW = "SLOW"


class DroneStatus(str, Enum):
    ACTIVE = "ACTIVE"
    ELIMINATED = "ELIMINATED"
    SCORED = "SCORED"


class DestructionReason(str, Enum):
    INTERCEPTION = "INTERCEPTION"
    OBSTACLE_CRASH = "OBSTACLE_CRASH"


@dataclass(frozen=True, slots=True)
class DroneSpec:
    max_speed: float
    max_acceleration: float
    max_jerk: float
    point_value: int


DRONE_SPECS = {
    DroneType.FAST: DroneSpec(5.0, 4.0, 16.0, 1),
    DroneType.SLOW: DroneSpec(2.5, 2.0, 8.0, 5),
}

DRONE_RADIUS = 0.25
INTERCEPT_RADIUS = 0.75
PHYSICS_DT = 0.05
CONTROLLER_PERIOD = 0.10
DEFAULT_MATCH_DURATION = 90.0
ARENA_WIDTH = 100.0
ARENA_HEIGHT = 60.0

Vec2 = tuple[float, float]

