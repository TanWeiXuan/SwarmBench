"""Immutable snapshots passed across the controller boundary."""

from __future__ import annotations

from dataclasses import dataclass

from .types import DestructionReason, DroneSpec, DroneStatus, DroneType, Team, Vec2


@dataclass(frozen=True, slots=True)
class GoalZone:
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    @property
    def center(self) -> Vec2:
        return ((self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2)

    def contains(self, point: Vec2) -> bool:
        return self.x_min <= point[0] <= self.x_max and self.y_min <= point[1] <= self.y_max


@dataclass(frozen=True, slots=True)
class CircleObstacle:
    center: Vec2
    radius: float


@dataclass(frozen=True, slots=True)
class RectangleObstacle:
    x_min: float
    x_max: float
    y_min: float
    y_max: float


Obstacle = CircleObstacle | RectangleObstacle


@dataclass(frozen=True, slots=True)
class DroneSnapshot:
    id: int
    team: Team
    drone_type: DroneType
    position: Vec2
    velocity: Vec2 = (0.0, 0.0)
    acceleration: Vec2 = (0.0, 0.0)
    status: DroneStatus = DroneStatus.ACTIVE
    destruction_reason: DestructionReason | None = None


@dataclass(frozen=True, slots=True)
class GameInfo:
    team: Team
    arena_width: float
    arena_height: float
    target_goal: GoalZone
    own_goal: GoalZone
    obstacles: tuple[Obstacle, ...]
    drone_specs: tuple[tuple[DroneType, DroneSpec], ...]
    own_initial_drones: tuple[DroneSnapshot, ...]
    opponent_initial_drones: tuple[DroneSnapshot, ...]
    scenario_seed: int
    generator_version: int
    controller_seed: int
    controller_api_version: int


@dataclass(frozen=True, slots=True)
class GameState:
    time: float
    own_drones: tuple[DroneSnapshot, ...]
    opponent_drones: tuple[DroneSnapshot, ...]
    own_score: int
    opponent_score: int

