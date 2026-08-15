from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from swarmbench.api import Team, Vec2


class EventType(str, Enum):
    OBSTACLE_CRASH = "OBSTACLE_CRASH"
    INTERCEPTION = "INTERCEPTION"
    GOAL = "GOAL"
    CONTROLLER_TIMEOUT = "CONTROLLER_TIMEOUT"
    CONTROLLER_EXCEPTION = "CONTROLLER_EXCEPTION"


EVENT_PRIORITY = {
    EventType.OBSTACLE_CRASH: 0,
    EventType.INTERCEPTION: 1,
    EventType.GOAL: 2,
    EventType.CONTROLLER_TIMEOUT: 3,
    EventType.CONTROLLER_EXCEPTION: 3,
}


@dataclass(frozen=True, slots=True)
class GameEvent:
    time: float
    event_type: EventType
    drone_ids: tuple[int, ...]
    position: Vec2
    team: Team | None = None
    points: int = 0

    @property
    def sort_key(self) -> tuple[float, int, tuple[int, ...]]:
        return (self.time, EVENT_PRIORITY[self.event_type], self.drone_ids)

