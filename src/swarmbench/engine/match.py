"""Headless authoritative game state and event resolution."""

from __future__ import annotations

from dataclasses import dataclass

from swarmbench.api import (
    DRONE_RADIUS,
    DRONE_SPECS,
    INTERCEPT_RADIUS,
    PHYSICS_DT,
    DestructionReason,
    DroneSnapshot,
    DroneStatus,
    Team,
    Vec2,
)

from .arena import Scenario
from .collisions import contact_position, swept_goal_contact, swept_obstacle_contact, swept_points_contact
from .dynamics import DynamicState, advance_dynamics
from .events import EventType, GameEvent


@dataclass(slots=True)
class _Drone:
    snapshot: DroneSnapshot
    desired_acceleration: Vec2 = (0.0, 0.0)


class Simulator:
    """Deterministic physics and rules with no controller or rendering dependency."""

    def __init__(self, scenario: Scenario, dt: float = PHYSICS_DT) -> None:
        self.scenario = scenario
        self.dt = dt
        self.time = 0.0
        self.scores = {Team.A: 0, Team.B: 0}
        self.drones = {drone.id: _Drone(drone) for drone in scenario.drones}
        self.events: list[GameEvent] = []

    def snapshots(self, team: Team | None = None) -> tuple[DroneSnapshot, ...]:
        return tuple(
            runtime.snapshot
            for drone_id, runtime in sorted(self.drones.items())
            if team is None or runtime.snapshot.team is team
        )

    def set_commands(self, team: Team, commands: dict[int, Vec2]) -> None:
        for drone_id, command in commands.items():
            runtime = self.drones.get(drone_id)
            if runtime is not None and runtime.snapshot.team is team and runtime.snapshot.status is DroneStatus.ACTIVE:
                runtime.desired_acceleration = command

    def _candidate_events(self, before: dict[int, DroneSnapshot], after: dict[int, DroneSnapshot]) -> list[GameEvent]:
        candidates: list[GameEvent] = []
        active_ids = sorted(before)
        for drone_id in active_ids:
            old, new = before[drone_id], after[drone_id]
            obstacle_times = [
                contact
                for obstacle in self.scenario.obstacles
                if (contact := swept_obstacle_contact(old.position, new.position, obstacle, DRONE_RADIUS)) is not None
            ]
            if obstacle_times:
                t = min(obstacle_times)
                candidates.append(GameEvent(t, EventType.OBSTACLE_CRASH, (drone_id,), contact_position(old.position, new.position, t)))
            goal_t = swept_goal_contact(old.position, new.position, self.scenario.target_goal(old.team))
            if goal_t is not None:
                candidates.append(
                    GameEvent(goal_t, EventType.GOAL, (drone_id,), contact_position(old.position, new.position, goal_t), old.team, DRONE_SPECS[old.drone_type].point_value)
                )

        team_a = [drone_id for drone_id in active_ids if before[drone_id].team is Team.A]
        team_b = [drone_id for drone_id in active_ids if before[drone_id].team is Team.B]
        for a_id in team_a:
            for b_id in team_b:
                contact = swept_points_contact(
                    before[a_id].position,
                    after[a_id].position,
                    before[b_id].position,
                    after[b_id].position,
                    INTERCEPT_RADIUS,
                )
                if contact is not None:
                    a_position = contact_position(before[a_id].position, after[a_id].position, contact)
                    b_position = contact_position(before[b_id].position, after[b_id].position, contact)
                    position = ((a_position[0] + b_position[0]) / 2, (a_position[1] + b_position[1]) / 2)
                    candidates.append(GameEvent(contact, EventType.INTERCEPTION, (a_id, b_id), position))
        return sorted(candidates, key=lambda event: event.sort_key)

    def step(self) -> tuple[GameEvent, ...]:
        before = {
            drone_id: runtime.snapshot
            for drone_id, runtime in self.drones.items()
            if runtime.snapshot.status is DroneStatus.ACTIVE
        }
        after: dict[int, DroneSnapshot] = {}
        for drone_id, old in sorted(before.items()):
            runtime = self.drones[drone_id]
            dynamic = advance_dynamics(
                DynamicState(old.position, old.velocity, old.acceleration),
                runtime.desired_acceleration,
                DRONE_SPECS[old.drone_type],
                self.dt,
            )
            after[drone_id] = DroneSnapshot(
                old.id,
                old.team,
                old.drone_type,
                dynamic.position,
                dynamic.velocity,
                dynamic.acceleration,
            )

        active = set(before)
        resolved: list[GameEvent] = []
        for candidate in self._candidate_events(before, after):
            if not all(drone_id in active for drone_id in candidate.drone_ids):
                continue
            event_time = self.time + candidate.time * self.dt
            event = GameEvent(event_time, candidate.event_type, candidate.drone_ids, candidate.position, candidate.team, candidate.points)
            if event.event_type is EventType.GOAL:
                drone_id = event.drone_ids[0]
                old = after[drone_id]
                after[drone_id] = DroneSnapshot(
                    old.id, old.team, old.drone_type, event.position, old.velocity, old.acceleration, DroneStatus.SCORED
                )
                self.scores[old.team] += event.points
            else:
                reason = DestructionReason.OBSTACLE_CRASH if event.event_type is EventType.OBSTACLE_CRASH else DestructionReason.INTERCEPTION
                for drone_id in event.drone_ids:
                    old = after[drone_id]
                    after[drone_id] = DroneSnapshot(
                        old.id, old.team, old.drone_type, event.position, old.velocity, old.acceleration, DroneStatus.ELIMINATED, reason
                    )
            active.difference_update(event.drone_ids)
            resolved.append(event)

        for drone_id, snapshot in after.items():
            self.drones[drone_id].snapshot = snapshot
        self.time += self.dt
        self.events.extend(resolved)
        return tuple(resolved)

