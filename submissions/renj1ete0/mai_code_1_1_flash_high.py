"""MAI Code 1.1 Flash High community controller.

This file was entirely coded by MAI Code 1.1 Flash High without human guidance.
"""

from __future__ import annotations

import math

from swarmbench import BaseSwarmController, DroneStatus


class SwarmController(BaseSwarmController):
    """Drive each drone toward a lane-aligned goal while preserving arena bounds."""

    def initialize(self, game_info):
        self.goal = game_info.target_goal
        self.own_goal = game_info.own_goal
        self.specs = dict(game_info.drone_specs)
        self.direction = 1.0 if self.goal.center[0] > self.own_goal.center[0] else -1.0

        ordered = sorted(game_info.own_initial_drones, key=lambda drone: (drone.position[1], drone.id))
        if not ordered:
            self.lanes = {}
            return

        low = self.goal.y_min + 0.75
        high = self.goal.y_max - 0.75
        span = max(0.0, high - low)
        if len(ordered) == 1:
            lane = self.goal.center[1]
            self.lanes = {drone.id: lane for drone in ordered}
        else:
            self.lanes = {
                drone.id: low + span * index / (len(ordered) - 1)
                for index, drone in enumerate(ordered)
            }

    @staticmethod
    def _clamp(value, minimum, maximum):
        return min(maximum, max(minimum, value))

    def _target(self, drone):
        lane = self.lanes.get(drone.id, self.goal.center[1])
        return (
            self.goal.center[0],
            self._clamp(lane, self.goal.y_min + 0.6, self.goal.y_max - 0.6),
        )

    def step(self, state):
        actions = {}
        for drone in state.own_drones:
            if drone.status is not DroneStatus.ACTIVE:
                continue

            target_x, target_y = self._target(drone)
            dx = target_x - drone.position[0]
            dy = target_y - drone.position[1]
            distance = math.hypot(dx, dy)

            if distance < 1e-6:
                actions[drone.id] = (0.0, 0.0)
                continue

            spec = self.specs[drone.drone_type]
            desired_speed = min(spec.max_speed, max(0.2, math.sqrt(2.0 * spec.max_acceleration * distance)))
            desired = (desired_speed * dx / distance, desired_speed * dy / distance)

            accel_x = 2.35 * (desired[0] - drone.velocity[0]) - 0.18 * drone.acceleration[0]
            accel_y = 2.35 * (desired[1] - drone.velocity[1]) - 0.18 * drone.acceleration[1]

            if drone.position[1] < 0.8:
                accel_y += spec.max_acceleration
            elif drone.position[1] > self.goal.center[1] + 26.0:
                accel_y -= spec.max_acceleration

            magnitude = math.hypot(accel_x, accel_y)
            if magnitude > spec.max_acceleration:
                scale = spec.max_acceleration / magnitude
                accel_x *= scale
                accel_y *= scale

            actions[drone.id] = (accel_x, accel_y)

        return actions
