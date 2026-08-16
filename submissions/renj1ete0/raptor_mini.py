"""Raptor Mini community controller with adaptive lane offense and defensive intercepts.

Authorship: this file and strategy were entirely coded by Raptor Mini without human guidance.
No human-authored code, strategy choices, or iterative guidance were used.
"""

from __future__ import annotations

from math import atan2, cos, hypot, pi, sin

from swarmbench import BaseSwarmController, CircleObstacle, DroneStatus, DroneType


class SwarmController(BaseSwarmController):
    """Advance lane-assigned drones while FAST units hunt nearby threats."""

    def initialize(self, game_info):
        self.goal = game_info.target_goal
        self.own_goal = game_info.own_goal
        self.obstacles = tuple(game_info.obstacles)
        self.specs = dict(game_info.drone_specs)
        self.direction = 1.0 if self.goal.center[0] > self.own_goal.center[0] else -1.0
        self.tick = 0

        ordered = sorted(game_info.own_initial_drones, key=lambda drone: (drone.position[1], drone.id))
        low = self.goal.y_min + 0.65
        high = self.goal.y_max - 0.65
        span = max(1.0, high - low)
        self.lane_y = {
            drone.id: low + span * (rank + 0.5) / max(1, len(ordered)) for rank, drone in enumerate(ordered)
        }

    @staticmethod
    def _distance(left, right):
        return hypot(left[0] - right[0], left[1] - right[1])

    @staticmethod
    def _angle_to(start, end):
        return atan2(end[1] - start[1], end[0] - start[0])

    def _steer(self, drone, target, spec, *, repulsion=1.0):
        dx = target[0] - drone.position[0]
        dy = target[1] - drone.position[1]
        distance = hypot(dx, dy)
        if distance < 1e-6:
            desired_velocity = (0.0, 0.0)
        else:
            desired_speed = min(spec.max_speed, sqrt(2.0 * spec.max_acceleration * distance))
            desired_velocity = (dx / distance * desired_speed, dy / distance * desired_speed)

        ax = 2.1 * (desired_velocity[0] - drone.velocity[0])
        ay = 2.1 * (desired_velocity[1] - drone.velocity[1])

        if distance > 0.0:
            forward = (dx / distance, dy / distance)
            for obstacle in self.obstacles:
                if isinstance(obstacle, CircleObstacle):
                    ox = obstacle.center[0] - drone.position[0]
                    oy = obstacle.center[1] - drone.position[1]
                    projection = ox * forward[0] + oy * forward[1]
                    lateral = ox * (-forward[1]) + oy * forward[0]
                    safe = obstacle.radius + 1.15
                    if -0.5 < projection < 8.0 and abs(lateral) < safe:
                        side = -1.0 if lateral > 0.0 else 1.0
                        strength = repulsion * spec.max_acceleration * (1.0 - max(0.0, projection) / 8.0)
                        ax += -forward[1] * side * strength
                        ay += forward[0] * side * strength
                    center_distance = hypot(ox, oy)
                    surface = center_distance - obstacle.radius
                    if 0.0 < surface < 3.0:
                        strength = repulsion * spec.max_acceleration * (3.0 - surface) / 3.0
                        ax -= ox / center_distance * strength
                        ay -= oy / center_distance * strength
                else:
                    corners = (
                        (obstacle.x_min, obstacle.y_min),
                        (obstacle.x_min, obstacle.y_max),
                        (obstacle.x_max, obstacle.y_min),
                        (obstacle.x_max, obstacle.y_max),
                    )
                    closest = min(corners, key=lambda point: hypot(point[0] - drone.position[0], point[1] - drone.position[1]))
                    ox = closest[0] - drone.position[0]
                    oy = closest[1] - drone.position[1]
                    center_distance = hypot(ox, oy)
                    if center_distance < 3.0 and center_distance > 1e-6:
                        strength = repulsion * spec.max_acceleration * (3.0 - center_distance) / 3.0
                        ax -= ox / center_distance * strength
                        ay -= oy / center_distance * strength

        limit = spec.max_acceleration * 1.15
        magnitude = hypot(ax, ay)
        if magnitude > limit and magnitude > 0.0:
            ax *= limit / magnitude
            ay *= limit / magnitude
        return (ax, ay)

    def _threat_score(self, opponent):
        if opponent.status is not DroneStatus.ACTIVE:
            return 0.0
        return max(0.0, 12.0 - abs(opponent.position[1] - self.own_goal.center[1])) + (12.0 if opponent.drone_type is DroneType.SLOW else 6.0)

    def _assign_fast_hunters(self, fast_drones, opponents):
        if not opponents or not fast_drones:
            return {}
        targets = sorted(opponents, key=lambda enemy: (-self._threat_score(enemy), enemy.position[0]))
        assignment = {}
        for index, drone in enumerate(sorted(fast_drones, key=lambda d: (d.position[1], d.id))):
            enemy = targets[index % len(targets)]
            assignment[drone.id] = enemy.position
        return assignment

    def step(self, state):
        self.tick += 1
        active = [drone for drone in state.own_drones if drone.status is DroneStatus.ACTIVE]
        enemies = [drone for drone in state.opponent_drones if drone.status is DroneStatus.ACTIVE]

        fast_drones = [drone for drone in active if drone.drone_type is DroneType.FAST]
        slow_drones = [drone for drone in active if drone.drone_type is DroneType.SLOW]

        hunters = self._assign_fast_hunters(fast_drones, enemies)
        actions = {}

        for drone in active:
            if drone.drone_type is DroneType.FAST and drone.id in hunters:
                target = hunters[drone.id]
            else:
                lane_y = self.lane_y.get(drone.id, drone.position[1])
                target = (self.goal.center[0], lane_y)
                if drone.drone_type is DroneType.SLOW and abs(drone.position[1] - lane_y) > 1.0:
                    target = (drone.position[0] + self.direction * 2.0, lane_y)

            repulsion = 1.35 if drone.drone_type is DroneType.FAST else 0.95
            actions[drone.id] = self._steer(drone, target, self.specs[drone.drone_type], repulsion=repulsion)

        return actions
