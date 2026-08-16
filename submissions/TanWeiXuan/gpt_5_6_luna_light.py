"""Luna Light: a deterministic adaptive escort controller.

Authorship: coded by GPT 5.6 Luna Light. Human guidance was limited to the
submission request; no human-authored implementation was incorporated.
"""

from __future__ import annotations

from math import hypot

from swarmbench import BaseSwarmController, CircleObstacle, DroneStatus, DroneType


class SwarmController(BaseSwarmController):
    """Send slow units through safe lanes while fast units screen the fleet."""

    def initialize(self, game_info):
        self.goal = game_info.target_goal
        self.own_goal = game_info.own_goal
        self.obstacles = game_info.obstacles
        self.specs = dict(game_info.drone_specs)
        self.width = game_info.arena_width
        self.height = game_info.arena_height
        self.tick = 0
        self.targets = {}
        drones = sorted(game_info.own_initial_drones, key=lambda d: (d.position[1], d.id))
        low = self.goal.y_min + 0.55
        high = self.goal.y_max - 0.55
        count = max(1, len(drones))
        self.lanes = {
            drone.id: low + (high - low) * (index + 0.5) / count
            for index, drone in enumerate(drones)
        }

    @staticmethod
    def _distance(a, b):
        return hypot(a[0] - b[0], a[1] - b[1])

    def _avoid_obstacles(self, position, limit):
        ax = ay = 0.0
        for obstacle in self.obstacles:
            if isinstance(obstacle, CircleObstacle):
                nearest, radius = obstacle.center, obstacle.radius
            else:
                nearest = (
                    min(obstacle.x_max, max(obstacle.x_min, position[0])),
                    min(obstacle.y_max, max(obstacle.y_min, position[1])),
                )
                radius = 0.0
            dx, dy = position[0] - nearest[0], position[1] - nearest[1]
            distance = hypot(dx, dy)
            clearance = distance - radius
            if clearance >= 2.8:
                continue
            if distance < 1e-8:
                dx, dy, distance = 0.0, (1.0 if position[1] < self.height / 2 else -1.0), 1.0
            strength = limit * min(1.0, (2.8 - clearance) / 2.8)
            ax += dx / distance * strength
            ay += dy / distance * strength
        return ax, ay

    def _steer(self, drone, target):
        spec = self.specs[drone.drone_type]
        dx, dy = target[0] - drone.position[0], target[1] - drone.position[1]
        distance = hypot(dx, dy)
        speed = min(spec.max_speed, (2.0 * spec.max_acceleration * distance) ** 0.5)
        if distance < 1e-8:
            desired = (0.0, 0.0)
        else:
            desired = (dx * speed / distance, dy * speed / distance)
        ax = 2.4 * (desired[0] - drone.velocity[0])
        ay = 2.4 * (desired[1] - drone.velocity[1])
        push_x, push_y = self._avoid_obstacles(drone.position, spec.max_acceleration)
        ax += push_x
        ay += push_y
        if drone.position[1] < 0.5:
            ay += spec.max_acceleration
        elif drone.position[1] > self.height - 0.5:
            ay -= spec.max_acceleration
        return ax, ay

    def _lead(self, hunter, enemy):
        speed = self.specs[hunter.drone_type].max_speed
        separation = self._distance(hunter.position, enemy.position)
        lead = min(3.5, separation / max(0.1, speed) + 0.25)
        return (
            min(self.width - 0.3, max(0.3, enemy.position[0] + enemy.velocity[0] * lead)),
            min(self.height - 0.3, max(0.3, enemy.position[1] + enemy.velocity[1] * lead)),
        )

    def step(self, state):
        self.tick += 1
        own = [d for d in state.own_drones if d.status is DroneStatus.ACTIVE]
        enemies = [d for d in state.opponent_drones if d.status is DroneStatus.ACTIVE]
        fast = [d for d in own if d.drone_type is DroneType.FAST]
        if self.tick == 1 or self.tick % 6 == 0:
            choices = []
            for hunter in fast:
                for enemy in enemies:
                    distance = self._distance(hunter.position, enemy.position)
                    urgency = max(0.0, 14.0 - self._distance(enemy.position, self.own_goal.center))
                    value = self.specs[enemy.drone_type].point_value
                    choices.append((distance - 2.5 * value - urgency, hunter.id, enemy.id))
            self.targets = {}
            used = set()
            for _, hunter_id, enemy_id in sorted(choices):
                if hunter_id not in self.targets and enemy_id not in used:
                    self.targets[hunter_id] = enemy_id
                    used.add(enemy_id)
        enemy_by_id = {d.id: d for d in enemies}
        actions = {}
        for drone in own:
            lane = self.lanes.get(drone.id, self.goal.center[1])
            target = (self.goal.center[0], min(self.goal.y_max - 0.35, max(self.goal.y_min + 0.35, lane)))
            enemy = enemy_by_id.get(self.targets.get(drone.id))
            if enemy is not None:
                target = self._lead(drone, enemy)
            actions[drone.id] = self._steer(drone, target)
        return actions
