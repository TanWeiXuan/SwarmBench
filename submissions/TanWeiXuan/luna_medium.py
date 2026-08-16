"""Luna Medium: a deterministic lane-and-intercept swarm controller.

Authorship: this file, including its strategy and implementation, was
entirely coded by GPT 5.6 Luna Medium without human guidance. No human-authored
code, strategy choices, or iterative guidance were used.
"""

from __future__ import annotations

from math import hypot

from swarmbench import BaseSwarmController, CircleObstacle, DroneStatus, DroneType


class SwarmController(BaseSwarmController):
    """Keep attackers spread across lanes and spend defenders deliberately."""

    def initialize(self, game_info):
        self.goal = game_info.target_goal
        self.own_goal = game_info.own_goal
        self.obstacles = game_info.obstacles
        self.specs = dict(game_info.drone_specs)
        self.width = game_info.arena_width
        self.height = game_info.arena_height
        self.direction = 1.0 if self.goal.center[0] > self.own_goal.center[0] else -1.0
        self.tick = 0
        self.assignments = {}

        drones = sorted(game_info.own_initial_drones, key=lambda item: (item.position[1], item.id))
        lane_count = max(1, len(drones))
        low = self.goal.y_min + 0.8
        high = self.goal.y_max - 0.8
        self.lanes = {
            drone.id: low + (high - low) * (index + 0.5) / lane_count
            for index, drone in enumerate(drones)
        }

    @staticmethod
    def _distance(left, right):
        return hypot(left[0] - right[0], left[1] - right[1])

    def _goal_target(self, drone):
        lane = self.lanes.get(drone.id, self.goal.center[1])
        return self.goal.center[0], min(self.goal.y_max - 0.45, max(self.goal.y_min + 0.45, lane))

    def _obstacle_push(self, position, maximum):
        push_x = push_y = 0.0
        for obstacle in self.obstacles:
            if isinstance(obstacle, CircleObstacle):
                nearest = obstacle.center
                radius = obstacle.radius
            else:
                nearest = (
                    min(obstacle.x_max, max(obstacle.x_min, position[0])),
                    min(obstacle.y_max, max(obstacle.y_min, position[1])),
                )
                radius = 0.0
            dx, dy = position[0] - nearest[0], position[1] - nearest[1]
            distance = hypot(dx, dy)
            clearance = distance - radius
            if clearance >= 3.0:
                continue
            if distance < 1e-9:
                dx, dy, distance = 0.0, (1.0 if position[1] < self.height / 2 else -1.0), 1.0
            strength = maximum * (3.0 - clearance) / 3.0
            push_x += dx / distance * strength
            push_y += dy / distance * strength
        return push_x, push_y

    def _steer(self, drone, target, scale=1.0):
        spec = self.specs[drone.drone_type]
        dx, dy = target[0] - drone.position[0], target[1] - drone.position[1]
        distance = hypot(dx, dy)
        if distance < 1e-8:
            desired_x = desired_y = 0.0
        else:
            speed = min(spec.max_speed * scale, (2.0 * spec.max_acceleration * distance) ** 0.5)
            desired_x, desired_y = dx / distance * speed, dy / distance * speed
        ax = 2.5 * (desired_x - drone.velocity[0])
        ay = 2.5 * (desired_y - drone.velocity[1])
        push_x, push_y = self._obstacle_push(drone.position, spec.max_acceleration)
        ax += push_x
        ay += push_y
        if drone.position[1] < 0.6:
            ay += spec.max_acceleration
        elif drone.position[1] > self.height - 0.6:
            ay -= spec.max_acceleration
        return ax, ay

    def _intercept_target(self, hunter, enemy):
        speed = self.specs[hunter.drone_type].max_speed
        dx = enemy.position[0] - hunter.position[0]
        dy = enemy.position[1] - hunter.position[1]
        separation = hypot(dx, dy)
        lead = min(4.0, separation / max(0.1, speed) + 0.35)
        return (
            min(self.width - 0.35, max(0.35, enemy.position[0] + enemy.velocity[0] * lead)),
            min(self.height - 0.35, max(0.35, enemy.position[1] + enemy.velocity[1] * lead)),
        )

    def _assign(self, defenders, enemies):
        candidates = []
        for defender in defenders:
            for enemy in enemies:
                distance = self._distance(defender.position, enemy.position)
                goal_distance = self._distance(enemy.position, self.own_goal.center)
                value = self.specs[enemy.drone_type].point_value
                urgency = max(0.0, 16.0 - goal_distance)
                candidates.append((distance - 3.0 * value - 0.75 * urgency, defender.id, enemy.id))
        chosen = {}
        used = set()
        for _, defender_id, enemy_id in sorted(candidates):
            if defender_id not in chosen and enemy_id not in used:
                chosen[defender_id] = enemy_id
                used.add(enemy_id)
        return chosen

    def step(self, state):
        self.tick += 1
        own = [drone for drone in state.own_drones if drone.status is DroneStatus.ACTIVE]
        enemies = [drone for drone in state.opponent_drones if drone.status is DroneStatus.ACTIVE]
        by_enemy = {drone.id: drone for drone in enemies}
        defenders = [drone for drone in own if drone.drone_type is DroneType.FAST]
        if self.tick == 1 or self.tick % 5 == 0:
            self.assignments = self._assign(defenders, enemies)
        else:
            self.assignments = {
                defender_id: enemy_id
                for defender_id, enemy_id in self.assignments.items()
                if enemy_id in by_enemy
            }

        actions = {}
        for drone in own:
            target = self._goal_target(drone)
            scale = 1.0
            enemy_id = self.assignments.get(drone.id)
            if enemy_id is not None:
                enemy = by_enemy[enemy_id]
                target = self._intercept_target(drone, enemy)
                scale = 1.15 if enemy.drone_type is DroneType.SLOW else 1.0
            elif drone.drone_type is DroneType.FAST and enemies:
                nearest = min(enemies, key=lambda enemy: (self._distance(drone.position, enemy.position), enemy.id))
                if self._distance(drone.position, nearest.position) < 12.0:
                    target = self._intercept_target(drone, nearest)
            actions[drone.id] = self._steer(drone, target, scale)
        return actions
