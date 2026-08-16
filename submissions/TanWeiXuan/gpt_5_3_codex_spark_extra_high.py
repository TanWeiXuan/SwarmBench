"""Lane-anchored dual-role controller authored by GPT 5.3 Codex Spark Extra High.

Authorship: this file, including its strategy and implementation, was entirely
coded by GPT 5.3 Codex Spark Extra High without human guidance. No human
authorship was used for the source code or strategy choices.
"""

from __future__ import annotations

from math import atan2, cos, sin, sqrt

from swarmbench import BaseSwarmController, CircleObstacle, DroneStatus, DroneType


class SwarmController(BaseSwarmController):
    """Escort slow drones with selected fast units and clear nearby threats."""

    def initialize(self, game_info):
        self.goal = game_info.target_goal
        self.own_goal = game_info.own_goal
        self.obstacles = game_info.obstacles
        self.specs = dict(game_info.drone_specs)
        self.width = game_info.arena_width
        self.height = game_info.arena_height
        self.direction = 1.0 if self.goal.center[0] > self.own_goal.center[0] else -1.0
        self.tick = 0
        self.hunter_targets: dict[int, int] = {}

        own = sorted(game_info.own_initial_drones, key=lambda drone: (drone.position[1], drone.id))
        low = self.goal.y_min + 0.7
        high = self.goal.y_max - 0.7
        usable = max(0.8, high - low)
        self.goal_lanes = {
            drone.id: low + usable * (index + 0.5) / max(1, len(own))
            for index, drone in enumerate(own)
        }

        fast = sorted((drone for drone in own if drone.drone_type is DroneType.FAST), key=lambda drone: (drone.position[1], drone.id))
        slow = sorted((drone for drone in own if drone.drone_type is DroneType.SLOW), key=lambda drone: (drone.position[1], drone.id))
        guard_count = max(3, len(fast) // 2)
        self.guard_for = {
            fast[index].id: slow[min(index, len(slow) - 1)].id
            for index in range(min(guard_count, len(fast), len(slow)))
        }

    @staticmethod
    def _distance(left, right):
        return ((left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2) ** 0.5

    def _goal_target(self, drone):
        return self.goal.center[0], self._clamp(
            self.goal_lanes.get(drone.id, self.goal.center[1]),
            self.goal.y_min + 0.4,
            self.goal.y_max - 0.4,
        )

    def _clamp(self, value, low, high):
        return max(low, min(high, value))

    def _obstacle_repulsion(self, position, maximum):
        repulse_x = repulse_y = 0.0
        for obstacle in self.obstacles:
            if isinstance(obstacle, CircleObstacle):
                closest = obstacle.center
                radius = obstacle.radius
            else:
                closest = (
                    self._clamp(position[0], obstacle.x_min, obstacle.x_max),
                    self._clamp(position[1], obstacle.y_min, obstacle.y_max),
                )
                radius = 0.0
            dx = position[0] - closest[0]
            dy = position[1] - closest[1]
            clearance = self._distance(position, closest) - radius
            if clearance >= 3.0:
                continue
            if clearance < 1e-6:
                angle = atan2(dy if abs(dy) > 1e-6 else 1.0, dx if abs(dx) > 1e-6 else 1.0)
                dx = cos(angle)
                dy = sin(angle)
                clearance = 1.0
            strength = maximum * (3.0 - clearance) / 3.0
            repulse_x += dx / (self._distance(position, closest) or 1.0) * strength
            repulse_y += dy / (self._distance(position, closest) or 1.0) * strength
        return repulse_x, repulse_y

    def _steer(self, drone, target, speed_scale=1.0):
        spec = self.specs[drone.drone_type]
        dx = target[0] - drone.position[0]
        dy = target[1] - drone.position[1]
        distance = self._distance(target, drone.position)
        if distance < 1e-8:
            desired_x = desired_y = 0.0
        else:
            desired_speed = min(spec.max_speed * speed_scale, sqrt(2.0 * spec.max_acceleration * distance))
            desired_x = dx / distance * desired_speed
            desired_y = dy / distance * desired_speed
        acceleration_x = 2.0 * (desired_x - drone.velocity[0])
        acceleration_y = 2.0 * (desired_y - drone.velocity[1])
        repulse_x, repulse_y = self._obstacle_repulsion(drone.position, spec.max_acceleration)
        acceleration_x += repulse_x
        acceleration_y += repulse_y
        if drone.position[1] < 0.7:
            acceleration_y += spec.max_acceleration
        elif drone.position[1] > self.height - 0.7:
            acceleration_y -= spec.max_acceleration
        magnitude = self._distance((0.0, 0.0), (acceleration_x, acceleration_y))
        if magnitude > spec.max_acceleration:
            scale = spec.max_acceleration / magnitude
            acceleration_x *= scale
            acceleration_y *= scale
        return acceleration_x, acceleration_y

    def _intercept_target(self, hunter, enemy):
        speed = self.specs[hunter.drone_type].max_speed
        dx = enemy.position[0] - hunter.position[0]
        dy = enemy.position[1] - hunter.position[1]
        lead = self._distance(hunter.position, enemy.position) / max(0.1, speed) + 0.2
        return (
            self._clamp(enemy.position[0] + enemy.velocity[0] * lead, 0.2, self.width - 0.2),
            self._clamp(enemy.position[1] + enemy.velocity[1] * lead, 0.2, self.height - 0.2),
        )

    def _assign_defenders(self, defenders, enemies):
        assignments = {}
        chosen: set[int] = set()
        options = []
        for defender in defenders:
            for enemy in enemies:
                value = self.specs[enemy.drone_type].point_value
                travel = self._distance(defender.position, enemy.position) / max(0.1, self.specs[defender.drone_type].max_speed)
                urgency = max(0.0, 35.0 - self._distance(enemy.position, self.own_goal.center))
                options.append((travel - 4.0 * value - 0.3 * urgency, defender.id, enemy.id))
        for _, defender_id, enemy_id in sorted(options):
            if defender_id not in assignments and enemy_id not in chosen:
                assignments[defender_id] = enemy_id
                chosen.add(enemy_id)
        return assignments

    def step(self, state):
        self.tick += 1
        own = {drone.id: drone for drone in state.own_drones if drone.status is DroneStatus.ACTIVE}
        enemies = [drone for drone in state.opponent_drones if drone.status is DroneStatus.ACTIVE]
        enemy_by_id = {drone.id: drone for drone in enemies}
        fast = [drone for drone in own.values() if drone.drone_type is DroneType.FAST]

        hunters = [drone for drone in fast if drone.id not in self.guard_for]
        if self.tick == 1 or self.tick % 4 == 0:
            self.hunter_targets = self._assign_defenders(hunters, enemies)
        else:
            self.hunter_targets = {
                hunter_id: enemy_id
                for hunter_id, enemy_id in self.hunter_targets.items()
                if hunter_id in own and enemy_id in enemy_by_id
            }

        actions = {}
        for drone in own.values():
            if drone.drone_type is DroneType.SLOW:
                actions[drone.id] = self._steer(drone, self._goal_target(drone), 1.0)
                continue

            escorted_id = self.guard_for.get(drone.id)
            escorted = own.get(escorted_id) if escorted_id is not None else None
            if escorted is not None:
                escort_target = self._goal_target(escorted)
                nearby_enemies = [
                    enemy
                    for enemy in enemies
                    if self._distance(enemy.position, escorted.position) < 11.0
                    and self._distance(enemy.position, drone.position) < 16.0
                ]
                if nearby_enemies:
                    threat = min(
                        nearby_enemies,
                        key=lambda enemy: (self._distance(enemy.position, escorted.position), enemy.id),
                    )
                    actions[drone.id] = self._steer(drone, self._intercept_target(drone, threat), 1.05)
                    continue
                if self.goal.contains(escorted.position):
                    actions[drone.id] = self._steer(drone, escort_target, 1.0)
                else:
                    escort_x = self._clamp(escort.position[0] + 2.0 * self.direction, 0.2, self.width - 0.2)
                    actions[drone.id] = self._steer(drone, (escort_x, escort.position[1]), 1.0)
                continue

            enemy_id = self.hunter_targets.get(drone.id)
            if enemy_id in enemy_by_id:
                actions[drone.id] = self._steer(drone, self._intercept_target(drone, enemy_by_id[enemy_id]), 1.08)
            elif enemies:
                nearest_enemy = min(enemies, key=lambda enemy: self._distance(enemy.position, drone.position))
                actions[drone.id] = self._steer(drone, self._intercept_target(drone, nearest_enemy), 1.02)
            else:
                actions[drone.id] = self._steer(drone, self._goal_target(drone), 1.0)
        return actions
