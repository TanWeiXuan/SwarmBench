"""Nemotron 3.5 Lightning - Strong swarm controller.

Strategy (role swap: SLOW attackers, FAST defenders):
- SLOW drones: primary attackers targeting opponent goal (5 pts each worth more)
- FAST drones: defensive interceptors targeting enemy threats
- Predictive interception for goal attacks and threat neutralization
- Periodic role reassessment every 3 control steps
- Obstacle-aware steering with repulsion forces
- Clear priority: score SLOW goals > prevent enemy goals > avoid obstacles

Borrowed concepts (with modification):
- steering logic adapted from baseline common.py steer() function
- defensive concepts from DefendController (SLOW attackers + FAST defenders)
- predictive leading inspired by rush-style interception
- assignment cost structure from AssignmentController

Authorship: Nemotron 3.5 Lightning controller, designed from game mechanics reasoning.
"""

from __future__ import annotations

from math import hypot

from swarmbench import BaseSwarmController, DroneStatus, DroneType, Team
from swarmbench.api.types import Vec2
from swarmbench.api import CircleObstacle, RectangleObstacle
from scipy.optimize import linear_sum_assignment


class SwarmController(BaseSwarmController):
    """Strong swarm controller with role swap: SLOW attackers, FAST defenders."""

    def initialize(self, game_info: GameInfo) -> None:
        self.team = game_info.team
        self.goal = game_info.target_goal
        self.own_goal = game_info.own_goal
        self.obstacles = game_info.obstacles
        self.specs = dict(game_info.drone_specs)
        self.width = game_info.arena_width
        self.height = game_info.arena_height
        self.tick = 0
        self.steps_since_assignment = 0

        # Lane assignments for FAST drones (defensive positions along own goal line)
        # Since FAST drones defend, we lane them along our own goal
        own_drones = sorted(game_info.own_initial_drones, key=lambda d: (d.position[1], d.id))
        fast_drones = [d for d in own_drones if d.drone_type is DroneType.FAST]
        low = self.own_goal.y_min + 0.7
        high = self.own_goal.y_max - 0.7
        usable = max(0.8, high - low)
        self.fast_lanes: dict[int, float] = {
            drone.id: low + usable * (i + 0.5) / max(1, len(fast_drones))
            for i, drone in enumerate(fast_drones)
        }

        # Lane assignments for SLOW drones (attacking positions on target goal)
        slow_drones = [d for d in own_drones if d.drone_type is DroneType.SLOW]
        low = self.goal.y_min + 0.7
        high = self.goal.y_max - 0.7
        usable = max(0.8, high - low)
        self.slow_lanes: dict[int, float] = {
            drone.id: low + usable * (i + 0.5) / max(1, len(slow_drones))
            for i, drone in enumerate(slow_drones)
        }

        # Guardian: each FAST drone guards one SLOW drone's position
        fast_drones_sorted = sorted(fast_drones, key=lambda d: (d.position[1], d.id))
        slow_drones_sorted = sorted(slow_drones, key=lambda d: (d.position[1], d.id))
        self.guard_for: dict[int, int] = {}
        if fast_drones_sorted and slow_drones_sorted:
            for i, fast in enumerate(fast_drones_sorted):
                self.guard_for[fast.id] = slow_drones_sorted[i % len(slow_drones_sorted)].id

        # Dynamic assignment of FAST defenders to enemy threats
        self.defender_assignments: dict[int, int] = {}  # fast_drone_id -> enemy_id

    @staticmethod
    def _distance(p1: Vec2, p2: Vec2) -> float:
        return hypot(p1[0] - p2[0], p1[1] - p2[1])

    def _goal_target(self, drone: DroneSnapshot, lane_y: float | None = None) -> Vec2:
        """Get target point on the target (opponent) goal line."""
        y = lane_y if lane_y is not None else self.goal.center[1]
        y = min(self.goal.y_max - 0.75, max(self.goal.y_min + 0.75, y))
        return (self.goal.center[0], y)

    def _own_goal_target(self, drone: DroneSnapshot, lane_y: float | None = None) -> Vec2:
        """Get target point on the own goal line (for defensive positioning)."""
        y = lane_y if lane_y is not None else self.own_goal.center[1]
        y = min(self.own_goal.y_max - 0.75, max(self.own_goal.y_min + 0.75, y))
        return (self.own_goal.center[0], y)

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def _steer_to_target(
        self,
        drone: DroneSnapshot,
        target: Vec2,
        spec: DroneSpec,
        obstacles: tuple[CircleObstacle | RectangleObstacle, ...],
        speed_scale: float = 1.0,
    ) -> Vec2:
        """Vector tracking with obstacle repulsion.

        Adapted from baseline common.py steer() function.
        """
        dx = target[0] - drone.position[0]
        dy = target[1] - drone.position[1]
        distance = hypot(dx, dy)

        if distance < 1e-8:
            desired_vel = (0.0, 0.0)
        else:
            desired_speed = min(spec.max_speed * speed_scale, (2.0 * spec.max_acceleration * distance) ** 0.5)
            desired_vel = (dx / distance * desired_speed, dy / distance * desired_speed)

        # Acceleration toward desired velocity (simple PD-like term)
        ax = 2.2 * (desired_vel[0] - drone.velocity[0])
        ay = 2.2 * (desired_vel[1] - drone.velocity[1])

        # Obstacle repulsion
        if distance > 0:
            forward = (dx / distance, dy / distance)
            for obstacle in obstacles:
                if isinstance(obstacle, CircleObstacle):
                    cx, cy = obstacle.center
                    radius = obstacle.radius
                else:
                    cx = (obstacle.x_min + obstacle.x_max) / 2.0
                    cy = (obstacle.y_min + obstacle.y_max) / 2.0
                    radius = hypot(obstacle.x_max - obstacle.x_min, obstacle.y_max - obstacle.y_min) / 2.0

                ox, oy = cx - drone.position[0], cy - drone.position[1]
                center_dist = hypot(ox, oy)

                # Surface proximity repulsion
                if 0 < center_dist < radius + 3.0:
                    surface_dist = center_dist - radius
                    strength = spec.max_acceleration * max(0.0, (3.0 - abs(surface_dist)) / 3.0)
                    if center_dist > 0:
                        ax -= ox / center_dist * strength
                        ay -= oy / center_dist * strength

                # Forward projection repulsion
                if center_dist > 0:
                    projection = ox * forward[0] + oy * forward[1]
                    if -0.5 < projection < 8.0:
                        lateral = ox * (-forward[1]) + oy * forward[0]
                        safe_radius = radius + 1.1
                        if abs(lateral) < safe_radius:
                            side = -1.0 if lateral > 0 else 1.0
                            if abs(lateral) < 1e-9:
                                side = 1.0 if drone.id % 2 == 0 else -1.0
                            strength = spec.max_acceleration * (1.0 - max(0.0, projection) / 8.0)
                            ax += -forward[1] * side * strength
                            ay += forward[0] * side * strength

        # Clamp acceleration magnitude
        mag = hypot(ax, ay)
        if mag > spec.max_acceleration and mag > 0:
            scale = spec.max_acceleration / mag
            ax *= scale
            ay *= scale

        return (ax, ay)

    def _predict_intercept(
        self,
        hunter: DroneSnapshot,
        target: DroneSnapshot,
        spec: DroneSpec,
    ) -> Vec2:
        """Predictive interception: lead the target based on distance and speed."""
        speed = spec.max_speed
        dx = target.position[0] - hunter.position[0]
        dy = target.position[1] - hunter.position[1]
        dist = hypot(dx, dy)

        if dist < 1e-6:
            return target.position

        lead = dist / max(0.1, speed) + 0.3

        pred_x = target.position[0] + target.velocity[0] * lead
        pred_y = target.position[1] + target.velocity[1] * lead

        pred_x = max(0.2, min(self.width - 0.2, pred_x))
        pred_y = max(0.2, min(self.height - 0.2, pred_y))

        return (pred_x, pred_y)

    def _assign_defenders(
        self,
        defenders: list[DroneSnapshot],
        enemies: list[DroneSnapshot],
    ) -> dict[int, int]:
        """Cost-based assignment of FAST defenders to enemy threats.

        Cost = travel_distance - 3.5 * point_value - 0.08 * threat_progress
        Higher point_value enemies cost less (more worth intercepting).
        Threat_progress: how close enemy is to our target goal.
        """
        if not defenders or not enemies:
            return {}

        # Compute threat progress: enemy closer to our target goal is more threatening
        threat_progress_fn = lambda pos: self._distance(pos, self.goal.center)

        costs: list[list[float]] = []
        for defender in defenders:
            row = []
            for enemy in enemies:
                travel = self._distance(defender.position, enemy.position)
                point_value = self.specs[enemy.drone_type].point_value
                threat_progress = threat_progress_fn(enemy.position)
                cost = travel - 3.5 * point_value - 0.08 * threat_progress
                row.append(cost)
            costs.append(row)

        rows, cols = linear_sum_assignment(costs)
        assignments: dict[int, int] = {}
        for r, c in zip(rows, cols, strict=True):
            defender_id = defenders[r].id
            enemy_id = enemies[c].id
            assignments[defender_id] = enemy_id

        return assignments

    def step(self, state: GameState) -> dict[int, Vec2]:
        """Return desired acceleration components keyed by own drone ID."""
        self.tick += 1
        self.steps_since_assignment += 1

        own_active: dict[int, DroneSnapshot] = {
            drone.id: drone for drone in state.own_drones if drone.status is DroneStatus.ACTIVE
        }
        enemies: list[DroneSnapshot] = [
            drone for drone in state.opponent_drones if drone.status is DroneStatus.ACTIVE
        ]
        enemy_by_id: dict[int, DroneSnapshot] = {drone.id: drone for drone in enemies}
        fast_drones: list[DroneSnapshot] = [
            drone for drone in state.own_drones
            if drone.status is DroneStatus.ACTIVE and drone.drone_type is DroneType.FAST
        ]
        slow_drones: list[DroneSnapshot] = [
            drone for drone in state.own_drones
            if drone.status is DroneStatus.ACTIVE and drone.drone_type is DroneType.SLOW
        ]

        # Periodic reassignment every 3 steps
        if self.steps_since_assignment >= 3:
            self.defender_assignments = self._assign_defenders(fast_drones, enemies)
            self.steps_since_assignment = 0

        actions: dict[int, Vec2] = {}

        # ---- Process FAST drones: defensive intercept ----
        for drone in fast_drones:
            assigned_enemy_id = self.defender_assignments.get(drone.id)
            has_assignment = assigned_enemy_id is not None and assigned_enemy_id in enemy_by_id

            if has_assignment:
                # Defender mode: intercept the assigned enemy threat
                threat_dist = self._distance(
                    enemy_by_id[assigned_enemy_id].position, self.own_goal.center
                )
                if threat_dist < 30.0:
                    target_enemy = enemy_by_id[assigned_enemy_id]
                    intercept_target = self._predict_intercept(drone, target_enemy, self.specs[drone.drone_type])
                    actions[drone.id] = self._steer_to_target(
                        drone, intercept_target, self.specs[drone.drone_type], self.obstacles,
                        speed_scale=1.05,
                    )
                else:
                    # Enemy no longer near our goal: go to defensive lane
                    target = self._own_goal_target(drone, self.fast_lanes.get(drone.id))
                    actions[drone.id] = self._steer_to_target(
                        drone, target, self.specs[drone.drone_type], self.obstacles,
                    )
            else:
                # No assignment: check if enemies are near our goal
                enemies_near_own_goal = [
                    e for e in enemies
                    if self._distance(e.position, self.own_goal.center) < 25.0
                ]

                if enemies_near_own_goal:
                    # Intercept enemies near our own goal
                    nearest_threat = min(enemies_near_own_goal,
                                        key=lambda e: self._distance(e.position, drone.position))
                    intercept_target = self._predict_intercept(drone, nearest_threat, self.specs[drone.drone_type])
                    actions[drone.id] = self._steer_to_target(
                        drone, intercept_target, self.specs[drone.drone_type], self.obstacles,
                        speed_scale=1.03,
                    )
                else:
                    # No nearby enemies: go to defensive lane
                    target = self._own_goal_target(drone, self.fast_lanes.get(drone.id))
                    actions[drone.id] = self._steer_to_target(
                        drone, target, self.specs[drone.drone_type], self.obstacles,
                    )

        # ---- Process SLOW drones: attacking the target goal ----
        for drone in slow_drones:
            # Check if enemies are near our target goal or own goal
            enemies_near_target = [
                e for e in enemies
                if self._distance(e.position, self.goal.center) < 25.0
            ]
            enemies_near_own_goal = [
                e for e in enemies
                if self._distance(e.position, self.own_goal.center) < 25.0
            ]

            if enemies_near_target or enemies_near_own_goal:
                # Enemies near either goal: we should be attacking the target goal
                # to counter the threat; go to attacking lane
                lane_y = self.slow_lanes.get(drone.id, self.goal.center[1])
                # If enemies are near own goal, also adjust lane
                if enemies_near_own_goal:
                    threat = min(enemies_near_own_goal,
                                key=lambda e: self._distance(e.position, self.own_goal.center))
                    lane_y = (lane_y + threat.position[1]) / 2.0
                target = self._goal_target(drone, lane_y)
            else:
                # No enemies near goals: attack target goal
                target = self._goal_target(drone, self.slow_lanes.get(drone.id))
            actions[drone.id] = self._steer_to_target(drone, target, self.specs[drone.drone_type], self.obstacles)

        return actions