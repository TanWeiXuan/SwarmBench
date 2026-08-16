"""Luna Max: a predictive escort-and-intercept swarm controller.

Authorship: this file, including its strategy and implementation, was
entirely coded by GPT 5.6 Luna Max without human guidance. No human-authored
code, strategy choices, or iterative guidance were used.
"""

from __future__ import annotations

from math import cos, hypot, pi, sin, sqrt

from swarmbench import BaseSwarmController, CircleObstacle, DroneStatus, DroneType


class SwarmController(BaseSwarmController):
    """Advance SLOW drones in lanes while FAST drones screen and intercept."""

    def initialize(self, game_info):
        self.goal = game_info.target_goal
        self.own_goal = game_info.own_goal
        self.obstacles = game_info.obstacles
        self.specs = dict(game_info.drone_specs)
        self.width = game_info.arena_width
        self.height = game_info.arena_height
        self.direction = 1.0 if self.goal.center[0] > self.own_goal.center[0] else -1.0
        self.tick = 0
        self.hunter_targets = {}

        initial = tuple(game_info.own_initial_drones)
        ordered = sorted(initial, key=lambda drone: (drone.position[1], drone.id))
        low = self.goal.y_min + 0.65
        high = self.goal.y_max - 0.65
        usable_height = max(1.5, high - low)
        self.lanes = {
            drone.id: low + usable_height * (rank + 0.5) / max(1, len(ordered))
            for rank, drone in enumerate(ordered)
        }

        fast = sorted(
            (drone for drone in initial if drone.drone_type is DroneType.FAST),
            key=lambda drone: (drone.position[1], drone.id),
        )
        slow = sorted(
            (drone for drone in initial if drone.drone_type is DroneType.SLOW),
            key=lambda drone: (drone.position[1], drone.id),
        )
        guard_count = min(len(fast), len(slow), max(4, len(fast) // 2))
        if guard_count == 1:
            guard_indices = [0]
        elif guard_count:
            guard_indices = sorted(
                {
                    round(index * (len(fast) - 1) / (guard_count - 1))
                    for index in range(guard_count)
                }
            )
        else:
            guard_indices = []
        self.escort_for = {
            fast[index].id: slow[index].id
            for index in guard_indices
            if index < len(slow)
        }
        self.guard_ids = set(self.escort_for)

    @staticmethod
    def _distance(left, right):
        return hypot(left[0] - right[0], left[1] - right[1])

    @staticmethod
    def _clamp(value, low, high):
        return min(high, max(low, value))

    @staticmethod
    def _segment_distance(start, end, point):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length_sq = dx * dx + dy * dy
        if length_sq <= 1e-12:
            return hypot(point[0] - start[0], point[1] - start[1]), 0.0
        t = (
            (point[0] - start[0]) * dx + (point[1] - start[1]) * dy
        ) / length_sq
        t = max(0.0, min(1.0, t))
        closest = (start[0] + t * dx, start[1] + t * dy)
        return hypot(point[0] - closest[0], point[1] - closest[1]), t

    @staticmethod
    def _segment_box_time(start, end, bounds):
        enter, leave = 0.0, 1.0
        for origin, delta, low, high in (
            (start[0], end[0] - start[0], bounds[0], bounds[1]),
            (start[1], end[1] - start[1], bounds[2], bounds[3]),
        ):
            if abs(delta) < 1e-12:
                if origin < low or origin > high:
                    return None
                continue
            first, second = (low - origin) / delta, (high - origin) / delta
            if first > second:
                first, second = second, first
            enter = max(enter, first)
            leave = min(leave, second)
            if enter > leave:
                return None
        if leave < 0.0 or enter > 1.0:
            return None
        return max(0.0, enter)

    def _segment_clear(self, start, end, margin):
        for obstacle in self.obstacles:
            if isinstance(obstacle, CircleObstacle):
                distance, _ = self._segment_distance(start, end, obstacle.center)
                if distance <= obstacle.radius + margin:
                    return False
            else:
                bounds = (
                    obstacle.x_min - margin,
                    obstacle.x_max + margin,
                    obstacle.y_min - margin,
                    obstacle.y_max + margin,
                )
                if self._segment_box_time(start, end, bounds) is not None:
                    return False
        return True

    def _first_blocker(self, start, end, margin):
        blockers = []
        for obstacle in self.obstacles:
            if isinstance(obstacle, CircleObstacle):
                distance, time = self._segment_distance(start, end, obstacle.center)
                if distance <= obstacle.radius + margin:
                    blockers.append((time, obstacle))
            else:
                bounds = (
                    obstacle.x_min - margin,
                    obstacle.x_max + margin,
                    obstacle.y_min - margin,
                    obstacle.y_max + margin,
                )
                time = self._segment_box_time(start, end, bounds)
                if time is not None:
                    blockers.append((time, obstacle))
        return min(blockers, key=lambda item: item[0])[1] if blockers else None

    def _route_target(self, start, finish, drone_id):
        if self._segment_clear(start, finish, 0.72):
            return finish

        obstacle = self._first_blocker(start, finish, 0.72)
        if obstacle is None:
            return finish

        clearance = 1.18
        candidates = []
        if isinstance(obstacle, CircleObstacle):
            radius = obstacle.radius + clearance
            for angle in (0.0, pi / 4, pi / 2, 3 * pi / 4, pi, 5 * pi / 4, 3 * pi / 2, 7 * pi / 4):
                candidates.append(
                    (
                        obstacle.center[0] + radius * cos(angle),
                        obstacle.center[1] + radius * sin(angle),
                    )
                )
        else:
            candidates = [
                (obstacle.x_min - clearance, obstacle.y_min - clearance),
                (obstacle.x_min - clearance, obstacle.y_max + clearance),
                (obstacle.x_max + clearance, obstacle.y_min - clearance),
                (obstacle.x_max + clearance, obstacle.y_max + clearance),
            ]

        lane = self.lanes.get(drone_id, self.goal.center[1])
        viable = []
        for candidate in candidates:
            candidate = (
                self._clamp(candidate[0], 0.45, self.width - 0.45),
                self._clamp(candidate[1], 0.45, self.height - 0.45),
            )
            if not self._segment_clear(start, candidate, 0.38):
                continue
            if not self._segment_clear(candidate, finish, 0.38):
                continue
            route_length = self._distance(start, candidate) + self._distance(candidate, finish)
            lane_bias = 0.018 * abs(candidate[1] - lane)
            viable.append((route_length + lane_bias, candidate[1], candidate))
        if viable:
            return min(viable)[2]

        # A second obstacle can hide the best exit.  Pick a safe first leg and
        # let the next control step route the remaining leg again.
        fallback = []
        for candidate in candidates:
            candidate = (
                self._clamp(candidate[0], 0.45, self.width - 0.45),
                self._clamp(candidate[1], 0.45, self.height - 0.45),
            )
            if self._segment_clear(start, candidate, 0.25):
                fallback.append((self._distance(start, candidate), candidate[1], candidate))
        return min(fallback)[2] if fallback else finish

    def _goal_target(self, drone):
        lane = self.lanes.get(drone.id, self.goal.center[1])
        return (
            self.goal.center[0],
            self._clamp(lane, self.goal.y_min + 0.4, self.goal.y_max - 0.4),
        )

    def _obstacle_repulsion(self, position, maximum):
        push_x = push_y = 0.0
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
            distance = hypot(dx, dy)
            surface = distance - radius
            if surface >= 2.6:
                continue
            if distance < 1e-9:
                dx, dy, distance = 0.0, (1.0 if position[1] < self.height / 2 else -1.0), 1.0
            strength = 0.75 * maximum * (2.6 - surface) / 2.6
            push_x += dx / distance * strength
            push_y += dy / distance * strength
        return push_x, push_y

    def _steer(self, drone, finish, speed_scale=1.0):
        spec = self.specs[drone.drone_type]
        target = self._route_target(drone.position, finish, drone.id)
        dx = target[0] - drone.position[0]
        dy = target[1] - drone.position[1]
        distance = hypot(dx, dy)
        if distance < 1e-8:
            desired_x = desired_y = 0.0
        else:
            desired_speed = min(
                spec.max_speed * speed_scale,
                sqrt(2.0 * spec.max_acceleration * distance),
            )
            desired_x = dx / distance * desired_speed
            desired_y = dy / distance * desired_speed

        ax = 2.65 * (desired_x - drone.velocity[0]) - 0.12 * drone.acceleration[0]
        ay = 2.65 * (desired_y - drone.velocity[1]) - 0.12 * drone.acceleration[1]
        repel_x, repel_y = self._obstacle_repulsion(drone.position, spec.max_acceleration)
        ax += repel_x
        ay += repel_y
        if drone.position[1] < 0.65:
            ay += spec.max_acceleration
        elif drone.position[1] > self.height - 0.65:
            ay -= spec.max_acceleration

        magnitude = hypot(ax, ay)
        if magnitude > spec.max_acceleration:
            scale = spec.max_acceleration / magnitude
            ax *= scale
            ay *= scale
        return ax, ay

    def _intercept_point(self, hunter, enemy):
        hunter_speed = self.specs[hunter.drone_type].max_speed
        rx = enemy.position[0] - hunter.position[0]
        ry = enemy.position[1] - hunter.position[1]
        vx, vy = enemy.velocity
        a = vx * vx + vy * vy - hunter_speed * hunter_speed
        b = 2.0 * (rx * vx + ry * vy)
        c = rx * rx + ry * ry
        time = None
        if abs(a) < 1e-9:
            if abs(b) > 1e-9:
                candidate = -c / b
                if candidate >= 0.0:
                    time = candidate
        else:
            discriminant = b * b - 4.0 * a * c
            if discriminant >= 0.0:
                root = sqrt(discriminant)
                roots = [
                    candidate
                    for candidate in ((-b - root) / (2.0 * a), (-b + root) / (2.0 * a))
                    if candidate >= 0.0
                ]
                if roots:
                    time = min(roots)
        if time is None:
            time = hypot(rx, ry) / max(0.1, hunter_speed)
        time = self._clamp(time + 0.2, 0.0, 5.0)
        return (
            self._clamp(
                enemy.position[0]
                + enemy.velocity[0] * time
                + 0.5 * enemy.acceleration[0] * time * time,
                0.3,
                self.width - 0.3,
            ),
            self._clamp(
                enemy.position[1]
                + enemy.velocity[1] * time
                + 0.5 * enemy.acceleration[1] * time * time,
                0.3,
                self.height - 0.3,
            ),
        )

    def _assign_hunters(self, hunters, enemies):
        candidates = []
        for hunter in hunters:
            for enemy in enemies:
                goal_distance = self._distance(enemy.position, self.own_goal.center)
                threshold = 70.0 if enemy.drone_type is DroneType.SLOW else 52.0
                if goal_distance > threshold:
                    continue
                intercept_time = self._distance(hunter.position, enemy.position) / max(
                    0.1, self.specs[hunter.drone_type].max_speed
                )
                value = self.specs[enemy.drone_type].point_value
                urgency = max(0.0, threshold - goal_distance)
                cost = intercept_time - 1.8 * value - 0.07 * urgency
                candidates.append((cost, enemy.id, hunter.id))

        assignments = {}
        claimed = set()
        for _, enemy_id, hunter_id in sorted(candidates):
            if hunter_id not in assignments and enemy_id not in claimed:
                assignments[hunter_id] = enemy_id
                claimed.add(enemy_id)
        return assignments

    def _guard_target(self, guard, escort, enemies):
        threats = []
        for enemy in enemies:
            near_escort = self._distance(enemy.position, escort.position) < 15.0
            near_guard = self._distance(enemy.position, guard.position) < 8.0
            near_goal = self._distance(enemy.position, self.own_goal.center) < 28.0
            if near_escort or near_guard or near_goal:
                value = self.specs[enemy.drone_type].point_value
                score = self._distance(enemy.position, escort.position) - 3.5 * value
                threats.append((score, enemy.id, enemy))
        if threats:
            return self._intercept_point(guard, min(threats)[2])

        if self.goal.contains(escort.position):
            return self._goal_target(guard)

        advance = min(10.0, max(4.0, self._distance(escort.position, self.goal.center) * 0.18))
        return (
            self._clamp(escort.position[0] + self.direction * advance, 0.45, self.width - 0.45),
            self._clamp(escort.position[1] + 0.35 * (self.goal.center[1] - escort.position[1]), 0.45, self.height - 0.45),
        )

    def step(self, state):
        self.tick += 1
        own = {
            drone.id: drone
            for drone in state.own_drones
            if drone.status is DroneStatus.ACTIVE
        }
        enemies = [
            drone
            for drone in state.opponent_drones
            if drone.status is DroneStatus.ACTIVE
        ]
        enemy_by_id = {drone.id: drone for drone in enemies}
        hunters = [
            drone
            for drone in own.values()
            if drone.drone_type is DroneType.FAST and drone.id not in self.guard_ids
        ]
        if self.tick == 1 or self.tick % 4 == 0:
            self.hunter_targets = self._assign_hunters(hunters, enemies)
        else:
            self.hunter_targets = {
                hunter_id: enemy_id
                for hunter_id, enemy_id in self.hunter_targets.items()
                if hunter_id in own and enemy_id in enemy_by_id
            }

        actions = {}
        for drone in own.values():
            if drone.drone_type is DroneType.SLOW:
                actions[drone.id] = self._steer(drone, self._goal_target(drone))
                continue

            escort_id = self.escort_for.get(drone.id)
            escort = own.get(escort_id) if escort_id is not None else None
            if escort is not None:
                target = self._guard_target(drone, escort, enemies)
                actions[drone.id] = self._steer(drone, target, 1.02)
                continue

            enemy_id = self.hunter_targets.get(drone.id)
            if enemy_id in enemy_by_id:
                target = self._intercept_point(drone, enemy_by_id[enemy_id])
                actions[drone.id] = self._steer(drone, target, 1.08)
            else:
                actions[drone.id] = self._steer(drone, self._goal_target(drone), 1.08)
        return actions
