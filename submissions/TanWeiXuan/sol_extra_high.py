"""Sol Extra High: a deterministic screen-and-intercept swarm controller.

Authorship: the code and strategy in this file were entirely produced by
GPT-5.6 Sol using extra-high reasoning.  The only human input was the request
to create a community controller; no human-authored code, strategy choices,
or iterative guidance were used.
"""

from __future__ import annotations

from math import hypot, sqrt

from swarmbench import BaseSwarmController, CircleObstacle, DroneStatus, DroneType


class SwarmController(BaseSwarmController):
    """Escort high-value attackers and hunt threats with predictive steering."""

    def initialize(self, game_info):
        self.goal = game_info.target_goal
        self.own_goal = game_info.own_goal
        self.obstacles = game_info.obstacles
        self.specs = dict(game_info.drone_specs)
        self.width = game_info.arena_width
        self.height = game_info.arena_height
        self.direction = 1.0 if self.goal.center[0] > self.own_goal.center[0] else -1.0
        self.step_number = 0
        self.hunter_targets = {}

        initial = game_info.own_initial_drones
        fast = sorted((drone for drone in initial if drone.drone_type is DroneType.FAST), key=lambda drone: (drone.position[1], drone.id))
        slow = sorted((drone for drone in initial if drone.drone_type is DroneType.SLOW), key=lambda drone: (drone.position[1], drone.id))

        # Six screens cover the full vertical spread; four FAST drones remain
        # free to remove opposing five-point drones or score when no chase pays.
        screen_indices = (0, 2, 4, 5, 7, 9)
        self.escort_for = {fast[index].id: slow[index].id for index in screen_indices}

        ordered = sorted(initial, key=lambda drone: (drone.position[1], drone.id))
        usable_height = max(2.0, self.goal.y_max - self.goal.y_min - 1.5)
        self.goal_lanes = {
            drone.id: self.goal.y_min + 0.75 + usable_height * (rank + 0.5) / len(ordered)
            for rank, drone in enumerate(ordered)
        }

    @staticmethod
    def _distance(left, right):
        return hypot(left[0] - right[0], left[1] - right[1])

    @staticmethod
    def _segment_distance(start, end, point):
        dx, dy = end[0] - start[0], end[1] - start[1]
        length_sq = dx * dx + dy * dy
        if length_sq <= 1e-12:
            return hypot(point[0] - start[0], point[1] - start[1]), 0.0
        t = max(0.0, min(1.0, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_sq))
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
            enter, leave = max(enter, first), min(leave, second)
            if enter > leave:
                return None
        return enter if 0.0 <= enter <= 1.0 else None

    def _blockers(self, start, end, margin):
        blockers = []
        for obstacle in self.obstacles:
            if isinstance(obstacle, CircleObstacle):
                clearance, time = self._segment_distance(start, end, obstacle.center)
                if clearance < obstacle.radius + margin:
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
        return sorted(blockers, key=lambda item: item[0])

    def _route_target(self, start, finish, drone_id):
        blockers = self._blockers(start, finish, 0.8)
        if not blockers:
            return finish

        obstacle = blockers[0][1]
        clearance = 1.15
        if isinstance(obstacle, CircleObstacle):
            radius = obstacle.radius + clearance
            candidates = (
                (obstacle.center[0], obstacle.center[1] - radius),
                (obstacle.center[0], obstacle.center[1] + radius),
                (obstacle.center[0] - radius, obstacle.center[1]),
                (obstacle.center[0] + radius, obstacle.center[1]),
            )
        else:
            candidates = (
                (obstacle.x_min - clearance, obstacle.y_min - clearance),
                (obstacle.x_min - clearance, obstacle.y_max + clearance),
                (obstacle.x_max + clearance, obstacle.y_min - clearance),
                (obstacle.x_max + clearance, obstacle.y_max + clearance),
            )

        viable = []
        for candidate in candidates:
            if not (0.4 < candidate[0] < self.width - 0.4 and 0.4 < candidate[1] < self.height - 0.4):
                continue
            if self._blockers(start, candidate, 0.55):
                continue
            cost = self._distance(start, candidate) + self._distance(candidate, finish)
            # A stable, tiny tie-break prevents mirrored agents from bunching.
            side_bias = 0.001 * abs(candidate[1] - self.goal_lanes.get(drone_id, self.goal.center[1]))
            viable.append((cost + side_bias, candidate[1], candidate))
        if viable:
            return min(viable)[2]

        # This is only an emergency fallback for a drone already inside a
        # planning margin.  Immediate repulsion in _steer moves it back out.
        return min(candidates, key=lambda point: (self._distance(start, point) + self._distance(point, finish), point[1]))

    def _obstacle_repulsion(self, position, maximum):
        push_x = push_y = 0.0
        for obstacle in self.obstacles:
            if isinstance(obstacle, CircleObstacle):
                closest = obstacle.center
                radius = obstacle.radius
            else:
                closest = (
                    min(obstacle.x_max, max(obstacle.x_min, position[0])),
                    min(obstacle.y_max, max(obstacle.y_min, position[1])),
                )
                radius = 0.0
            dx, dy = position[0] - closest[0], position[1] - closest[1]
            center_distance = hypot(dx, dy)
            surface_distance = center_distance - radius
            if surface_distance < 2.5:
                if center_distance < 1e-9:
                    dx, dy, center_distance = 0.0, 1.0 if position[1] >= self.height / 2 else -1.0, 1.0
                strength = maximum * max(0.0, 2.5 - surface_distance) / 2.5
                push_x += dx / center_distance * strength
                push_y += dy / center_distance * strength
        return push_x, push_y

    def _steer(self, drone, finish, speed_scale=1.0):
        spec = self.specs[drone.drone_type]
        target = self._route_target(drone.position, finish, drone.id)
        dx, dy = target[0] - drone.position[0], target[1] - drone.position[1]
        remaining = hypot(dx, dy)
        if remaining < 1e-8:
            desired_x = desired_y = 0.0
        else:
            desired_speed = min(spec.max_speed * speed_scale, sqrt(2.0 * spec.max_acceleration * remaining))
            desired_x = dx / remaining * desired_speed
            desired_y = dy / remaining * desired_speed

        ax = 2.7 * (desired_x - drone.velocity[0]) - 0.12 * drone.acceleration[0]
        ay = 2.7 * (desired_y - drone.velocity[1]) - 0.12 * drone.acceleration[1]
        repel_x, repel_y = self._obstacle_repulsion(drone.position, spec.max_acceleration)
        ax += repel_x
        ay += repel_y

        if drone.position[1] < 0.7:
            ay += spec.max_acceleration
        elif drone.position[1] > self.height - 0.7:
            ay -= spec.max_acceleration
        return ax, ay

    def _intercept_point(self, hunter, enemy):
        speed = self.specs[hunter.drone_type].max_speed
        rx = enemy.position[0] - hunter.position[0]
        ry = enemy.position[1] - hunter.position[1]
        vx, vy = enemy.velocity
        a = vx * vx + vy * vy - speed * speed
        b = 2.0 * (rx * vx + ry * vy)
        c = rx * rx + ry * ry
        time = None
        if abs(a) < 1e-9:
            if b < -1e-9:
                time = -c / b
        else:
            discriminant = b * b - 4.0 * a * c
            if discriminant >= 0.0:
                root = sqrt(discriminant)
                roots = [value for value in ((-b - root) / (2.0 * a), (-b + root) / (2.0 * a)) if value >= 0.0]
                if roots:
                    time = min(roots)
        if time is None:
            time = hypot(rx, ry) / max(0.1, speed)
        time = max(0.0, min(5.0, time + 0.2))
        return (
            min(self.width - 0.3, max(0.3, enemy.position[0] + enemy.velocity[0] * time + 0.25 * enemy.acceleration[0] * time * time)),
            min(self.height - 0.3, max(0.3, enemy.position[1] + enemy.velocity[1] * time + 0.25 * enemy.acceleration[1] * time * time)),
        )

    def _assign_hunters(self, hunters, enemies):
        candidates = []
        for hunter in hunters:
            for enemy in enemies:
                separation = self._distance(hunter.position, enemy.position)
                intercept_time = separation / max(0.1, self.specs[hunter.drone_type].max_speed)
                enemy_goal_distance = abs(enemy.position[0] - self.own_goal.center[0])
                enemy_goal_time = enemy_goal_distance / max(0.1, self.specs[enemy.drone_type].max_speed)
                if enemy.drone_type is DroneType.SLOW and intercept_time > enemy_goal_time + 1.5:
                    continue
                value_bonus = 24.0 if enemy.drone_type is DroneType.SLOW else 0.0
                urgency = max(0.0, 18.0 - enemy_goal_distance) * 0.7
                cost = intercept_time - value_bonus - urgency
                candidates.append((cost, enemy.id, hunter.id))

        assignments = {}
        claimed = set()
        for _, enemy_id, hunter_id in sorted(candidates):
            if hunter_id not in assignments and enemy_id not in claimed:
                assignments[hunter_id] = enemy_id
                claimed.add(enemy_id)
        return assignments

    def _goal_target(self, drone):
        lane = min(self.goal.y_max - 0.45, max(self.goal.y_min + 0.45, self.goal_lanes[drone.id]))
        return self.goal.center[0], lane

    def step(self, state):
        self.step_number += 1
        own = {drone.id: drone for drone in state.own_drones if drone.status is DroneStatus.ACTIVE}
        enemies = {drone.id: drone for drone in state.opponent_drones if drone.status is DroneStatus.ACTIVE}
        fast = [drone for drone in own.values() if drone.drone_type is DroneType.FAST]

        active_escorts = {
            fast_id: slow_id
            for fast_id, slow_id in self.escort_for.items()
            if fast_id in own and slow_id in own
        }
        hunters = [drone for drone in fast if drone.id not in active_escorts]
        if self.step_number == 1 or self.step_number % 4 == 0:
            self.hunter_targets = self._assign_hunters(hunters, list(enemies.values()))
        else:
            self.hunter_targets = {
                hunter_id: enemy_id
                for hunter_id, enemy_id in self.hunter_targets.items()
                if hunter_id in own and enemy_id in enemies
            }

        actions = {}
        for drone in own.values():
            if drone.drone_type is DroneType.SLOW:
                actions[drone.id] = self._steer(drone, self._goal_target(drone))
                continue

            escorted_id = active_escorts.get(drone.id)
            if escorted_id is not None:
                escorted = own[escorted_id]
                nearby = [
                    enemy
                    for enemy in enemies.values()
                    if self._distance(enemy.position, escorted.position) < 10.0
                    and self.direction * (enemy.position[0] - escorted.position[0]) > -3.0
                ]
                if nearby:
                    threat = min(
                        nearby,
                        key=lambda enemy: (
                            self._distance(enemy.position, escorted.position)
                            - (4.0 if enemy.drone_type is DroneType.FAST else 2.0),
                            enemy.id,
                        ),
                    )
                    target = self._intercept_point(drone, threat)
                elif self.goal.contains(escorted.position):
                    target = self._goal_target(drone)
                else:
                    target = (
                        min(self.width - 0.4, max(0.4, escorted.position[0] + self.direction * 3.0)),
                        escorted.position[1],
                    )
                actions[drone.id] = self._steer(drone, target)
                continue

            enemy_id = self.hunter_targets.get(drone.id)
            if enemy_id in enemies:
                actions[drone.id] = self._steer(drone, self._intercept_point(drone, enemies[enemy_id]))
            else:
                actions[drone.id] = self._steer(drone, self._goal_target(drone))
        return actions
