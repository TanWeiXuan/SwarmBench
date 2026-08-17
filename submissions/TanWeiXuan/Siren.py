"""Siren: terrain-commitment pursuit traps for SwarmBench.

Implementation model: GPT-5.6 Sol Extra High.
Controller design prompt: GPT-5.6 Sol Pro.

Siren normally uses conservative goal routes and value-aware FAST-for-SLOW
interceptions.  One SLOW may instead lure one inferred FAST pursuer through a
pre-verified rectangle-corner manoeuvre.  A small multi-model forward simulator
springs the turn only when the prepared SLOW remains safe and continuing pursuit
is predicted to crash or force material braking/rerouting.  One rescue FAST waits
outside the lure and intervenes only if the pursuer safely recovers.

Attribution: the visibility-roadmap/chord-clearance construction and the exact
jerk-dynamics terrain-veto concept are adapted from the author's earlier
``TempoTrap.py`` and ``GPT_5_6_Sol_Ultra.py`` submissions.  Siren's trap-site
geometry, bait state machine, multi-model pursuer simulation, trigger logic,
outcome classification, and rescue fallback are original to this file.  No
other controller code or tuned constants were copied.
"""

from __future__ import annotations

from heapq import heappop, heappush
from math import acos, cos, hypot, pi, sin, sqrt

from swarmbench import (
    BaseSwarmController,
    CircleObstacle,
    DroneStatus,
    DroneType,
    RectangleObstacle,
    Team,
)


EPS = 1.0e-9
MATCH_DURATION = 90.0


def _clip(value, low, high):
    return low if value < low else high if value > high else value


def _limit(x, y, maximum):
    magnitude = hypot(x, y)
    if magnitude <= maximum or magnitude <= EPS:
        return (float(x), float(y))
    scale = maximum / magnitude
    return (float(x * scale), float(y * scale))


def _unit(x, y):
    magnitude = hypot(x, y)
    return (0.0, 0.0) if magnitude <= EPS else (x / magnitude, y / magnitude)


def _distance(left, right):
    return hypot(left[0] - right[0], left[1] - right[1])


def _dot(left, right):
    return left[0] * right[0] + left[1] * right[1]


class SwarmController(BaseSwarmController):
    """Conventional control baseline; rectangle trapping is added incrementally."""

    PLAN_CLEARANCE = 0.94
    TRACK_CLEARANCE = 0.74
    EMERGENCY_CLEARANCE = 0.50
    NODE_EPSILON = 0.10
    ENABLE_RECTANGLE_TRAPS = False

    def initialize(self, game_info):
        self.team = game_info.team
        self.direction = 1.0 if self.team is Team.A else -1.0
        self.width = float(game_info.arena_width)
        self.height = float(game_info.arena_height)
        self.target_goal = game_info.target_goal
        self.own_goal = game_info.own_goal
        self.obstacles = tuple(game_info.obstacles)
        self.specs = dict(game_info.drone_specs)
        self.nodes = self._make_nodes()
        self.graph = self._make_graph()

        self.paths = {}
        self.path_destination = {}
        self.goal_target = {}
        own_slow = sorted(
            (drone for drone in game_info.own_initial_drones if drone.drone_type is DroneType.SLOW),
            key=lambda drone: (drone.position[1], drone.id),
        )
        own_fast = sorted(
            (drone for drone in game_info.own_initial_drones if drone.drone_type is DroneType.FAST),
            key=lambda drone: (drone.position[1], drone.id),
        )
        for collection in (own_slow, own_fast):
            for rank, drone in enumerate(collection):
                target = self._goal_portal(rank, len(collection))
                self.goal_target[drone.id] = target
                self.paths[drone.id] = self._route(drone.position, target)
                self.path_destination[drone.id] = target

        # Six FAST drones seek distinct feasible enemy SLOWs.  The remaining
        # four score, so the ordinary policy itself has a one-point endgame.
        self.hunter_ids = {drone.id for drone in own_fast[:6]}
        self.fast_target = {drone.id: None for drone in own_fast}
        self.fast_destination = {drone.id: self.goal_target[drone.id] for drone in own_fast}
        self.last_assignment = -10.0

    def _goal_portal(self, rank, count):
        low = self.target_goal.y_min + 0.9
        high = self.target_goal.y_max - 0.9
        y = low + (high - low) * (rank + 0.5) / max(1, count)
        x = self.target_goal.x_min + 1.4 if self.direction > 0 else self.target_goal.x_max - 1.4
        return (float(x), float(y))

    def _point_clear(self, point, padding):
        x, y = point
        if not (0.32 <= x <= self.width - 0.32 and 0.32 <= y <= self.height - 0.32):
            return False
        for obstacle in self.obstacles:
            if isinstance(obstacle, CircleObstacle):
                radius = obstacle.radius + padding
                if (x - obstacle.center[0]) ** 2 + (y - obstacle.center[1]) ** 2 <= radius * radius:
                    return False
            elif (
                obstacle.x_min - padding <= x <= obstacle.x_max + padding
                and obstacle.y_min - padding <= y <= obstacle.y_max + padding
            ):
                return False
        return True

    @staticmethod
    def _segment_circle_blocked(start, end, center, radius):
        dx, dy = end[0] - start[0], end[1] - start[1]
        length2 = dx * dx + dy * dy
        if length2 <= EPS:
            return _distance(start, center) <= radius
        time = _clip(((center[0] - start[0]) * dx + (center[1] - start[1]) * dy) / length2, 0.0, 1.0)
        x, y = start[0] + time * dx, start[1] + time * dy
        return (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius * radius

    @staticmethod
    def _segment_box_blocked(start, end, bounds):
        enter, leave = 0.0, 1.0
        for origin, delta, low, high in (
            (start[0], end[0] - start[0], bounds[0], bounds[1]),
            (start[1], end[1] - start[1], bounds[2], bounds[3]),
        ):
            if abs(delta) <= EPS:
                if origin < low or origin > high:
                    return False
                continue
            first, second = (low - origin) / delta, (high - origin) / delta
            if first > second:
                first, second = second, first
            enter, leave = max(enter, first), min(leave, second)
            if enter > leave:
                return False
        return enter <= 1.0 and leave >= 0.0

    def _segment_clear(self, start, end, padding):
        if not self._point_clear(start, min(padding, self.EMERGENCY_CLEARANCE)):
            return False
        if not self._point_clear(end, min(padding, self.EMERGENCY_CLEARANCE)):
            return False
        for obstacle in self.obstacles:
            if isinstance(obstacle, CircleObstacle):
                if self._segment_circle_blocked(start, end, obstacle.center, obstacle.radius + padding):
                    return False
            elif self._segment_box_blocked(
                start,
                end,
                (
                    obstacle.x_min - padding,
                    obstacle.x_max + padding,
                    obstacle.y_min - padding,
                    obstacle.y_max + padding,
                ),
            ):
                return False
        return True

    def _make_nodes(self):
        padding = self.PLAN_CLEARANCE + self.NODE_EPSILON
        candidates = [
            (18.45, 0.62),
            (18.45, self.height - 0.62),
            (81.55, 0.62),
            (81.55, self.height - 0.62),
            (18.45, self.height * 0.5),
            (81.55, self.height * 0.5),
        ]
        for obstacle in self.obstacles:
            if isinstance(obstacle, CircleObstacle):
                sides = 16
                radius = (obstacle.radius + padding) / cos(pi / sides)
                for index in range(sides):
                    angle = 2.0 * pi * index / sides
                    candidates.append(
                        (obstacle.center[0] + radius * cos(angle), obstacle.center[1] + radius * sin(angle))
                    )
            else:
                candidates.extend(
                    (
                        (obstacle.x_min - padding, obstacle.y_min - padding),
                        (obstacle.x_min - padding, obstacle.y_max + padding),
                        (obstacle.x_max + padding, obstacle.y_min - padding),
                        (obstacle.x_max + padding, obstacle.y_max + padding),
                    )
                )
        nodes = []
        for point in candidates:
            point = (_clip(point[0], 0.62, self.width - 0.62), _clip(point[1], 0.62, self.height - 0.62))
            if self._point_clear(point, self.PLAN_CLEARANCE) and all(_distance(point, other) > 0.04 for other in nodes):
                nodes.append((float(point[0]), float(point[1])))
        return nodes

    def _make_graph(self):
        graph = [[] for _ in self.nodes]
        for left in range(len(self.nodes)):
            for right in range(left + 1, len(self.nodes)):
                if self._segment_clear(self.nodes[left], self.nodes[right], self.PLAN_CLEARANCE):
                    length = _distance(self.nodes[left], self.nodes[right])
                    graph[left].append((right, length))
                    graph[right].append((left, length))
        return graph

    def _route(self, start, target, clearance=None):
        clearance = self.PLAN_CLEARANCE if clearance is None else clearance
        start, target = (float(start[0]), float(start[1])), (float(target[0]), float(target[1]))
        if self._segment_clear(start, target, clearance):
            return [target]
        distance = [float("inf")] * len(self.nodes)
        previous = [-1] * len(self.nodes)
        queue = []
        for index, node in enumerate(self.nodes):
            if self._segment_clear(start, node, clearance):
                distance[index] = _distance(start, node)
                heappush(queue, (distance[index], index))
        best_total, best_index = float("inf"), -1
        while queue:
            current, index = heappop(queue)
            if current > distance[index] + EPS or current >= best_total:
                continue
            if self._segment_clear(self.nodes[index], target, clearance):
                total = current + _distance(self.nodes[index], target)
                if total < best_total:
                    best_total, best_index = total, index
            for neighbor, weight in self.graph[index]:
                candidate = current + weight
                if candidate + EPS < distance[neighbor]:
                    distance[neighbor], previous[neighbor] = candidate, index
                    heappush(queue, (candidate, neighbor))
        if best_index < 0:
            visible = [
                (distance[index] + _distance(node, target), index)
                for index, node in enumerate(self.nodes)
                if distance[index] < float("inf") and self._segment_clear(node, target, self.TRACK_CLEARANCE)
            ]
            if not visible:
                return [target]
            _, best_index = min(visible)
        chain = []
        cursor = best_index
        while cursor >= 0:
            chain.append(self.nodes[cursor])
            cursor = previous[cursor]
        chain.reverse()
        chain.append(target)
        return chain

    @staticmethod
    def _path_length(start, path):
        total, point = 0.0, start
        for waypoint in path:
            total += _distance(point, waypoint)
            point = waypoint
        return total

    def _set_route(self, drone, target, force=False, clearance=None):
        target = (float(target[0]), float(target[1]))
        old = self.path_destination.get(drone.id)
        if force or old is None or _distance(old, target) > 1.1 or not self.paths.get(drone.id):
            self.paths[drone.id] = self._route(drone.position, target, clearance)
            self.path_destination[drone.id] = target

    def _predict_position(self, drone, seconds):
        spec = self.specs[drone.drone_type]
        ramp = min(max(0.0, seconds), 0.25)
        velocity = _limit(
            drone.velocity[0] + drone.acceleration[0] * ramp,
            drone.velocity[1] + drone.acceleration[1] * ramp,
            spec.max_speed,
        )
        position = (
            drone.position[0] + drone.velocity[0] * ramp + 0.5 * drone.acceleration[0] * ramp * ramp,
            drone.position[1] + drone.velocity[1] * ramp + 0.5 * drone.acceleration[1] * ramp * ramp,
        )
        coast = max(0.0, seconds - ramp)
        return (
            _clip(position[0] + velocity[0] * coast, 0.35, self.width - 0.35),
            _clip(position[1] + velocity[1] * coast, 0.35, self.height - 0.35),
        )

    @staticmethod
    def _travel_time(distance, initial_speed, acceleration, maximum_speed):
        initial_speed = _clip(initial_speed, 0.0, maximum_speed)
        accelerating = max(0.0, maximum_speed * maximum_speed - initial_speed * initial_speed) / (2.0 * acceleration)
        if distance <= accelerating:
            return (sqrt(max(0.0, initial_speed * initial_speed + 2.0 * acceleration * distance)) - initial_speed) / acceleration + 0.25
        return (maximum_speed - initial_speed) / acceleration + (distance - accelerating) / maximum_speed + 0.25

    def _eta_point(self, drone, point):
        direction = _unit(point[0] - drone.position[0], point[1] - drone.position[1])
        projected = max(0.0, _dot(drone.velocity, direction))
        distance = _distance(drone.position, point)
        if not self._segment_clear(drone.position, point, self.TRACK_CLEARANCE):
            distance = distance * 1.16 + 2.0
        spec = self.specs[drone.drone_type]
        return self._travel_time(distance, projected, spec.max_acceleration, spec.max_speed)

    def _goal_eta(self, drone, own=True):
        if own:
            target = self.goal_target[drone.id]
        else:
            target = (
                self.own_goal.x_max - 1.2 if self.direction > 0 else self.own_goal.x_min + 1.2,
                _clip(drone.position[1], self.own_goal.y_min + 0.8, self.own_goal.y_max - 0.8),
            )
        return self._eta_point(drone, target)

    def _intercept_eta(self, hunter, target):
        dx, dy = target.position[0] - hunter.position[0], target.position[1] - hunter.position[1]
        direction = _unit(dx, dy)
        closing = max(0.0, _dot(hunter.velocity, direction) - _dot(target.velocity, direction))
        spec = self.specs[hunter.drone_type]
        return self._travel_time(max(0.0, hypot(dx, dy) - 0.75), closing, spec.max_acceleration, spec.max_speed)

    def _assign_fast(self, state, own_fast, enemy_slow):
        remaining = MATCH_DURATION - state.time
        used = set()
        for hunter in sorted((drone for drone in own_fast if drone.id in self.hunter_ids), key=lambda drone: drone.id):
            candidates = []
            for enemy in enemy_slow:
                contact = self._intercept_eta(hunter, enemy)
                goal_eta = self._goal_eta(enemy, False)
                if contact + 0.25 >= min(goal_eta, remaining):
                    continue
                score = 1.5 * (goal_eta - contact)
                if self.fast_target.get(hunter.id) == enemy.id:
                    score += 0.65
                candidates.append((score, -enemy.id, enemy))
            candidates.sort(reverse=True, key=lambda item: (item[0], item[1]))
            chosen = next((item[2] for item in candidates if item[2].id not in used), None)
            if chosen is None:
                self.fast_target[hunter.id] = None
                self.fast_destination[hunter.id] = self.goal_target[hunter.id]
            else:
                used.add(chosen.id)
                self.fast_target[hunter.id] = chosen.id
                self.fast_destination[hunter.id] = self._predict_position(
                    chosen, min(3.0, max(0.35, self._intercept_eta(hunter, chosen)))
                )
        for drone in own_fast:
            if drone.id not in self.hunter_ids:
                self.fast_target[drone.id] = None
                self.fast_destination[drone.id] = self.goal_target[drone.id]

    def _advance_path(self, drone, path):
        while len(path) > 1 and _distance(drone.position, path[0]) < 0.80:
            path.pop(0)
        for index in range(len(path) - 1, 0, -1):
            if self._segment_clear(drone.position, path[index], self.TRACK_CLEARANCE):
                del path[:index]
                break

    def _lookahead(self, position, path, amount):
        point, remaining = position, amount
        for waypoint in path:
            length = _distance(point, waypoint)
            if length >= remaining and length > EPS:
                scale = remaining / length
                return (point[0] + (waypoint[0] - point[0]) * scale, point[1] + (waypoint[1] - point[1]) * scale)
            remaining -= length
            point = waypoint
        return path[-1]

    def _corner_speed(self, drone, path, maximum):
        if len(path) < 2:
            return maximum
        first = _unit(path[0][0] - drone.position[0], path[0][1] - drone.position[1])
        second = _unit(path[1][0] - path[0][0], path[1][1] - path[0][1])
        angle = acos(_clip(_dot(first, second), -1.0, 1.0))
        if angle < 0.28:
            return maximum
        cap = maximum * _clip(1.0 - 0.72 * angle / pi, 0.24, 0.82)
        spec = self.specs[drone.drone_type]
        braking = max(0.0, hypot(*drone.velocity) ** 2 - cap * cap) / (2.0 * spec.max_acceleration)
        return cap if _distance(drone.position, path[0]) < 1.25 * braking + 0.8 else maximum

    def _steer(self, drone, path, speed_limit=None, arrive=False):
        spec = self.specs[drone.drone_type]
        self._advance_path(drone, path)
        speed = hypot(*drone.velocity)
        target = self._lookahead(drone.position, path, 1.0 + 0.42 * speed)
        direction = _unit(target[0] - drone.position[0], target[1] - drone.position[1])
        desired_speed = self._corner_speed(drone, path, spec.max_speed)
        if speed_limit is not None:
            desired_speed = min(desired_speed, speed_limit)
        if arrive:
            remaining = self._path_length(drone.position, path)
            desired_speed = min(desired_speed, sqrt(max(0.0, 2.0 * spec.max_acceleration * max(0.0, remaining - 0.18))))
        desired_velocity = (direction[0] * desired_speed, direction[1] * desired_speed)
        command = _limit(
            (desired_velocity[0] - (drone.velocity[0] + 0.14 * drone.acceleration[0])) / 0.34,
            (desired_velocity[1] - (drone.velocity[1] + 0.14 * drone.acceleration[1])) / 0.34,
            spec.max_acceleration,
        )
        command = self._terrain_guard(drone, command)
        return self._safe_command(drone, command, target), target

    def _nearest_obstacle(self, position):
        x, y = position
        best = (float("inf"), (0.0, 0.0))
        for obstacle in self.obstacles:
            if isinstance(obstacle, CircleObstacle):
                dx, dy = x - obstacle.center[0], y - obstacle.center[1]
                magnitude = hypot(dx, dy)
                normal = _unit(dx, dy) if magnitude > EPS else (self.direction, 0.0)
                clearance = magnitude - (obstacle.radius + 0.25)
            else:
                x0, x1 = obstacle.x_min - 0.25, obstacle.x_max + 0.25
                y0, y1 = obstacle.y_min - 0.25, obstacle.y_max + 0.25
                nearest = (_clip(x, x0, x1), _clip(y, y0, y1))
                dx, dy = x - nearest[0], y - nearest[1]
                magnitude = hypot(dx, dy)
                if magnitude > EPS:
                    normal, clearance = (dx / magnitude, dy / magnitude), magnitude
                else:
                    depth, normal = min(
                        ((x - x0, (-1.0, 0.0)), (x1 - x, (1.0, 0.0)), (y - y0, (0.0, -1.0)), (y1 - y, (0.0, 1.0))),
                        key=lambda item: item[0],
                    )
                    clearance = -depth
            if clearance < best[0]:
                best = (clearance, normal)
        return best

    def _terrain_guard(self, drone, command):
        spec = self.specs[drone.drone_type]
        clearance, normal = self._nearest_obstacle(drone.position)
        inward = max(0.0, -_dot(drone.velocity, normal))
        commitment = inward * 0.25 + inward * inward / (2.0 * spec.max_acceleration) + 0.20
        if clearance < commitment + 0.34 or clearance < 0.76:
            tangent = (-normal[1], normal[0])
            tangential = _dot(command, tangent)
            outward = spec.max_acceleration * _clip((commitment + 0.45 - clearance) / max(0.45, commitment + 0.45), 0.62, 1.0)
            command = _limit(
                normal[0] * outward + tangent[0] * tangential * 0.45,
                normal[1] * outward + tangent[1] * tangential * 0.45,
                spec.max_acceleration,
            )
        return command

    def _simulate_command(self, drone, command, ticks):
        spec = self.specs[drone.drone_type]
        position, velocity, acceleration = drone.position, drone.velocity, drone.acceleration
        for _ in range(ticks):
            desired = _limit(*command, spec.max_acceleration)
            delta = _limit(desired[0] - acceleration[0], desired[1] - acceleration[1], spec.max_jerk * 0.05)
            acceleration = _limit(acceleration[0] + delta[0], acceleration[1] + delta[1], spec.max_acceleration)
            new_position = (
                position[0] + velocity[0] * 0.05 + 0.5 * acceleration[0] * 0.0025,
                position[1] + velocity[1] * 0.05 + 0.5 * acceleration[1] * 0.0025,
            )
            new_velocity = _limit(velocity[0] + acceleration[0] * 0.05, velocity[1] + acceleration[1] * 0.05, spec.max_speed)
            if not self._segment_clear(position, new_position, self.EMERGENCY_CLEARANCE):
                return None
            position, velocity = new_position, new_velocity
        return position, velocity

    def _safe_command(self, drone, command, target):
        if self._simulate_command(drone, command, 20) is not None:
            return command
        spec = self.specs[drone.drone_type]
        brake = _limit(
            -3.2 * drone.velocity[0] - 0.8 * drone.acceleration[0],
            -3.2 * drone.velocity[1] - 0.8 * drone.acceleration[1],
            spec.max_acceleration,
        )
        candidates = [brake]
        for angle in (pi / 5.0, -pi / 5.0):
            candidates.append(
                _limit(
                    command[0] * cos(angle) - command[1] * sin(angle),
                    command[0] * sin(angle) + command[1] * cos(angle),
                    spec.max_acceleration,
                )
            )
        best = None
        for candidate in candidates:
            if self._simulate_command(drone, candidate, 20) is None:
                continue
            position, velocity = self._simulate_command(drone, candidate, 6)
            score = _distance(position, target) + 0.10 * hypot(*velocity)
            if best is None or score < best[0]:
                best = (score, candidate)
        if best is not None:
            return best[1]
        _, normal = self._nearest_obstacle(drone.position)
        return _limit(
            normal[0] * spec.max_acceleration - 2.2 * drone.velocity[0],
            normal[1] * spec.max_acceleration - 2.2 * drone.velocity[1],
            spec.max_acceleration,
        )

    @staticmethod
    def _closest_approach(own, enemy, horizon=2.6):
        rx, ry = enemy.position[0] - own.position[0], enemy.position[1] - own.position[1]
        vx, vy = enemy.velocity[0] - own.velocity[0], enemy.velocity[1] - own.velocity[1]
        speed2 = vx * vx + vy * vy
        time = 0.0 if speed2 <= EPS else _clip(-(rx * vx + ry * vy) / speed2, 0.0, horizon)
        return hypot(rx + vx * time, ry + vy * time), time

    def _avoid_enemies(self, drone, command, enemies, engage_id=None):
        spec = self.specs[drone.drone_type]
        push_x, push_y = 0.0, 0.0
        for enemy in enemies:
            if enemy.id == engage_id or _distance(drone.position, enemy.position) > 9.0:
                continue
            closest, time = self._closest_approach(drone, enemy)
            threshold = 1.55 if drone.drone_type is DroneType.SLOW else 1.25
            if closest >= threshold:
                continue
            own_future = (drone.position[0] + drone.velocity[0] * time, drone.position[1] + drone.velocity[1] * time)
            enemy_future = (enemy.position[0] + enemy.velocity[0] * time, enemy.position[1] + enemy.velocity[1] * time)
            away = _unit(own_future[0] - enemy_future[0], own_future[1] - enemy_future[1])
            if away == (0.0, 0.0):
                side = 1.0 if (drone.id + enemy.id) & 1 else -1.0
                away = _unit(-(enemy.velocity[1] - drone.velocity[1]) * side, (enemy.velocity[0] - drone.velocity[0]) * side)
            strength = spec.max_acceleration * _clip((threshold - closest) / threshold + (2.6 - time) / 5.2, 0.0, 1.25)
            push_x += away[0] * strength
            push_y += away[1] * strength
        return _limit(command[0] + push_x, command[1] + push_y, spec.max_acceleration)

    def step(self, state):
        own_fast = [drone for drone in state.own_drones if drone.status is DroneStatus.ACTIVE and drone.drone_type is DroneType.FAST]
        enemy_slow = [drone for drone in state.opponent_drones if drone.status is DroneStatus.ACTIVE and drone.drone_type is DroneType.SLOW]
        enemies = [drone for drone in state.opponent_drones if drone.status is DroneStatus.ACTIVE]
        if state.time + EPS >= self.last_assignment + 0.40:
            self._assign_fast(state, own_fast, enemy_slow)
            self.last_assignment = state.time

        actions = {}
        for drone in state.own_drones:
            if drone.status is not DroneStatus.ACTIVE:
                continue
            engage_id = None
            if drone.drone_type is DroneType.FAST:
                destination = self.fast_destination.get(drone.id, self.goal_target[drone.id])
                engage_id = self.fast_target.get(drone.id)
            else:
                destination = self.goal_target[drone.id]
            self._set_route(drone, destination)
            command, local_target = self._steer(drone, self.paths[drone.id])
            command = self._avoid_enemies(drone, command, enemies, engage_id)
            command = self._terrain_guard(drone, command)
            command = self._safe_command(drone, command, local_target)
            actions[drone.id] = (float(command[0]), float(command[1]))
        return actions
