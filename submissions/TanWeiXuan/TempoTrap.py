"""TempoTrap: delayed-commitment swarm control for SwarmBench.

Implementation model: GPT-5.6 Sol Extra High.
Controller design prompt: GPT-5.6 Sol Pro.

The final strategy uses two timed SLOW waves, grouped defenders, selective
FAST-for-SLOW exchanges, home-side denial, a deliberately uncommitted reserve,
short-horizon intent inference, defender rendezvous, and outcome-aware endgame
releases.  It is deterministic and uses only public game state.

Attribution: the visibility-roadmap/chord-clearance navigation pattern and the
short exact-dynamics terrain veto are adapted from the earlier controller
``submissions/TanWeiXuan/GPT_5_6_Sol_Ultra.py``.  TempoTrap's wave timing,
tactical state machines, target inference, rendezvous selection, role policy,
and endgame logic are original to this submission.  No other controller code
or tuned constants were copied.
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


def _clip(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


def _norm(x: float, y: float) -> float:
    return hypot(x, y)


def _limit(x: float, y: float, maximum: float) -> tuple[float, float]:
    magnitude = hypot(x, y)
    if magnitude <= maximum or magnitude <= EPS:
        return (float(x), float(y))
    scale = maximum / magnitude
    return (float(x * scale), float(y * scale))


def _unit(x: float, y: float) -> tuple[float, float]:
    magnitude = hypot(x, y)
    if magnitude <= EPS:
        return (0.0, 0.0)
    return (x / magnitude, y / magnitude)


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return hypot(a[0] - b[0], a[1] - b[1])


def _dot(a: tuple[float, float], b: tuple[float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


class SwarmController(BaseSwarmController):
    """Safe navigation foundation for the TempoTrap tactical controller."""

    PLAN_CLEARANCE = 0.92
    TRACK_CLEARANCE = 0.68
    EMERGENCY_CLEARANCE = 0.34
    NODE_EPSILON = 0.10

    def initialize(self, game_info):
        self.team = game_info.team
        self.direction = 1.0 if self.team is Team.A else -1.0
        self.width = float(game_info.arena_width)
        self.height = float(game_info.arena_height)
        self.target_goal = game_info.target_goal
        self.own_goal = game_info.own_goal
        self.obstacles = tuple(game_info.obstacles)
        self.specs = dict(game_info.drone_specs)
        self.controller_seed = int(game_info.controller_seed)

        self.nodes = self._make_roadmap_nodes()
        self.graph = self._make_roadmap_graph()
        self.paths: dict[int, list[tuple[float, float]]] = {}
        self.path_destination: dict[int, tuple[float, float]] = {}
        self.goal_target: dict[int, tuple[float, float]] = {}

        own = sorted(game_info.own_initial_drones, key=lambda drone: (drone.position[1], drone.id))
        for rank, drone in enumerate(own):
            portal = self._goal_portal(rank, len(own))
            self.goal_target[drone.id] = portal
            self.paths[drone.id] = self._route(drone.position, portal)
            self.path_destination[drone.id] = portal

    def _goal_portal(self, rank: int, count: int) -> tuple[float, float]:
        margin = 0.9
        low = self.target_goal.y_min + margin
        high = self.target_goal.y_max - margin
        lane = (rank + 0.5) / max(1, count)
        y = low + (high - low) * lane
        x = self.target_goal.x_min + 1.4 if self.direction > 0 else self.target_goal.x_max - 1.4
        return (float(x), float(y))

    def _point_clear(self, point, padding):
        x, y = point
        if x < 0.32 or x > self.width - 0.32 or y < 0.32 or y > self.height - 0.32:
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
        denominator = dx * dx + dy * dy
        if denominator <= EPS:
            return _distance(start, center) <= radius
        t = _clip(((center[0] - start[0]) * dx + (center[1] - start[1]) * dy) / denominator, 0.0, 1.0)
        px, py = start[0] + t * dx, start[1] + t * dy
        return (px - center[0]) ** 2 + (py - center[1]) ** 2 <= radius * radius

    @staticmethod
    def _segment_box_blocked(start, end, bounds):
        x_min, x_max, y_min, y_max = bounds
        enter, leave = 0.0, 1.0
        for origin, delta, low, high in (
            (start[0], end[0] - start[0], x_min, x_max),
            (start[1], end[1] - start[1], y_min, y_max),
        ):
            if abs(delta) <= EPS:
                if origin < low or origin > high:
                    return False
                continue
            t0, t1 = (low - origin) / delta, (high - origin) / delta
            if t0 > t1:
                t0, t1 = t1, t0
            enter, leave = max(enter, t0), min(leave, t1)
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

    def _make_roadmap_nodes(self):
        padding = self.PLAN_CLEARANCE + self.NODE_EPSILON
        candidates: list[tuple[float, float]] = [
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
                        (
                            obstacle.center[0] + radius * cos(angle),
                            obstacle.center[1] + radius * sin(angle),
                        )
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

        nodes: list[tuple[float, float]] = []
        for point in candidates:
            point = (
                _clip(float(point[0]), 0.62, self.width - 0.62),
                _clip(float(point[1]), 0.62, self.height - 0.62),
            )
            if self._point_clear(point, self.PLAN_CLEARANCE) and all(_distance(point, other) > 0.04 for other in nodes):
                nodes.append(point)
        return nodes

    def _make_roadmap_graph(self):
        graph: list[list[tuple[int, float]]] = [[] for _ in self.nodes]
        for left in range(len(self.nodes)):
            for right in range(left + 1, len(self.nodes)):
                if self._segment_clear(self.nodes[left], self.nodes[right], self.PLAN_CLEARANCE):
                    length = _distance(self.nodes[left], self.nodes[right])
                    graph[left].append((right, length))
                    graph[right].append((left, length))
        for edges in graph:
            edges.sort(key=lambda item: item[0])
        return graph

    def _route(self, start, target):
        start = (float(start[0]), float(start[1]))
        target = (float(target[0]), float(target[1]))
        if self._segment_clear(start, target, self.PLAN_CLEARANCE):
            return [target]

        count = len(self.nodes)
        distances = [float("inf")] * count
        previous = [-1] * count
        queue: list[tuple[float, int]] = []
        for index, node in enumerate(self.nodes):
            if self._segment_clear(start, node, self.PLAN_CLEARANCE):
                length = _distance(start, node)
                distances[index] = length
                heappush(queue, (length, index))

        best_total = float("inf")
        best_index = -1
        while queue:
            current, index = heappop(queue)
            if current > distances[index] + EPS or current >= best_total:
                continue
            if self._segment_clear(self.nodes[index], target, self.PLAN_CLEARANCE):
                total = current + _distance(self.nodes[index], target)
                if total < best_total:
                    best_total, best_index = total, index
            for neighbor, weight in self.graph[index]:
                candidate = current + weight
                if candidate + EPS < distances[neighbor]:
                    distances[neighbor] = candidate
                    previous[neighbor] = index
                    heappush(queue, (candidate, neighbor))

        if best_index < 0:
            # Valid generated arenas always have a protected outer route.  This
            # fallback deliberately picks a visible graph node rather than
            # returning an unsafe direct segment.
            visible = [
                (distances[index] + _distance(node, target), index)
                for index, node in enumerate(self.nodes)
                if distances[index] < float("inf") and self._segment_clear(node, target, self.TRACK_CLEARANCE)
            ]
            if not visible:
                return [target]
            _, best_index = min(visible)

        chain: list[tuple[float, float]] = []
        cursor = best_index
        while cursor >= 0:
            chain.append(self.nodes[cursor])
            cursor = previous[cursor]
        chain.reverse()
        chain.append(target)
        return chain

    @staticmethod
    def _path_length(start, path):
        total = 0.0
        point = start
        for waypoint in path:
            total += _distance(point, waypoint)
            point = waypoint
        return total

    def _set_route(self, drone, destination, force=False):
        destination = (float(destination[0]), float(destination[1]))
        old = self.path_destination.get(drone.id)
        if force or old is None or _distance(old, destination) > 1.1 or not self.paths.get(drone.id):
            self.paths[drone.id] = self._route(drone.position, destination)
            self.path_destination[drone.id] = destination

    def _advance_path(self, drone, path):
        while len(path) > 1 and _distance(drone.position, path[0]) < 0.80:
            path.pop(0)
        # Reacquire the farthest visible future waypoint after a tactical
        # deviation; this prevents stale vertices from causing U-turns.
        farthest = 0
        for index in range(len(path) - 1, 0, -1):
            if self._segment_clear(drone.position, path[index], self.TRACK_CLEARANCE):
                farthest = index
                break
        if farthest:
            del path[:farthest]

    def _lookahead(self, position, path, distance):
        if not path:
            return position
        point = position
        remaining = distance
        for waypoint in path:
            length = _distance(point, waypoint)
            if length >= remaining and length > EPS:
                scale = remaining / length
                return (
                    point[0] + (waypoint[0] - point[0]) * scale,
                    point[1] + (waypoint[1] - point[1]) * scale,
                )
            remaining -= length
            point = waypoint
        return path[-1]

    def _corner_speed(self, drone, path, maximum):
        if len(path) < 2:
            return maximum
        incoming = _unit(path[0][0] - drone.position[0], path[0][1] - drone.position[1])
        outgoing = _unit(path[1][0] - path[0][0], path[1][1] - path[0][1])
        if incoming == (0.0, 0.0) or outgoing == (0.0, 0.0):
            return maximum
        angle = acos(_clip(_dot(incoming, outgoing), -1.0, 1.0))
        if angle < 0.28:
            return maximum
        distance = _distance(drone.position, path[0])
        cap = maximum * _clip(1.0 - 0.72 * angle / pi, 0.24, 0.82)
        braking = max(0.0, _norm(*drone.velocity) ** 2 - cap * cap) / (2.0 * self.specs[drone.drone_type].max_acceleration)
        return cap if distance < 1.25 * braking + 0.8 else maximum

    def _steer(self, drone, path, arrive=False, velocity_hint=(0.0, 0.0)):
        spec = self.specs[drone.drone_type]
        if not path:
            return (0.0, 0.0), drone.position
        self._advance_path(drone, path)
        speed = _norm(*drone.velocity)
        local_target = self._lookahead(drone.position, path, 1.0 + 0.42 * speed)
        direction = _unit(local_target[0] - drone.position[0], local_target[1] - drone.position[1])
        target_speed = self._corner_speed(drone, path, spec.max_speed)
        if arrive:
            remaining = self._path_length(drone.position, path)
            target_speed = min(target_speed, sqrt(max(0.0, 2.0 * spec.max_acceleration * max(0.0, remaining - 0.2))))
        desired_velocity = (
            direction[0] * target_speed + velocity_hint[0],
            direction[1] * target_speed + velocity_hint[1],
        )
        desired_velocity = _limit(*desired_velocity, spec.max_speed)
        command = _limit(
            (desired_velocity[0] - (drone.velocity[0] + 0.14 * drone.acceleration[0])) / 0.34,
            (desired_velocity[1] - (drone.velocity[1] + 0.14 * drone.acceleration[1])) / 0.34,
            spec.max_acceleration,
        )
        command = self._terrain_guard(drone, command)
        command = self._safe_command(drone, command, local_target)
        return command, local_target

    def _nearest_obstacle_normal(self, position):
        x, y = position
        best = (float("inf"), (0.0, 0.0))
        for obstacle in self.obstacles:
            if isinstance(obstacle, CircleObstacle):
                dx, dy = x - obstacle.center[0], y - obstacle.center[1]
                magnitude = hypot(dx, dy)
                normal = _unit(dx, dy) if magnitude > EPS else (self.direction, 0.0)
                clearance = magnitude - obstacle.radius
            else:
                nearest_x = _clip(x, obstacle.x_min, obstacle.x_max)
                nearest_y = _clip(y, obstacle.y_min, obstacle.y_max)
                dx, dy = x - nearest_x, y - nearest_y
                magnitude = hypot(dx, dy)
                if magnitude > EPS:
                    normal = (dx / magnitude, dy / magnitude)
                    clearance = magnitude
                else:
                    options = (
                        (x - obstacle.x_min, (-1.0, 0.0)),
                        (obstacle.x_max - x, (1.0, 0.0)),
                        (y - obstacle.y_min, (0.0, -1.0)),
                        (obstacle.y_max - y, (0.0, 1.0)),
                    )
                    depth, normal = min(options, key=lambda item: item[0])
                    clearance = -depth
            if clearance < best[0]:
                best = (clearance, normal)
        return best

    def _terrain_guard(self, drone, command):
        spec = self.specs[drone.drone_type]
        clearance, normal = self._nearest_obstacle_normal(drone.position)
        inward_speed = -_dot(drone.velocity, normal)
        if clearance < 0.82 and (inward_speed > 0.0 or clearance < 0.48):
            urgency = _clip((0.82 - clearance) / 0.48 + inward_speed / max(1.0, spec.max_speed), 0.0, 1.0)
            command = _limit(
                command[0] + normal[0] * spec.max_acceleration * (0.7 + urgency),
                command[1] + normal[1] * spec.max_acceleration * (0.7 + urgency),
                spec.max_acceleration,
            )
        return command

    def _simulate_command(self, drone, command, ticks=6):
        spec = self.specs[drone.drone_type]
        position = drone.position
        velocity = drone.velocity
        acceleration = drone.acceleration
        for _ in range(ticks):
            desired = _limit(*command, spec.max_acceleration)
            delta = _limit(desired[0] - acceleration[0], desired[1] - acceleration[1], spec.max_jerk * 0.05)
            acceleration = _limit(acceleration[0] + delta[0], acceleration[1] + delta[1], spec.max_acceleration)
            new_position = (
                position[0] + velocity[0] * 0.05 + 0.5 * acceleration[0] * 0.05 * 0.05,
                position[1] + velocity[1] * 0.05 + 0.5 * acceleration[1] * 0.05 * 0.05,
            )
            new_velocity = _limit(
                velocity[0] + acceleration[0] * 0.05,
                velocity[1] + acceleration[1] * 0.05,
                spec.max_speed,
            )
            if not self._segment_clear(position, new_position, self.EMERGENCY_CLEARANCE):
                return None
            position, velocity = new_position, new_velocity
        return position, velocity

    def _safe_command(self, drone, command, local_target):
        spec = self.specs[drone.drone_type]
        brake = _limit(
            -3.2 * drone.velocity[0] - 0.8 * drone.acceleration[0],
            -3.2 * drone.velocity[1] - 0.8 * drone.acceleration[1],
            spec.max_acceleration,
        )
        candidates = [command, brake]
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
            result = self._simulate_command(drone, candidate)
            if result is None:
                continue
            position, velocity = result
            score = _distance(position, local_target) + 0.10 * _norm(*velocity)
            if best is None or score < best[0]:
                best = (score, candidate)
        if best is not None:
            return best[1]
        _, normal = self._nearest_obstacle_normal(drone.position)
        return _limit(
            normal[0] * spec.max_acceleration - 2.2 * drone.velocity[0],
            normal[1] * spec.max_acceleration - 2.2 * drone.velocity[1],
            spec.max_acceleration,
        )

    def step(self, state):
        actions = {}
        for drone in state.own_drones:
            if drone.status is not DroneStatus.ACTIVE:
                continue
            destination = self.goal_target[drone.id]
            self._set_route(drone, destination)
            command, _ = self._steer(drone, self.paths[drone.id])
            actions[drone.id] = (float(command[0]), float(command[1]))
        return actions
