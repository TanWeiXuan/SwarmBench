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

    PLAN_CLEARANCE = 0.94
    TRACK_CLEARANCE = 0.74
    EMERGENCY_CLEARANCE = 0.50
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
        self.initial_enemy_fast_ids = {
            drone.id
            for drone in game_info.opponent_initial_drones
            if drone.drone_type is DroneType.FAST
        }

        self.nodes = self._make_roadmap_nodes()
        self.graph = self._make_roadmap_graph()
        self.paths: dict[int, list[tuple[float, float]]] = {}
        self.path_destination: dict[int, tuple[float, float]] = {}
        self.goal_target: dict[int, tuple[float, float]] = {}

        own_slow = sorted(
            (drone for drone in game_info.own_initial_drones if drone.drone_type is DroneType.SLOW),
            key=lambda drone: (drone.position[1], drone.id),
        )
        own_fast = sorted(
            (drone for drone in game_info.own_initial_drones if drone.drone_type is DroneType.FAST),
            key=lambda drone: (drone.position[1], drone.id),
        )

        # The two corridors are deliberately separated in y.  Which wave gets
        # the upper route is controller-seed deterministic, avoiding a fixed
        # exploitable side without tuning to scenario seeds.
        corridor_sign = 1.0 if self.controller_seed & 1 else -1.0
        goal_center_y = self.target_goal.center[1]
        self.wave_corridor_y = {
            0: _clip(goal_center_y + corridor_sign * 2.8, self.target_goal.y_min + 1.2, self.target_goal.y_max - 1.2),
            1: _clip(goal_center_y - corridor_sign * 3.5, self.target_goal.y_min + 1.2, self.target_goal.y_max - 1.2),
        }
        first_wave_ids = {
            drone.id
            for drone in sorted(
                own_slow,
                key=lambda drone: (abs(drone.position[1] - self.wave_corridor_y[0]), drone.id),
            )[:6]
        }
        wave_members = {
            0: sorted((drone for drone in own_slow if drone.id in first_wave_ids), key=lambda drone: (drone.position[1], drone.id)),
            1: sorted((drone for drone in own_slow if drone.id not in first_wave_ids), key=lambda drone: (drone.position[1], drone.id)),
        }

        self.slow_wave: dict[int, int] = {}
        self.slow_state: dict[int, str] = {}
        self.stage_target: dict[int, tuple[float, float]] = {}
        self.latest_departure: dict[int, float] = {}
        self.departed_at: dict[int, float | None] = {}
        self.initial_route_time: dict[int, float] = {}
        stage_x = 17.0 if self.direction > 0 else self.width - 17.0
        for wave, members in wave_members.items():
            for rank, drone in enumerate(members):
                offset = (rank - (len(members) - 1) * 0.5) * 0.72
                goal_y = _clip(
                    self.wave_corridor_y[wave] + offset,
                    self.target_goal.y_min + 0.9,
                    self.target_goal.y_max - 0.9,
                )
                goal_x = self.target_goal.x_min + 1.4 if self.direction > 0 else self.target_goal.x_max - 1.4
                portal = (float(goal_x), float(goal_y))
                self.goal_target[drone.id] = portal
                goal_path = self._route(drone.position, portal)
                route_time = self._conservative_route_time(drone, goal_path)
                self.initial_route_time[drone.id] = route_time
                self.latest_departure[drone.id] = max(0.0, MATCH_DURATION - route_time - 5.0)
                self.slow_wave[drone.id] = wave
                self.slow_state[drone.id] = "SAFE" if wave == 0 else "DELAYED"
                self.departed_at[drone.id] = 0.0 if wave == 0 else None
                stage_y = _clip(
                    self.wave_corridor_y[wave] + offset * 1.25,
                    1.2,
                    self.height - 1.2,
                )
                self.stage_target[drone.id] = (stage_x, float(stage_y))
                destination = portal if wave == 0 else self.stage_target[drone.id]
                self.paths[drone.id] = goal_path if wave == 0 else self._route(drone.position, destination)
                self.path_destination[drone.id] = destination

        for rank, drone in enumerate(own_fast):
            portal = self._goal_portal(rank, len(own_fast))
            self.goal_target[drone.id] = portal
            self.paths[drone.id] = self._route(drone.position, portal)
            self.path_destination[drone.id] = portal

        # Allocate the starting roles by geometry, not ID: one guard begins
        # near each wave, three denial drones begin near the defended goal, and
        # the central remaining drone is the reserve.  The other four hunt.
        remaining = list(own_fast)

        def take_nearest(point):
            chosen = min(remaining, key=lambda drone: (_distance(drone.position, point), drone.id))
            remaining.remove(chosen)
            return chosen

        wave_centers = {
            wave: (
                sum(drone.position[0] for drone in members) / max(1, len(members)),
                sum(drone.position[1] for drone in members) / max(1, len(members)),
            )
            for wave, members in wave_members.items()
        }
        guard_zero = take_nearest(wave_centers[0])
        guard_one = take_nearest(wave_centers[1])
        home_anchor = (14.5 if self.direction > 0 else self.width - 14.5, self.own_goal.center[1])
        home = [take_nearest(home_anchor) for _ in range(3)]
        reserve_anchor = (20.0 if self.direction > 0 else self.width - 20.0, (self.wave_corridor_y[0] + self.wave_corridor_y[1]) * 0.5)
        reserve = take_nearest(reserve_anchor)

        self.base_role: dict[int, str] = {}
        self.guard_wave = {guard_zero.id: 0, guard_one.id: 1}
        for drone in remaining:
            self.base_role[drone.id] = "HUNTER"
        for drone in home:
            self.base_role[drone.id] = "HOME"
        self.base_role[guard_zero.id] = "GUARD"
        self.base_role[guard_one.id] = "GUARD"
        self.base_role[reserve.id] = "RESERVE"

        self.task = {drone.id: self.base_role[drone.id] for drone in own_fast}
        self.task_target: dict[int, int | None] = {drone.id: None for drone in own_fast}
        self.task_destination: dict[int, tuple[float, float]] = {}
        self.task_since = {drone.id: 0.0 for drone in own_fast}
        self.role_switches = 0
        self.intent_score: dict[tuple[int, int], float] = {}
        self.intent_target: dict[int, tuple[int | None, float]] = {}
        self.threat_for_slow: dict[int, tuple[int, float]] = {}
        self.rendezvous: dict[int, tuple[tuple[float, float], int, int, float]] = {}
        self.release_delayed = False
        self.outcome = "LIKELY_DRAW"
        self.projected_margin = 0.0
        self.last_high_update = -10.0

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

    def _conservative_route_time(self, drone, path):
        spec = self.specs[drone.drone_type]
        length = self._path_length(drone.position, path)
        # Cruise time plus acceleration/jerk response and explicit turn tax.
        return length / spec.max_speed + spec.max_speed / (2.0 * spec.max_acceleration) + 0.25 + 0.42 * max(0, len(path) - 1)

    @staticmethod
    def _travel_time(distance, initial_speed, acceleration, maximum_speed):
        distance = max(0.0, distance)
        initial_speed = _clip(initial_speed, 0.0, maximum_speed)
        acceleration_distance = max(0.0, maximum_speed * maximum_speed - initial_speed * initial_speed) / (2.0 * acceleration)
        if distance <= acceleration_distance:
            return (sqrt(max(0.0, initial_speed * initial_speed + 2.0 * acceleration * distance)) - initial_speed) / acceleration + 0.25
        return (maximum_speed - initial_speed) / acceleration + (distance - acceleration_distance) / maximum_speed + 0.25

    def _eta_point(self, drone, point, detour=True):
        dx, dy = point[0] - drone.position[0], point[1] - drone.position[1]
        direction = _unit(dx, dy)
        projected_speed = max(0.0, _dot(drone.velocity, direction))
        distance = hypot(dx, dy)
        if detour and not self._segment_clear(drone.position, point, self.TRACK_CLEARANCE):
            distance = distance * 1.16 + 2.0
        spec = self.specs[drone.drone_type]
        return self._travel_time(distance, projected_speed, spec.max_acceleration, spec.max_speed)

    def _goal_eta(self, drone, own_drone=True):
        if own_drone:
            target = self.goal_target.get(drone.id, self.target_goal.center)
            path = self.paths.get(drone.id, ())
            if self.path_destination.get(drone.id) == target and path:
                distance = self._path_length(drone.position, path)
                direction = _unit(path[0][0] - drone.position[0], path[0][1] - drone.position[1])
            else:
                distance = _distance(drone.position, target)
                if not self._segment_clear(drone.position, target, self.TRACK_CLEARANCE):
                    distance = distance * 1.16 + 2.0
                direction = _unit(target[0] - drone.position[0], target[1] - drone.position[1])
        else:
            goal = self.own_goal
            target = (
                goal.x_max - 1.2 if self.direction > 0 else goal.x_min + 1.2,
                _clip(drone.position[1], goal.y_min + 0.8, goal.y_max - 0.8),
            )
            distance = _distance(drone.position, target)
            if not self._segment_clear(drone.position, target, self.TRACK_CLEARANCE):
                distance = distance * 1.16 + 2.0
            direction = _unit(target[0] - drone.position[0], target[1] - drone.position[1])
        projected_speed = max(0.0, _dot(drone.velocity, direction))
        spec = self.specs[drone.drone_type]
        return self._travel_time(distance, projected_speed, spec.max_acceleration, spec.max_speed)

    def _predict_position(self, drone, seconds):
        # Observed acceleration is trusted only for one response quantum; then
        # the target coasts at its clipped predicted velocity.  This avoids the
        # large overprediction caused by integrating acceleration past vmax.
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

    def _intercept_eta(self, pursuer, target):
        dx, dy = target.position[0] - pursuer.position[0], target.position[1] - pursuer.position[1]
        direction = _unit(dx, dy)
        closing = max(0.0, _dot(pursuer.velocity, direction) - _dot(target.velocity, direction))
        spec = self.specs[pursuer.drone_type]
        distance = max(0.0, hypot(dx, dy) - 0.75)
        return self._travel_time(distance, closing, spec.max_acceleration, spec.max_speed)

    @staticmethod
    def _closest_approach(first, second, horizon=2.6):
        rx = second.position[0] - first.position[0]
        ry = second.position[1] - first.position[1]
        vx = second.velocity[0] - first.velocity[0]
        vy = second.velocity[1] - first.velocity[1]
        speed2 = vx * vx + vy * vy
        time = 0.0 if speed2 <= EPS else _clip(-(rx * vx + ry * vy) / speed2, 0.0, horizon)
        return hypot(rx + vx * time, ry + vy * time), time

    def _update_intentions(self, own_slow, enemy_fast):
        active_pairs = set()
        for enemy in enemy_fast:
            best_id = None
            best_value = -1.0
            previous_id = self.intent_target.get(enemy.id, (None, 0.0))[0]
            for slow in own_slow:
                key = (enemy.id, slow.id)
                active_pairs.add(key)
                dx, dy = slow.position[0] - enemy.position[0], slow.position[1] - enemy.position[1]
                direction = _unit(dx, dy)
                distance = hypot(dx, dy)
                closest, _ = self._closest_approach(enemy, slow)
                velocity_alignment = max(0.0, _dot(_unit(*enemy.velocity), direction))
                acceleration_alignment = max(0.0, _dot(_unit(*enemy.acceleration), direction))
                contact_eta = self._intercept_eta(enemy, slow)
                goal_eta = self._goal_eta(slow, True)
                feasibility = _clip((goal_eta - contact_eta + 2.5) / 6.0, 0.0, 1.0)
                raw = (
                    0.24 / (1.0 + distance / 7.0)
                    + 0.28 * _clip((9.0 - closest) / 9.0, 0.0, 1.0)
                    + 0.20 * velocity_alignment
                    + 0.10 * acceleration_alignment
                    + 0.18 * feasibility
                )
                if slow.id == previous_id:
                    raw += 0.08
                if distance < 6.0:
                    raw = max(raw, 0.78)
                old = self.intent_score.get(key, raw)
                value = 0.64 * old + 0.36 * raw
                self.intent_score[key] = value
                if value > best_value or (abs(value - best_value) <= EPS and (best_id is None or slow.id < best_id)):
                    best_id, best_value = slow.id, value
            self.intent_target[enemy.id] = (best_id, best_value)
        self.intent_score = {key: value for key, value in self.intent_score.items() if key in active_pairs}

    def _group_center(self, own_slow, wave):
        members = [slow for slow in own_slow if self.slow_wave.get(slow.id) == wave]
        if not members:
            return (
                18.0 if self.direction > 0 else self.width - 18.0,
                self.wave_corridor_y[wave],
            )
        return (
            sum(drone.position[0] for drone in members) / len(members),
            sum(drone.position[1] for drone in members) / len(members),
        )

    def _safe_dynamic_target(self, point):
        point = (_clip(point[0], 0.45, self.width - 0.45), _clip(point[1], 0.45, self.height - 0.45))
        if self._point_clear(point, self.TRACK_CLEARANCE):
            return point
        clear_nodes = [node for node in self.nodes if self._point_clear(node, self.TRACK_CLEARANCE)]
        return min(clear_nodes, key=lambda node: (_distance(node, point), node[0], node[1])) if clear_nodes else point

    def _choose_rendezvous(self, slow, enemy, guards, remaining):
        candidates = [slow.position]
        path = self.paths.get(slow.id, ())
        candidates.extend(path[:4])
        forward = self._predict_position(slow, 1.5)
        candidates.append(forward)
        nearby_nodes = sorted(
            (node for node in self.nodes if _distance(node, slow.position) < 9.0),
            key=lambda node: (_distance(node, slow.position), node[0], node[1]),
        )
        candidates.extend(nearby_nodes[:8])

        best = None
        for guard in guards:
            for point in candidates:
                point = self._safe_dynamic_target(point)
                slow_eta = self._eta_point(slow, point)
                guard_eta = self._eta_point(guard, point)
                enemy_eta = self._eta_point(enemy, point)
                distance_to_goal = _distance(point, self.goal_target[slow.id])
                remaining_goal = distance_to_goal / self.specs[DroneType.SLOW].max_speed + 1.2
                if slow_eta + remaining_goal > remaining - 1.5 or guard_eta + 0.35 >= enemy_eta:
                    continue
                guard_margin = enemy_eta - guard_eta
                progress = self.direction * (point[0] - slow.position[0])
                cover = 0.55 if not self._segment_clear(enemy.position, point, self.TRACK_CLEARANCE) else 0.0
                score = guard_margin + 0.035 * progress - 0.08 * slow_eta + cover
                candidate = (score, -guard.id, point, guard.id)
                if best is None or candidate[:2] > best[:2]:
                    best = candidate
        if best is None:
            return None
        return best[2], best[3]

    def _pursuit_point(self, hunter, target, horizon=3.0):
        eta = min(horizon, max(0.35, self._intercept_eta(hunter, target)))
        return self._safe_dynamic_target(self._predict_position(target, eta))

    def _set_task(self, drone_id, task, target_id, destination, now, emergency=False):
        changed = self.task.get(drone_id) != task or self.task_target.get(drone_id) != target_id
        if changed and drone_id in self.task_destination and not emergency and now < self.task_since.get(drone_id, 0.0) + 0.8:
            return
        if changed:
            self.role_switches += 1
            self.task_since[drone_id] = now
        self.task[drone_id] = task
        self.task_target[drone_id] = target_id
        self.task_destination[drone_id] = self._safe_dynamic_target(destination)

    def _estimate_outcome(self, state, own_fast, own_slow, enemy_fast, enemy_slow):
        remaining = MATCH_DURATION - state.time
        own_expected = float(state.own_score)
        enemy_expected = float(state.opponent_score)

        for slow in own_slow:
            eta = self._goal_eta(slow, True)
            if eta > remaining - 0.5:
                continue
            slow_state = self.slow_state.get(slow.id, "SAFE")
            chance = 0.90
            if slow_state == "THREATENED":
                chance = 0.76 if slow.id in self.rendezvous else 0.28
            elif slow_state == "DOOMED":
                chance = 0.08
            elif slow_state == "DELAYED" and not self.release_delayed:
                chance = 0.84 if state.time < self.latest_departure.get(slow.id, 0.0) else 0.35
            own_expected += 5.0 * chance

        targeted_enemy = {
            target_id
            for drone_id, target_id in self.task_target.items()
            if target_id is not None and self.task.get(drone_id) in {"HUNT_SLOW", "HOME_DENY"}
        }
        for enemy in enemy_slow:
            eta = self._goal_eta(enemy, False)
            if eta < remaining - 0.5:
                enemy_expected += 5.0 * (0.34 if enemy.id in targeted_enemy else 0.88)
        for drone in own_fast:
            if self._goal_eta(drone, True) < remaining - 0.5:
                task = self.task.get(drone.id)
                own_expected += 0.82 if task == "SCORE" else 0.30
        for enemy in enemy_fast:
            if self._goal_eta(enemy, False) < remaining - 0.5:
                enemy_expected += 0.68

        self.projected_margin = own_expected - enemy_expected
        if self.projected_margin > 3.0:
            self.outcome = "LIKELY_WIN"
        elif self.projected_margin < -3.0:
            self.outcome = "LIKELY_LOSS"
        else:
            self.outcome = "LIKELY_DRAW"

    def _assign_fast_tasks(self, state, own_fast, own_slow, enemy_fast, enemy_slow):
        remaining_time = MATCH_DURATION - state.time
        own_by_id = {drone.id: drone for drone in own_fast}
        slow_by_id = {drone.id: drone for drone in own_slow}
        enemy_by_id = {drone.id: drone for drone in enemy_fast + enemy_slow}
        used_enemy_slow: set[int] = set()
        protected_enemy_fast: set[int] = set()

        # Bind guards to rendezvous plans for their own wave.  A guard that is
        # not needed remains with the whole group rather than chasing noise.
        rendezvous_by_guard = {
            plan[1]: (slow_id, plan)
            for slow_id, plan in self.rendezvous.items()
            if plan[1] in own_by_id
        }
        for guard_id, wave in sorted(self.guard_wave.items()):
            guard = own_by_id.get(guard_id)
            if guard is None:
                continue
            assigned = rendezvous_by_guard.get(guard_id)
            if assigned is not None:
                slow_id, plan = assigned
                point, _, enemy_id, _ = plan
                enemy = enemy_by_id.get(enemy_id)
                aim = point if enemy is None else self._pursuit_point(guard, enemy, 2.2)
                self._set_task(guard.id, "PROTECT", enemy_id, aim, state.time, emergency=True)
                protected_enemy_fast.add(enemy_id)
            else:
                threats = sorted(
                    (
                        (urgency, slow_id, enemy_id)
                        for slow_id, (enemy_id, urgency) in self.threat_for_slow.items()
                        if self.slow_wave.get(slow_id) == wave and enemy_id not in protected_enemy_fast
                    ),
                    reverse=True,
                )
                direct_plan = None
                for urgency, slow_id, enemy_id in threats:
                    slow = slow_by_id.get(slow_id)
                    enemy = enemy_by_id.get(enemy_id)
                    if slow is None or enemy is None:
                        continue
                    contact = self._intercept_eta(enemy, slow)
                    cut = self._predict_position(enemy, min(2.4, contact))
                    if self._eta_point(guard, cut) < contact + 0.55:
                        direct_plan = (enemy_id, self._safe_dynamic_target(cut))
                        break
                if direct_plan is not None:
                    enemy_id, aim = direct_plan
                    protected_enemy_fast.add(enemy_id)
                    self._set_task(guard.id, "PROTECT", enemy_id, aim, state.time, emergency=True)
                elif not enemy_fast and self._goal_eta(guard, True) < remaining_time - 1.0:
                    self._set_task(guard.id, "SCORE", None, self.goal_target[guard.id], state.time)
                else:
                    center = self._group_center(own_slow, wave)
                    escort = (center[0] + self.direction * 2.8, center[1])
                    self._set_task(guard.id, "ESCORT", None, escort, state.time)

        # Four forward hunters bid only on enemy SLOW drones.  Continuing a
        # feasible target receives a small hysteresis bonus.
        hunters = sorted((drone for drone in own_fast if self.base_role.get(drone.id) == "HUNTER"), key=lambda drone: drone.id)
        for hunter in hunters:
            candidates = []
            for enemy in enemy_slow:
                contact = self._intercept_eta(hunter, enemy)
                goal_eta = self._goal_eta(enemy, False)
                if contact >= min(goal_eta - 0.20, remaining_time - 0.35):
                    continue
                progress = max(0.0, 45.0 - self.direction * (enemy.position[0] - self.own_goal.center[0]))
                score = 1.55 * (goal_eta - contact) + 0.03 * progress
                if self.task_target.get(hunter.id) == enemy.id:
                    score += 0.65
                candidates.append((score, -enemy.id, enemy))
            candidates.sort(reverse=True, key=lambda item: (item[0], item[1]))
            chosen = next((item[2] for item in candidates if item[2].id not in used_enemy_slow), None)
            if chosen is not None:
                used_enemy_slow.add(chosen.id)
                self._set_task(hunter.id, "HUNT_SLOW", chosen.id, self._pursuit_point(hunter, chosen), state.time)
            else:
                self._set_task(hunter.id, "SCORE", None, self.goal_target[hunter.id], state.time)

        # Home defenders ignore leading enemy FAST drones and wait for a
        # valuable SLOW.  They spend themselves on an enemy FAST only when a
        # concrete, imminent SLOW save is feasible.  The x leash prevents
        # midfield baiting.
        home = sorted((drone for drone in own_fast if self.base_role.get(drone.id) == "HOME"), key=lambda drone: drone.id)
        for slot, defender in enumerate(home):
            protection = None
            threat_jobs = sorted(
                (
                    (urgency, slow_id, enemy_id)
                    for slow_id, (enemy_id, urgency) in self.threat_for_slow.items()
                    if enemy_id not in protected_enemy_fast
                ),
                reverse=True,
            )
            for urgency, slow_id, enemy_id in threat_jobs:
                slow = slow_by_id.get(slow_id)
                enemy = enemy_by_id.get(enemy_id)
                if slow is None or enemy is None:
                    continue
                contact = self._intercept_eta(enemy, slow)
                if contact > 6.2:
                    continue
                cut = self._predict_position(enemy, min(2.8, contact))
                if self._eta_point(defender, cut) + 0.25 < contact + 0.65:
                    protection = (enemy_id, self._safe_dynamic_target(cut))
                    break
            if protection is not None:
                enemy_id, aim = protection
                protected_enemy_fast.add(enemy_id)
                self._set_task(defender.id, "PROTECT", enemy_id, aim, state.time, emergency=True)
                continue
            candidates = []
            for enemy in enemy_slow:
                front = self.direction * (enemy.position[0] - self.own_goal.center[0])
                goal_eta = self._goal_eta(enemy, False)
                contact = self._intercept_eta(defender, enemy)
                if (front > 44.0 and goal_eta > 19.0) or contact >= min(goal_eta - 0.15, remaining_time - 0.30):
                    continue
                score = 1.7 * (goal_eta - contact) + 0.04 * (44.0 - front)
                if self.task_target.get(defender.id) == enemy.id:
                    score += 0.7
                candidates.append((score, -enemy.id, enemy))
            candidates.sort(reverse=True, key=lambda item: (item[0], item[1]))
            chosen = next((item[2] for item in candidates if item[2].id not in used_enemy_slow), None)
            if chosen is not None:
                used_enemy_slow.add(chosen.id)
                self._set_task(defender.id, "HOME_DENY", chosen.id, self._pursuit_point(defender, chosen), state.time)
            elif ((not enemy_slow and not enemy_fast) or state.time > 64.0) and self._goal_eta(defender, True) < remaining_time - 1.2:
                self._set_task(defender.id, "SCORE", None, self.goal_target[defender.id], state.time)
            else:
                fraction = (slot + 1.0) / (len(home) + 1.0)
                station_y = self.own_goal.y_min + 1.1 + fraction * max(0.0, self.own_goal.y_max - self.own_goal.y_min - 2.2)
                station = (14.5 if self.direction > 0 else self.width - 14.5, station_y)
                self._set_task(defender.id, "HOME_STATION", None, station, state.time)

        reserve = next((drone for drone in own_fast if self.base_role.get(drone.id) == "RESERVE"), None)
        if reserve is not None:
            emergency = []
            for slow_id, (enemy_id, urgency) in self.threat_for_slow.items():
                if slow_id in self.rendezvous or enemy_id in protected_enemy_fast:
                    continue
                slow = slow_by_id.get(slow_id)
                enemy = enemy_by_id.get(enemy_id)
                if slow is None or enemy is None:
                    continue
                contact = self._intercept_eta(enemy, slow)
                if contact < 3.0 and self._intercept_eta(reserve, enemy) < contact + 0.35:
                    emergency.append((urgency, slow_id, enemy_id))
            emergency.sort(reverse=True)
            if emergency and emergency[0][0] > 0.82:
                _, slow_id, enemy_id = emergency[0]
                enemy = enemy_by_id.get(enemy_id)
                aim = self._pursuit_point(reserve, enemy, 2.2) if enemy is not None else self._group_center(own_slow, self.slow_wave.get(slow_id, 0))
                self._set_task(reserve.id, "PROTECT", enemy_id, aim, state.time, emergency=True)
            elif not enemy_slow and not enemy_fast and self._goal_eta(reserve, True) < remaining_time - 1.0:
                self._set_task(reserve.id, "SCORE", None, self.goal_target[reserve.id], state.time)
            elif state.time < 55.0 and self.outcome != "LIKELY_LOSS":
                center = self._group_center(own_slow, 1 if not self.release_delayed else 0)
                station = (center[0] - self.direction * 3.0, center[1])
                self._set_task(reserve.id, "RESERVE", None, station, state.time)
            elif self.outcome == "LIKELY_LOSS":
                candidates = []
                for enemy in enemy_slow:
                    contact = self._intercept_eta(reserve, enemy)
                    goal_eta = self._goal_eta(enemy, False)
                    if contact < min(goal_eta, remaining_time) - 0.15:
                        candidates.append((goal_eta - contact, -enemy.id, enemy))
                chosen = max(candidates, default=None, key=lambda item: (item[0], item[1]))
                if chosen is not None:
                    enemy = chosen[2]
                    self._set_task(reserve.id, "HUNT_SLOW", enemy.id, self._pursuit_point(reserve, enemy), state.time)
                else:
                    self._set_task(reserve.id, "SCORE", None, self.goal_target[reserve.id], state.time)
            elif self._goal_eta(reserve, True) < remaining_time - 1.0:
                self._set_task(reserve.id, "SCORE", None, self.goal_target[reserve.id], state.time)

    def _high_level_update(self, state, own_fast, own_slow, enemy_fast, enemy_slow):
        remaining = MATCH_DURATION - state.time
        self._update_intentions(own_slow, enemy_fast)

        active_enemy_fast_ids = {drone.id for drone in enemy_fast}
        resolved_enemy_fast = len(self.initial_enemy_fast_ids - active_enemy_fast_ids)
        committed_to_first = sum(
            target_id is not None
            and self.slow_wave.get(target_id) == 0
            and confidence > 0.52
            for target_id, confidence in self.intent_target.values()
        )
        delayed = [slow for slow in own_slow if self.slow_wave.get(slow.id) == 1]
        deadline_reached = any(state.time + 1.0 >= self.latest_departure.get(slow.id, 0.0) for slow in delayed)
        if delayed and (
            deadline_reached
            or resolved_enemy_fast >= 3
            or (state.time >= 12.0 and committed_to_first >= 3)
            or state.time >= 28.0
        ):
            self.release_delayed = True
            for slow in delayed:
                if self.departed_at.get(slow.id) is None:
                    self.departed_at[slow.id] = state.time

        self.threat_for_slow.clear()
        for slow in own_slow:
            goal_eta = self._goal_eta(slow, True)
            best = None
            for enemy in enemy_fast:
                target_id, confidence = self.intent_target.get(enemy.id, (None, 0.0))
                contact = self._intercept_eta(enemy, slow)
                distance = _distance(enemy.position, slow.position)
                floor = 0.25 * _clip((goal_eta - contact + 2.0) / 5.0, 0.0, 1.0)
                likelihood = max(floor, confidence if target_id == slow.id else self.intent_score.get((enemy.id, slow.id), 0.0))
                if distance < 7.0:
                    likelihood = max(likelihood, 0.82)
                if contact < goal_eta + 0.8 and likelihood > 0.34:
                    urgency = likelihood + 0.35 * _clip((goal_eta - contact) / max(1.0, goal_eta), 0.0, 1.0)
                    candidate = (urgency, -enemy.id, enemy)
                    if best is None or candidate[:2] > best[:2]:
                        best = candidate
            if best is not None:
                self.threat_for_slow[slow.id] = (best[2].id, best[0])

        # Keep still-valid rendezvous briefly; then compute at most one plan per
        # wave because each wave has one dedicated guard.
        active_slow_ids = {slow.id for slow in own_slow}
        active_enemy_ids = {enemy.id for enemy in enemy_fast}
        self.rendezvous = {
            slow_id: plan
            for slow_id, plan in self.rendezvous.items()
            if slow_id in active_slow_ids and plan[2] in active_enemy_ids and plan[3] > state.time
        }
        own_fast_by_id = {drone.id: drone for drone in own_fast}
        slow_by_id = {drone.id: drone for drone in own_slow}
        enemy_by_id = {drone.id: drone for drone in enemy_fast}
        for wave in (0, 1):
            if any(self.slow_wave.get(slow_id) == wave for slow_id in self.rendezvous):
                continue
            threats = sorted(
                (
                    (urgency, slow_id, enemy_id)
                    for slow_id, (enemy_id, urgency) in self.threat_for_slow.items()
                    if self.slow_wave.get(slow_id) == wave
                ),
                reverse=True,
            )
            if not threats:
                continue
            _, slow_id, enemy_id = threats[0]
            guard_id = next((drone_id for drone_id, assigned_wave in self.guard_wave.items() if assigned_wave == wave), None)
            guards = [own_fast_by_id[guard_id]] if guard_id in own_fast_by_id else []
            slow = slow_by_id.get(slow_id)
            enemy = enemy_by_id.get(enemy_id)
            if slow is not None and enemy is not None and guards:
                plan = self._choose_rendezvous(slow, enemy, guards, remaining)
                if plan is not None:
                    point, selected_guard = plan
                    self.rendezvous[slow.id] = (point, selected_guard, enemy.id, state.time + 1.4)

        for slow in own_slow:
            if self.slow_wave.get(slow.id) == 1 and not self.release_delayed:
                self.slow_state[slow.id] = "DELAYED"
                continue
            goal_eta = self._goal_eta(slow, True)
            threat = self.threat_for_slow.get(slow.id)
            if goal_eta > remaining - 0.5:
                self.slow_state[slow.id] = "DOOMED"
            elif threat is None:
                self.slow_state[slow.id] = "SAFE"
            elif slow.id in self.rendezvous:
                self.slow_state[slow.id] = "THREATENED"
            else:
                enemy = enemy_by_id.get(threat[0])
                contact = self._intercept_eta(enemy, slow) if enemy is not None else remaining
                self.slow_state[slow.id] = "DOOMED" if contact < min(3.4, goal_eta - 0.4) else "THREATENED"

        self._estimate_outcome(state, own_fast, own_slow, enemy_fast, enemy_slow)
        self._assign_fast_tasks(state, own_fast, own_slow, enemy_fast, enemy_slow)

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
                clearance = magnitude - (obstacle.radius + 0.25)
            else:
                x_min = obstacle.x_min - 0.25
                x_max = obstacle.x_max + 0.25
                y_min = obstacle.y_min - 0.25
                y_max = obstacle.y_max + 0.25
                nearest_x = _clip(x, x_min, x_max)
                nearest_y = _clip(y, y_min, y_max)
                dx, dy = x - nearest_x, y - nearest_y
                magnitude = hypot(dx, dy)
                if magnitude > EPS:
                    normal = (dx / magnitude, dy / magnitude)
                    clearance = magnitude
                else:
                    options = (
                        (x - x_min, (-1.0, 0.0)),
                        (x_max - x, (1.0, 0.0)),
                        (y - y_min, (0.0, -1.0)),
                        (y_max - y, (0.0, 1.0)),
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
        stopping_commitment = max(0.0, inward_speed) * 0.25 + max(0.0, inward_speed) ** 2 / (2.0 * spec.max_acceleration) + 0.20
        usable = clearance
        if usable < stopping_commitment + 0.34 or clearance < 0.76:
            # Once the normal stopping corridor is consumed, tactical lateral
            # pushes are discarded.  Drive outward while retaining only the
            # safe tangential component of the previous command.
            tangent = (-normal[1], normal[0])
            tangential = _dot(command, tangent)
            outward = spec.max_acceleration * _clip(
                (stopping_commitment + 0.45 - usable) / max(0.45, stopping_commitment + 0.45),
                0.62,
                1.0,
            )
            command = _limit(
                normal[0] * outward + tangent[0] * tangential * 0.45,
                normal[1] * outward + tangent[1] * tangential * 0.45,
                spec.max_acceleration,
            )
        return command

    def _simulate_command(self, drone, command, ticks=20):
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
        # Preserve the tactical command exactly when its full one-second
        # constant-command commitment remains terrain-safe.  Long rollout is
        # a veto, not an objective; ranking every safe command at t+1 changed
        # interception timing even though control will update again in 0.1 s.
        if self._simulate_command(drone, command, ticks=20) is not None:
            return command
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
            if self._simulate_command(drone, candidate, ticks=20) is None:
                continue
            position, velocity = self._simulate_command(drone, candidate, ticks=6)
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

    def _avoid_enemies(self, drone, command, enemies, engage_id=None):
        spec = self.specs[drone.drone_type]
        push_x = 0.0
        push_y = 0.0
        for enemy in enemies:
            if enemy.id == engage_id:
                continue
            distance = _distance(drone.position, enemy.position)
            if distance > 9.0:
                continue
            closest, time = self._closest_approach(drone, enemy, 2.6)
            threshold = 1.55 if drone.drone_type is DroneType.SLOW else 1.25
            if closest >= threshold or time > 2.6:
                continue
            own_future = (
                drone.position[0] + drone.velocity[0] * time,
                drone.position[1] + drone.velocity[1] * time,
            )
            enemy_future = (
                enemy.position[0] + enemy.velocity[0] * time,
                enemy.position[1] + enemy.velocity[1] * time,
            )
            away = _unit(own_future[0] - enemy_future[0], own_future[1] - enemy_future[1])
            if away == (0.0, 0.0):
                relative = (enemy.velocity[0] - drone.velocity[0], enemy.velocity[1] - drone.velocity[1])
                side = 1.0 if (drone.id + enemy.id) & 1 else -1.0
                away = _unit(-relative[1] * side, relative[0] * side)
            strength = spec.max_acceleration * _clip((threshold - closest) / threshold + (2.6 - time) / 5.2, 0.0, 1.25)
            push_x += away[0] * strength
            push_y += away[1] * strength
        return _limit(command[0] + push_x, command[1] + push_y, spec.max_acceleration)

    def _salvage_target(self, slow, own_fast, enemy_slow):
        useful_trades = []
        for enemy in enemy_slow:
            eta = self._intercept_eta(slow, enemy)
            goal_eta = self._goal_eta(enemy, False)
            if eta < goal_eta and _distance(slow.position, enemy.position) < 16.0:
                useful_trades.append((goal_eta - eta, -enemy.id, enemy.position))
        if useful_trades:
            return max(useful_trades)[2], None
        if own_fast:
            defender = min(own_fast, key=lambda drone: (_distance(drone.position, slow.position), drone.id))
            return defender.position, None
        if enemy_slow:
            enemy = min(enemy_slow, key=lambda drone: (_distance(drone.position, slow.position), drone.id))
            return enemy.position, enemy.id
        return self.goal_target[slow.id], None

    def step(self, state):
        own_fast = [
            drone
            for drone in state.own_drones
            if drone.status is DroneStatus.ACTIVE and drone.drone_type is DroneType.FAST
        ]
        own_slow = [
            drone
            for drone in state.own_drones
            if drone.status is DroneStatus.ACTIVE and drone.drone_type is DroneType.SLOW
        ]
        enemy_fast = [
            drone
            for drone in state.opponent_drones
            if drone.status is DroneStatus.ACTIVE and drone.drone_type is DroneType.FAST
        ]
        enemy_slow = [
            drone
            for drone in state.opponent_drones
            if drone.status is DroneStatus.ACTIVE and drone.drone_type is DroneType.SLOW
        ]
        if state.time + EPS >= self.last_high_update + 0.40:
            self._high_level_update(state, own_fast, own_slow, enemy_fast, enemy_slow)
            self.last_high_update = state.time

        enemy_active = enemy_fast + enemy_slow
        actions = {}
        for drone in state.own_drones:
            if drone.status is not DroneStatus.ACTIVE:
                continue

            arrive = False
            velocity_hint = (0.0, 0.0)
            engage_id = None
            if drone.drone_type is DroneType.SLOW:
                slow_state = self.slow_state.get(drone.id, "SAFE")
                if slow_state == "DELAYED" and not self.release_delayed:
                    destination = self.stage_target[drone.id]
                    arrive = True
                elif slow_state == "THREATENED" and drone.id in self.rendezvous:
                    destination = self.rendezvous[drone.id][0]
                    arrive = True
                elif slow_state == "DOOMED":
                    destination, engage_id = self._salvage_target(drone, own_fast, enemy_slow)
                else:
                    destination = self.goal_target[drone.id]
            else:
                destination = self.task_destination.get(drone.id, self.goal_target[drone.id])
                task = self.task.get(drone.id, "SCORE")
                engage_id = self.task_target.get(drone.id)
                arrive = task in {"ESCORT", "HOME_STATION", "RESERVE"}
                if task == "ESCORT":
                    wave = self.guard_wave.get(drone.id, 0)
                    members = [slow for slow in own_slow if self.slow_wave.get(slow.id) == wave]
                    if members:
                        velocity_hint = (
                            sum(slow.velocity[0] for slow in members) / len(members) * 0.45,
                            sum(slow.velocity[1] for slow in members) / len(members) * 0.45,
                        )

            self._set_route(drone, destination)
            command, local_target = self._steer(
                drone,
                self.paths[drone.id],
                arrive=arrive,
                velocity_hint=velocity_hint,
            )
            command = self._avoid_enemies(drone, command, enemy_active, engage_id)
            command = self._terrain_guard(drone, command)
            command = self._safe_command(drone, command, local_target)
            actions[drone.id] = (float(command[0]), float(command[1]))
        return actions
