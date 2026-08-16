"""GPT-5.6 Sol Ultra controller for SwarmBench.

Designed and implemented by OpenAI GPT-5.6 Sol (ultra reasoning).  The strategy
and constants below were derived from the public game mechanics.  Existing
submissions were used only as black-box evaluation opponents; no controller
code, tuned constants, or non-trivial implementation structure was borrowed.

The controller gives terrain safety first priority, because a single SLOW crash
is a five-point swing.  It builds a conservative visibility roadmap, follows it
with jerk-aware velocity control, and continuously assigns FAST drones to the
best of scoring, denying an enemy SLOW, or screening a threatened friendly SLOW.
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


_EPS = 1.0e-9


def _clip(x: float, low: float, high: float) -> float:
    return low if x < low else high if x > high else x


def _norm_clip(x: float, y: float, maximum: float) -> tuple[float, float]:
    magnitude = hypot(x, y)
    if magnitude <= maximum or magnitude <= _EPS:
        return (float(x), float(y))
    scale = maximum / magnitude
    return (float(x * scale), float(y * scale))


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return hypot(a[0] - b[0], a[1] - b[1])


def _unit(x: float, y: float) -> tuple[float, float]:
    magnitude = hypot(x, y)
    if magnitude <= _EPS:
        return (0.0, 0.0)
    return (x / magnitude, y / magnitude)


def _dot(a: tuple[float, float], b: tuple[float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


class SwarmController(BaseSwarmController):
    """Safe global routing plus value-aware, predictive swarm tactics."""

    # A center following this roadmap stays 1.05 m from raw terrain: 0.25 m is
    # the physical radius and the rest absorbs jerk/turn tracking.  This also
    # closes deceptive narrow gaps where repulsion from one obstacle can push a
    # drone into its neighbor; protected outer rails remain globally connected.
    PLAN_CLEARANCE = 1.05
    TRACK_CLEARANCE = 0.82
    EMERGENCY_CLEARANCE = 0.40
    ROADMAP_EPSILON = 0.18

    def initialize(self, game_info):
        self.team = game_info.team
        self.direction = 1.0 if self.team is Team.A else -1.0
        self.goal = game_info.target_goal
        self.own_goal = game_info.own_goal
        self.width = float(game_info.arena_width)
        self.height = float(game_info.arena_height)
        self.obstacles = tuple(game_info.obstacles)
        self.specs = dict(game_info.drone_specs)

        self.nodes = self._make_roadmap_nodes()
        self.graph = self._make_roadmap_graph()

        self.goal_portals = self._make_portals(self.goal, 10)
        self.enemy_goal_portals = self._make_portals(self.own_goal, 10)
        self.goal_target = {}
        self.paths = {}
        self.path_destination = {}
        self.roles = {}
        self.evasion_side = {}
        self.last_role_time = -10.0
        self.last_slow_replan = -10.0
        self.last_pursuit_replan = {}

        slow = sorted(
            (d for d in game_info.own_initial_drones if d.drone_type is DroneType.SLOW),
            key=lambda d: (d.position[1], d.id),
        )
        fast = sorted(
            (d for d in game_info.own_initial_drones if d.drone_type is DroneType.FAST),
            key=lambda d: (d.position[1], d.id),
        )
        for index, drone in enumerate(slow):
            portal = self.goal_portals[index]
            self.goal_target[drone.id] = portal
            self.paths[drone.id] = self._route(drone.position, portal)
            self.path_destination[drone.id] = portal
        # Offset FAST goal lanes from the valuable SLOW lanes.  Friendly drones
        # may overlap, but spreading them reduces accidental enemy multi-threats.
        for index, drone in enumerate(fast):
            portal = self.goal_portals[(index * 3 + 1) % len(self.goal_portals)]
            self.goal_target[drone.id] = portal
            self.paths[drone.id] = self._route(drone.position, portal)
            self.path_destination[drone.id] = portal
            self.roles[drone.id] = ("score", None, None)

    # ------------------------------------------------------------------
    # Exact conservative geometry and static visibility roadmap

    def _point_clear(self, point, padding):
        x, y = point
        for obstacle in self.obstacles:
            if isinstance(obstacle, CircleObstacle):
                if hypot(x - obstacle.center[0], y - obstacle.center[1]) <= obstacle.radius + padding + _EPS:
                    return False
            elif (
                obstacle.x_min - padding - _EPS <= x <= obstacle.x_max + padding + _EPS
                and obstacle.y_min - padding - _EPS <= y <= obstacle.y_max + padding + _EPS
            ):
                return False
        return True

    @staticmethod
    def _segment_circle_blocked(start, end, center, radius):
        dx, dy = end[0] - start[0], end[1] - start[1]
        denominator = dx * dx + dy * dy
        if denominator <= _EPS:
            return (start[0] - center[0]) ** 2 + (start[1] - center[1]) ** 2 <= radius * radius
        t = ((center[0] - start[0]) * dx + (center[1] - start[1]) * dy) / denominator
        t = _clip(t, 0.0, 1.0)
        px, py = start[0] + t * dx, start[1] + t * dy
        return (px - center[0]) ** 2 + (py - center[1]) ** 2 <= radius * radius

    @staticmethod
    def _segment_box_blocked(start, end, bounds):
        enter, leave = 0.0, 1.0
        for origin, delta, low, high in (
            (start[0], end[0] - start[0], bounds[0], bounds[1]),
            (start[1], end[1] - start[1], bounds[2], bounds[3]),
        ):
            if abs(delta) <= _EPS:
                if origin < low or origin > high:
                    return False
                continue
            t0, t1 = (low - origin) / delta, (high - origin) / delta
            if t0 > t1:
                t0, t1 = t1, t0
            enter, leave = max(enter, t0), min(leave, t1)
            if enter > leave:
                return False
        return 0.0 <= enter <= 1.0

    def _segment_clear(self, start, end, padding):
        for obstacle in self.obstacles:
            if isinstance(obstacle, CircleObstacle):
                if self._segment_circle_blocked(
                    start, end, obstacle.center, obstacle.radius + padding + _EPS
                ):
                    return False
            else:
                bounds = (
                    obstacle.x_min - padding - _EPS,
                    obstacle.x_max + padding + _EPS,
                    obstacle.y_min - padding - _EPS,
                    obstacle.y_max + padding + _EPS,
                )
                if self._segment_box_blocked(start, end, bounds):
                    return False
        return True

    def _project_safe(self, point, padding=0.90):
        """Move a dynamic pursuit/screen endpoint outside inflated terrain."""
        x = _clip(point[0], 0.52, self.width - 0.52)
        y = _clip(point[1], 0.52, self.height - 0.52)
        for _ in range(3):
            changed = False
            for obstacle in self.obstacles:
                if isinstance(obstacle, CircleObstacle):
                    dx, dy = x - obstacle.center[0], y - obstacle.center[1]
                    distance = hypot(dx, dy)
                    required = obstacle.radius + padding
                    if distance <= required:
                        direction = _unit(dx, dy)
                        if direction == (0.0, 0.0):
                            direction = (self.direction, 0.0)
                        x = obstacle.center[0] + direction[0] * (required + 0.03)
                        y = obstacle.center[1] + direction[1] * (required + 0.03)
                        changed = True
                else:
                    left = obstacle.x_min - padding
                    right = obstacle.x_max + padding
                    bottom = obstacle.y_min - padding
                    top = obstacle.y_max + padding
                    if left <= x <= right and bottom <= y <= top:
                        choices = (
                            (x - left, (left - 0.03, y)),
                            (right - x, (right + 0.03, y)),
                            (y - bottom, (x, bottom - 0.03)),
                            (top - y, (x, top + 0.03)),
                        )
                        _, (x, y) = min(choices, key=lambda item: item[0])
                        changed = True
            x = _clip(x, 0.52, self.width - 0.52)
            y = _clip(y, 0.52, self.height - 0.52)
            if not changed:
                break
        result = (float(x), float(y))
        if not self._point_clear(result, self.EMERGENCY_CLEARANCE + 0.08):
            # Inflated neighboring obstacles can overlap and make sequential
            # projection oscillate.  Roadmap nodes are globally verified, so
            # the nearest one is a deterministic safe fallback.
            result = min(self.nodes, key=lambda node: (_distance(node, result), node))
        return result

    def _make_roadmap_nodes(self):
        nodes = [
            (18.5, 0.55),
            (81.5, 0.55),
            (18.5, 59.45),
            (81.5, 59.45),
        ]
        clearance = self.PLAN_CLEARANCE
        extra = self.ROADMAP_EPSILON
        for obstacle in self.obstacles:
            if isinstance(obstacle, CircleObstacle):
                samples = 16
                # This radius certifies that even the chord between adjacent
                # samples remains outside the clearance circle.
                radius = (obstacle.radius + clearance + extra) / cos(pi / samples)
                for k in range(samples):
                    angle = 2.0 * pi * k / samples
                    nodes.append(
                        (
                            obstacle.center[0] + radius * cos(angle),
                            obstacle.center[1] + radius * sin(angle),
                        )
                    )
            else:
                offset = clearance + extra
                nodes.extend(
                    (
                        (obstacle.x_min - offset, obstacle.y_min - offset),
                        (obstacle.x_min - offset, obstacle.y_max + offset),
                        (obstacle.x_max + offset, obstacle.y_min - offset),
                        (obstacle.x_max + offset, obstacle.y_max + offset),
                    )
                )

        unique = []
        seen = set()
        for x, y in nodes:
            point = (float(_clip(x, 0.52, self.width - 0.52)), float(_clip(y, 0.52, self.height - 0.52)))
            key = (round(point[0], 7), round(point[1], 7))
            if key not in seen and self._point_clear(point, clearance):
                seen.add(key)
                unique.append(point)
        return tuple(unique)

    def _make_roadmap_graph(self):
        graph = [[] for _ in self.nodes]
        for i, left in enumerate(self.nodes):
            for j in range(i + 1, len(self.nodes)):
                right = self.nodes[j]
                if self._segment_clear(left, right, self.PLAN_CLEARANCE):
                    length = _distance(left, right)
                    graph[i].append((j, length))
                    graph[j].append((i, length))
        return tuple(tuple(edges) for edges in graph)

    @staticmethod
    def _make_portals(goal, count):
        margin = 0.9
        low, high = goal.y_min + margin, goal.y_max - margin
        x = goal.center[0]
        if count == 1:
            return ((float(x), float((low + high) * 0.5)),)
        return tuple((float(x), float(low + (high - low) * i / (count - 1))) for i in range(count))

    @staticmethod
    def _point_segment_distance(point, start, end):
        dx, dy = end[0] - start[0], end[1] - start[1]
        denominator = dx * dx + dy * dy
        if denominator <= _EPS:
            return _distance(point, start)
        t = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / denominator
        t = _clip(t, 0.0, 1.0)
        return hypot(point[0] - start[0] - t * dx, point[1] - start[1] - t * dy)

    def _danger_cost(self, start, end, length, dangers):
        cost = length
        for point, weight in dangers:
            separation = self._point_segment_distance(point, start, end)
            if separation < 8.0:
                risk = (8.0 - separation) / 8.0
                cost += length * weight * risk * risk
        return cost

    def _route(self, start, target, dangers=()):
        direct_clear = self._segment_clear(start, target, self.PLAN_CLEARANCE)
        if direct_clear and not dangers:
            return [target]
        direct_cost = (
            self._danger_cost(start, target, _distance(start, target), dangers)
            if direct_clear
            else float("inf")
        )

        start_links = []
        target_links = {}
        for index, node in enumerate(self.nodes):
            if self._segment_clear(start, node, self.PLAN_CLEARANCE):
                length = _distance(start, node)
                start_links.append((index, self._danger_cost(start, node, length, dangers) if dangers else length))
            if self._segment_clear(node, target, self.PLAN_CLEARANCE):
                length = _distance(node, target)
                target_links[index] = self._danger_cost(node, target, length, dangers) if dangers else length

        # A drone displaced inside the generous planning envelope still needs an
        # escape route.  Static edges remain 1.05-clear; only its first/last link
        # may spend some reserve.
        if not start_links or not target_links:
            for padding in (self.TRACK_CLEARANCE, 0.40):
                if not start_links:
                    start_links = [
                        (i, _distance(start, node))
                        for i, node in enumerate(self.nodes)
                        if self._segment_clear(start, node, padding)
                    ]
                if not target_links:
                    target_links = {
                        i: _distance(node, target)
                        for i, node in enumerate(self.nodes)
                        if self._segment_clear(node, target, padding)
                    }
                if start_links and target_links:
                    break

        count = len(self.nodes)
        distances = [float("inf")] * count
        previous = [-1] * count
        queue = []
        for index, cost in start_links:
            if cost < distances[index]:
                distances[index] = cost
                heappush(queue, (cost, index))

        best_total = direct_cost
        best_end = -1
        while queue:
            cost, node_index = heappop(queue)
            if cost != distances[node_index] or cost >= best_total:
                continue
            if node_index in target_links:
                total = cost + target_links[node_index]
                if total < best_total:
                    best_total, best_end = total, node_index
            node = self.nodes[node_index]
            for neighbor, length in self.graph[node_index]:
                edge_cost = self._danger_cost(node, self.nodes[neighbor], length, dangers) if dangers else length
                candidate = cost + edge_cost
                if candidate + _EPS < distances[neighbor]:
                    distances[neighbor] = candidate
                    previous[neighbor] = node_index
                    heappush(queue, (candidate, neighbor))

        if best_end < 0 and direct_clear:
            return [target]
        if best_end < 0:
            # The generated arena always has the protected outer rails.  This is
            # a last-resort recovery for a dynamically displaced endpoint.
            return [target]
        indices = []
        cursor = best_end
        while cursor >= 0:
            indices.append(cursor)
            cursor = previous[cursor]
        indices.reverse()
        route = [self.nodes[index] for index in indices]
        route.append(target)
        return route

    @staticmethod
    def _polyline_length(start, route):
        total = 0.0
        cursor = start
        for point in route:
            total += _distance(cursor, point)
            cursor = point
        return total

    # ------------------------------------------------------------------
    # Travel/contact estimates and dynamic FAST job allocation

    @staticmethod
    def _travel_time(distance, drone, spec, destination):
        distance = max(0.0, distance)
        if distance <= _EPS:
            return 0.0
        direction = _unit(destination[0] - drone.position[0], destination[1] - drone.position[1])
        projected_speed = _clip(_dot(drone.velocity, direction), 0.0, spec.max_speed)
        acceleration_distance = max(0.0, (spec.max_speed**2 - projected_speed**2) / (2.0 * spec.max_acceleration))
        if distance <= acceleration_distance:
            time = (sqrt(projected_speed**2 + 2.0 * spec.max_acceleration * distance) - projected_speed) / spec.max_acceleration
        else:
            time = (spec.max_speed - projected_speed) / spec.max_acceleration
            time += (distance - acceleration_distance) / spec.max_speed
        # Both types need 0.25 s to jerk from zero to full acceleration.  Add a
        # direction-dependent fraction as an optimism guard for reversals.
        velocity_size = hypot(*drone.velocity)
        alignment = _dot(_unit(*drone.velocity), direction) if velocity_size > 0.1 else 0.0
        return time + 0.12 + 0.18 * max(0.0, -alignment)

    def _goal_eta(self, drone, own_drone):
        goal = self.goal if own_drone else self.own_goal
        travel_direction = self.direction if own_drone else -self.direction
        # Tactical auctions run frequently, so use a cheap conservative route
        # estimate here.  Exact graph paths are still used for every commanded
        # trajectory.  The target is the first scoring plane, not goal center.
        target_x = goal.x_min if travel_direction > 0.0 else goal.x_max
        target_y = _clip(drone.position[1], goal.y_min + 0.45, goal.y_max - 0.45)
        target = (target_x, target_y)
        length = _distance(drone.position, target)
        if not self._segment_clear(drone.position, target, 0.40):
            # Cheap coarse detour allowance; feasibility probabilities retain a
            # broad uncertainty band because compound routes can be longer.
            length += 4.0
        spec = self.specs[drone.drone_type]
        return self._travel_time(length, drone, spec, target)

    def _predict_position(self, drone, seconds, fallback_direction):
        seconds = max(0.0, seconds)
        dynamic_time = min(seconds, 0.60)
        x, y = drone.position
        vx, vy = drone.velocity
        ax, ay = drone.acceleration
        spec = self.specs[drone.drone_type]
        remaining = dynamic_time
        # Continue the observed acceleration, but reproduce the engine's 0.05 s
        # position/update order and per-tick velocity clipping.
        while remaining > _EPS:
            dt = min(0.05, remaining)
            x += vx * dt + 0.5 * ax * dt * dt
            y += vy * dt + 0.5 * ay * dt * dt
            vx, vy = _norm_clip(vx + ax * dt, vy + ay * dt, spec.max_speed)
            remaining -= dt
        if seconds > dynamic_time:
            velocity = _unit(vx, vy)
            if velocity == (0.0, 0.0):
                velocity = fallback_direction
            x += velocity[0] * spec.max_speed * (seconds - dynamic_time)
            y += velocity[1] * spec.max_speed * (seconds - dynamic_time)
        return (
            float(_clip(x, 0.75, self.width - 0.75)),
            float(_clip(y, 0.75, self.height - 0.75)),
        )

    def _contact_estimate(self, interceptor, target, deadline):
        interceptor_spec = self.specs[interceptor.drone_type]
        target_forward = (self.direction, 0.0) if target.team is self.team else (-self.direction, 0.0)
        estimate = max(0.0, _distance(interceptor.position, target.position) - 0.75) / max(interceptor_spec.max_speed, 0.1)
        intercept_point = target.position
        for _ in range(3):
            horizon = min(max(0.0, estimate), max(0.0, deadline), 8.0)
            intercept_point = self._predict_position(target, horizon, target_forward)
            length = max(0.0, _distance(interceptor.position, intercept_point) - 0.75)
            if not self._segment_clear(interceptor.position, intercept_point, 0.40):
                length += 3.0
            estimate = self._travel_time(length, interceptor, interceptor_spec, intercept_point)
        slack = deadline - estimate
        probability = _clip(0.5 + slack / 1.25, 0.0, 1.0)
        return estimate, probability, intercept_point

    def _threats(self, own_slow, enemy_fast, slow_goal_eta):
        """Return the best threatened SLOW for each enemy FAST and per-SLOW risk."""
        by_enemy = {}
        risk_by_slow = {drone.id: 0.0 for drone in own_slow}
        for enemy in enemy_fast:
            best = None
            for slow in own_slow:
                deadline = slow_goal_eta[slow.id]
                contact_time, reach, point = self._contact_estimate(enemy, slow, deadline)
                to_slow = _unit(slow.position[0] - enemy.position[0], slow.position[1] - enemy.position[1])
                speed = max(0.1, self.specs[enemy.drone_type].max_speed)
                alignment = _clip(_dot(enemy.velocity, to_slow) / speed, -1.0, 1.0)
                distance = _distance(enemy.position, slow.position)
                # Reachability is a nonzero threat floor; observed alignment only
                # raises it, so a late feint cannot make a hunter invisible.
                intent = 0.28 + 0.72 * max(0.0, alignment)
                if distance < 10.0 or contact_time < 2.0:
                    intent = 1.0
                risk = reach * intent
                if best is None or (risk, -contact_time, -slow.id) > (best[0], -best[1], -best[2].id):
                    best = (risk, contact_time, slow, point)
            if best is not None:
                by_enemy[enemy.id] = best
                risk_by_slow[best[2].id] = max(risk_by_slow[best[2].id], best[0])
        return by_enemy, risk_by_slow

    def _assign_fast_roles(self, state, own_fast, own_slow, enemy_fast, enemy_slow):
        remaining = max(0.0, 90.0 - state.time)
        slow_goal_eta = {drone.id: self._goal_eta(drone, True) for drone in own_slow}
        enemy_slow_eta = {drone.id: self._goal_eta(drone, False) for drone in enemy_slow}
        threats, slow_risk = self._threats(own_slow, enemy_fast, slow_goal_eta)
        own_by_id = {drone.id: drone for drone in own_slow}
        enemy_by_id = {drone.id: drone for drone in enemy_fast + enemy_slow}

        jobs = []
        # Denial is worth five points, but cap speculative hunters so the whole
        # FAST team cannot be pulled away by ten simultaneously reachable SLOWs.
        deny_rank = []
        for enemy in enemy_slow:
            urgency = 1.0 - _clip(enemy_slow_eta[enemy.id] / max(remaining, 1.0), 0.0, 1.0)
            deny_rank.append((urgency, enemy.id))
        deny_rank.sort(reverse=True)
        max_denials = min(len(own_fast), 6 if state.own_score <= state.opponent_score + 5 else 5)
        allowed_denials = {enemy_id for _, enemy_id in deny_rank[:max_denials]}
        for enemy in enemy_slow:
            if enemy.id in allowed_denials:
                jobs.append(("deny", enemy.id, None))

        # Concrete protection jobs consume the threatening FAST one-for-one.
        for enemy_id, (risk, contact_time, slow, _) in threats.items():
            if risk >= 0.20 or contact_time < 4.0:
                jobs.append(("protect", enemy_id, slow.id))

        # Proactive screens cover the highest-risk distinct SLOWs.  Four reserves
        # establish a balanced opening; concrete threats can raise this to ten.
        screened = {aux for role, _, aux in jobs if role == "protect"}
        escort_candidates = sorted(
            ((risk, slow.id) for slow in own_slow if slow.id not in screened for risk in (slow_risk[slow.id],)),
            key=lambda item: (-item[0], item[1]),
        )
        proactive = max(0, min(4, len(own_fast)) - len(screened))
        for _, slow_id in escort_candidates[:proactive]:
            jobs.append(("escort", slow_id, None))

        candidates = {drone.id: [] for drone in own_fast}
        for fast in own_fast:
            score_eta = self._goal_eta(fast, True)
            score_probability = _clip(0.5 + (remaining - score_eta) / 1.25, 0.0, 1.0)
            previous = self.roles.get(fast.id, ("score", None, None))
            for job in jobs:
                role, target_id, auxiliary = job
                sticky = 0.32 if previous == job else 0.0
                if role == "deny":
                    target = enemy_by_id[target_id]
                    deadline = min(remaining, enemy_slow_eta[target_id])
                    contact, probability, point = self._contact_estimate(fast, target, deadline)
                    utility = 5.0 * probability - score_probability + sticky
                elif role == "protect":
                    target = enemy_by_id[target_id]
                    risk, threat_contact, slow, _ = threats[target_id]
                    contact, probability, point = self._contact_estimate(fast, target, min(remaining, threat_contact))
                    utility = 5.0 * risk * probability - score_probability + sticky
                else:
                    slow = own_by_id[target_id]
                    risk = max(0.35, slow_risk[target_id])
                    distance = _distance(fast.position, slow.position)
                    rendezvous = self._travel_time(
                        max(0.0, distance - 2.5), fast, self.specs[fast.drone_type], slow.position
                    )
                    probability = _clip(1.0 - rendezvous / max(slow_goal_eta[target_id], 1.0), 0.0, 1.0)
                    utility = 5.0 * risk * 0.72 * probability - score_probability + sticky
                    point = slow.position
                    contact = rendezvous
                if utility > -0.2:
                    candidates[fast.id].append((utility, job, contact, point))

        # Deterministic regret matching avoids the worst globally-greedy conflict
        # while keeping the 10x~20 auction tiny and dependency-free.
        unassigned = {drone.id for drone in own_fast}
        claimed = set()
        chosen = {}
        while unassigned:
            bids = []
            for fast_id in sorted(unassigned):
                options = [item for item in candidates[fast_id] if item[1] not in claimed]
                options.sort(key=lambda item: (-item[0], item[1], item[2]))
                if not options or options[0][0] <= 0.0:
                    continue
                second = max(0.0, options[1][0] if len(options) > 1 else 0.0)
                regret = options[0][0] - second
                bids.append((regret, options[0][0], -fast_id, fast_id, options[0]))
            if not bids:
                break
            _, _, _, fast_id, option = max(bids)
            chosen[fast_id] = option[1]
            claimed.add(option[1])
            unassigned.remove(fast_id)

        for fast in own_fast:
            self.roles[fast.id] = chosen.get(fast.id, ("score", None, None))
        return threats, slow_risk

    # ------------------------------------------------------------------
    # Path following, evasive shaping, and jerk-safe command selection

    def _advance_path(self, drone, path):
        if not path:
            return
        while len(path) > 1:
            reached = _distance(drone.position, path[0]) < 0.75
            if not reached:
                break
            path.pop(0)

    def _corner_speed(self, drone, path, spec):
        if len(path) < 2:
            return spec.max_speed
        first = _unit(path[0][0] - drone.position[0], path[0][1] - drone.position[1])
        second = _unit(path[1][0] - path[0][0], path[1][1] - path[0][1])
        cosine = _clip(_dot(first, second), -1.0, 1.0)
        theta = acos(cosine)
        if theta < 0.08:
            return spec.max_speed
        half = theta * 0.5
        secant_loss = 1.0 / max(cos(half), 0.05) - 1.0
        radius_by_margin = 0.35 / max(secant_loss, 1.0e-4)
        incoming = _distance(drone.position, path[0])
        outgoing = _distance(path[0], path[1])
        radius_by_length = 0.45 * min(incoming, outgoing) / max(sin(half) / max(cos(half), 0.05), 1.0e-4)
        radius = max(0.02, min(radius_by_margin, radius_by_length))
        return min(spec.max_speed, 0.75 * sqrt(spec.max_acceleration * radius))

    def _steer(self, drone, path, *, moving_velocity=(0.0, 0.0), stop_at_target=False):
        spec = self.specs[drone.drone_type]
        self._advance_path(drone, path)
        if not path:
            return (0.0, 0.0), drone.position
        target = path[0]
        dx, dy = target[0] - drone.position[0], target[1] - drone.position[1]
        distance = hypot(dx, dy)
        direction = _unit(dx, dy)
        speed = spec.max_speed
        if len(path) > 1:
            corner_speed = self._corner_speed(drone, path, spec)
            velocity = hypot(*drone.velocity)
            braking = max(0.0, (velocity * velocity - corner_speed * corner_speed) / (2.0 * spec.max_acceleration))
            braking += 0.25 * velocity + 0.35
            if distance < 1.25 * braking:
                available = max(0.0, distance - 0.35)
                speed = min(speed, sqrt(corner_speed * corner_speed + 2.0 * spec.max_acceleration * available))
        elif stop_at_target:
            speed = min(speed, sqrt(max(0.0, 2.0 * spec.max_acceleration * distance)))

        terrain_clearance, _ = self._obstacle_escape(drone)
        if terrain_clearance < 5.5:
            # Route vertices are certified, but a velocity controller needs time
            # to remove cross-track error before a face/corner.  This cap makes
            # the reserve grow with braking distance instead of relying on the
            # short-horizon emergency override.
            safe_run = max(0.0, terrain_clearance - 0.55)
            terrain_speed = 0.60 * sqrt(2.0 * spec.max_acceleration * safe_run)
            speed = min(speed, max(0.45, terrain_speed))

        desired_vx = direction[0] * speed + moving_velocity[0]
        desired_vy = direction[1] * speed + moving_velocity[1]
        desired_vx, desired_vy = _norm_clip(desired_vx, desired_vy, spec.max_speed)
        predicted_vx = drone.velocity[0] + 0.15 * drone.acceleration[0]
        predicted_vy = drone.velocity[1] + 0.15 * drone.acceleration[1]
        command = _norm_clip((desired_vx - predicted_vx) / 0.35, (desired_vy - predicted_vy) / 0.35, spec.max_acceleration)
        return command, target

    def _closest_approach(self, own, enemy, horizon=3.0):
        rx, ry = enemy.position[0] - own.position[0], enemy.position[1] - own.position[1]
        vx, vy = enemy.velocity[0] - own.velocity[0], enemy.velocity[1] - own.velocity[1]
        speed_squared = vx * vx + vy * vy
        time = _clip(-(rx * vx + ry * vy) / speed_squared, 0.0, horizon) if speed_squared > _EPS else 0.0
        px, py = rx + vx * time, ry + vy * time
        return time, hypot(px, py), (px, py)

    def _avoid_enemies(self, drone, command, enemies, engage_id, path_direction, aggressive):
        spec = self.specs[drone.drone_type]
        ax, ay = command
        best = None
        for enemy in enemies:
            if enemy.id == engage_id:
                continue
            time, miss, miss_vector = self._closest_approach(drone, enemy)
            separation = _distance(drone.position, enemy.position)
            danger = 0.0
            if time > 0.0 and miss < 3.2:
                danger = (3.2 - miss) / 3.2 * (1.0 - time / 3.4)
            if separation < 3.0:
                danger = max(danger, (3.0 - separation) / 3.0)
            if danger > 0.0 and (best is None or danger > best[0]):
                best = (danger, enemy, time, miss_vector)

        if best is None:
            return command
        danger, enemy, time, miss_vector = best
        away = _unit(-miss_vector[0], -miss_vector[1])
        if away == (0.0, 0.0):
            away = _unit(drone.position[0] - enemy.position[0], drone.position[1] - enemy.position[1])
        left = (-path_direction[1], path_direction[0])
        preferred = 1.0 if _dot(left, away) >= 0.0 else -1.0
        old_side = self.evasion_side.get(drone.id, preferred)
        if old_side * preferred < 0.0 and danger < 0.72:
            preferred = old_side
        # Never choose an evasive side blindly: probe far enough to expose a
        # nearby inflated obstacle before lateral acceleration commits the drone.
        probe_length = 1.25 + 0.35 * hypot(*drone.velocity)
        def side_is_clear(side):
            lateral_probe = (left[0] * side, left[1] * side)
            probe = (
                drone.position[0] + 0.45 * path_direction[0] * probe_length + lateral_probe[0] * probe_length,
                drone.position[1] + 0.45 * path_direction[1] * probe_length + lateral_probe[1] * probe_length,
            )
            return (
                0.35 <= probe[0] <= self.width - 0.35
                and 0.35 <= probe[1] <= self.height - 0.35
                and self._point_clear(probe, self.TRACK_CLEARANCE)
                and self._segment_clear(drone.position, probe, self.EMERGENCY_CLEARANCE)
            )
        preferred_clear = side_is_clear(preferred)
        alternate_clear = side_is_clear(-preferred)
        if not preferred_clear and alternate_clear:
            preferred = -preferred
        self.evasion_side[drone.id] = preferred
        lateral = (left[0] * preferred, left[1] * preferred)
        strength = spec.max_acceleration * danger * (0.65 if aggressive else 1.0)
        if not preferred_clear and not alternate_clear:
            # A narrow corridor is a poor place to dodge.  Preserve the safe
            # path and use only a modest radial correction until it opens.
            lateral = (0.0, 0.0)
            strength *= 0.35
        ax += strength * (0.62 * away[0] + 0.78 * lateral[0])
        ay += strength * (0.62 * away[1] + 0.78 * lateral[1])
        return _norm_clip(ax, ay, spec.max_acceleration)

    def _simulate_command(self, drone, command):
        spec = self.specs[drone.drone_type]
        position = drone.position
        velocity = drone.velocity
        acceleration = drone.acceleration
        segments = []
        # Hold each candidate for 1.00 s.  Looking only through the next control
        # period is insufficient when current velocity plus jerk lag already
        # commits a drone to a rectangle face.
        for _ in range(20):
            delta = (command[0] - acceleration[0], command[1] - acceleration[1])
            change = _norm_clip(delta[0], delta[1], spec.max_jerk * 0.05)
            acceleration = _norm_clip(acceleration[0] + change[0], acceleration[1] + change[1], spec.max_acceleration)
            new_position = (
                position[0] + velocity[0] * 0.05 + 0.5 * acceleration[0] * 0.05**2,
                position[1] + velocity[1] * 0.05 + 0.5 * acceleration[1] * 0.05**2,
            )
            velocity = _norm_clip(velocity[0] + acceleration[0] * 0.05, velocity[1] + acceleration[1] * 0.05, spec.max_speed)
            segments.append((position, new_position))
            position = new_position
        safe = all(
            self._segment_clear(start, end, self.EMERGENCY_CLEARANCE)
            and 0.30 <= end[0] <= self.width - 0.30
            and 0.30 <= end[1] <= self.height - 0.30
            for start, end in segments
        )
        return safe, position, velocity

    def _obstacle_escape(self, drone):
        best_distance = float("inf")
        vector = (0.0, 0.0)
        x, y = drone.position
        for obstacle in self.obstacles:
            if isinstance(obstacle, CircleObstacle):
                dx, dy = x - obstacle.center[0], y - obstacle.center[1]
                center_distance = hypot(dx, dy)
                clearance = center_distance - obstacle.radius
                outward = _unit(dx, dy)
                if outward == (0.0, 0.0):
                    outward = (self.direction, 0.0)
            else:
                # The engine expands rectangle axes independently by 0.25 m;
                # measure against that AABB, then convert back to the raw-surface
                # clearance convention used by the circle branch/callers.
                left, right = obstacle.x_min - 0.25, obstacle.x_max + 0.25
                bottom, top = obstacle.y_min - 0.25, obstacle.y_max + 0.25
                nearest_x = _clip(x, left, right)
                nearest_y = _clip(y, bottom, top)
                dx, dy = x - nearest_x, y - nearest_y
                hard_clearance = hypot(dx, dy)
                outward = _unit(dx, dy)
                if outward == (0.0, 0.0):
                    options = (
                        (abs(x - left), (-1.0, 0.0)),
                        (abs(right - x), (1.0, 0.0)),
                        (abs(y - bottom), (0.0, -1.0)),
                        (abs(top - y), (0.0, 1.0)),
                    )
                    penetration_to_edge, outward = min(options)
                    clearance = 0.25 - penetration_to_edge
                else:
                    clearance = hard_clearance + 0.25
            if clearance < best_distance:
                best_distance, vector = clearance, outward
        return best_distance, vector

    def _terrain_guard(self, drone, command):
        """Override inward commands before jerk-limited braking becomes late."""
        spec = self.specs[drone.drone_type]
        clearance, outward = self._obstacle_escape(drone)
        inward_speed = max(0.0, -_dot(drone.velocity, outward))
        inward_acceleration = max(0.0, -_dot(drone.acceleration, outward))
        stop_distance = 0.25 * inward_speed + inward_speed * inward_speed / (2.0 * spec.max_acceleration)
        stop_distance += 0.5 * inward_acceleration * 0.25**2 + 0.12
        required = 0.25 + stop_distance
        if clearance >= required + 1.25:
            return command
        urgency = _clip((required + 1.25 - clearance) / 1.25, 0.0, 1.0)
        inward_command = max(0.0, -_dot(command, outward))
        ax = command[0] + outward[0] * (inward_command + urgency * spec.max_acceleration * 1.35)
        ay = command[1] + outward[1] * (inward_command + urgency * spec.max_acceleration * 1.35)
        return _norm_clip(ax, ay, spec.max_acceleration)

    def _safe_command(self, drone, command, target):
        spec = self.specs[drone.drone_type]
        candidates = [command]
        brake = _norm_clip(
            -3.0 * drone.velocity[0] - 0.8 * drone.acceleration[0],
            -3.0 * drone.velocity[1] - 0.8 * drone.acceleration[1],
            spec.max_acceleration,
        )
        candidates.append(brake)
        clearance, outward = self._obstacle_escape(drone)
        candidates.append((outward[0] * spec.max_acceleration, outward[1] * spec.max_acceleration))
        base_angle = pi / 6.0
        for sign in (-1.0, 1.0):
            c, s = cos(sign * base_angle), sin(sign * base_angle)
            candidates.append(_norm_clip(command[0] * c - command[1] * s, command[0] * s + command[1] * c, spec.max_acceleration))

        best = None
        for candidate in candidates:
            safe, position, velocity = self._simulate_command(drone, candidate)
            if not safe:
                continue
            error = _distance(position, target) + 0.035 * hypot(*velocity)
            if candidate is brake:
                error += 0.12
            if best is None or error < best[0]:
                best = (error, candidate)
        if best is not None:
            return best[1]
        # If all long-horizon rollouts are already unsafe, maximize outward response;
        # this is finite and deterministic even for a nearly lost drone.
        return (float(outward[0] * spec.max_acceleration), float(outward[1] * spec.max_acceleration))

    def _pursuit_point(self, hunter, target, deadline=8.0):
        estimate, _, point = self._contact_estimate(hunter, target, deadline)
        # Far-horizon acceleration extrapolation is unreliable around terrain;
        # use at most a two-second lead for the actual moving target.
        point = self._predict_position(target, min(estimate, 2.0), (-self.direction, 0.0))
        return self._project_safe(point, 0.90)

    def _screen_point(self, guard, slow, enemies, threat_id=None):
        threat = next((enemy for enemy in enemies if enemy.id == threat_id), None)
        if threat is None and enemies:
            threat = min(enemies, key=lambda enemy: (_distance(enemy.position, slow.position), enemy.id))
        if threat is not None:
            direction = _unit(threat.position[0] - slow.position[0], threat.position[1] - slow.position[1])
            separation = _distance(threat.position, slow.position)
            lead = min(3.4, max(1.4, separation * 0.28))
        else:
            direction = (self.direction, 0.0)
            lead = 2.8
        point = (slow.position[0] + direction[0] * lead, slow.position[1] + direction[1] * lead)
        point = (
            float(_clip(point[0], 0.8, self.width - 0.8)),
            float(_clip(point[1], 0.8, self.height - 0.8)),
        )
        return self._project_safe(point, 0.90)

    def step(self, state):
        own_active = [drone for drone in state.own_drones if drone.status is DroneStatus.ACTIVE]
        enemy_active = [drone for drone in state.opponent_drones if drone.status is DroneStatus.ACTIVE]
        own_fast = [drone for drone in own_active if drone.drone_type is DroneType.FAST]
        own_slow = [drone for drone in own_active if drone.drone_type is DroneType.SLOW]
        enemy_fast = [drone for drone in enemy_active if drone.drone_type is DroneType.FAST]
        enemy_slow = [drone for drone in enemy_active if drone.drone_type is DroneType.SLOW]
        enemy_by_id = {drone.id: drone for drone in enemy_active}
        own_slow_by_id = {drone.id: drone for drone in own_slow}

        if state.time + _EPS >= self.last_role_time + 0.40:
            threats, slow_risk = self._assign_fast_roles(state, own_fast, own_slow, enemy_fast, enemy_slow)
            self.last_role_time = state.time
        else:
            slow_goal_eta = {drone.id: self._goal_eta(drone, True) for drone in own_slow}
            threats, slow_risk = self._threats(own_slow, enemy_fast, slow_goal_eta)

        # Threatened SLOWs periodically select a danger-weighted route.  Enemy
        # predicted positions bend the shortest path toward obstacle cover while
        # the globally safe roadmap continues to enforce terrain clearance.
        if state.time + _EPS >= self.last_slow_replan + 0.60:
            for slow in own_slow:
                destination = self.goal_target[slow.id]
                if slow_risk.get(slow.id, 0.0) > 0.22:
                    current_index = min(range(len(self.goal_portals)), key=lambda i: abs(self.goal_portals[i][1] - destination[1]))
                    indices = sorted({max(0, current_index - 2), current_index, min(len(self.goal_portals) - 1, current_index + 2)})
                    predicted = [
                        self._predict_position(enemy, 1.2, (-self.direction, 0.0))
                        for enemy in enemy_fast
                    ]
                    def lane_cost(index):
                        portal = self.goal_portals[index]
                        exposure = sum(max(0.0, 7.0 - abs(portal[1] - point[1])) for point in predicted)
                        return (_distance(slow.position, portal) + 0.32 * exposure, index)
                    chosen_index = min(indices, key=lane_cost)
                    destination = self.goal_portals[chosen_index]
                    nearest = sorted(
                        enemy_fast,
                        key=lambda enemy: (_distance(enemy.position, slow.position), enemy.id),
                    )[:3]
                    local_dangers = []
                    for enemy in nearest:
                        local_dangers.append((enemy.position, 1.55))
                        local_dangers.append((self._predict_position(enemy, 1.2, (-self.direction, 0.0)), 0.90))
                    route = self._route(slow.position, destination, local_dangers)
                    self.goal_target[slow.id] = destination
                    self.paths[slow.id] = route
                    self.path_destination[slow.id] = destination
                elif not self.paths.get(slow.id):
                    self.paths[slow.id] = self._route(slow.position, destination)
            self.last_slow_replan = state.time

        actions = {}
        for drone in own_active:
            engage_id = None
            stop_at_target = False
            moving_velocity = (0.0, 0.0)
            aggressive = False

            if drone.drone_type is DroneType.SLOW:
                destination = self.goal_target[drone.id]
                path = self.paths.get(drone.id)
                if not path:
                    path = self._route(drone.position, destination)
                    self.paths[drone.id] = path
            else:
                role, target_id, auxiliary = self.roles.get(drone.id, ("score", None, None))
                if role in ("deny", "protect") and target_id in enemy_by_id:
                    target = enemy_by_id[target_id]
                    engage_id = target_id
                    aggressive = True
                    destination = self._pursuit_point(drone, target)
                    last = self.last_pursuit_replan.get(drone.id, -10.0)
                    if (
                        state.time + _EPS >= last + 0.30
                        or _distance(destination, self.path_destination.get(drone.id, destination)) > 1.5
                    ):
                        self.paths[drone.id] = self._route(drone.position, destination)
                        self.path_destination[drone.id] = destination
                        self.last_pursuit_replan[drone.id] = state.time
                    path = self.paths[drone.id]
                elif role == "escort" and target_id in own_slow_by_id:
                    slow = own_slow_by_id[target_id]
                    threat_id = None
                    for enemy_id, (risk, _, threatened, _) in threats.items():
                        if threatened.id == slow.id and (threat_id is None or risk > threats[threat_id][0]):
                            threat_id = enemy_id
                    destination = self._screen_point(drone, slow, enemy_fast, threat_id)
                    moving_velocity = slow.velocity
                    stop_at_target = True
                    if threat_id is not None and threat_id in enemy_by_id and _distance(enemy_by_id[threat_id].position, slow.position) < 8.0:
                        engage_id = threat_id
                        destination = self._pursuit_point(drone, enemy_by_id[threat_id], 4.0)
                        moving_velocity = (0.0, 0.0)
                        stop_at_target = False
                        aggressive = True
                    last = self.last_pursuit_replan.get(drone.id, -10.0)
                    if (
                        state.time + _EPS >= last + 0.30
                        or _distance(destination, self.path_destination.get(drone.id, destination)) > 1.25
                    ):
                        self.paths[drone.id] = self._route(drone.position, destination)
                        self.path_destination[drone.id] = destination
                        self.last_pursuit_replan[drone.id] = state.time
                    path = self.paths[drone.id]
                else:
                    destination = self.goal_target[drone.id]
                    if self.path_destination.get(drone.id) != destination or not self.paths.get(drone.id):
                        self.paths[drone.id] = self._route(drone.position, destination)
                        self.path_destination[drone.id] = destination
                    path = self.paths[drone.id]

            command, local_target = self._steer(
                drone, path, moving_velocity=moving_velocity, stop_at_target=stop_at_target
            )
            path_direction = _unit(local_target[0] - drone.position[0], local_target[1] - drone.position[1])
            if path_direction == (0.0, 0.0):
                path_direction = (self.direction, 0.0)

            # SLOWs and free scorers evade likely contacts.  Assigned hunters
            # evade every non-target enemy so an escort cannot cheaply deflect
            # them from their intended five-point exchange.
            command = self._avoid_enemies(
                drone,
                command,
                enemy_active,
                engage_id,
                path_direction,
                aggressive,
            )
            command = self._terrain_guard(drone, command)
            command = self._safe_command(drone, command, local_target)
            actions[drone.id] = (float(command[0]), float(command[1]))

        return actions
