"""FastBait: obstacle-refuge SLOW bait with predictive FAST hunters.

SLOW drones deliberately stop at close obstacle perches.  They move to another
refuge only when a route-time/interception-time test says the crossing is safe,
or when their conservative goal ETA makes waiting unaffordable.  FAST drones
primarily make distinct lead-pursuit attacks on enemy SLOW drones.

Attribution
-----------
The visibility-roadmap layout, jerk-aware velocity tracking, and exact
short-horizon terrain veto are adapted from ``TanWeiXuan/Siren.py`` (which in
turn attributes its roadmap/veto foundation to TempoTrap and Sol Ultra).
The constant-speed interception quadratic, distinct-target assignment with
dummy scoring jobs, assignment hysteresis, and damped acceleration lead are
adapted from ``TanWeiXuan/gpt_5_6_sol_extra_high_aegis_weave.py`` and
``renj1ete0/claude_opus_5_apex.py``.  FastBait's refuge network, close-perimeter
perches, pursuit-occlusion scoring, active tangential bait, transition proof,
and deadline release are specific to this controller.
"""

from __future__ import annotations

from heapq import heappop, heappush
from math import acos, cos, hypot, inf, pi, sin, sqrt

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
DRONE_RADIUS = 0.25
INTERCEPT_RADIUS = 0.75


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
    """Asymmetric controller: every SLOW baits; every FAST hunts first."""

    # Padding is measured from the raw obstacle, so 0.43 leaves only 0.18 m
    # beyond the physical drone contact boundary.  This is intentionally much
    # tighter than ordinary scoring routes while retaining dynamics margin.
    BAIT_PADDING = 0.43
    BAIT_TRACK_PADDING = 0.285
    ROUTE_CLEARANCE = 0.56
    NODE_PADDING = 0.82
    SAFETY_PADDING = 0.265
    TRANSITION_MARGIN = 1.15
    DEADLINE_MARGIN = 4.2
    ACTIVE_BAIT = True

    def initialize(self, game_info):
        self.team = game_info.team
        self.direction = 1.0 if self.team is Team.A else -1.0
        self.width = float(game_info.arena_width)
        self.height = float(game_info.arena_height)
        self.goal = game_info.target_goal
        self.own_goal = game_info.own_goal
        self.obstacles = tuple(game_info.obstacles)
        self.specs = dict(game_info.drone_specs)

        own_slow = sorted(
            (d for d in game_info.own_initial_drones if d.drone_type is DroneType.SLOW),
            key=lambda d: (d.position[1], d.id),
        )
        own_fast = sorted(
            (d for d in game_info.own_initial_drones if d.drone_type is DroneType.FAST),
            key=lambda d: (d.position[1], d.id),
        )
        self.goal_target = {}
        for group in (own_slow, own_fast):
            for rank, drone in enumerate(group):
                self.goal_target[drone.id] = self._goal_portal(rank, len(group))

        self.nodes = self._make_nodes()
        self.graph = self._make_graph()
        self.refuges = self._make_refuges()

        # Greedy capacitated placement distributes ten SLOWs across the six to
        # twelve generated obstacles instead of sending all of them to the same
        # nearest body.  The second occupant costs roughly another obstacle hop.
        occupancy = [0] * len(self.refuges)
        initial_refuge = {}
        for slow in own_slow:
            if not self.refuges:
                initial_refuge[slow.id] = None
                continue
            choices = []
            for index, refuge in enumerate(self.refuges):
                center = refuge[1]
                forward = self.direction * (center[0] - slow.position[0])
                cost = _distance(slow.position, center) + 9.0 * occupancy[index]
                cost += 0.20 * abs(center[1] - slow.position[1]) - 0.10 * max(0.0, forward)
                choices.append((cost, occupancy[index], index))
            index = min(choices)[2]
            occupancy[index] += 1
            initial_refuge[slow.id] = index

        self.chain = {}
        self.chain_index = {}
        self.slow_phase = {}
        self.perch_index = {}
        self.last_perch_change = {}
        self.paths = {}
        self.path_destination = {}
        self.transition_cache = {}
        self.goal_eta_refuge = {}
        for slow in own_slow:
            if initial_refuge[slow.id] is None:
                self.chain[slow.id] = ()
                self.chain_index[slow.id] = 0
                self.slow_phase[slow.id] = "FINAL"
                self.perch_index[slow.id] = 0
                self.last_perch_change[slow.id] = -10.0
                destination = self.goal_target[slow.id]
                self.paths[slow.id] = self._route(slow.position, destination)
                self.path_destination[slow.id] = destination
                continue
            chain = self._build_chain(initial_refuge[slow.id], self.goal_target[slow.id])
            self.chain[slow.id] = chain
            self.chain_index[slow.id] = 0
            self.slow_phase[slow.id] = "TRANSIT"
            first = self._closest_perch(chain[0], slow.position)
            self.perch_index[slow.id] = first
            self.last_perch_change[slow.id] = -10.0
            destination = self.refuges[chain[0]][2][first]
            self.paths[slow.id] = self._route(slow.position, destination, self.BAIT_TRACK_PADDING)
            self.path_destination[slow.id] = destination

        # Keep all repeated Dijkstra work in the 10 s initialization window.
        # Transition proofs then re-evaluate moving enemies every 0.1 s using a
        # fixed obstacle-safe route, without synchronized step-time spikes.
        edges = {(chain[i], chain[i + 1]) for chain in self.chain.values() for i in range(len(chain) - 1)}
        for source, target in sorted(edges):
            for source_index, source_point in enumerate(self.refuges[source][2]):
                target_index = self._closest_perch(target, source_point)
                target_point = self.refuges[target][2][target_index]
                key = (source, target, source_index, target_index)
                self.transition_cache[key] = tuple(self._route(source_point, target_point, self.BAIT_TRACK_PADDING))
        for slow in own_slow:
            for refuge_index in self.chain[slow.id]:
                start = min(
                    self.refuges[refuge_index][2],
                    key=lambda point: (_distance(point, self.goal_target[slow.id]), point),
                )
                path = self._route(start, self.goal_target[slow.id], self.ROUTE_CLEARANCE)
                length = self._path_length(start, path)
                eta = self._travel_time(length, 0.0, 2.0, 2.5) + 0.32 * max(0, len(path) - 1) + 1.5
                self.goal_eta_refuge[(slow.id, refuge_index)] = eta

        for fast in own_fast:
            destination = self.goal_target[fast.id]
            self.paths[fast.id] = self._route(fast.position, destination)
            self.path_destination[fast.id] = destination

        self.fast_target = {fast.id: None for fast in own_fast}
        self.previous_assignment = {}
        self.last_assignment = -10.0
        self.hunter_route_time = {fast.id: -10.0 for fast in own_fast}
        self.hunter_route_target = {fast.id: None for fast in own_fast}

    # ------------------------------------------------------------------
    # Static geometry and visibility routing.  Adapted from Siren's compact
    # roadmap/chord-clearance construction; constants are retuned for FastBait.

    def _goal_portal(self, rank, count):
        y = self.goal.y_min + 0.75 + (self.goal.y_max - self.goal.y_min - 1.5) * (rank + 0.5) / max(1, count)
        x = self.goal.x_min + 1.2 if self.direction > 0 else self.goal.x_max - 1.2
        return (float(x), float(y))

    @staticmethod
    def _segment_circle_blocked(start, end, center, radius):
        dx, dy = end[0] - start[0], end[1] - start[1]
        length2 = dx * dx + dy * dy
        if length2 <= EPS:
            return _distance(start, center) <= radius
        t = _clip(((center[0] - start[0]) * dx + (center[1] - start[1]) * dy) / length2, 0.0, 1.0)
        x, y = start[0] + t * dx, start[1] + t * dy
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

    def _obstacle_blocks(self, obstacle, start, end, padding):
        if isinstance(obstacle, CircleObstacle):
            return self._segment_circle_blocked(start, end, obstacle.center, obstacle.radius + padding)
        return self._segment_box_blocked(
            start,
            end,
            (
                obstacle.x_min - padding,
                obstacle.x_max + padding,
                obstacle.y_min - padding,
                obstacle.y_max + padding,
            ),
        )

    def _point_clear(self, point, padding, ignore=None):
        x, y = point
        if not (0.30 <= x <= self.width - 0.30 and 0.30 <= y <= self.height - 0.30):
            return False
        return not any(
            obstacle is not ignore and self._obstacle_blocks(obstacle, point, point, padding)
            for obstacle in self.obstacles
        )

    def _segment_clear(self, start, end, padding, ignore=None):
        if not self._point_clear(start, min(padding, self.SAFETY_PADDING), ignore):
            return False
        if not self._point_clear(end, min(padding, self.SAFETY_PADDING), ignore):
            return False
        return not any(
            obstacle is not ignore and self._obstacle_blocks(obstacle, start, end, padding)
            for obstacle in self.obstacles
        )

    def _make_nodes(self):
        candidates = []
        for obstacle in self.obstacles:
            if isinstance(obstacle, CircleObstacle):
                sides = 12
                radius = (obstacle.radius + self.NODE_PADDING) / cos(pi / sides)
                for index in range(sides):
                    angle = 2.0 * pi * index / sides
                    candidates.append((obstacle.center[0] + radius * cos(angle), obstacle.center[1] + radius * sin(angle)))
            else:
                pad = self.NODE_PADDING
                candidates.extend(
                    (
                        (obstacle.x_min - pad, obstacle.y_min - pad),
                        (obstacle.x_min - pad, obstacle.y_max + pad),
                        (obstacle.x_max + pad, obstacle.y_min - pad),
                        (obstacle.x_max + pad, obstacle.y_max + pad),
                    )
                )
        nodes = []
        for point in candidates:
            point = (_clip(point[0], 0.45, self.width - 0.45), _clip(point[1], 0.45, self.height - 0.45))
            if self._point_clear(point, self.ROUTE_CLEARANCE) and all(_distance(point, other) > 0.05 for other in nodes):
                nodes.append(point)
        return tuple(nodes)

    def _make_graph(self):
        graph = [[] for _ in self.nodes]
        for left in range(len(self.nodes)):
            for right in range(left + 1, len(self.nodes)):
                if self._segment_clear(self.nodes[left], self.nodes[right], self.ROUTE_CLEARANCE):
                    length = _distance(self.nodes[left], self.nodes[right])
                    graph[left].append((right, length))
                    graph[right].append((left, length))
        return tuple(tuple(row) for row in graph)

    def _route(self, start, target, clearance=None):
        clearance = self.ROUTE_CLEARANCE if clearance is None else clearance
        start, target = tuple(start), tuple(target)
        if self._segment_clear(start, target, clearance):
            return [target]
        distance = [inf] * len(self.nodes)
        previous = [-1] * len(self.nodes)
        queue = []
        for index, node in enumerate(self.nodes):
            if self._segment_clear(start, node, clearance):
                distance[index] = _distance(start, node)
                heappush(queue, (distance[index], index))
        best_total, best_index = inf, -1
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
            return [target]
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

    # ------------------------------------------------------------------
    # Refuge construction, distribution, and bait geometry.

    def _make_refuges(self):
        refuges = []
        for obstacle in self.obstacles:
            if isinstance(obstacle, CircleObstacle):
                count = 16
                radius = obstacle.radius + self.BAIT_PADDING
                perches = tuple(
                    (
                        obstacle.center[0] + radius * cos(2.0 * pi * k / count),
                        obstacle.center[1] + radius * sin(2.0 * pi * k / count),
                    )
                    for k in range(count)
                )
                center = obstacle.center
                quality = obstacle.radius + 0.45
            else:
                p = self.BAIT_PADDING
                x0, x1 = obstacle.x_min - p, obstacle.x_max + p
                y0, y1 = obstacle.y_min - p, obstacle.y_max + p
                xm, ym = (x0 + x1) * 0.5, (y0 + y1) * 0.5
                # Clockwise ordering makes adjacent moves stay outside the
                # inflated rectangle rather than cutting through a corner.
                perches = ((xm, y0), (x1, y0), (x1, ym), (x1, y1), (xm, y1), (x0, y1), (x0, ym), (x0, y0))
                center = ((obstacle.x_min + obstacle.x_max) * 0.5, (obstacle.y_min + obstacle.y_max) * 0.5)
                quality = 0.25 * ((obstacle.x_max - obstacle.x_min) + (obstacle.y_max - obstacle.y_min)) + 0.65
            valid = tuple(point for point in perches if self._point_clear(point, self.SAFETY_PADDING, obstacle))
            if valid:
                refuges.append((obstacle, center, valid, quality))
        return tuple(refuges)

    def _closest_perch(self, refuge_index, position):
        perches = self.refuges[refuge_index][2]
        return min(range(len(perches)), key=lambda index: (_distance(position, perches[index]), index))

    def _build_chain(self, first, goal):
        chain = [first]
        used = {first}
        current = first
        while True:
            center = self.refuges[current][1]
            candidates = []
            for index, refuge in enumerate(self.refuges):
                if index in used:
                    continue
                progress = self.direction * (refuge[1][0] - center[0])
                if progress < 4.0:
                    continue
                span = _distance(center, refuge[1])
                if span > 36.0:
                    continue
                score = span + 0.16 * abs(refuge[1][1] - goal[1]) - 0.18 * progress - 0.55 * refuge[3]
                candidates.append((score, index))
            if not candidates:
                break
            current = min(candidates)[1]
            chain.append(current)
            used.add(current)
        return tuple(chain)

    @staticmethod
    def _closest_approach(first, second, horizon=3.0):
        rx, ry = second.position[0] - first.position[0], second.position[1] - first.position[1]
        vx, vy = second.velocity[0] - first.velocity[0], second.velocity[1] - first.velocity[1]
        speed2 = vx * vx + vy * vy
        time = 0.0 if speed2 <= EPS else _clip(-(rx * vx + ry * vy) / speed2, 0.0, horizon)
        return hypot(rx + vx * time, ry + vy * time), time

    def _threats_for(self, slow, enemy_fast):
        threats = []
        for enemy in enemy_fast:
            distance = _distance(slow.position, enemy.position)
            closest, time = self._closest_approach(slow, enemy)
            radial = _unit(slow.position[0] - enemy.position[0], slow.position[1] - enemy.position[1])
            closing = -_dot((enemy.velocity[0] - slow.velocity[0], enemy.velocity[1] - slow.velocity[1]), radial)
            eta = self._intercept_time(enemy, slow.position, slow.velocity)
            if distance < 18.0 or closest < 5.0 or (closing > 0.4 and eta < 8.0):
                threats.append((eta + 0.12 * time, enemy.id, enemy))
        threats.sort(key=lambda item: (item[0], item[1]))
        return threats

    def _bait_perch(self, slow, refuge_index, threats, now):
        obstacle, center, perches, _ = self.refuges[refuge_index]
        current = self.perch_index[slow.id] % len(perches)
        if not threats:
            return current
        relevant = [item[2] for item in threats[:3]]
        scores = []
        for index, point in enumerate(perches):
            score = -0.18 * _distance(slow.position, point)
            outward = _unit(point[0] - center[0], point[1] - center[1])
            for rank, enemy in enumerate(relevant):
                future = (
                    enemy.position[0] + 0.65 * enemy.velocity[0] + 0.10 * enemy.acceleration[0],
                    enemy.position[1] + 0.65 * enemy.velocity[1] + 0.10 * enemy.acceleration[1],
                )
                weight = 1.0 / (rank + 1.0)
                blocked = self._obstacle_blocks(obstacle, future, point, DRONE_RADIUS)
                away = _unit(center[0] - future[0], center[1] - future[1])
                score += weight * (5.2 if blocked else -1.8)
                score += weight * 1.4 * _dot(outward, away)
                # A pursuer already on a collision course with the body is the
                # best bait candidate; do not pull the SLOW into its near side.
                ray_end = (enemy.position[0] + 1.2 * enemy.velocity[0], enemy.position[1] + 1.2 * enemy.velocity[1])
                if self._obstacle_blocks(obstacle, enemy.position, ray_end, DRONE_RADIUS):
                    score += weight * (1.0 if blocked else -1.0)
            scores.append((score, -index, index))
        desired = max(scores)[2]
        if not self.ACTIVE_BAIT or now < self.last_perch_change[slow.id] + 0.65:
            return current
        count = len(perches)
        clockwise = (desired - current) % count
        counter = (current - desired) % count
        if min(clockwise, counter) == 0:
            # A small peek when a committed hunter is close makes a stationary
            # hidden target visible without abandoning the far-side geometry.
            if threats[0][0] < 3.6:
                first, second = (current + 1) % count, (current - 1) % count
                desired = max((scores[first][0], -first, first), (scores[second][0], -second, second))[2]
            else:
                return current
        else:
            desired = (current + (1 if clockwise <= counter else -1)) % count
        start, end = perches[current], perches[desired]
        if not self._segment_clear(start, end, self.SAFETY_PADDING, obstacle):
            return current
        self.last_perch_change[slow.id] = now
        return desired

    # ------------------------------------------------------------------
    # Dynamics-aware ETAs and the transition/deadline policy.

    @staticmethod
    def _travel_time(distance, initial_speed, acceleration, maximum_speed):
        initial_speed = _clip(initial_speed, 0.0, maximum_speed)
        accelerating = max(0.0, maximum_speed * maximum_speed - initial_speed * initial_speed) / (2.0 * acceleration)
        if distance <= accelerating:
            return (sqrt(max(0.0, initial_speed * initial_speed + 2.0 * acceleration * distance)) - initial_speed) / acceleration + 0.22
        return (maximum_speed - initial_speed) / acceleration + (distance - accelerating) / maximum_speed + 0.22

    def _path_time(self, drone, path):
        if not path:
            return 0.0
        direction = _unit(path[0][0] - drone.position[0], path[0][1] - drone.position[1])
        projected = max(0.0, _dot(drone.velocity, direction))
        spec = self.specs[drone.drone_type]
        length = self._path_length(drone.position, path)
        turns = max(0, len(path) - 1)
        return self._travel_time(length, projected, spec.max_acceleration, spec.max_speed) + 0.32 * turns

    def _intercept_time(self, pursuer, target_position, target_velocity=(0.0, 0.0)):
        # Constant-speed quadratic adapted from Aegis Weave/Apex, with the
        # physical interception radius removed before solving and a response tax.
        rx = target_position[0] - pursuer.position[0]
        ry = target_position[1] - pursuer.position[1]
        gap = hypot(rx, ry)
        if gap <= INTERCEPT_RADIUS:
            return 0.0
        scale = (gap - INTERCEPT_RADIUS) / gap
        rx, ry = rx * scale, ry * scale
        vx, vy = target_velocity
        speed = self.specs[DroneType.FAST].max_speed * 0.96
        qa = vx * vx + vy * vy - speed * speed
        qb = 2.0 * (rx * vx + ry * vy)
        qc = rx * rx + ry * ry
        roots = []
        if abs(qa) <= EPS:
            if qb < -EPS:
                roots.append(-qc / qb)
        else:
            discriminant = qb * qb - 4.0 * qa * qc
            if discriminant >= 0.0:
                root = sqrt(discriminant)
                roots.extend(value for value in ((-qb - root) / (2.0 * qa), (-qb + root) / (2.0 * qa)) if value >= 0.0)
        base = min(roots) if roots else gap / speed
        toward = _unit(target_position[0] - pursuer.position[0], target_position[1] - pursuer.position[1])
        projected = max(0.0, _dot(pursuer.velocity, toward))
        acceleration_eta = self._travel_time(max(0.0, gap - INTERCEPT_RADIUS), projected, 4.0, 5.0)
        return max(base, acceleration_eta - 0.18)

    def _position_along(self, start, path, fraction):
        total = self._path_length(start, path)
        remaining = total * fraction
        point = start
        for waypoint in path:
            span = _distance(point, waypoint)
            if span >= remaining and span > EPS:
                ratio = remaining / span
                return (point[0] + ratio * (waypoint[0] - point[0]), point[1] + ratio * (waypoint[1] - point[1]))
            remaining -= span
            point = waypoint
        return path[-1]

    def _commit_delay(self, enemy, slow, own_slow):
        if not own_slow:
            return 0.0
        projected = (enemy.position[0] + 1.0 * enemy.velocity[0], enemy.position[1] + 1.0 * enemy.velocity[1])
        chosen = min(own_slow, key=lambda other: (_distance(projected, other.position), other.id))
        if chosen.id == slow.id:
            return 0.0
        heading = _unit(enemy.velocity[0], enemy.velocity[1])
        to_chosen = _unit(chosen.position[0] - enemy.position[0], chosen.position[1] - enemy.position[1])
        commitment = max(0.0, _dot(heading, to_chosen)) * min(1.0, hypot(*enemy.velocity) / 4.0)
        return 0.55 * commitment

    def _transition_safe(self, slow, path, enemy_fast, own_slow):
        if not enemy_fast:
            return True
        cover_time = self._path_time(slow, path)
        total_length = max(EPS, self._path_length(slow.position, path))
        for fraction in (0.25, 0.50, 0.75, 1.0):
            point = self._position_along(slow.position, path, fraction)
            slow_time = cover_time * fraction
            for enemy in enemy_fast:
                # Use the direct chord as an optimistic lower bound on enemy
                # travel time, even when that chord crosses an obstacle.  This
                # is conservative for the bait: a transition is approved only
                # if even an impossible terrain-ignoring pursuer arrives late.
                # The SLOW time still follows its real obstacle-safe route.
                direction = _unit(point[0] - enemy.position[0], point[1] - enemy.position[1])
                projected = max(0.0, _dot(enemy.velocity, direction))
                length = max(0.0, _distance(enemy.position, point) - INTERCEPT_RADIUS)
                enemy_time = self._travel_time(length, projected, 4.0, 5.0)
                enemy_time += self._commit_delay(enemy, slow, own_slow)
                uncertainty = self.TRANSITION_MARGIN + 0.18 * fraction + 0.04 * total_length
                if enemy_time <= slow_time + uncertainty:
                    return False
        return True

    def _deadline_push(self, slow, remaining, now):
        del now
        chain = self.chain.get(slow.id, ())
        if not chain:
            return False
        index = min(self.chain_index[slow.id], len(chain) - 1)
        refuge_index = chain[index]
        eta = self.goal_eta_refuge[(slow.id, refuge_index)]
        if self.slow_phase[slow.id] == "TRANSIT":
            eta += self._path_time(slow, self.paths[slow.id])
        return remaining <= eta + self.DEADLINE_MARGIN

    # ------------------------------------------------------------------
    # Distinct value-five hunter assignment and predictive pursuit.

    @staticmethod
    def _hungarian_maximize(scores):
        # Shortest-augmenting-path assignment adapted from Aegis Weave.  Dummy
        # goal columns make every row feasible and prevent low-value FAST trades.
        rows = len(scores)
        columns = len(scores[0]) if rows else 0
        if not rows or not columns:
            return []
        u, v = [0.0] * (rows + 1), [0.0] * (columns + 1)
        match, way = [0] * (columns + 1), [0] * (columns + 1)
        for row in range(1, rows + 1):
            match[0] = row
            minimum, used = [inf] * (columns + 1), [False] * (columns + 1)
            column = 0
            while True:
                used[column] = True
                active = match[column]
                delta, next_column = inf, 0
                for candidate in range(1, columns + 1):
                    if used[candidate]:
                        continue
                    reduced = -scores[active - 1][candidate - 1] - u[active] - v[candidate]
                    if reduced < minimum[candidate]:
                        minimum[candidate], way[candidate] = reduced, column
                    if minimum[candidate] < delta:
                        delta, next_column = minimum[candidate], candidate
                for candidate in range(columns + 1):
                    if used[candidate]:
                        u[match[candidate]] += delta
                        v[candidate] -= delta
                    else:
                        minimum[candidate] -= delta
                column = next_column
                if match[column] == 0:
                    break
            while True:
                previous = way[column]
                match[column] = match[previous]
                column = previous
                if column == 0:
                    break
        result = [-1] * rows
        for column in range(1, columns + 1):
            if match[column]:
                result[match[column] - 1] = column - 1
        return result

    def _enemy_goal_eta(self, enemy):
        x = self.own_goal.x_max - 0.8 if self.direction > 0 else self.own_goal.x_min + 0.8
        target = (x, _clip(enemy.position[1], self.own_goal.y_min + 0.6, self.own_goal.y_max - 0.6))
        path = self._route(enemy.position, target)
        direction = _unit(path[0][0] - enemy.position[0], path[0][1] - enemy.position[1])
        projected = max(0.0, _dot(enemy.velocity, direction))
        return self._travel_time(self._path_length(enemy.position, path), projected, 2.0, 2.5) + 0.25 * max(0, len(path) - 1)

    def _assign_fast(self, fast, enemy_slow, remaining):
        if not fast or not enemy_slow:
            self.fast_target = {drone.id: None for drone in fast}
            self.previous_assignment = {}
            return
        goal_eta_by_id = {enemy.id: self._enemy_goal_eta(enemy) for enemy in enemy_slow}
        scores = []
        for hunter in fast:
            row = []
            for enemy in enemy_slow:
                eta = self._intercept_time(hunter, enemy.position, enemy.velocity)
                goal_eta = goal_eta_by_id[enemy.id]
                feasible = eta + 0.35 < min(remaining, goal_eta + 1.0)
                score = (32.0 - 1.20 * eta + 0.75 * max(0.0, goal_eta - eta)) if feasible else -10.0 - eta
                if self.previous_assignment.get(hunter.id) == enemy.id:
                    score += 2.4
                row.append(score)
            row.extend([5.0] * len(fast))
            scores.append(row)
        columns = self._hungarian_maximize(scores)
        assignments = {}
        for row, column in enumerate(columns):
            if 0 <= column < len(enemy_slow) and scores[row][column] > 5.0:
                assignments[fast[row].id] = enemy_slow[column].id
        self.fast_target = {drone.id: assignments.get(drone.id) for drone in fast}
        self.previous_assignment = assignments

    def _lead_point(self, hunter, target):
        horizon = _clip(self._intercept_time(hunter, target.position, target.velocity), 0.0, 4.0)
        # Damped acceleration lead adapted from Apex; full constant acceleration
        # overreacts to an evasive SLOW's short-lived jerk-limited turn.
        point = (
            target.position[0] + horizon * target.velocity[0] + 0.28 * horizon * horizon * target.acceleration[0],
            target.position[1] + horizon * target.velocity[1] + 0.28 * horizon * horizon * target.acceleration[1],
        )
        return (_clip(point[0], 0.35, self.width - 0.35), _clip(point[1], 0.35, self.height - 0.35))

    # ------------------------------------------------------------------
    # Low-level jerk-aware steering and non-negotiable terrain survival.

    def _advance_path(self, drone, path):
        while len(path) > 1 and _distance(drone.position, path[0]) < 0.72:
            path.pop(0)
        for index in range(len(path) - 1, 0, -1):
            if self._segment_clear(drone.position, path[index], self.ROUTE_CLEARANCE):
                del path[:index]
                break

    def _lookahead(self, position, path, amount):
        point, remaining = position, amount
        for waypoint in path:
            length = _distance(point, waypoint)
            if length >= remaining and length > EPS:
                scale = remaining / length
                return (point[0] + scale * (waypoint[0] - point[0]), point[1] + scale * (waypoint[1] - point[1]))
            remaining -= length
            point = waypoint
        return path[-1]

    def _nearest_surface(self, position):
        x, y = position
        best = (inf, (0.0, 1.0))
        for obstacle in self.obstacles:
            if isinstance(obstacle, CircleObstacle):
                dx, dy = x - obstacle.center[0], y - obstacle.center[1]
                magnitude = hypot(dx, dy)
                normal = _unit(dx, dy) if magnitude > EPS else (self.direction, 0.0)
                clearance = magnitude - obstacle.radius - DRONE_RADIUS
            else:
                x0, x1 = obstacle.x_min - DRONE_RADIUS, obstacle.x_max + DRONE_RADIUS
                y0, y1 = obstacle.y_min - DRONE_RADIUS, obstacle.y_max + DRONE_RADIUS
                qx, qy = _clip(x, x0, x1), _clip(y, y0, y1)
                dx, dy = x - qx, y - qy
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
        clearance, normal = self._nearest_surface(drone.position)
        inward = max(0.0, -_dot(drone.velocity, normal))
        commitment = inward * 0.20 + inward * inward / (2.0 * spec.max_acceleration) + 0.055
        if clearance < commitment:
            tangent = (-normal[1], normal[0])
            tangential = _dot(command, tangent)
            outward = spec.max_acceleration * _clip((commitment + 0.10 - clearance) / max(0.10, commitment + 0.10), 0.45, 1.0)
            command = _limit(
                normal[0] * outward + tangent[0] * tangential * 0.62,
                normal[1] * outward + tangent[1] * tangential * 0.62,
                spec.max_acceleration,
            )
        return command

    def _simulate_command(self, drone, command, ticks=20):
        spec = self.specs[drone.drone_type]
        position, velocity, acceleration = drone.position, drone.velocity, drone.acceleration
        for _ in range(ticks):
            desired = _limit(command[0], command[1], spec.max_acceleration)
            delta = _limit(desired[0] - acceleration[0], desired[1] - acceleration[1], spec.max_jerk * 0.05)
            acceleration = _limit(acceleration[0] + delta[0], acceleration[1] + delta[1], spec.max_acceleration)
            new_position = (
                position[0] + velocity[0] * 0.05 + 0.5 * acceleration[0] * 0.0025,
                position[1] + velocity[1] * 0.05 + 0.5 * acceleration[1] * 0.0025,
            )
            if new_position[1] < 0.26 or new_position[1] > self.height - 0.26:
                return None
            if not self._segment_clear(position, new_position, self.SAFETY_PADDING):
                return None
            velocity = _limit(velocity[0] + acceleration[0] * 0.05, velocity[1] + acceleration[1] * 0.05, spec.max_speed)
            position = new_position
        return (position, velocity)

    def _safe_command(self, drone, command, target):
        command = self._terrain_guard(drone, command)
        if self._simulate_command(drone, command) is not None:
            return command
        spec = self.specs[drone.drone_type]
        brake = _limit(-3.4 * drone.velocity[0] - 0.7 * drone.acceleration[0], -3.4 * drone.velocity[1] - 0.7 * drone.acceleration[1], spec.max_acceleration)
        candidates = [brake]
        for angle in (pi / 4.0, -pi / 4.0, pi / 2.0, -pi / 2.0):
            candidates.append(_limit(command[0] * cos(angle) - command[1] * sin(angle), command[0] * sin(angle) + command[1] * cos(angle), spec.max_acceleration))
        best = None
        for candidate in candidates:
            future = self._simulate_command(drone, candidate)
            if future is None:
                continue
            score = _distance(future[0], target) + 0.08 * hypot(*future[1])
            if best is None or score < best[0]:
                best = (score, candidate)
        if best is not None:
            return best[1]
        _, normal = self._nearest_surface(drone.position)
        return _limit(normal[0] * spec.max_acceleration - 2.5 * drone.velocity[0], normal[1] * spec.max_acceleration - 2.5 * drone.velocity[1], spec.max_acceleration)

    def _steer(self, drone, path, arrive=False, speed_limit=None):
        spec = self.specs[drone.drone_type]
        self._advance_path(drone, path)
        speed = hypot(*drone.velocity)
        target = self._lookahead(drone.position, path, 0.85 + 0.38 * speed)
        direction = _unit(target[0] - drone.position[0], target[1] - drone.position[1])
        desired_speed = spec.max_speed
        if len(path) > 1:
            first = _unit(path[0][0] - drone.position[0], path[0][1] - drone.position[1])
            second = _unit(path[1][0] - path[0][0], path[1][1] - path[0][1])
            angle = acos(_clip(_dot(first, second), -1.0, 1.0))
            if angle > 0.30:
                desired_speed *= _clip(1.0 - 0.70 * angle / pi, 0.30, 0.82)
        if speed_limit is not None:
            desired_speed = min(desired_speed, speed_limit)
        if arrive:
            remaining = self._path_length(drone.position, path)
            desired_speed = min(desired_speed, sqrt(max(0.0, 2.0 * spec.max_acceleration * max(0.0, remaining - 0.08))))
            if remaining < 0.30:
                desired_speed = min(desired_speed, remaining / 0.24)
        desired_velocity = (direction[0] * desired_speed, direction[1] * desired_speed)
        command = _limit(
            (desired_velocity[0] - (drone.velocity[0] + 0.12 * drone.acceleration[0])) / 0.34,
            (desired_velocity[1] - (drone.velocity[1] + 0.12 * drone.acceleration[1])) / 0.34,
            spec.max_acceleration,
        )
        return self._safe_command(drone, command, target)

    def _avoid_non_target_enemies(self, drone, command, enemies, target_id):
        spec = self.specs[drone.drone_type]
        push_x = push_y = 0.0
        for enemy in enemies:
            if enemy.id == target_id or _distance(drone.position, enemy.position) > 8.0:
                continue
            closest, time = self._closest_approach(drone, enemy, 2.2)
            if closest >= 1.35:
                continue
            own_future = (drone.position[0] + time * drone.velocity[0], drone.position[1] + time * drone.velocity[1])
            enemy_future = (enemy.position[0] + time * enemy.velocity[0], enemy.position[1] + time * enemy.velocity[1])
            away = _unit(own_future[0] - enemy_future[0], own_future[1] - enemy_future[1])
            strength = spec.max_acceleration * _clip((1.35 - closest) / 1.35, 0.0, 0.85)
            push_x += strength * away[0]
            push_y += strength * away[1]
        return _limit(command[0] + push_x, command[1] + push_y, spec.max_acceleration)

    # ------------------------------------------------------------------

    def _set_route(self, drone, destination, clearance=None, force=False):
        old = self.path_destination.get(drone.id)
        if force or old is None or _distance(old, destination) > 0.20 or not self.paths.get(drone.id):
            self.paths[drone.id] = self._route(drone.position, destination, clearance)
            self.path_destination[drone.id] = destination

    def _slow_command(self, slow, enemy_fast, own_slow, state):
        remaining = MATCH_DURATION - state.time
        if self.slow_phase[slow.id] != "FINAL" and self._deadline_push(slow, remaining, state.time):
            self.slow_phase[slow.id] = "FINAL"

        if self.slow_phase[slow.id] == "FINAL":
            destination = self.goal_target[slow.id]
            self._set_route(slow, destination, self.ROUTE_CLEARANCE)
            return self._steer(slow, self.paths[slow.id])

        chain = self.chain[slow.id]
        index = self.chain_index[slow.id]
        refuge_index = chain[index]
        obstacle, _, perches, _ = self.refuges[refuge_index]
        threats = self._threats_for(slow, enemy_fast)

        if self.slow_phase[slow.id] == "TRANSIT":
            destination = perches[self.perch_index[slow.id] % len(perches)]
            self._set_route(slow, destination, self.BAIT_TRACK_PADDING)
            distance = _distance(slow.position, destination)
            if distance < 0.48 and hypot(*slow.velocity) < 0.70:
                self.slow_phase[slow.id] = "HOLD"
            return self._steer(slow, self.paths[slow.id], arrive=True, speed_limit=2.05 if distance < 2.0 else None)

        # HOLD: normally remain at close cover, with only adjacent perimeter
        # steps.  Re-evaluate a next-refuge proof every control instant.
        desired_index = self._bait_perch(slow, refuge_index, threats, state.time)
        if desired_index != self.perch_index[slow.id]:
            self.perch_index[slow.id] = desired_index
        destination = perches[self.perch_index[slow.id] % len(perches)]

        if index + 1 < len(chain):
            next_refuge = chain[index + 1]
            next_perch = self._closest_perch(next_refuge, slow.position)
            next_destination = self.refuges[next_refuge][2][next_perch]
            cache_key = (refuge_index, next_refuge, self.perch_index[slow.id], next_perch)
            transition_path = self.transition_cache.get(cache_key)
            if transition_path is None:
                transition_path = self._route(destination, next_destination, self.BAIT_TRACK_PADDING)
                self.transition_cache[cache_key] = tuple(transition_path)
            transition_path = list(transition_path)
            if self._transition_safe(slow, transition_path, enemy_fast, own_slow):
                self.chain_index[slow.id] += 1
                self.perch_index[slow.id] = next_perch
                self.slow_phase[slow.id] = "TRANSIT"
                self.paths[slow.id] = transition_path
                self.path_destination[slow.id] = next_destination
                return self._steer(slow, self.paths[slow.id], arrive=True)
        elif not enemy_fast:
            self.slow_phase[slow.id] = "FINAL"
            self._set_route(slow, self.goal_target[slow.id], self.ROUTE_CLEARANCE, True)
            return self._steer(slow, self.paths[slow.id])

        # Direct adjacent-perimeter steering is what makes active bait stay
        # close; routing it through ordinary clearance nodes would erase cover.
        path = [destination]
        return self._steer(slow, path, arrive=True, speed_limit=1.15)

    def step(self, state):
        own = [d for d in state.own_drones if d.status is DroneStatus.ACTIVE]
        enemies = [d for d in state.opponent_drones if d.status is DroneStatus.ACTIVE]
        own_fast = [d for d in own if d.drone_type is DroneType.FAST]
        own_slow = [d for d in own if d.drone_type is DroneType.SLOW]
        enemy_fast = [d for d in enemies if d.drone_type is DroneType.FAST]
        enemy_slow = [d for d in enemies if d.drone_type is DroneType.SLOW]
        enemy_by_id = {d.id: d for d in enemies}

        if state.time >= self.last_assignment + 0.35:
            self._assign_fast(own_fast, enemy_slow, MATCH_DURATION - state.time)
            self.last_assignment = state.time

        actions = {}
        for drone in own:
            if drone.drone_type is DroneType.SLOW:
                command = self._slow_command(drone, enemy_fast, own_slow, state)
            else:
                target_id = self.fast_target.get(drone.id)
                target = enemy_by_id.get(target_id)
                if target is not None and target.drone_type is DroneType.SLOW:
                    destination = self._lead_point(drone, target)
                    close_direct = _distance(drone.position, target.position) < 6.0 and self._segment_clear(drone.position, destination, 0.32)
                    changed_target = self.hunter_route_target.get(drone.id) != target.id
                    if close_direct:
                        self.paths[drone.id] = [destination]
                        self.path_destination[drone.id] = destination
                    elif changed_target or state.time >= self.hunter_route_time[drone.id] + 0.35:
                        self._set_route(drone, destination, 0.40, True)
                        self.hunter_route_time[drone.id] = state.time
                        self.hunter_route_target[drone.id] = target.id
                    command = self._steer(drone, self.paths[drone.id])
                    command = self._avoid_non_target_enemies(drone, command, enemies, target.id)
                    command = self._safe_command(drone, command, destination)
                else:
                    destination = self.goal_target[drone.id]
                    self._set_route(drone, destination, self.ROUTE_CLEARANCE)
                    command = self._steer(drone, self.paths[drone.id])
                    command = self._avoid_non_target_enemies(drone, command, enemies, None)
                    command = self._safe_command(drone, command, destination)
            actions[drone.id] = (float(command[0]), float(command[1]))
        return actions
