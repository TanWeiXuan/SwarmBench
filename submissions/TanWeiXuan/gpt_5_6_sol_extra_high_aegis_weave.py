"""Aegis Weave -- an original SwarmBench controller.

Coding model: GPT-5.6 Sol Extra High.

The controller was designed directly from the documented game mechanics.  Its
visibility graph, allocation objective, motion controller, constants, and code
are original.  After the first working version, I inspected only function and
constant outlines (not implementation bodies) in
``submissions/TanWeiXuan/GPT_5_6_Sol_Ultra.py`` and the Claude Sonnet/Opus 5 Max
submissions.  I then incorporated the high-level ideas of a multi-second
closest-approach dodge and forward-simulated terrain guarding; those two ideas
are attributed to those sources, while their implementation here was written
independently from the public physics equations.

Its three main ideas are:

* route all scoring drones on a precomputed obstacle visibility graph;
* use FAST drones for predictive, value-aware interceptions that also form a
  moving screen when enemy hunters approach the scoring formation; and
* validate evasive commands against the terrain before issuing them.
"""

from heapq import heappop, heappush
from math import cos, hypot, inf, pi, sin, sqrt

from swarmbench import (
    BaseSwarmController,
    CircleObstacle,
    DroneStatus,
    DroneType,
    RectangleObstacle,
)


EPSILON = 1.0e-9


def _distance(a, b):
    return hypot(a[0] - b[0], a[1] - b[1])


def _clip_length(x, y, limit):
    length = hypot(x, y)
    if length <= limit or length < EPSILON:
        return (x, y)
    scale = limit / length
    return (x * scale, y * scale)


def _segment_point_distance_sq(a, b, p):
    dx, dy = b[0] - a[0], b[1] - a[1]
    denominator = dx * dx + dy * dy
    if denominator < EPSILON:
        return (p[0] - a[0]) ** 2 + (p[1] - a[1]) ** 2
    t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / denominator
    t = min(1.0, max(0.0, t))
    qx, qy = a[0] + t * dx, a[1] + t * dy
    return (p[0] - qx) ** 2 + (p[1] - qy) ** 2


def _segment_aabb_intersects(a, b, x0, x1, y0, y1):
    enter, leave = 0.0, 1.0
    for origin, delta, low, high in (
        (a[0], b[0] - a[0], x0, x1),
        (a[1], b[1] - a[1], y0, y1),
    ):
        if abs(delta) < EPSILON:
            if origin < low or origin > high:
                return False
            continue
        t0, t1 = (low - origin) / delta, (high - origin) / delta
        if t0 > t1:
            t0, t1 = t1, t0
        enter, leave = max(enter, t0), min(leave, t1)
        if enter > leave:
            return False
    return True


class SwarmController(BaseSwarmController):
    """Obstacle-routed scoring formation with escorts and active interception."""

    def initialize(self, game_info):
        self.team = game_info.team
        self.goal = game_info.target_goal
        self.own_goal = game_info.own_goal
        self.width = game_info.arena_width
        self.height = game_info.arena_height
        self.obstacles = game_info.obstacles
        self.specs = dict(game_info.drone_specs)
        self.forward = 1.0 if self.goal.center[0] > self.own_goal.center[0] else -1.0
        self.route_clearance = 0.46
        self.node_padding = 0.82
        self.previous_assignments = {}
        self.previous_nav_node = {}

        own_fast = sorted(
            (d for d in game_info.own_initial_drones if d.drone_type is DroneType.FAST),
            key=lambda d: (d.position[1], d.id),
        )
        own_slow = sorted(
            (d for d in game_info.own_initial_drones if d.drone_type is DroneType.SLOW),
            key=lambda d: (d.position[1], d.id),
        )

        # The job allocator begins with every FAST drone as a hunter.  In this
        # game, attacking the enemy SLOW formation naturally makes the two FAST
        # lines screen their own SLOW formations as they cross.  The guard
        # machinery below remains available for late reactive screening.
        guard_ranks = ()
        self.guard_ids = {own_fast[index].id for index in guard_ranks}
        self.guard_rank = {
            own_fast[index].id: rank for rank, index in enumerate(guard_ranks)
        }
        self.initial_slow_order = tuple(d.id for d in own_slow)

        usable_height = max(1.0, self.goal.y_max - self.goal.y_min - 1.6)
        self.goal_points = []
        for rank in range(10):
            y = self.goal.y_min + 0.8 + usable_height * (rank + 0.5) / 10.0
            self.goal_points.append((self.goal.center[0], y))

        slow_rank = {d.id: rank for rank, d in enumerate(own_slow)}
        fast_rank = {d.id: rank for rank, d in enumerate(own_fast)}
        self.lane_by_id = {}
        for drone in game_info.own_initial_drones:
            if drone.drone_type is DroneType.SLOW:
                self.lane_by_id[drone.id] = slow_rank[drone.id]
            else:
                # Interleave FAST goal lanes with the SLOW formation.
                self.lane_by_id[drone.id] = (3 * fast_rank[drone.id] + 1) % 10

        self._build_navigation_graph()

    # ------------------------------------------------------------------
    # Static obstacle routing

    def _point_safe(self, point, clearance=None):
        clearance = self.route_clearance if clearance is None else clearance
        if not (0.35 <= point[0] <= self.width - 0.35 and 0.35 <= point[1] <= self.height - 0.35):
            return False
        for obstacle in self.obstacles:
            if isinstance(obstacle, CircleObstacle):
                if _distance(point, obstacle.center) <= obstacle.radius + clearance:
                    return False
            elif (
                obstacle.x_min - clearance <= point[0] <= obstacle.x_max + clearance
                and obstacle.y_min - clearance <= point[1] <= obstacle.y_max + clearance
            ):
                return False
        return True

    def _clear_segment(self, a, b, clearance=None):
        clearance = self.route_clearance if clearance is None else clearance
        for obstacle in self.obstacles:
            if isinstance(obstacle, CircleObstacle):
                radius = obstacle.radius + clearance
                if _segment_point_distance_sq(a, b, obstacle.center) <= radius * radius:
                    return False
            elif _segment_aabb_intersects(
                a,
                b,
                obstacle.x_min - clearance,
                obstacle.x_max + clearance,
                obstacle.y_min - clearance,
                obstacle.y_max + clearance,
            ):
                return False
        return True

    def _build_navigation_graph(self):
        nodes = list(self.goal_points)
        pad = self.node_padding
        for obstacle in self.obstacles:
            candidates = []
            if isinstance(obstacle, CircleObstacle):
                radius = obstacle.radius + pad
                candidates = [
                    (
                        obstacle.center[0] + radius * cos(2.0 * pi * k / 12.0),
                        obstacle.center[1] + radius * sin(2.0 * pi * k / 12.0),
                    )
                    for k in range(12)
                ]
            else:
                candidates = [
                    (obstacle.x_min - pad, obstacle.y_min - pad),
                    (obstacle.x_min - pad, obstacle.y_max + pad),
                    (obstacle.x_max + pad, obstacle.y_min - pad),
                    (obstacle.x_max + pad, obstacle.y_max + pad),
                ]
            nodes.extend(point for point in candidates if self._point_safe(point))

        self.nodes = tuple(nodes)
        count = len(nodes)
        adjacency = [[] for _ in range(count)]
        for left in range(count):
            for right in range(left + 1, count):
                if self._clear_segment(nodes[left], nodes[right]):
                    length = _distance(nodes[left], nodes[right])
                    adjacency[left].append((right, length))
                    adjacency[right].append((left, length))
        self.adjacency = tuple(tuple(row) for row in adjacency)
        self.goal_fields = tuple(self._dijkstra(((lane, 0.0),)) for lane in range(10))

    def _dijkstra(self, sources):
        distances = [inf] * len(self.nodes)
        following = [-1] * len(self.nodes)
        heap = []
        for node, distance in sources:
            if distance < distances[node]:
                distances[node] = distance
                following[node] = -1
                heappush(heap, (distance, node))
        while heap:
            distance, node = heappop(heap)
            if distance != distances[node]:
                continue
            for neighbor, edge in self.adjacency[node]:
                candidate = distance + edge
                if candidate + 1.0e-10 < distances[neighbor]:
                    distances[neighbor] = candidate
                    following[neighbor] = node
                    heappush(heap, (candidate, neighbor))
        return (tuple(distances), tuple(following))

    def _best_visible_entry(self, position, distances, drone_id=None):
        best_node, best_cost = -1, inf
        old_node = self.previous_nav_node.get(drone_id, -1)
        old_cost = inf
        for node, point in enumerate(self.nodes):
            if distances[node] == inf or not self._clear_segment(position, point):
                continue
            cost = _distance(position, point) + distances[node]
            if node == old_node:
                old_cost = cost
            if cost < best_cost:
                best_node, best_cost = node, cost
        # A small hysteresis band prevents equal-length routes around a circle
        # from causing left/right command chatter.
        if old_node >= 0 and old_cost <= best_cost + 0.28:
            best_node = old_node
        if drone_id is not None and best_node >= 0:
            self.previous_nav_node[drone_id] = best_node
        return best_node

    def _route_to_goal(self, drone):
        lane = self.lane_by_id.get(drone.id, drone.id % 10)
        target = self.goal_points[lane]
        if self._clear_segment(drone.position, target):
            self.previous_nav_node.pop(drone.id, None)
            return target, True
        distances, _ = self.goal_fields[lane]
        node = self._best_visible_entry(drone.position, distances, drone.id)
        if node >= 0:
            return self.nodes[node], node < 10
        # This should only be reached in a clearance pinch.  Reactive obstacle
        # steering below still makes this safer than withholding the command.
        return target, True

    def _route_to_dynamic_target(self, position, target):
        if self._clear_segment(position, target):
            return target, True
        visible_sources = [
            (node, _distance(point, target))
            for node, point in enumerate(self.nodes)
            if self._clear_segment(point, target)
        ]
        if not visible_sources:
            return target, True
        distances, _ = self._dijkstra(visible_sources)
        node = self._best_visible_entry(position, distances)
        if node >= 0:
            return self.nodes[node], False
        return target, True

    # ------------------------------------------------------------------
    # Interception and role allocation

    def _lead_time(self, pursuer, target):
        rx = target.position[0] - pursuer.position[0]
        ry = target.position[1] - pursuer.position[1]
        vx, vy = target.velocity
        speed = self.specs[pursuer.drone_type].max_speed * 0.94
        a = vx * vx + vy * vy - speed * speed
        b = 2.0 * (rx * vx + ry * vy)
        c = rx * rx + ry * ry
        if c < 0.75 * 0.75:
            return 0.0
        if abs(a) < EPSILON:
            if b < -EPSILON:
                return max(0.0, -c / b)
            return sqrt(c) / max(speed, EPSILON)
        discriminant = b * b - 4.0 * a * c
        if discriminant >= 0.0:
            root = sqrt(discriminant)
            roots = [value for value in ((-b - root) / (2.0 * a), (-b + root) / (2.0 * a)) if value >= 0.0]
            if roots:
                return min(roots)
        return sqrt(c) / max(speed, EPSILON)

    def _remaining_enemy_time(self, enemy):
        entry_x = self.own_goal.x_max if self.forward > 0.0 else self.own_goal.x_min
        distance = abs(enemy.position[0] - entry_x)
        return distance / max(0.4, self.specs[enemy.drone_type].max_speed)

    def _closest_slow_threat(self, enemy, own_slow):
        if not own_slow:
            return (inf, inf)
        best_now, best_forecast = inf, inf
        for slow in own_slow:
            rx = enemy.position[0] - slow.position[0]
            ry = enemy.position[1] - slow.position[1]
            vx = enemy.velocity[0] - slow.velocity[0]
            vy = enemy.velocity[1] - slow.velocity[1]
            now = hypot(rx, ry)
            vv = vx * vx + vy * vy
            time = min(2.5, max(0.0, -(rx * vx + ry * vy) / vv)) if vv > EPSILON else 0.0
            forecast = hypot(rx + vx * time, ry + vy * time)
            best_now = min(best_now, now)
            best_forecast = min(best_forecast, forecast)
        return (best_now, best_forecast)

    def _hungarian_maximize(self, scores):
        """Return an optimal row-to-column assignment for a small dense matrix."""
        rows = len(scores)
        columns = len(scores[0]) if rows else 0
        if not rows or not columns:
            return []
        # The classic shortest-augmenting-path form minimizes costs.  There are
        # always at least as many columns as FAST rows because dummy goal tasks
        # are appended by the caller.
        potential_row = [0.0] * (rows + 1)
        potential_col = [0.0] * (columns + 1)
        matched_row = [0] * (columns + 1)
        way = [0] * (columns + 1)
        for row in range(1, rows + 1):
            matched_row[0] = row
            minimum = [inf] * (columns + 1)
            used = [False] * (columns + 1)
            column = 0
            while True:
                used[column] = True
                active_row = matched_row[column]
                delta, next_column = inf, 0
                for candidate in range(1, columns + 1):
                    if used[candidate]:
                        continue
                    cost = -scores[active_row - 1][candidate - 1]
                    reduced = cost - potential_row[active_row] - potential_col[candidate]
                    if reduced < minimum[candidate]:
                        minimum[candidate] = reduced
                        way[candidate] = column
                    if minimum[candidate] < delta:
                        delta, next_column = minimum[candidate], candidate
                for candidate in range(columns + 1):
                    if used[candidate]:
                        potential_row[matched_row[candidate]] += delta
                        potential_col[candidate] -= delta
                    else:
                        minimum[candidate] -= delta
                column = next_column
                if matched_row[column] == 0:
                    break
            while True:
                previous = way[column]
                matched_row[column] = matched_row[previous]
                column = previous
                if column == 0:
                    break
        result = [-1] * rows
        for column in range(1, columns + 1):
            if matched_row[column]:
                result[matched_row[column] - 1] = column - 1
        return result

    def _assign_hunters(self, hunters, enemies, own_slow, reserved_enemy_ids, urgent):
        available = [enemy for enemy in enemies if enemy.id not in reserved_enemy_ids]
        if not hunters or not available:
            self.previous_assignments = {}
            return {}

        scores = []
        for hunter in hunters:
            row = []
            for enemy in available:
                intercept_time = self._lead_time(hunter, enemy)
                remaining = self._remaining_enemy_time(enemy)
                now, forecast = self._closest_slow_threat(enemy, own_slow)
                if enemy.drone_type is DroneType.SLOW:
                    score = 31.0 - 1.05 * intercept_time
                    if intercept_time > remaining + 1.0:
                        score -= 18.0 + 2.0 * (intercept_time - remaining)
                else:
                    threat = max(0.0, 14.0 - min(now, forecast))
                    score = -5.0 + 3.0 * threat - 0.8 * intercept_time
                    if remaining < 7.0:
                        score += 8.0
                if urgent and enemy.drone_type is DroneType.FAST:
                    score += 5.0
                if self.previous_assignments.get(hunter.id) == enemy.id:
                    score += 2.6
                row.append(score)
            # One dummy goal task per hunter keeps the rectangular assignment
            # feasible and lets a poor/unreachable interception be declined.
            row.extend([4.0] * len(hunters))
            scores.append(row)

        columns = self._hungarian_maximize(scores)
        assignments = {}
        for row, column in enumerate(columns):
            if 0 <= column < len(available) and scores[row][column] > 4.0:
                assignments[hunters[row].id] = available[column].id
        self.previous_assignments = assignments
        return assignments

    def _guard_plan(self, guards, own_slow, enemies):
        if not guards or not own_slow:
            return {}, {}, set()
        ordered_slow = sorted(own_slow, key=lambda d: (d.position[1], d.id))
        guard_slots = {}
        candidates = []
        for guard in guards:
            rank = self.guard_rank.get(guard.id, 0)
            index = round((rank + 0.5) * (len(ordered_slow) - 1) / max(1, len(guards)))
            protected = ordered_slow[min(len(ordered_slow) - 1, index)]
            route_point, _ = self._route_to_goal(protected)
            dx = route_point[0] - protected.position[0]
            dy = route_point[1] - protected.position[1]
            length = hypot(dx, dy)
            if length < EPSILON:
                dx, dy, length = self.forward, 0.0, 1.0
            ux, uy = dx / length, dy / length
            offset = -0.45 if rank % 2 == 0 else 0.45
            slot = (
                protected.position[0] + 2.7 * ux - offset * uy,
                protected.position[1] + 2.7 * uy + offset * ux,
            )
            slot = (min(self.width - 0.5, max(0.5, slot[0])), min(self.height - 0.5, max(0.5, slot[1])))
            guard_slots[guard.id] = (protected, slot)
            for enemy in enemies:
                if enemy.drone_type is not DroneType.FAST:
                    continue
                distance = _distance(enemy.position, protected.position)
                forecast = _distance(
                    (enemy.position[0] + 1.1 * enemy.velocity[0], enemy.position[1] + 1.1 * enemy.velocity[1]),
                    (protected.position[0] + 1.1 * protected.velocity[0], protected.position[1] + 1.1 * protected.velocity[1]),
                )
                danger = min(distance, forecast)
                if danger < 13.5:
                    candidates.append((danger + 0.18 * _distance(guard.position, enemy.position), guard.id, enemy.id))

        intercepts = {}
        used_guards, used_enemies = set(), set()
        for _, guard_id, enemy_id in sorted(candidates):
            if guard_id not in used_guards and enemy_id not in used_enemies:
                intercepts[guard_id] = enemy_id
                used_guards.add(guard_id)
                used_enemies.add(enemy_id)
        return guard_slots, intercepts, used_enemies

    # ------------------------------------------------------------------
    # Motion control and collision avoidance

    def _lead_point(self, pursuer, target):
        lead = min(6.0, max(0.0, self._lead_time(pursuer, target)))
        # Full constant-velocity lead overreacts to evasive lateral motion; a
        # damped lead remains aggressive while being much less oscillatory.
        lead *= 0.82
        point = (
            target.position[0] + lead * target.velocity[0],
            target.position[1] + lead * target.velocity[1],
        )
        point = (min(self.width - 0.4, max(0.4, point[0])), min(self.height - 0.4, max(0.4, point[1])))
        if not self._point_safe(point, 0.27):
            return target.position
        return point

    def _avoid_enemies(self, drone, desired_velocity, enemies, target_id=None):
        vx, vy = desired_velocity
        speed = self.specs[drone.drone_type].max_speed
        threat_range = 13.0 if drone.drone_type is DroneType.SLOW else 10.0
        horizon = 3.1 if drone.drone_type is DroneType.SLOW else 2.7
        safety = 2.5 if drone.drone_type is DroneType.SLOW else 2.0
        for enemy in enemies:
            if enemy.id == target_id:
                continue
            rx = enemy.position[0] - drone.position[0]
            ry = enemy.position[1] - drone.position[1]
            distance = hypot(rx, ry)
            if distance > threat_range:
                continue
            dvx = enemy.velocity[0] - drone.velocity[0]
            dvy = enemy.velocity[1] - drone.velocity[1]
            vv = dvx * dvx + dvy * dvy
            time = min(horizon, max(0.0, -(rx * dvx + ry * dvy) / vv)) if vv > EPSILON else 0.0
            cx, cy = rx + time * dvx, ry + time * dvy
            closest = hypot(cx, cy)
            if closest >= safety and distance >= 2.8:
                continue
            if distance < EPSILON:
                rx, ry, distance = 0.0, 1.0, 1.0
            away_x, away_y = -rx / distance, -ry / distance
            # Mirrored teams choose mirrored global lateral directions.  This
            # creates separation even when both controllers use reciprocal
            # avoidance, unlike a shared "toward arena center" convention.
            side_x, side_y = -ry / distance, rx / distance
            if side_y * self.forward < 0.0:
                side_x, side_y = -side_x, -side_y
            if abs(side_y) < 0.08 and (drone.id + enemy.id) % 2:
                side_x, side_y = -side_x, -side_y

            # Prefer the reciprocal side unless terrain or the arena boundary
            # offers materially more room on the other side.
            probe = 4.5 if drone.drone_type is DroneType.SLOW else 5.5
            positive = (
                drone.position[0] + probe * side_x,
                drone.position[1] + probe * side_y,
            )
            negative = (
                drone.position[0] - probe * side_x,
                drone.position[1] - probe * side_y,
            )
            positive_safe = self._point_safe(positive, 0.30) and self._clear_segment(drone.position, positive, 0.30)
            negative_safe = self._point_safe(negative, 0.30) and self._clear_segment(drone.position, negative, 0.30)
            if negative_safe and not positive_safe:
                side_x, side_y = -side_x, -side_y

            urgency = min(1.18, max(0.0, (safety - closest) / safety) + max(0.0, (2.8 - distance) / 2.8))
            lateral_share = 0.88 if drone.drone_type is DroneType.SLOW else 0.84
            vx += speed * urgency * (lateral_share * side_x + (1.0 - lateral_share) * away_x)
            vy += speed * urgency * (lateral_share * side_y + (1.0 - lateral_share) * away_y)
        return _clip_length(vx, vy, speed)

    def _obstacle_repulsion(self, drone):
        ax = ay = 0.0
        influence = 2.0
        for obstacle in self.obstacles:
            if isinstance(obstacle, CircleObstacle):
                dx = drone.position[0] - obstacle.center[0]
                dy = drone.position[1] - obstacle.center[1]
                center_distance = hypot(dx, dy)
                surface = center_distance - obstacle.radius - 0.25
                if surface < influence and center_distance > EPSILON:
                    strength = max(0.0, (influence - surface) / influence)
                    ax += dx / center_distance * strength
                    ay += dy / center_distance * strength
            else:
                qx = min(obstacle.x_max, max(obstacle.x_min, drone.position[0]))
                qy = min(obstacle.y_max, max(obstacle.y_min, drone.position[1]))
                dx, dy = drone.position[0] - qx, drone.position[1] - qy
                center_distance = hypot(dx, dy)
                surface = center_distance - 0.25
                if surface < influence and center_distance > EPSILON:
                    strength = max(0.0, (influence - surface) / influence)
                    ax += dx / center_distance * strength
                    ay += dy / center_distance * strength
        if drone.position[1] < 1.0:
            ay += 1.0 - drone.position[1]
        elif drone.position[1] > self.height - 1.0:
            ay -= drone.position[1] - (self.height - 1.0)
        return (ax, ay)

    def _terrain_safe(self, drone, command):
        """Conservatively roll a command forward through jerk-limited physics."""
        spec = self.specs[drone.drone_type]
        command = _clip_length(command[0], command[1], spec.max_acceleration)
        px, py = drone.position
        vx, vy = drone.velocity
        ax, ay = drone.acceleration
        dt = 0.10
        steps = 11 if drone.drone_type is DroneType.SLOW else 9
        for _ in range(steps):
            dax, day = command[0] - ax, command[1] - ay
            dax, day = _clip_length(dax, day, spec.max_jerk * dt)
            ax, ay = _clip_length(ax + dax, ay + day, spec.max_acceleration)
            next_position = (
                px + vx * dt + 0.5 * ax * dt * dt,
                py + vy * dt + 0.5 * ay * dt * dt,
            )
            if next_position[1] < 0.30 or next_position[1] > self.height - 0.30:
                return False
            if not self._clear_segment((px, py), next_position, 0.265):
                return False
            vx, vy = _clip_length(vx + ax * dt, vy + ay * dt, spec.max_speed)
            px, py = next_position
        return True

    def _velocity_command(self, drone, desired_velocity, obstacle_push):
        gain = 2.75 if drone.drone_type is DroneType.FAST else 2.45
        spec = self.specs[drone.drone_type]
        ax = gain * (desired_velocity[0] - drone.velocity[0]) - 0.32 * drone.acceleration[0]
        ay = gain * (desired_velocity[1] - drone.velocity[1]) - 0.32 * drone.acceleration[1]
        ax += spec.max_acceleration * 1.35 * obstacle_push[0]
        ay += spec.max_acceleration * 1.35 * obstacle_push[1]
        return _clip_length(ax, ay, spec.max_acceleration)

    def _steer(self, drone, waypoint, final, enemies=(), target_id=None, velocity_hint=None):
        spec = self.specs[drone.drone_type]
        dx, dy = waypoint[0] - drone.position[0], waypoint[1] - drone.position[1]
        distance = hypot(dx, dy)
        if velocity_hint is not None and final:
            desired_velocity = velocity_hint
        elif distance < EPSILON:
            desired_velocity = (0.0, 0.0)
        else:
            if final:
                desired_speed = min(spec.max_speed, sqrt(max(0.0, 2.0 * spec.max_acceleration * distance)))
            else:
                # Slow near a graph corner enough for jerk-limited lateral
                # acceleration to bend the trajectory inside its clearance.
                corner_speed = 0.8 + sqrt(max(0.0, 1.35 * spec.max_acceleration * distance))
                desired_speed = min(spec.max_speed, corner_speed)
            desired_velocity = (dx / distance * desired_speed, dy / distance * desired_speed)

        nominal_velocity = desired_velocity
        obstacle_push = self._obstacle_repulsion(drone)
        desired_velocity = self._avoid_enemies(drone, desired_velocity, enemies, target_id)
        evasive_command = self._velocity_command(drone, desired_velocity, obstacle_push)
        if self._terrain_safe(drone, evasive_command):
            return evasive_command

        # Enemy avoidance is optional; terrain survival is not.  First fall
        # back to the visibility-graph velocity, then to an emergency brake.
        nominal_command = self._velocity_command(drone, nominal_velocity, obstacle_push)
        if self._terrain_safe(drone, nominal_command):
            return nominal_command
        brake = _clip_length(
            -3.8 * drone.velocity[0] - 0.5 * drone.acceleration[0] + spec.max_acceleration * 1.8 * obstacle_push[0],
            -3.8 * drone.velocity[1] - 0.5 * drone.acceleration[1] + spec.max_acceleration * 1.8 * obstacle_push[1],
            spec.max_acceleration,
        )
        return brake

    def _escort_command(self, guard, protected, slot, enemies):
        route_point, final = self._route_to_dynamic_target(guard.position, slot)
        if final:
            error_x, error_y = slot[0] - guard.position[0], slot[1] - guard.position[1]
            desired = (
                protected.velocity[0] + 1.45 * error_x,
                protected.velocity[1] + 1.45 * error_y,
            )
            desired = _clip_length(desired[0], desired[1], self.specs[DroneType.FAST].max_speed)
            return self._steer(guard, slot, True, enemies, velocity_hint=desired)
        return self._steer(guard, route_point, False, enemies)

    # ------------------------------------------------------------------

    def step(self, state):
        own_active = [drone for drone in state.own_drones if drone.status is DroneStatus.ACTIVE]
        enemies = [drone for drone in state.opponent_drones if drone.status is DroneStatus.ACTIVE]
        enemy_by_id = {drone.id: drone for drone in enemies}
        own_slow = [drone for drone in own_active if drone.drone_type is DroneType.SLOW]
        own_fast = [drone for drone in own_active if drone.drone_type is DroneType.FAST]

        # Guards are released once the scoring core is gone, or late when no
        # enemy FAST drone is remotely close enough to threaten it.
        nearest_fast_threat = min(
            (_distance(enemy.position, slow.position) for enemy in enemies if enemy.drone_type is DroneType.FAST for slow in own_slow),
            default=inf,
        )
        release_guards = not own_slow or (state.time > 48.0 and nearest_fast_threat > 18.0)
        guards = [] if release_guards else [drone for drone in own_fast if drone.id in self.guard_ids]
        hunters = [drone for drone in own_fast if drone not in guards]

        guard_slots, guard_intercepts, reserved = self._guard_plan(guards, own_slow, enemies)
        urgent = state.opponent_score > state.own_score or state.time > 62.0
        assignments = self._assign_hunters(hunters, enemies, own_slow, reserved, urgent)

        actions = {}
        for drone in own_active:
            target_id = guard_intercepts.get(drone.id, assignments.get(drone.id))
            if target_id is not None and target_id in enemy_by_id:
                target = enemy_by_id[target_id]
                lead_point = self._lead_point(drone, target)
                waypoint, final = self._route_to_dynamic_target(drone.position, lead_point)
                actions[drone.id] = self._steer(drone, waypoint, final, enemies, target_id)
            elif drone.id in guard_slots:
                protected, slot = guard_slots[drone.id]
                actions[drone.id] = self._escort_command(drone, protected, slot, enemies)
            else:
                waypoint, final = self._route_to_goal(drone)
                actions[drone.id] = self._steer(drone, waypoint, final, enemies)
        return actions
