"""Gemini 3.5 Flash High community controller.

Authorship: this file, including its strategy and implementation, was
entirely coded by Gemini 3.5 Flash High without human guidance. No human-authored
code, strategy choices, or iterative guidance were used.
"""

from __future__ import annotations

import heapq
from math import atan2, cos, hypot, pi, sin, sqrt
from scipy.optimize import linear_sum_assignment

from swarmbench import BaseSwarmController, CircleObstacle, DroneStatus, DroneType, Team


class SwarmController(BaseSwarmController):
    """Dynamic bipartite matched escorts and hunters with shortest-path visibility graph routing."""

    def initialize(self, game_info):
        self.team = game_info.team
        self.goal = game_info.target_goal
        self.own_goal = game_info.own_goal
        self.obstacles = tuple(game_info.obstacles)
        self.specs = dict(game_info.drone_specs)
        self.width = game_info.arena_width
        self.height = game_info.arena_height
        self.direction = 1.0 if self.goal.center[0] > self.own_goal.center[0] else -1.0
        self.tick = 0
        self.hunter_targets = {}
        
        # Lane assignments to avoid congestion (for both SLOW and FAST own drones)
        initial = tuple(game_info.own_initial_drones)
        ordered = sorted(initial, key=lambda drone: (drone.position[1], drone.id))
        low = self.goal.y_min + 0.65
        high = self.goal.y_max - 0.65
        span = max(1.4, high - low)
        self.goal_lanes = {
            drone.id: low + span * (rank + 0.5) / max(1, len(ordered))
            for rank, drone in enumerate(ordered)
        }

        # Build stable escort assignments at start by pairing vertically sorted drones
        fast_init = sorted(
            [d for d in initial if d.drone_type is DroneType.FAST],
            key=lambda d: (d.position[1], d.id),
        )
        slow_init = sorted(
            [d for d in initial if d.drone_type is DroneType.SLOW],
            key=lambda d: (d.position[1], d.id),
        )
        
        escort_count = min(len(fast_init), len(slow_init), max(4, (len(fast_init) + 1) // 2))
        if escort_count <= 1:
            indices = range(escort_count)
        else:
            indices = sorted(
                {
                    round(index * (len(fast_init) - 1) / (escort_count - 1))
                    for index in range(escort_count)
                }
            )
        self.escort_for = {
            fast_init[idx].id: slow_init[idx].id
            for idx in indices
            if idx < len(fast_init) and idx < len(slow_init)
        }

    @staticmethod
    def _clamp(value, low, high):
        return min(high, max(low, value))

    @staticmethod
    def _distance(left, right):
        return hypot(left[0] - right[0], left[1] - right[1])

    @staticmethod
    def _segment_distance(start, end, point):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length_sq = dx * dx + dy * dy
        if length_sq < 1e-12:
            return hypot(point[0] - start[0], point[1] - start[1]), 0.0
        fraction = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_sq
        fraction = max(0.0, min(1.0, fraction))
        closest = (start[0] + fraction * dx, start[1] + fraction * dy)
        return hypot(point[0] - closest[0], point[1] - closest[1]), fraction

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
            first = (low - origin) / delta
            second = (high - origin) / delta
            if first > second:
                first, second = second, first
            enter = max(enter, first)
            leave = min(leave, second)
            if enter > leave:
                return None
        if leave < 0.0 or enter > 1.0:
            return None
        return max(0.0, enter)

    def _line_clear(self, start, end, margin):
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

    def _escape_target(self, start, finish, drone_id):
        heading = atan2(finish[1] - start[1], finish[0] - start[0])
        angles = [heading, heading - pi / 4, heading + pi / 4, heading - pi / 2, heading + pi / 2]
        lane = self.goal_lanes.get(drone_id, self.goal.center[1])
        options = []
        for step_dist in (0.8, 1.5, 2.3):
            for angle in angles:
                candidate = (
                    self._clamp(start[0] + step_dist * cos(angle), 0.35, self.width - 0.35),
                    self._clamp(start[1] + step_dist * sin(angle), 0.35, self.height - 0.35),
                )
                if not self._line_clear(start, candidate, 0.25):
                    continue
                cost = self._distance(candidate, finish)
                if not self._line_clear(candidate, finish, 0.55):
                    cost += 1.2
                cost += 0.02 * abs(candidate[1] - lane)
                options.append((cost, candidate))
        return min(options)[1] if options else finish

    def _route_target(self, start, finish, drone_id):
        # 1. First check if direct line is clear. We do this to bypass graph search completely in standard cases.
        if self._line_clear(start, finish, 0.70):
            return finish

        # 2. Direct path blocked. Build Waypoints around the obstacles.
        waypoints = [start, finish]
        clearance = 0.95

        for obstacle in self.obstacles:
            if isinstance(obstacle, CircleObstacle):
                # Generate 6 points around the circle
                cx, cy = obstacle.center
                r = obstacle.radius + clearance
                for i in range(6):
                    angle = i * pi / 3.0
                    waypoints.append((cx + r * cos(angle), cy + r * sin(angle)))
            else:
                # Generate 4 corners
                for wx, wy in (
                    (obstacle.x_min - clearance, obstacle.y_min - clearance),
                    (obstacle.x_min - clearance, obstacle.y_max + clearance),
                    (obstacle.x_max + clearance, obstacle.y_min - clearance),
                    (obstacle.x_max + clearance, obstacle.y_max + clearance),
                ):
                    waypoints.append((wx, wy))

        # Filter waypoints. Must be within arena and not blocked by any obstacle.
        valid_waypoints = []
        for pt in waypoints:
            if 0.45 <= pt[0] <= self.width - 0.45 and 0.45 <= pt[1] <= self.height - 0.45:
                # Check point spacing from blockers
                blocked = False
                for obs in self.obstacles:
                    if isinstance(obs, CircleObstacle):
                        if hypot(pt[0] - obs.center[0], pt[1] - obs.center[1]) <= obs.radius + 0.45:
                            blocked = True
                            break
                    else:
                        if (obs.x_min - 0.45 <= pt[0] <= obs.x_max + 0.45 and
                            obs.y_min - 0.45 <= pt[1] <= obs.y_max + 0.45):
                            blocked = True
                            break
                if not blocked:
                    valid_waypoints.append(pt)

        # Ensure start and finish are cataloged
        if start not in valid_waypoints:
            valid_waypoints.insert(0, start)
        if finish not in valid_waypoints:
            valid_waypoints.append(finish)

        n = len(valid_waypoints)
        try:
            start_idx = valid_waypoints.index(start)
            finish_idx = valid_waypoints.index(finish)
        except ValueError:
            return self._escape_target(start, finish, drone_id)

        # Dijkstra on visibility graph
        pq = [(0.0, start_idx, -1)]  # (cost, current, parent)
        dist = {start_idx: 0.0}
        parent = {}

        while pq:
            d, u, p = heapq.heappop(pq)
            if d > dist.get(u, float('inf')):
                continue
            parent[u] = p
            if u == finish_idx:
                break
            for v in range(n):
                if v == u:
                    continue
                cost_uv = self._distance(valid_waypoints[u], valid_waypoints[v])
                # Bias to encourage staying in designated lanes
                lane = self.goal_lanes.get(drone_id, self.goal.center[1])
                lane_bias = 0.015 * abs(valid_waypoints[v][1] - lane)

                new_dist = d + cost_uv + lane_bias
                if new_dist < dist.get(v, float('inf')):
                    # Use lower edge safety margin during actual travel
                    if self._line_clear(valid_waypoints[u], valid_waypoints[v], 0.38):
                        dist[v] = new_dist
                        heapq.heappush(pq, (new_dist, v, u))

        if finish_idx not in parent:
            return self._escape_target(start, finish, drone_id)

        # Trace parent list to extract first step waypoint
        curr = finish_idx
        path = []
        while curr != start_idx:
            path.append(valid_waypoints[curr])
            curr = parent[curr]
        path.reverse()
        return path[0] if path else finish

    def _obstacle_push(self, position, maximum):
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
            if surface >= 2.5:
                continue
            if distance < 1e-8:
                dx = 0.0
                dy = 1.0 if position[1] <= self.height / 2 else -1.0
                distance = 1.0
            strength = 0.7 * maximum * max(0.0, 2.5 - surface) / 2.5
            push_x += dx / distance * strength
            push_y += dy / distance * strength
        return push_x, push_y

    def _steer(self, drone, finish, pace=1.0):
        spec = self.specs[drone.drone_type]
        target = self._route_target(drone.position, finish, drone.id)
        dx = target[0] - drone.position[0]
        dy = target[1] - drone.position[1]
        distance = hypot(dx, dy)
        if distance < 1e-8:
            desired_x = desired_y = 0.0
        else:
            desired_speed = min(
                spec.max_speed * pace,
                sqrt(2.0 * spec.max_acceleration * distance),
            )
            desired_x = desired_speed * dx / distance
            desired_y = desired_speed * dy / distance

        acceleration_x = 2.65 * (desired_x - drone.velocity[0]) - 0.12 * drone.acceleration[0]
        acceleration_y = 2.65 * (desired_y - drone.velocity[1]) - 0.12 * drone.acceleration[1]
        push_x, push_y = self._obstacle_push(drone.position, spec.max_acceleration)
        acceleration_x += push_x
        acceleration_y += push_y
        
        # Bound safety padding with arena edge boundaries
        if drone.position[1] < 0.7:
            acceleration_y += spec.max_acceleration
        elif drone.position[1] > self.height - 0.7:
            acceleration_y -= spec.max_acceleration

        magnitude = hypot(acceleration_x, acceleration_y)
        if magnitude > spec.max_acceleration:
            scale = spec.max_acceleration / magnitude
            acceleration_x *= scale
            acceleration_y *= scale
        return acceleration_x, acceleration_y

    def _goal_target(self, drone):
        lane = self.goal_lanes.get(drone.id, self.goal.center[1])
        return (
            self.goal.center[0],
            self._clamp(lane, self.goal.y_min + 0.4, self.goal.y_max - 0.4),
        )

    def _intercept_point(self, hunter, enemy):
        speed = self.specs[hunter.drone_type].max_speed
        relative_x = enemy.position[0] - hunter.position[0]
        relative_y = enemy.position[1] - hunter.position[1]
        velocity_x, velocity_y = enemy.velocity
        quadratic = velocity_x * velocity_x + velocity_y * velocity_y - speed * speed
        linear = 2.0 * (relative_x * velocity_x + relative_y * velocity_y)
        constant = relative_x * relative_x + relative_y * relative_y
        
        time = None
        if abs(quadratic) < 1e-9:
            if linear < -1e-9:
                time = -constant / linear
        else:
            discriminant = linear * linear - 4.0 * quadratic * constant
            if discriminant >= 0.0:
                root = sqrt(discriminant)
                candidates = [
                    candidate
                    for candidate in (
                        (-linear - root) / (2.0 * quadratic),
                        (-linear + root) / (2.0 * quadratic),
                    )
                    if candidate >= 0.0
                ]
                if candidates:
                    time = min(candidates)
        if time is None:
            time = hypot(relative_x, relative_y) / max(0.1, speed)
        
        time = self._clamp(time + 0.18, 0.0, 4.5)
        return (
            self._clamp(
                enemy.position[0] + enemy.velocity[0] * time + 0.25 * enemy.acceleration[0] * time * time,
                0.35,
                self.width - 0.35,
            ),
            self._clamp(
                enemy.position[1] + enemy.velocity[1] * time + 0.25 * enemy.acceleration[1] * time * time,
                0.35,
                self.height - 0.35,
            ),
        )

    def _guard_target(self, guard, escorted, enemies):
        threats = []
        for enemy in enemies:
            escort_distance = self._distance(enemy.position, escorted.position)
            guard_distance = self._distance(enemy.position, guard.position)
            goal_distance = self._distance(enemy.position, self.own_goal.center)
            incoming = self.direction * (enemy.position[0] - escorted.position[0]) > -4.0
            if (incoming and escort_distance < 15.0) or guard_distance < 8.5 or goal_distance < 24.0:
                value = self.specs[enemy.drone_type].point_value
                threats.append((escort_distance - 2.8 * value, enemy))
        
        if threats:
            # Intercept closest or most valuable threat
            return self._intercept_point(guard, min(threats, key=lambda item: item[0])[1])

        if self.goal.contains(escorted.position):
            return self._goal_target(guard)
            
        lead = min(8.0, max(2.8, 0.16 * self._distance(escorted.position, self.goal.center)))
        return (
            self._clamp(escorted.position[0] + self.direction * lead, 0.45, self.width - 0.45),
            self._clamp(
                escorted.position[1] + 0.45 * (self.goal_lanes.get(escorted.id, self.goal.center[1]) - escorted.position[1]),
                0.45,
                self.height - 0.45,
            ),
        )

    def step(self, state):
        self.tick += 1
        own_active = {
            drone.id: drone
            for drone in state.own_drones
            if drone.status is DroneStatus.ACTIVE
        }
        enemies_active = [
            drone
            for drone in state.opponent_drones
            if drone.status is DroneStatus.ACTIVE
        ]
        
        own_slow = [d for d in own_active.values() if d.drone_type is DroneType.SLOW]
        own_fast = [d for d in own_active.values() if d.drone_type is DroneType.FAST]

        # 1. Filter stable escort assignments, checking if both guard and escorted slow are active
        escort_map = {
            fast_id: slow_id
            for fast_id, slow_id in self.escort_for.items()
            if fast_id in own_active and slow_id in own_active
        }

        # 2. Identify Hunters (FAST drones that are not escorts)
        hunters = [d for d in own_fast if d.id not in escort_map]

        # 3. Designate Hunter target matching to active enemies (updates every 4 ticks)
        enemy_by_id = {drone.id: drone for drone in enemies_active}
        if self.tick == 1 or self.tick % 4 == 0:
            self.hunter_targets = {}
            if hunters and enemies_active:
                cost_matrix = []
                for h in hunters:
                    row = []
                    for e in enemies_active:
                        dist = self._distance(h.position, e.position)
                        intercept_time = dist / max(0.1, self.specs[DroneType.FAST].max_speed)
                        
                        own_goal_distance = self._distance(e.position, self.own_goal.center)
                        e_type = e.drone_type
                        
                        threshold = 76.0 if e_type is DroneType.SLOW else 54.0
                        if own_goal_distance > threshold and e_type is DroneType.FAST:
                            cost = 1e6
                        else:
                            val_bonus = 7.5 * self.specs[e_type].point_value
                            urgency = max(0.0, threshold - own_goal_distance)
                            cost = intercept_time - val_bonus - 0.12 * urgency
                            
                        row.append(cost)
                    cost_matrix.append(row)
                    
                h_ind, e_ind = linear_sum_assignment(cost_matrix)
                for r, c in zip(h_ind, e_ind):
                    if cost_matrix[r][c] < 1e5:
                        self.hunter_targets[hunters[r].id] = enemies_active[c].id
        else:
            # Retain existing matches if both hunter and target enemy are still active
            self.hunter_targets = {
                h_id: e_id
                for h_id, e_id in self.hunter_targets.items()
                if h_id in own_active and e_id in enemy_by_id
            }

        # 4. Generate acceleration command results for each active own drone
        actions = {}
        for drone in own_active.values():
            if drone.drone_type is DroneType.SLOW:
                actions[drone.id] = self._steer(drone, self._goal_target(drone))
                continue

            # Drone is FAST
            if drone.id in escort_map:
                # Acting as escort guard
                escorted_slow = own_active.get(escort_map[drone.id])
                if escorted_slow is not None:
                    actions[drone.id] = self._steer(drone, self._guard_target(drone, escorted_slow, enemies_active), 1.05)
                else:
                    actions[drone.id] = self._steer(drone, self._goal_target(drone), 1.08)
            else:
                # Acting as hunter
                target_enemy_id = self.hunter_targets.get(drone.id)
                target_enemy = next((e for e in enemies_active if e.id == target_enemy_id), None)
                if target_enemy is not None:
                    actions[drone.id] = self._steer(drone, self._intercept_point(drone, target_enemy), 1.08)
                else:
                    # Fallback to scoring if no dynamic enemy target mapped
                    actions[drone.id] = self._steer(drone, self._goal_target(drone), 1.08)

        return actions
