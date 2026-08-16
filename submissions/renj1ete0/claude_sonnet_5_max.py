"""Claude Sonnet 5 Max: visibility-graph routing with cover-seeking evasion.

Authorship: this file — its strategy, its numerical models, and every line of
its implementation — was coded entirely by Claude Sonnet 5 Max without human
guidance. No human authored, edited, or iteratively steered the code, the
tactics, or the constants below.

Design notes
------------
Three ideas carry this controller.

1. *Routing is an exact graph, not a discretised field.*  ``initialize``
   inflates every obstacle into a convex polygon (rectangle corners, or a
   circumscribed octagon for circles) and builds a visibility graph over the
   resulting vertices plus the goal mouth.  A single Dijkstra pass from the
   goal labels every vertex with its true remaining path length.  Each control
   tick then does a cheap local visibility scan from the drone's *current*
   position to pick whichever visible vertex minimises "hop here, then walk
   the precomputed shortest remainder" — an exact shortest route with no grid
   resolution error, recomputed from wherever the drone actually is instead of
   being baked into a fixed polyline.

2. *Evasion prefers breaking line of sight over out-turning the chaser.*  A
   pursued runner does not just react in velocity space: among the visibility
   graph vertices that keep it making reasonable progress, it favours ones an
   approaching hunter cannot see, so a detour that ducks behind an obstacle is
   chosen over one that stays in the open even when both cost similar path
   length.  A short-range kinematic repulsion layer is kept underneath as a
   last-resort safety net for close quarters where cover is not available.

3. *Interception is an auction.*  Every enemy is priced by the points it is
   about to bank (it is close to scoring, weighted by drone value) plus the
   points it is about to deny us (it is closing on one of our runners), then
   discounted by how confidently a given FAST drone can actually reach it in
   time.  Each hunter also gets a private "attack" column priced at rushing
   the goal itself, so a hunter with no profitable intercept converts to a
   scorer instead of shadowing an enemy that is out of reach.  `scipy`'s
   Hungarian algorithm clears the whole board at once so two hunters never
   pile onto one enemy while a five-point runner walks in unmarked.
"""

from __future__ import annotations

from heapq import heappop, heappush
from math import cos, exp, hypot, inf, pi, sin, sqrt

from scipy.optimize import linear_sum_assignment

from swarmbench import BaseSwarmController, CircleObstacle, DroneStatus, DroneType

MATCH_DURATION = 90.0
CLEARANCE = 0.62
"""Drone radius (0.25) plus enough skin to clear the generator's own margin."""

CIRCLE_SIDES = 8
"""Vertices used to circumscribe a circular obstacle for the visibility graph."""

GAIN_VELOCITY = 3.3
GAIN_ACCELERATION = 0.20
CORNER_BRAKE_ANGLE = 0.55
"""Radians of upcoming heading change beyond which speed is throttled back."""

WALL_MARGIN = 1.5
AVOID_SKIN = 0.5

DODGE_HORIZON = 3.2
DODGE_RADIUS = 2.0
DODGE_REPULSION_RADIUS = 9.0
COVER_DETOUR_TOLERANCE = 1.35
"""A shielded vertex is accepted if its route cost is within this factor of best."""

DENY_HORIZON = 6.0
RAID_VALUE = 5.0
RAID_HORIZON = 11.0
SCORE_WEIGHT = 2.0


class SwarmController(BaseSwarmController):
    """Runners follow an exact visibility-graph route; hunters bid on threats."""

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------

    def initialize(self, game_info):
        self.width = game_info.arena_width
        self.height = game_info.arena_height
        self.goal = game_info.target_goal
        self.own_goal = game_info.own_goal
        self.specs = dict(game_info.drone_specs)
        self.assignments = {}

        self.circles = tuple(
            (obstacle.center[0], obstacle.center[1], obstacle.radius)
            for obstacle in game_info.obstacles
            if isinstance(obstacle, CircleObstacle)
        )
        self.rectangles = tuple(
            (obstacle.x_min, obstacle.x_max, obstacle.y_min, obstacle.y_max)
            for obstacle in game_info.obstacles
            if not isinstance(obstacle, CircleObstacle)
        )

        inner_x = self.goal.x_max if self.goal.x_min <= 1e-6 else self.goal.x_min
        self.goal_mouth = (inner_x, self.goal.center[1])

        self._build_visibility_graph()

        # Spread the goal approach so a single defender cannot shadow the
        # whole squad, and so a detour never funnels two runners together.
        ordered = sorted(game_info.own_initial_drones, key=lambda drone: (drone.position[1], drone.id))
        low = self.goal.y_min + 0.9
        high = self.goal.y_max - 0.9
        span = max(0.0, high - low)
        count = max(1, len(ordered))
        self.lanes = {
            drone.id: low + span * (rank + 0.5) / count
            for rank, drone in enumerate(ordered)
        }

    # ------------------------------------------------------------------
    # visibility graph
    # ------------------------------------------------------------------

    def _build_visibility_graph(self):
        nodes = [self.goal_mouth]
        for center_x, center_y, radius in self.circles:
            reach = radius / cos(pi / CIRCLE_SIDES) + CLEARANCE
            for index in range(CIRCLE_SIDES):
                angle = 2.0 * pi * index / CIRCLE_SIDES
                nodes.append((center_x + reach * cos(angle), center_y + reach * sin(angle)))
        for x_min, x_max, y_min, y_max in self.rectangles:
            nodes.append((x_min - CLEARANCE, y_min - CLEARANCE))
            nodes.append((x_max + CLEARANCE, y_min - CLEARANCE))
            nodes.append((x_max + CLEARANCE, y_max + CLEARANCE))
            nodes.append((x_min - CLEARANCE, y_max + CLEARANCE))

        # Keep nodes inside the arena; a vertex projected outside the field
        # cannot help routing and only costs visibility checks.
        nodes = [
            node for node in nodes
            if -1.0 <= node[0] <= self.width + 1.0 and -1.0 <= node[1] <= self.height + 1.0
        ]
        self.nodes = nodes

        count = len(nodes)
        adjacency = [[] for _ in range(count)]
        for i in range(count):
            for j in range(i + 1, count):
                if self._clear(nodes[i], nodes[j], CLEARANCE):
                    weight = hypot(nodes[i][0] - nodes[j][0], nodes[i][1] - nodes[j][1])
                    adjacency[i].append((j, weight))
                    adjacency[j].append((i, weight))
        self.adjacency = adjacency

        distance = [inf] * count
        distance[0] = 0.0
        frontier = [(0.0, 0)]
        while frontier:
            cost, node = heappop(frontier)
            if cost > distance[node]:
                continue
            for neighbour, weight in adjacency[node]:
                candidate = cost + weight
                if candidate < distance[neighbour]:
                    distance[neighbour] = candidate
                    heappush(frontier, (candidate, neighbour))
        self.goal_distance = distance

    # ------------------------------------------------------------------
    # geometry
    # ------------------------------------------------------------------

    @staticmethod
    def _unit(origin, target):
        dx = target[0] - origin[0]
        dy = target[1] - origin[1]
        magnitude = hypot(dx, dy)
        if magnitude < 1e-9:
            return 0.0, 0.0
        return dx / magnitude, dy / magnitude

    @staticmethod
    def _clamp(value, low, high):
        return low if value < low else (high if value > high else value)

    @staticmethod
    def _segment_hits_circle(start, end, center_x, center_y, radius):
        offset_x = start[0] - center_x
        offset_y = start[1] - center_y
        if offset_x * offset_x + offset_y * offset_y <= radius * radius:
            return True
        delta_x = end[0] - start[0]
        delta_y = end[1] - start[1]
        quadratic = delta_x * delta_x + delta_y * delta_y
        if quadratic < 1e-12:
            return False
        linear = 2.0 * (offset_x * delta_x + offset_y * delta_y)
        constant = offset_x * offset_x + offset_y * offset_y - radius * radius
        discriminant = linear * linear - 4.0 * quadratic * constant
        if discriminant < 0.0:
            return False
        root = sqrt(discriminant)
        near = (-linear - root) / (2.0 * quadratic)
        far = (-linear + root) / (2.0 * quadratic)
        return near <= 1.0 and far >= 0.0

    @staticmethod
    def _segment_hits_box(start, end, x_min, x_max, y_min, y_max):
        entry, exit_time = 0.0, 1.0
        for origin, delta, low, high in (
            (start[0], end[0] - start[0], x_min, x_max),
            (start[1], end[1] - start[1], y_min, y_max),
        ):
            if abs(delta) < 1e-12:
                if origin < low or origin > high:
                    return False
                continue
            first = (low - origin) / delta
            second = (high - origin) / delta
            if first > second:
                first, second = second, first
            entry = max(entry, first)
            exit_time = min(exit_time, second)
            if entry > exit_time:
                return False
        return exit_time >= 0.0 and entry <= 1.0

    def _clear(self, start, end, margin):
        for center_x, center_y, radius in self.circles:
            if self._segment_hits_circle(start, end, center_x, center_y, radius + margin):
                return False
        for x_min, x_max, y_min, y_max in self.rectangles:
            if self._segment_hits_box(start, end, x_min - margin, x_max + margin, y_min - margin, y_max + margin):
                return False
        return True

    def _nearest_blocker_center(self, start, direction, lookahead):
        end = (start[0] + direction[0] * lookahead, start[1] + direction[1] * lookahead)
        best = None
        for center_x, center_y, radius in self.circles:
            reach = radius + AVOID_SKIN
            if self._segment_hits_circle(start, end, center_x, center_y, reach):
                distance = hypot(center_x - start[0], center_y - start[1])
                if best is None or distance < best[0]:
                    best = (distance, center_x, center_y, reach)
        for x_min, x_max, y_min, y_max in self.rectangles:
            if self._segment_hits_box(start, end, x_min - AVOID_SKIN, x_max + AVOID_SKIN, y_min - AVOID_SKIN, y_max + AVOID_SKIN):
                center_x = (x_min + x_max) / 2.0
                center_y = (y_min + y_max) / 2.0
                reach = 0.5 * hypot(x_max - x_min, y_max - y_min) + AVOID_SKIN
                distance = hypot(center_x - start[0], center_y - start[1])
                if best is None or distance < best[0]:
                    best = (distance, center_x, center_y, reach)
        return best

    # ------------------------------------------------------------------
    # routing
    # ------------------------------------------------------------------

    def _route_options(self, position, goal_point, margin):
        """Every reachable aim point from here, cheapest-total-route first.

        Each option is (route_cost, point): either the goal directly (if it
        is visible) or a graph vertex reached by one visibility hop plus its
        precomputed shortest remainder to the goal.
        """
        options = []
        if self._clear(position, goal_point, margin):
            options.append((hypot(goal_point[0] - position[0], goal_point[1] - position[1]), goal_point))
        for index, node in enumerate(self.nodes):
            if self.goal_distance[index] == inf:
                continue
            if self._clear(position, node, margin):
                cost = hypot(node[0] - position[0], node[1] - position[1]) + self.goal_distance[index]
                options.append((cost, node))
        options.sort(key=lambda item: item[0])
        return options

    def _route_target(self, position, goal_point, threats):
        """Next aim point toward the goal, biased away from watching enemies.

        A drone with no active pursuer always takes the cheapest option. A
        pursued drone instead compares nearby options and accepts a modestly
        longer one if it breaks every threat's line of sight.
        """
        options = self._route_options(position, goal_point, CLEARANCE)
        if not options:
            return goal_point
        best_cost = options[0][0]

        if threats:
            for cost, point in options:
                if cost > best_cost * COVER_DETOUR_TOLERANCE:
                    break
                if all(not self._clear(threat, point, 0.0) for threat in threats):
                    return point

        return options[0][1]

    def _path_length(self, position, goal_point):
        options = self._route_options(position, goal_point, CLEARANCE)
        if options:
            return options[0][0]
        return hypot(position[0] - goal_point[0], position[1] - goal_point[1])

    # ------------------------------------------------------------------
    # low-level control
    # ------------------------------------------------------------------

    def _avoid(self, position, direction, speed):
        if direction[0] == 0.0 and direction[1] == 0.0:
            return direction
        lookahead = 1.4 + 0.75 * speed
        blocker = self._nearest_blocker_center(position, direction, lookahead)
        if blocker is None:
            return direction
        _, center_x, center_y, radius = blocker
        toward_x = center_x - position[0]
        toward_y = center_y - position[1]
        distance = hypot(toward_x, toward_y)
        if distance < 1e-6:
            return direction
        toward_x /= distance
        toward_y /= distance
        if distance <= radius:
            tangent = (-toward_y, toward_x)
            if tangent[0] * direction[0] + tangent[1] * direction[1] < 0.0:
                tangent = (toward_y, -toward_x)
            escape_x = tangent[0] - 0.7 * toward_x
            escape_y = tangent[1] - 0.7 * toward_y
            magnitude = hypot(escape_x, escape_y)
            return (escape_x / magnitude, escape_y / magnitude) if magnitude > 1e-9 else direction
        sine = self._clamp(radius / distance, -1.0, 1.0)
        cosine = sqrt(max(0.0, 1.0 - sine * sine))
        left = (toward_x * cosine - toward_y * sine, toward_x * sine + toward_y * cosine)
        right = (toward_x * cosine + toward_y * sine, -toward_x * sine + toward_y * cosine)
        if left[0] * direction[0] + left[1] * direction[1] >= right[0] * direction[0] + right[1] * direction[1]:
            return left
        return right

    def _drive(self, drone, aim_point, threats=None):
        spec = self.specs[drone.drone_type]
        position = drone.position
        speed = hypot(drone.velocity[0], drone.velocity[1])
        direction = self._unit(position, aim_point)
        if threats:
            direction = self._kinematic_dodge(drone, direction, threats)
        direction = self._avoid(position, direction, speed)

        # Throttle back before a sharp turn instead of overshooting past a
        # corner and paying for it with a second detour next tick.
        target_speed = spec.max_speed
        if speed > 0.4:
            heading = (drone.velocity[0] / speed, drone.velocity[1] / speed)
            turn = heading[0] * direction[0] + heading[1] * direction[1]
            if turn < 1.0 - CORNER_BRAKE_ANGLE:
                target_speed *= max(0.35, 0.5 + 0.5 * turn)

        desired_x = direction[0] * target_speed
        desired_y = direction[1] * target_speed
        command_x = GAIN_VELOCITY * (desired_x - drone.velocity[0]) - GAIN_ACCELERATION * drone.acceleration[0]
        command_y = GAIN_VELOCITY * (desired_y - drone.velocity[1]) - GAIN_ACCELERATION * drone.acceleration[1]

        if position[1] < WALL_MARGIN:
            command_y += spec.max_acceleration * (WALL_MARGIN - position[1]) / WALL_MARGIN
        elif position[1] > self.height - WALL_MARGIN:
            command_y -= spec.max_acceleration * (position[1] - self.height + WALL_MARGIN) / WALL_MARGIN

        magnitude = hypot(command_x, command_y)
        if magnitude > spec.max_acceleration:
            scale = spec.max_acceleration / magnitude
            command_x *= scale
            command_y *= scale
        return command_x, command_y

    # ------------------------------------------------------------------
    # threat model
    # ------------------------------------------------------------------

    def _predicted_velocity(self, drone, goal_point):
        spec = self.specs[drone.drone_type]
        toward_x, toward_y = self._unit(drone.position, goal_point)
        return (
            0.6 * drone.velocity[0] + 0.4 * toward_x * spec.max_speed,
            0.6 * drone.velocity[1] + 0.4 * toward_y * spec.max_speed,
        )

    def _intercept(self, hunter, target_position, target_velocity):
        speed = self.specs[hunter.drone_type].max_speed
        offset_x = target_position[0] - hunter.position[0]
        offset_y = target_position[1] - hunter.position[1]
        quadratic = target_velocity[0] ** 2 + target_velocity[1] ** 2 - speed * speed
        linear = 2.0 * (offset_x * target_velocity[0] + offset_y * target_velocity[1])
        constant = offset_x * offset_x + offset_y * offset_y

        time = None
        if abs(quadratic) < 1e-9:
            if linear < -1e-9:
                time = -constant / linear
        else:
            discriminant = linear * linear - 4.0 * quadratic * constant
            if discriminant >= 0.0:
                root = sqrt(discriminant)
                roots = [value for value in ((-linear - root) / (2.0 * quadratic), (-linear + root) / (2.0 * quadratic)) if value >= 0.0]
                if roots:
                    time = min(roots)
        if time is None or time > 30.0:
            time = sqrt(constant) / max(0.5, speed)
        time += 0.15
        point = (
            self._clamp(target_position[0] + target_velocity[0] * time, 0.3, self.width - 0.3),
            self._clamp(target_position[1] + target_velocity[1] * time, 0.3, self.height - 0.3),
        )
        return time, point

    def _time_to_own_goal(self, enemy):
        distance = hypot(enemy.position[0] - self.own_goal.center[0], enemy.position[1] - self.own_goal.center[1])
        return 1.12 * distance / self.specs[enemy.drone_type].max_speed

    def _raid_chance(self, enemy, runners):
        if not runners or enemy.drone_type is DroneType.SLOW:
            return 0.0
        best = inf
        for runner in runners:
            velocity = self._predicted_velocity(runner, self.goal_mouth)
            time, _ = self._intercept(enemy, runner.position, velocity)
            if time < best:
                best = time
        if best == inf:
            return 0.0
        return self._clamp(1.0 - best / RAID_HORIZON, 0.0, 1.0)

    # ------------------------------------------------------------------
    # interception auction
    # ------------------------------------------------------------------

    def _deny_value(self, enemy, remaining, runners):
        spec = self.specs[enemy.drone_type]
        margin = remaining - self._time_to_own_goal(enemy)
        scoring = self._clamp(margin / 2.5, 0.0, 1.0)
        return spec.point_value * scoring + RAID_VALUE * self._raid_chance(enemy, runners)

    def _intercept_bid(self, hunter, enemy, remaining, runners):
        value = self._deny_value(enemy, remaining, runners)
        if value <= 0.05:
            return 0.0, None
        velocity = self._predicted_velocity(enemy, self.own_goal.center)
        time, point = self._intercept(hunter, enemy.position, velocity)
        if time >= remaining:
            return 0.0, point
        confidence = exp(-time / DENY_HORIZON)
        if time > self._time_to_own_goal(enemy) + 0.6:
            confidence *= 0.12
        if not self._clear(hunter.position, point, 0.3):
            confidence *= 0.85
        if self.assignments.get(hunter.id) == enemy.id:
            confidence *= 1.15
        return value * confidence, point

    def _attack_bid(self, hunter, remaining, urgency):
        spec = self.specs[hunter.drone_type]
        travel = 1.15 * self._path_length(hunter.position, self.goal_mouth) / spec.max_speed
        reachable = self._clamp((remaining - travel) / 2.5, 0.0, 1.0)
        return SCORE_WEIGHT * spec.point_value * reachable * urgency

    def _clear_the_board(self, hunters, enemies, runners, remaining, urgency):
        if not hunters:
            return {}, {}
        rows = len(hunters)
        enemy_count = len(enemies)
        columns = enemy_count + rows
        cost = [[0.0] * columns for _ in range(rows)]
        points = [[None] * columns for _ in range(rows)]
        for row, hunter in enumerate(hunters):
            for column, enemy in enumerate(enemies):
                bid, point = self._intercept_bid(hunter, enemy, remaining, runners)
                cost[row][column] = -bid
                points[row][column] = point
            fallback = self._attack_bid(hunter, remaining, urgency)
            for slot in range(rows):
                cost[row][enemy_count + slot] = -fallback if slot == row else 1e6

        row_indices, column_indices = linear_sum_assignment(cost)
        targets, aims = {}, {}
        for row, column in zip(row_indices, column_indices, strict=True):
            if column < enemy_count and -cost[row][column] > max(0.03, self._attack_bid(hunters[row], remaining, urgency)):
                targets[hunters[row].id] = enemies[column].id
                aims[hunters[row].id] = points[row][column]
        return targets, aims

    # ------------------------------------------------------------------
    # behaviours
    # ------------------------------------------------------------------

    def _goal_point(self, drone):
        lane = self.lanes.get(drone.id, self.goal_mouth[1])
        return (self.goal_mouth[0], self._clamp(lane, self.goal.y_min + 0.5, self.goal.y_max - 0.5))

    def _threats(self, drone, enemies):
        found = []
        for enemy in enemies:
            offset_x = enemy.position[0] - drone.position[0]
            offset_y = enemy.position[1] - drone.position[1]
            if abs(offset_x) > 14.0 or abs(offset_y) > 14.0:
                continue
            closing_x = enemy.velocity[0] - drone.velocity[0]
            closing_y = enemy.velocity[1] - drone.velocity[1]
            closing = closing_x * closing_x + closing_y * closing_y
            if closing < 1e-6:
                continue
            time = -(offset_x * closing_x + offset_y * closing_y) / closing
            if time <= 0.0 or time > DODGE_HORIZON:
                continue
            miss = hypot(offset_x + closing_x * time, offset_y + closing_y * time)
            if miss <= DODGE_RADIUS:
                found.append((time, enemy))
        found.sort(key=lambda item: item[0])
        return [enemy.position for _, enemy in found[:2]]

    def _kinematic_dodge(self, drone, direction, threats_positions):
        if not threats_positions:
            return direction
        closest = min(threats_positions, key=lambda point: hypot(point[0] - drone.position[0], point[1] - drone.position[1]))
        offset_x = drone.position[0] - closest[0]
        offset_y = drone.position[1] - closest[1]
        distance = hypot(offset_x, offset_y)
        if distance < 1e-6 or distance > DODGE_REPULSION_RADIUS:
            return direction
        weight = self._clamp(1.0 - distance / DODGE_REPULSION_RADIUS, 0.0, 1.0) * 0.8
        blended_x = direction[0] + weight * offset_x / distance
        blended_y = direction[1] + weight * offset_y / distance
        magnitude = hypot(blended_x, blended_y)
        return (blended_x / magnitude, blended_y / magnitude) if magnitude > 1e-9 else direction

    def _run(self, drone, enemies):
        goal_point = self._goal_point(drone)
        threats = self._threats(drone, enemies)
        aim = self._route_target(drone.position, goal_point, threats)
        return self._drive(drone, aim, threats)

    def _pursue(self, hunter, enemy, aim):
        distance = hypot(enemy.position[0] - hunter.position[0], enemy.position[1] - hunter.position[1])
        if distance < 2.0:
            aim = (enemy.position[0] + enemy.velocity[0] * 0.12, enemy.position[1] + enemy.velocity[1] * 0.12)
        elif aim is None:
            aim = enemy.position
        return self._drive(hunter, aim)

    # ------------------------------------------------------------------
    # control loop
    # ------------------------------------------------------------------

    def step(self, state):
        remaining = max(0.0, MATCH_DURATION - state.time)
        own = [drone for drone in state.own_drones if drone.status is DroneStatus.ACTIVE]
        enemies = [drone for drone in state.opponent_drones if drone.status is DroneStatus.ACTIVE]
        runners = [drone for drone in own if drone.drone_type is DroneType.SLOW]
        hunters = [drone for drone in own if drone.drone_type is DroneType.FAST]
        by_id = {enemy.id: enemy for enemy in enemies}

        urgency = self._clamp(1.0 + 0.09 * (state.opponent_score - state.own_score), 0.65, 1.7)
        targets, aims = self._clear_the_board(hunters, enemies, runners, remaining, urgency)
        self.assignments = targets

        actions = {}
        for drone in runners:
            actions[drone.id] = self._run(drone, enemies)
        for drone in hunters:
            enemy = by_id.get(targets.get(drone.id))
            if enemy is None:
                actions[drone.id] = self._run(drone, enemies)
            else:
                actions[drone.id] = self._pursue(drone, enemy, aims.get(drone.id))
        return actions
