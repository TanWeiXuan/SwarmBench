"""Claude Opus 5 Max: flow-field runners with value-weighted interception.

Authorship: this file — its strategy, its numerical models, and every line of
its implementation — was coded entirely by Claude Opus 5 Max without human
guidance. No human authored, edited, or iteratively steered the code, the
tactics, or the constants below.

Design notes
------------
Three ideas carry this controller.

1. *Navigation is solved once, offline.*  ``initialize`` marches a fast
   marching method (Eikonal) front outward from the target goal across an
   inflated occupancy grid, which yields both an exact remaining path length
   and a smooth descent direction for every point of the arena.  Runners then
   navigate by an O(1) interpolated field lookup instead of re-deriving a
   detour every control tick, so they never stall in a concave pocket and
   never plan through an obstacle.

2. *Interception is an economics problem.*  Eliminations are one-for-one, so a
   FAST drone (1 point) that removes a SLOW drone (5 points) converts a
   1-point asset into 5 denied points, and a FAST drone that removes an enemy
   hunter rescues a 5-point runner.  Every enemy is therefore scored by the
   points it is expected to take from us, damped by the probability that this
   particular pursuer actually reaches it in time, and FAST drones are matched
   to enemies by an exact minimum-cost assignment rather than a greedy pass.
   Scoring is modelled as just another job on that same board, so hunters
   convert to runners exactly when denial stops paying for itself.

3. *Runners dodge along the miss vector.*  A pursued drone accelerates along
   the predicted closest-approach offset, which is the direction that grows
   the miss distance fastest, while the goal term keeps the detour productive.
"""

from __future__ import annotations

from heapq import heappop, heappush
from math import ceil, exp, floor, hypot, inf, sqrt

from swarmbench import BaseSwarmController, CircleObstacle, DroneStatus, DroneType

GRID_STEP = 0.5
"""Occupancy/field resolution in metres; matches the generator's own grid."""

PLAN_CLEARANCE = 0.60
"""Drone radius plus planning margin, i.e. the generator's traversability test."""

AVOID_MARGIN = 0.55
"""Extra skin around obstacles used by the reactive lookahead layer."""

MATCH_DURATION = 90.0
GAIN_VELOCITY = 3.4
GAIN_ACCELERATION = 0.22
WALL_MARGIN = 1.6
DODGE_HORIZON = 3.0
DODGE_RADIUS = 1.9
DENY_HORIZON = 6.0
RAID_VALUE = 5.0
RAID_HORIZON = 12.0
SCORE_WEIGHT = 2.0
"""A running FAST drone banks one point *and* baits a hunter off a runner, so
its job is priced above the single point that reaches the scoreboard."""


def _assign_minimum_cost(cost: list[list[float]]) -> list[int]:
    """Jonker-Volgenant shortest augmenting path assignment (rows <= columns).

    Returns the column chosen for each row.  Greedy matching routinely wastes
    two hunters on one enemy while a five-point runner walks in unopposed, so
    the exact solution is worth its (tiny, n <= 10) cost.
    """
    rows = len(cost)
    if rows == 0:
        return []
    columns = len(cost[0])
    potential_row = [0.0] * (rows + 1)
    potential_column = [0.0] * (columns + 1)
    match = [0] * (columns + 1)
    parent = [0] * (columns + 1)
    for row in range(1, rows + 1):
        match[0] = row
        column = 0
        minimum = [inf] * (columns + 1)
        used = [False] * (columns + 1)
        while True:
            used[column] = True
            current_row = match[column]
            delta = inf
            next_column = 0
            for candidate in range(1, columns + 1):
                if used[candidate]:
                    continue
                reduced = cost[current_row - 1][candidate - 1] - potential_row[current_row] - potential_column[candidate]
                if reduced < minimum[candidate]:
                    minimum[candidate] = reduced
                    parent[candidate] = column
                if minimum[candidate] < delta:
                    delta = minimum[candidate]
                    next_column = candidate
            for candidate in range(columns + 1):
                if used[candidate]:
                    potential_row[match[candidate]] += delta
                    potential_column[candidate] -= delta
                else:
                    minimum[candidate] -= delta
            column = next_column
            if match[column] == 0:
                break
        while column:
            previous = parent[column]
            match[column] = match[previous]
            column = previous
    result = [-1] * rows
    for candidate in range(1, columns + 1):
        if match[candidate]:
            result[match[candidate] - 1] = candidate - 1
    return result


class SwarmController(BaseSwarmController):
    """Runners follow a precomputed flow field; hunters buy denied points."""

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------

    def initialize(self, game_info):
        self.width = game_info.arena_width
        self.height = game_info.arena_height
        self.goal = game_info.target_goal
        self.own_goal = game_info.own_goal
        self.specs = dict(game_info.drone_specs)
        self.duration = MATCH_DURATION
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

        self._build_flow_field()

        # Spread the goal approach so a single defender cannot shadow the
        # whole squad, and so one obstacle never claims two runners at once.
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
    # navigation field
    # ------------------------------------------------------------------

    def _build_flow_field(self):
        """Fast-march arrival distance from the target goal over free space.

        The Eikonal update produces a field whose gradient points along a
        genuinely shortest route rather than along the eight compass headings
        a Dijkstra grid would give, so the interpolated descent direction is
        smooth enough to feed straight into the velocity controller.
        """
        step = GRID_STEP
        size_x = int(round(self.width / step)) + 1
        size_y = int(round(self.height / step)) + 1
        self.size_x = size_x
        self.size_y = size_y
        blocked = bytearray(size_x * size_y)

        for center_x, center_y, radius in self.circles:
            reach = radius + PLAN_CLEARANCE
            for index_x in range(max(0, int(floor((center_x - reach) / step))), min(size_x - 1, int(ceil((center_x + reach) / step))) + 1):
                for index_y in range(max(0, int(floor((center_y - reach) / step))), min(size_y - 1, int(ceil((center_y + reach) / step))) + 1):
                    if hypot(index_x * step - center_x, index_y * step - center_y) <= reach:
                        blocked[index_x * size_y + index_y] = 1
        for x_min, x_max, y_min, y_max in self.rectangles:
            for index_x in range(max(0, int(floor((x_min - PLAN_CLEARANCE) / step))), min(size_x - 1, int(ceil((x_max + PLAN_CLEARANCE) / step))) + 1):
                for index_y in range(max(0, int(floor((y_min - PLAN_CLEARANCE) / step))), min(size_y - 1, int(ceil((y_max + PLAN_CLEARANCE) / step))) + 1):
                    point_x = index_x * step
                    point_y = index_y * step
                    if x_min - PLAN_CLEARANCE <= point_x <= x_max + PLAN_CLEARANCE and y_min - PLAN_CLEARANCE <= point_y <= y_max + PLAN_CLEARANCE:
                        blocked[index_x * size_y + index_y] = 1

        field = [inf] * (size_x * size_y)
        frontier = []
        for index_x in range(size_x):
            point_x = index_x * step
            if not self.goal.x_min <= point_x <= self.goal.x_max:
                continue
            for index_y in range(size_y):
                point_y = index_y * step
                if not self.goal.y_min <= point_y <= self.goal.y_max:
                    continue
                cell = index_x * size_y + index_y
                if not blocked[cell]:
                    field[cell] = 0.0
                    heappush(frontier, (0.0, cell))

        while frontier:
            arrival, cell = heappop(frontier)
            if arrival > field[cell]:
                continue
            index_x, index_y = divmod(cell, size_y)
            for neighbour_x, neighbour_y in (
                (index_x - 1, index_y),
                (index_x + 1, index_y),
                (index_x, index_y - 1),
                (index_x, index_y + 1),
            ):
                if not (0 <= neighbour_x < size_x and 0 <= neighbour_y < size_y):
                    continue
                neighbour = neighbour_x * size_y + neighbour_y
                if blocked[neighbour] or field[neighbour] <= arrival:
                    continue
                candidate = self._eikonal(field, size_x, size_y, neighbour_x, neighbour_y, step)
                if candidate < field[neighbour]:
                    field[neighbour] = candidate
                    heappush(frontier, (candidate, neighbour))

        self.field = field
        self.blocked = blocked
        self._build_gradient(step)

    @staticmethod
    def _eikonal(field, size_x, size_y, index_x, index_y, step):
        """Solve |grad T| = 1 at one cell from its two upwind neighbours."""
        horizontal = inf
        if index_x > 0:
            horizontal = field[(index_x - 1) * size_y + index_y]
        if index_x + 1 < size_x:
            horizontal = min(horizontal, field[(index_x + 1) * size_y + index_y])
        vertical = inf
        if index_y > 0:
            vertical = field[index_x * size_y + index_y - 1]
        if index_y + 1 < size_y:
            vertical = min(vertical, field[index_x * size_y + index_y + 1])
        if horizontal == inf and vertical == inf:
            return inf
        if horizontal == inf or vertical == inf or abs(horizontal - vertical) >= step:
            return min(horizontal, vertical) + step
        difference = horizontal - vertical
        return 0.5 * (horizontal + vertical + sqrt(2.0 * step * step - difference * difference))

    def _build_gradient(self, step):
        """Cache the normalised descent direction of the arrival field."""
        size_x, size_y = self.size_x, self.size_y
        field = self.field
        flow_x = [0.0] * (size_x * size_y)
        flow_y = [0.0] * (size_x * size_y)
        for index_x in range(size_x):
            for index_y in range(size_y):
                cell = index_x * size_y + index_y
                here = field[cell]
                if here == inf:
                    continue
                gradient_x = self._difference(field, here, cell - size_y if index_x > 0 else None, cell + size_y if index_x + 1 < size_x else None, step)
                gradient_y = self._difference(field, here, cell - 1 if index_y > 0 else None, cell + 1 if index_y + 1 < size_y else None, step)
                magnitude = hypot(gradient_x, gradient_y)
                if magnitude > 1e-9:
                    flow_x[cell] = -gradient_x / magnitude
                    flow_y[cell] = -gradient_y / magnitude
        self.flow_x = flow_x
        self.flow_y = flow_y

    @staticmethod
    def _difference(field, here, lower, upper, step):
        low = field[lower] if lower is not None else inf
        high = field[upper] if upper is not None else inf
        if low != inf and high != inf:
            return (high - low) / (2.0 * step)
        if low != inf:
            return (here - low) / step
        if high != inf:
            return (high - here) / step
        return 0.0

    def _cell_weights(self, position):
        """Bilinear weights over the four cells surrounding a point."""
        raw_x = min(max(position[0] / GRID_STEP, 0.0), self.size_x - 1.0)
        raw_y = min(max(position[1] / GRID_STEP, 0.0), self.size_y - 1.0)
        base_x = min(int(raw_x), self.size_x - 2) if self.size_x > 1 else 0
        base_y = min(int(raw_y), self.size_y - 2) if self.size_y > 1 else 0
        offset_x = raw_x - base_x
        offset_y = raw_y - base_y
        return (
            (base_x * self.size_y + base_y, (1.0 - offset_x) * (1.0 - offset_y)),
            ((base_x + 1) * self.size_y + base_y, offset_x * (1.0 - offset_y)),
            (base_x * self.size_y + base_y + 1, (1.0 - offset_x) * offset_y),
            ((base_x + 1) * self.size_y + base_y + 1, offset_x * offset_y),
        )

    def _flow_direction(self, position):
        """Interpolated unit heading that descends the arrival field."""
        total_x = total_y = 0.0
        for cell, weight in self._cell_weights(position):
            if weight <= 0.0 or self.blocked[cell]:
                continue
            total_x += weight * self.flow_x[cell]
            total_y += weight * self.flow_y[cell]
        magnitude = hypot(total_x, total_y)
        if magnitude > 1e-6:
            return total_x / magnitude, total_y / magnitude
        return self._unit(position, self.goal.center)

    def _path_length(self, position):
        """Remaining route length to the goal, or a straight-line estimate."""
        total = weight_sum = 0.0
        for cell, weight in self._cell_weights(position):
            value = self.field[cell]
            if weight <= 0.0 or value == inf:
                continue
            total += weight * value
            weight_sum += weight
        if weight_sum > 1e-6:
            return total / weight_sum
        return hypot(position[0] - self.goal.center[0], position[1] - self.goal.center[1])

    # ------------------------------------------------------------------
    # geometry
    # ------------------------------------------------------------------

    @staticmethod
    def _unit(origin, target):
        delta_x = target[0] - origin[0]
        delta_y = target[1] - origin[1]
        magnitude = hypot(delta_x, delta_y)
        if magnitude < 1e-9:
            return 0.0, 0.0
        return delta_x / magnitude, delta_y / magnitude

    @staticmethod
    def _clamp(value, low, high):
        return low if value < low else (high if value > high else value)

    @staticmethod
    def _disc_entry(start, end, center_x, center_y, radius):
        offset_x = start[0] - center_x
        offset_y = start[1] - center_y
        if offset_x * offset_x + offset_y * offset_y <= radius * radius:
            return 0.0
        delta_x = end[0] - start[0]
        delta_y = end[1] - start[1]
        quadratic = delta_x * delta_x + delta_y * delta_y
        if quadratic < 1e-12:
            return None
        linear = 2.0 * (offset_x * delta_x + offset_y * delta_y)
        constant = offset_x * offset_x + offset_y * offset_y - radius * radius
        discriminant = linear * linear - 4.0 * quadratic * constant
        if discriminant < 0.0:
            return None
        entry = (-linear - sqrt(discriminant)) / (2.0 * quadratic)
        return entry if 0.0 <= entry <= 1.0 else None

    @staticmethod
    def _box_entry(start, end, x_min, x_max, y_min, y_max):
        entry, exit_time = 0.0, 1.0
        for origin, delta, low, high in (
            (start[0], end[0] - start[0], x_min, x_max),
            (start[1], end[1] - start[1], y_min, y_max),
        ):
            if abs(delta) < 1e-12:
                if origin < low or origin > high:
                    return None
                continue
            first = (low - origin) / delta
            second = (high - origin) / delta
            if first > second:
                first, second = second, first
            entry = max(entry, first)
            exit_time = min(exit_time, second)
            if entry > exit_time:
                return None
        return entry if exit_time >= 0.0 and entry <= 1.0 else None

    def _first_blocker(self, start, end, margin):
        """Nearest obstacle the segment enters, as a conservative disc."""
        best = None
        for center_x, center_y, radius in self.circles:
            entry = self._disc_entry(start, end, center_x, center_y, radius + margin)
            if entry is not None and (best is None or entry < best[0]):
                best = (entry, center_x, center_y, radius + margin)
        for x_min, x_max, y_min, y_max in self.rectangles:
            entry = self._box_entry(start, end, x_min - margin, x_max + margin, y_min - margin, y_max + margin)
            if entry is not None and (best is None or entry < best[0]):
                best = (
                    entry,
                    (x_min + x_max) / 2.0,
                    (y_min + y_max) / 2.0,
                    0.5 * hypot(x_max - x_min, y_max - y_min) + margin,
                )
        return best

    def _clear(self, start, end, margin):
        return self._first_blocker(start, end, margin) is None

    def _avoid(self, position, direction, speed):
        """Graze the first obstacle on the lookahead ray instead of entering it.

        The flow field already routes runners around static geometry; this
        layer exists for pursuers, whose targets move and whose straight-line
        chase has no such guarantee.
        """
        if direction[0] == 0.0 and direction[1] == 0.0:
            return direction
        lookahead = 1.5 + 0.8 * speed
        end = (position[0] + direction[0] * lookahead, position[1] + direction[1] * lookahead)
        blocker = self._first_blocker(position, end, AVOID_MARGIN)
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
            # Already inside the inflated skin: leave along the tangent that
            # still makes progress, with a radial component to get clear.
            tangent = (-toward_y, toward_x)
            if tangent[0] * direction[0] + tangent[1] * direction[1] < 0.0:
                tangent = (toward_y, -toward_x)
            escape_x = tangent[0] - 0.7 * toward_x
            escape_y = tangent[1] - 0.7 * toward_y
            magnitude = hypot(escape_x, escape_y)
            return (escape_x / magnitude, escape_y / magnitude) if magnitude > 1e-9 else direction

        sine = radius / distance
        cosine = sqrt(max(0.0, 1.0 - sine * sine))
        left = (toward_x * cosine - toward_y * sine, toward_x * sine + toward_y * cosine)
        right = (toward_x * cosine + toward_y * sine, -toward_x * sine + toward_y * cosine)
        if left[0] * direction[0] + left[1] * direction[1] >= right[0] * direction[0] + right[1] * direction[1]:
            return left
        return right

    # ------------------------------------------------------------------
    # low-level control
    # ------------------------------------------------------------------

    def _drive(self, drone, direction):
        """Velocity-tracking command with jerk damping and arena containment.

        Nothing here ever asks for less than full speed: goal entry is a swept
        test, so arriving fast is free, and an interception point that has
        moved is re-solved on the next control tick anyway.
        """
        spec = self.specs[drone.drone_type]
        speed = hypot(drone.velocity[0], drone.velocity[1])
        direction = self._avoid(drone.position, direction, speed)
        desired_x = direction[0] * spec.max_speed
        desired_y = direction[1] * spec.max_speed

        # Acceleration slews at a finite jerk, so damping the current
        # acceleration is what keeps a turn from ringing past its heading.
        command_x = GAIN_VELOCITY * (desired_x - drone.velocity[0]) - GAIN_ACCELERATION * drone.acceleration[0]
        command_y = GAIN_VELOCITY * (desired_y - drone.velocity[1]) - GAIN_ACCELERATION * drone.acceleration[1]

        if drone.position[1] < WALL_MARGIN:
            command_y += spec.max_acceleration * (WALL_MARGIN - drone.position[1]) / WALL_MARGIN
        elif drone.position[1] > self.height - WALL_MARGIN:
            command_y -= spec.max_acceleration * (drone.position[1] - self.height + WALL_MARGIN) / WALL_MARGIN

        magnitude = hypot(command_x, command_y)
        if magnitude > spec.max_acceleration:
            scale = spec.max_acceleration / magnitude
            command_x *= scale
            command_y *= scale
        return command_x, command_y

    # ------------------------------------------------------------------
    # threat model
    # ------------------------------------------------------------------

    def _predicted_velocity(self, drone, goal_center):
        """Blend measured motion with the objective the drone must pursue.

        Instantaneous velocity alone over-reacts to a dodge; the goal term
        remembers that every enemy is ultimately walking to one rectangle.
        """
        spec = self.specs[drone.drone_type]
        toward_x, toward_y = self._unit(drone.position, goal_center)
        return (
            0.62 * drone.velocity[0] + 0.38 * toward_x * spec.max_speed,
            0.62 * drone.velocity[1] + 0.38 * toward_y * spec.max_speed,
        )

    def _intercept(self, hunter, target, target_velocity):
        """Earliest time and point at which the hunter can meet the target."""
        speed = self.specs[hunter.drone_type].max_speed
        offset_x = target.position[0] - hunter.position[0]
        offset_y = target.position[1] - hunter.position[1]
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

        # A stationary pursuer needs a moment to spin its acceleration around.
        time += 0.15
        point = (
            self._clamp(target.position[0] + target_velocity[0] * time, 0.3, self.width - 0.3),
            self._clamp(target.position[1] + target_velocity[1] * time, 0.3, self.height - 0.3),
        )
        return time, point

    def _time_to_our_goal(self, enemy):
        """Optimistic time for an enemy to reach the rectangle it is aiming at."""
        distance = hypot(enemy.position[0] - self.own_goal.center[0], enemy.position[1] - self.own_goal.center[1])
        return 1.12 * distance / self.specs[enemy.drone_type].max_speed

    def _raid_chance(self, enemy, runners):
        """Probability that this enemy takes one of our five-point runners."""
        if not runners or enemy.drone_type is DroneType.SLOW:
            return 0.0
        best = inf
        for runner in runners:
            velocity = self._predicted_velocity(runner, self.goal.center)
            time, _ = self._intercept(enemy, runner, velocity)
            if time < best:
                best = time
        if best == inf:
            return 0.0
        return self._clamp(1.0 - best / RAID_HORIZON, 0.0, 1.0)

    # ------------------------------------------------------------------
    # job market
    # ------------------------------------------------------------------

    def _deny_value(self, enemy, remaining, runners):
        """Points this enemy is expected to take from us if left alone."""
        spec = self.specs[enemy.drone_type]
        margin = remaining - self._time_to_our_goal(enemy)
        scoring = self._clamp(margin / 2.5, 0.0, 1.0)
        return spec.point_value * scoring + RAID_VALUE * self._raid_chance(enemy, runners)

    def _intercept_benefit(self, hunter, enemy, remaining, runners):
        value = self._deny_value(enemy, remaining, runners)
        if value <= 0.05:
            return 0.0, None
        velocity = self._predicted_velocity(enemy, self.own_goal.center)
        time, point = self._intercept(hunter, enemy, velocity)
        if time >= remaining:
            return 0.0, point
        chance = exp(-time / DENY_HORIZON)
        if time > self._time_to_our_goal(enemy) + 0.6:
            # The enemy banks its points before we arrive; the chase is a
            # gift of one more free runner to the other side.
            chance *= 0.12
        if not self._clear(hunter.position, point, 0.3):
            chance *= 0.85
        if self.assignments.get(hunter.id) == enemy.id:
            chance *= 1.15
        return value * chance, point

    def _score_benefit(self, drone, remaining, urgency):
        spec = self.specs[drone.drone_type]
        travel = 1.15 * self._path_length(drone.position) / spec.max_speed
        reachable = self._clamp((remaining - travel) / 2.5, 0.0, 1.0)
        return SCORE_WEIGHT * spec.point_value * reachable * urgency

    def _match_hunters(self, hunters, enemies, runners, remaining, urgency):
        """Exact assignment of FAST drones to enemies, with scoring as a job."""
        if not hunters:
            return {}, {}
        columns = len(enemies) + len(hunters)
        benefits = []
        points = []
        for hunter in hunters:
            row = []
            row_points = []
            for enemy in enemies:
                benefit, point = self._intercept_benefit(hunter, enemy, remaining, runners)
                row.append(benefit)
                row_points.append(point)
            fallback = self._score_benefit(hunter, remaining, urgency)
            row.extend([fallback] * len(hunters))
            row_points.extend([None] * len(hunters))
            benefits.append(row)
            points.append(row_points)

        cost = [[-value for value in row] for row in benefits]
        matching = _assign_minimum_cost(cost)
        targets = {}
        aim = {}
        for index, column in enumerate(matching):
            if 0 <= column < len(enemies) and benefits[index][column] > max(0.03, benefits[index][columns - 1]):
                targets[hunters[index].id] = enemies[column].id
                aim[hunters[index].id] = points[index][column]
        return targets, aim

    # ------------------------------------------------------------------
    # behaviours
    # ------------------------------------------------------------------

    def _goal_aim(self, drone):
        lane = self.lanes.get(drone.id, self.goal.center[1])
        return (
            self.goal.center[0],
            self._clamp(lane, self.goal.y_min + 0.5, self.goal.y_max - 0.5),
        )

    def _dodge(self, drone, direction, enemies):
        """Push along the predicted miss vector, which widens it fastest."""
        best = None
        for enemy in enemies:
            offset_x = enemy.position[0] - drone.position[0]
            offset_y = enemy.position[1] - drone.position[1]
            if abs(offset_x) > 16.0 or abs(offset_y) > 16.0:
                continue
            closing_x = enemy.velocity[0] - drone.velocity[0]
            closing_y = enemy.velocity[1] - drone.velocity[1]
            closing = closing_x * closing_x + closing_y * closing_y
            if closing < 1e-6:
                continue
            time = -(offset_x * closing_x + offset_y * closing_y) / closing
            if time <= 0.0 or time > DODGE_HORIZON:
                continue
            miss_x = offset_x + closing_x * time
            miss_y = offset_y + closing_y * time
            miss = hypot(miss_x, miss_y)
            if miss > DODGE_RADIUS:
                continue
            if best is None or time < best[0]:
                best = (time, miss, miss_x, miss_y, offset_x, offset_y)
        if best is None:
            return direction

        time, miss, miss_x, miss_y, offset_x, offset_y = best
        if miss > 1e-3:
            dodge_x, dodge_y = -miss_x / miss, -miss_y / miss
        else:
            span = hypot(offset_x, offset_y) or 1.0
            dodge_x, dodge_y = -offset_y / span, offset_x / span
            if dodge_x * direction[0] + dodge_y * direction[1] < 0.0:
                dodge_x, dodge_y = -dodge_x, -dodge_y

        # Never dodge into a wall: a pinned runner is a caught runner.
        projected = drone.position[1] + dodge_y * 4.0
        if projected < 0.8 or projected > self.height - 0.8:
            dodge_y = -dodge_y

        weight = 1.25 * (1.0 - time / DODGE_HORIZON) * (1.0 - miss / DODGE_RADIUS)
        blended_x = direction[0] + weight * dodge_x
        blended_y = direction[1] + weight * dodge_y
        magnitude = hypot(blended_x, blended_y)
        return (blended_x / magnitude, blended_y / magnitude) if magnitude > 1e-9 else direction

    def _run(self, drone, enemies, evade):
        aim = self._goal_aim(drone)
        if self._clear(drone.position, aim, 0.35):
            direction = self._unit(drone.position, aim)
        else:
            direction = self._flow_direction(drone.position)
        if evade:
            direction = self._dodge(drone, direction, enemies)
        return self._drive(drone, direction)

    def _pursue(self, hunter, enemy, aim):
        distance = hypot(enemy.position[0] - hunter.position[0], enemy.position[1] - hunter.position[1])
        if distance < 2.0:
            # Inside the endgame of a chase, lead by a single control period
            # rather than by the full interception solution.
            aim = (enemy.position[0] + enemy.velocity[0] * 0.12, enemy.position[1] + enemy.velocity[1] * 0.12)
        elif aim is None:
            aim = enemy.position
        return self._drive(hunter, self._unit(hunter.position, aim))

    # ------------------------------------------------------------------
    # control loop
    # ------------------------------------------------------------------

    def step(self, state):
        remaining = max(0.0, self.duration - state.time)
        own = [drone for drone in state.own_drones if drone.status is DroneStatus.ACTIVE]
        enemies = [drone for drone in state.opponent_drones if drone.status is DroneStatus.ACTIVE]
        runners = [drone for drone in own if drone.drone_type is DroneType.SLOW]
        hunters = [drone for drone in own if drone.drone_type is DroneType.FAST]
        by_id = {enemy.id: enemy for enemy in enemies}

        # Trailing teams should buy points; leading teams should deny them.
        urgency = self._clamp(1.0 + 0.09 * (state.opponent_score - state.own_score), 0.65, 1.7)
        targets, aims = self._match_hunters(hunters, enemies, runners, remaining, urgency)
        self.assignments = targets

        actions = {}
        for drone in runners:
            actions[drone.id] = self._run(drone, enemies, True)
        for drone in hunters:
            enemy = by_id.get(targets.get(drone.id))
            if enemy is None:
                actions[drone.id] = self._run(drone, enemies, True)
            else:
                actions[drone.id] = self._pursue(drone, enemy, aims.get(drone.id))
        return actions
