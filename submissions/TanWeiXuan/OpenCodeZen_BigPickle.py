"""OpenCodeZen_BigPickle: flow-field runners with a value auction of FAST hunters.

Coding agent: OpenCode Zen, model "big-pickle" (opencode / opencode.ai).

Strategy overview (designed from the documented SwarmBench v1 mechanics)
-----------------------------------------------------------------------
Each team has ten FAST drones (1 point, 5 m/s) and ten SLOW drones (5 points,
2.5 m/s).  Enemy interception is one-for-one at 0.75 m, so the central trade of
the game is "sacrifice one FAST to deny an enemy SLOW".  The controller splits
cleanly into two layers:

1. NAVIGATION (all drones): a fast-marching (Eikonal) flow field is built once
   in ``initialize`` from the target goal over an occupancy grid in which every
   obstacle is inflated by a generous planning clearance.  The gradient of the
   arrival field (lightly smoothed) is the obstacle-free descent direction a
   runner should follow; it is interpolated at O(1) per control tick.  A
   reactive tangent-steering layer (lookahead ray against inflated obstacles,
   graze along the tangent that keeps progress) plus wall repulsion catches any
   corner clipping the coarse grid cannot see.

2. ASSIGNMENT OF FAST DRONES (per tick): every active FAST drone is an
   auctioneer of jobs.  Each enemy is priced by the points it is expected to
   take from us:
     * SLOW enemies are worth their 5-point swing if they can still reach our
       goal inside the match time.
     * FAST enemies additionally carry a "protection" value: 5 points for each
       of our SLOW runners they can realistically catch.
   Each hunter bids (value x catch-confidence) for each enemy; the confidence
   decays with the hunter's intercept time, is cut if the enemy resolves before
   the hunter arrives, and is penalised when an enemy FAST escort could reach
   the hunter first (deep hunts into defended territory).  The whole board is
   cleared with SciPy's exact linear-sum assignment, so two hunters never pile
   onto one enemy while a 5-point runner walks in unmarked.  A modest
   stickiness bonus keeps a hunter on its current target across ticks.

3. DEFENSIVE POSTURE: an unassigned hunter does not rush across the field to
   bank a 1-point goal -- interception is mutual, so a FAST-for-FAST trade is
   worthless and only bleeds the defensive core that must deny the enemy SLOWs
   arriving around the 35-45s mark.  Instead it loiters on a patrol line a few
   metres in front of our own goal, spread across the goal's approach corridor,
   where it converts the one good trade in the game (FAST-for-SLOW) the moment
   an enemy runner reaches its window.  Only in the last seconds do hunters
   switch to scoring themselves.

4. RUNNER EVASION: a pursued SLOW runner dodges along the predicted
   closest-approach miss vector (the direction that grows the miss distance
   fastest), blended with its progress direction, weighted harder the closer
   the pursuer is, and never dodges into a wall.

Attribution: the general algorithmic ideas above (flow-field routing, a
deny/protect/score auction solved by linear-sum assignment, and
closest-approach miss-vector dodging) are standard techniques that also appear
in the community controllers in this repository (notably renj1ete0's flow
field / auction architecture and the baseline "assignment" controller).  I
studied those files to understand the API and the value of these ideas, but
every line below is an original implementation written by OpenCode Zen with
the big-pickle model; no code was copied from any other controller.
"""

from __future__ import annotations

from heapq import heappop, heappush
from math import ceil, exp, floor, hypot, inf, sqrt

from scipy.optimize import linear_sum_assignment

from swarmbench import BaseSwarmController, CircleObstacle, DroneStatus, DroneType

MATCH_DURATION = 90.0
GRID_STEP = 0.5
PLAN_CLEARANCE = 1.0
AVOID_SKIN = 0.55
GAIN_VELOCITY = 3.4
GAIN_ACCELERATION = 0.22
WALL_MARGIN = 1.5
DODGE_HORIZON = 5.0
DODGE_RADIUS = 2.5
DENY_HORIZON = 6.0
PROTECT_HORIZON = 8.0
SCORE_WINDOW = 2.5
SCORE_WEIGHT = 1.3
SLOW_VALUE = 5.0
INTERCEPT_RADIUS = 0.75
DEFEND_VALUE = 1.0
DEFEND_PACE = 0.85
ENDGAME_HORIZON = 12.0


class SwarmController(BaseSwarmController):
    """Flow-field runners plus a per-tick value auction of FAST hunters."""

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------

    def initialize(self, game_info):
        self.width = game_info.arena_width
        self.height = game_info.arena_height
        self.goal = game_info.target_goal
        self.own_goal = game_info.own_goal
        self.team = game_info.team
        self.specs = dict(game_info.drone_specs)
        self.assignments = {}
        self.team_is_a = game_info.team.value == "A"

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

    # ------------------------------------------------------------------
    # navigation field (fast marching / Eikonal)
    # ------------------------------------------------------------------

    def _build_flow_field(self):
        step = GRID_STEP
        size_x = int(round(self.width / step)) + 1
        size_y = int(round(self.height / step)) + 1
        self.size_x = size_x
        self.size_y = size_y
        blocked = bytearray(size_x * size_y)

        for center_x, center_y, radius in self.circles:
            reach = radius + PLAN_CLEARANCE
            for ix in range(max(0, int(floor((center_x - reach) / step))), min(size_x - 1, int(ceil((center_x + reach) / step))) + 1):
                for iy in range(max(0, int(floor((center_y - reach) / step))), min(size_y - 1, int(ceil((center_y + reach) / step))) + 1):
                    if hypot(ix * step - center_x, iy * step - center_y) <= reach:
                        blocked[ix * size_y + iy] = 1
        for x0, x1, y0, y1 in self.rectangles:
            for ix in range(max(0, int(floor((x0 - PLAN_CLEARANCE) / step))), min(size_x - 1, int(ceil((x1 + PLAN_CLEARANCE) / step))) + 1):
                for iy in range(max(0, int(floor((y0 - PLAN_CLEARANCE) / step))), min(size_y - 1, int(ceil((y1 + PLAN_CLEARANCE) / step))) + 1):
                    px = ix * step
                    py = iy * step
                    if x0 - PLAN_CLEARANCE <= px <= x1 + PLAN_CLEARANCE and y0 - PLAN_CLEARANCE <= py <= y1 + PLAN_CLEARANCE:
                        blocked[ix * size_y + iy] = 1

        field = [inf] * (size_x * size_y)
        frontier = []
        for ix in range(size_x):
            px = ix * step
            if not self.goal.x_min <= px <= self.goal.x_max:
                continue
            for iy in range(size_y):
                py = iy * step
                if not self.goal.y_min <= py <= self.goal.y_max:
                    continue
                cell = ix * size_y + iy
                if not blocked[cell]:
                    field[cell] = 0.0
                    heappush(frontier, (0.0, cell))

        while frontier:
            arrival, cell = heappop(frontier)
            if arrival > field[cell]:
                continue
            ix, iy = divmod(cell, size_y)
            for nx, ny in ((ix - 1, iy), (ix + 1, iy), (ix, iy - 1), (ix, iy + 1)):
                if not (0 <= nx < size_x and 0 <= ny < size_y):
                    continue
                neighbour = nx * size_y + ny
                if blocked[neighbour] or field[neighbour] <= arrival:
                    continue
                candidate = self._eikonal(field, size_x, size_y, nx, ny, step)
                if candidate < field[neighbour]:
                    field[neighbour] = candidate
                    heappush(frontier, (candidate, neighbour))

        self.field = field
        self.blocked = blocked
        self._build_gradient(step)

    @staticmethod
    def _eikonal(field, size_x, size_y, ix, iy, step):
        """Solve |grad T| = 1 at one cell from its two upwind neighbours."""
        horizontal = inf
        if ix > 0:
            horizontal = min(horizontal, field[(ix - 1) * size_y + iy])
        if ix + 1 < size_x:
            horizontal = min(horizontal, field[(ix + 1) * size_y + iy])
        vertical = inf
        if iy > 0:
            vertical = min(vertical, field[ix * size_y + iy - 1])
        if iy + 1 < size_y:
            vertical = min(vertical, field[ix * size_y + iy + 1])
        if horizontal == inf and vertical == inf:
            return inf
        if horizontal == inf or vertical == inf or abs(horizontal - vertical) >= step:
            return min(horizontal, vertical) + step
        difference = horizontal - vertical
        return 0.5 * (horizontal + vertical + sqrt(2.0 * step * step - difference * difference))

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
    def _distance(left, right):
        return hypot(left[0] - right[0], left[1] - right[1])

    def _clear(self, start, end, margin):
        return self._first_blocker(start, end, margin) is None

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
    def _box_entry(start, end, x0, x1, y0, y1):
        entry, exit_time = 0.0, 1.0
        for origin, delta, low, high in (
            (start[0], end[0] - start[0], x0, x1),
            (start[1], end[1] - start[1], y0, y1),
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
        """Nearest inflated obstacle the segment enters, as a disc."""
        best = None
        for center_x, center_y, radius in self.circles:
            entry = self._disc_entry(start, end, center_x, center_y, radius + margin)
            if entry is not None and (best is None or entry < best[0]):
                best = (entry, center_x, center_y, radius + margin)
        for x0, x1, y0, y1 in self.rectangles:
            entry = self._box_entry(start, end, x0 - margin, x1 + margin, y0 - margin, y1 + margin)
            if entry is not None and (best is None or entry < best[0]):
                best = (
                    entry,
                    (x0 + x1) / 2.0,
                    (y0 + y1) / 2.0,
                    0.5 * hypot(x1 - x0, y1 - y0) + margin,
                )
        return best

    def _avoid(self, position, direction, speed):
        """Grazing tangent steering around the first obstacle on the ray."""
        if direction[0] == 0.0 and direction[1] == 0.0:
            return direction
        lookahead = 3.5 + 1.2 * speed
        blocker = self._first_blocker(
            position,
            (position[0] + direction[0] * lookahead, position[1] + direction[1] * lookahead),
            AVOID_SKIN,
        )
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
            escape_x = tangent[0] - 1.1 * toward_x
            escape_y = tangent[1] - 1.1 * toward_y
            magnitude = hypot(escape_x, escape_y)
            return (escape_x / magnitude, escape_y / magnitude) if magnitude > 1e-9 else direction
        sine = self._clamp(radius / distance, -1.0, 1.0)
        cosine = sqrt(max(0.0, 1.0 - sine * sine))
        left = (toward_x * cosine - toward_y * sine, toward_x * sine + toward_y * cosine)
        right = (toward_x * cosine + toward_y * sine, -toward_x * sine + toward_y * cosine)
        if left[0] * direction[0] + left[1] * direction[1] >= right[0] * direction[0] + right[1] * direction[1]:
            return left
        return right

    # ------------------------------------------------------------------
    # low-level control
    # ------------------------------------------------------------------

    def _drive(self, drone, direction, pace=1.0):
        spec = self.specs[drone.drone_type]
        speed = hypot(drone.velocity[0], drone.velocity[1])
        direction = self._avoid(drone.position, direction, speed)
        desired_speed = spec.max_speed * pace
        command_x = GAIN_VELOCITY * (direction[0] * desired_speed - drone.velocity[0]) - GAIN_ACCELERATION * drone.acceleration[0]
        command_y = GAIN_VELOCITY * (direction[1] * desired_speed - drone.velocity[1]) - GAIN_ACCELERATION * drone.acceleration[1]
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
    # flow-field interpolation
    # ------------------------------------------------------------------

    def _build_gradient(self, step):
        size_x, size_y = self.size_x, self.size_y
        field = self.field
        flow_x = [0.0] * (size_x * size_y)
        flow_y = [0.0] * (size_x * size_y)
        for ix in range(size_x):
            for iy in range(size_y):
                cell = ix * size_y + iy
                here = field[cell]
                if here == inf:
                    continue
                gradient_x = self._difference(field, here, cell - size_y if ix > 0 else None, cell + size_y if ix + 1 < size_x else None, step)
                gradient_y = self._difference(field, here, cell - 1 if iy > 0 else None, cell + 1 if iy + 1 < size_y else None, step)
                magnitude = hypot(gradient_x, gradient_y)
                if magnitude > 1e-9:
                    flow_x[cell] = -gradient_x / magnitude
                    flow_y[cell] = -gradient_y / magnitude
        for pass_number in range(2):
            previous_x, previous_y = flow_x, flow_y
            flow_x = [0.0] * (size_x * size_y)
            flow_y = [0.0] * (size_x * size_y)
            for ix in range(size_x):
                for iy in range(size_y):
                    cell = ix * size_y + iy
                    if self.blocked[cell] or previous_x[cell] == 0.0:
                        continue
                    total_x, total_y = previous_x[cell], previous_y[cell]
                    count = 1
                    for nx, ny in ((ix - 1, iy), (ix + 1, iy), (ix, iy - 1), (ix, iy + 1)):
                        if 0 <= nx < size_x and 0 <= ny < size_y:
                            neighbour = nx * size_y + ny
                            if not self.blocked[neighbour] and previous_x[neighbour] != 0.0:
                                total_x += previous_x[neighbour]
                                total_y += previous_y[neighbour]
                                count += 1
                    magnitude = hypot(total_x, total_y)
                    if magnitude > 1e-9:
                        flow_x[cell] = total_x / magnitude
                        flow_y[cell] = total_y / magnitude
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
        raw_x = self._clamp(position[0] / GRID_STEP, 0.0, self.size_x - 1.0)
        raw_y = self._clamp(position[1] / GRID_STEP, 0.0, self.size_y - 1.0)
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
    # threat detection and runner evasion
    # ------------------------------------------------------------------

    def _threats(self, drone, enemies):
        threats = []
        for enemy in enemies:
            rx = drone.position[0] - enemy.position[0]
            ry = drone.position[1] - enemy.position[1]
            distance = hypot(rx, ry)
            if distance > 16.0 or distance < 0.05:
                continue
            vx = drone.velocity[0] - enemy.velocity[0]
            vy = drone.velocity[1] - enemy.velocity[1]
            v2 = vx * vx + vy * vy
            if v2 < 1e-6:
                if distance < 4.0:
                    threats.append((0.0, enemy, rx / distance, ry / distance))
                continue
            time = -(rx * vx + ry * vy) / v2
            if time <= 0.0 or time > DODGE_HORIZON:
                closing = enemy.velocity[0] * (rx / distance) + enemy.velocity[1] * (ry / distance)
                if closing > 1.0 and distance < 12.0:
                    contact = distance / closing
                    if contact <= DODGE_HORIZON:
                        threats.append((contact, enemy, rx / distance, ry / distance))
                continue
            miss_x = rx + vx * time
            miss_y = ry + vy * time
            if hypot(miss_x, miss_y) <= DODGE_RADIUS:
                threats.append((time, enemy, rx / distance, ry / distance))
        threats.sort(key=lambda item: item[0])
        return threats[:2]

    def _dodge(self, drone, direction, threats):
        _, enemy, toward_x, toward_y = threats[0]
        rx = drone.position[0] - enemy.position[0]
        ry = drone.position[1] - enemy.position[1]
        distance = hypot(rx, ry)
        vx = drone.velocity[0] - enemy.velocity[0]
        vy = drone.velocity[1] - enemy.velocity[1]
        v2 = vx * vx + vy * vy
        if v2 < 1e-6:
            time = 0.0
            miss = 0.0
            dodge_x, dodge_y = -toward_y, toward_x
            if dodge_x * direction[0] + dodge_y * direction[1] < 0.0:
                dodge_x, dodge_y = -dodge_x, -dodge_y
        else:
            time = self._clamp(-(rx * vx + ry * vy) / v2, 0.0, DODGE_HORIZON)
            miss_x = rx + vx * time
            miss_y = ry + vy * time
            miss = hypot(miss_x, miss_y)
            if miss > 1e-3:
                dodge_x, dodge_y = miss_x / miss, miss_y / miss
            else:
                dodge_x, dodge_y = -toward_y, toward_x
                if dodge_x * direction[0] + dodge_y * direction[1] < 0.0:
                    dodge_x, dodge_y = -dodge_x, -dodge_y
                miss = 0.0
        if drone.position[1] + dodge_y * 4.0 < 1.0 or drone.position[1] + dodge_y * 4.0 > self.height - 1.0:
            dodge_y = -dodge_y
        proximity = self._clamp(1.0 - distance / 10.0, 0.0, 1.0)
        weight = (1.4 + 1.8 * proximity) * (1.0 - time / DODGE_HORIZON) * (1.0 - miss / DODGE_RADIUS)
        blended_x = direction[0] + weight * dodge_x
        blended_y = direction[1] + weight * dodge_y
        magnitude = hypot(blended_x, blended_y)
        return (blended_x / magnitude, blended_y / magnitude) if magnitude > 1e-9 else direction

    def _steer(self, position, target):
        if self._clear(position, target, 0.4):
            return self._unit(position, target)
        return self._flow_direction(position)

    def _run(self, drone, enemies):
        direction = self._flow_direction(drone.position)
        threats = self._threats(drone, enemies)
        if threats:
            direction = self._dodge(drone, direction, threats)
        return self._drive(drone, direction)

    def _defend_waypoint(self, hunter, slot=None):
        own = self.own_goal
        if own.x_min <= 3.0:
            wait_x = own.x_max + 9.0
        else:
            wait_x = own.x_min - 9.0
        if slot is None:
            slot = hunter.id % 6
        y = own.center[1] + (slot - 2.5) * 2.4
        return (wait_x, self._clamp(y, 3.0, self.height - 3.0))

    def _defend(self, hunter, enemies):
        slot = hunter.id % 6
        waypoint = self._defend_waypoint(hunter, slot)
        if self._distance(hunter.position, waypoint) < 2.5:
            waypoint = self._defend_waypoint(hunter, (slot + 1) % 6)
        direction = self._steer(hunter.position, waypoint)
        threats = self._threats(hunter, enemies)
        if threats:
            direction = self._dodge(hunter, direction, threats)
        return self._drive(hunter, direction, pace=DEFEND_PACE)

    # ------------------------------------------------------------------
    # threat model and interception
    # ------------------------------------------------------------------

    def _predicted_velocity(self, drone, toward):
        spec = self.specs[drone.drone_type]
        ux, uy = self._unit(drone.position, toward)
        return (
            0.62 * drone.velocity[0] + 0.38 * ux * spec.max_speed,
            0.62 * drone.velocity[1] + 0.38 * uy * spec.max_speed,
        )

    def _intercept(self, hunter, target, target_velocity):
        speed = self.specs[hunter.drone_type].max_speed
        ox = target.position[0] - hunter.position[0]
        oy = target.position[1] - hunter.position[1]
        vx, vy = target_velocity
        quadratic = vx * vx + vy * vy - speed * speed
        linear = 2.0 * (ox * vx + oy * vy)
        constant = ox * ox + oy * oy
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
            self._clamp(target.position[0] + vx * time, 0.3, self.width - 0.3),
            self._clamp(target.position[1] + vy * time, 0.3, self.height - 0.3),
        )
        return time, point

    def _catch_time(self, enemy, runner):
        """Time for an enemy FAST to catch one of our SLOW runners."""
        dx = runner.position[0] - enemy.position[0]
        dy = runner.position[1] - enemy.position[1]
        distance = hypot(dx, dy)
        if distance < 0.05:
            return 0.0
        ux, uy = dx / distance, dy / distance
        v_r = self._predicted_velocity(runner, self.goal.center)
        closing = (enemy.velocity[0] - v_r[0]) * ux + (enemy.velocity[1] - v_r[1]) * uy
        if closing <= 0.5:
            return inf
        return (distance - INTERCEPT_RADIUS) / closing

    def _time_to_score(self, enemy):
        distance = self._distance(enemy.position, self.own_goal.center)
        return 1.12 * distance / self.specs[enemy.drone_type].max_speed

    def _runner_score_potential(self, runner, remaining):
        spec = self.specs[runner.drone_type]
        travel = self._path_length(runner.position) / spec.max_speed
        return self._clamp((remaining - travel) / SCORE_WINDOW, 0.0, 1.0)

    def _escort_danger(self, hunter, target, enemies, t_catch):
        """Deep hunts into defended territory: enemy FASTs may reach the hunter first."""
        if target.drone_type is not DroneType.SLOW:
            return 1.0
        in_our_half = target.position[0] < 50.0 if self.team_is_a else target.position[0] > 50.0
        if in_our_half:
            return 1.0
        worst = 1.0
        for enemy in enemies:
            if enemy.drone_type is not DroneType.FAST:
                continue
            if self._distance(enemy.position, hunter.position) > 22.0:
                continue
            enemy_vel = self._predicted_velocity(enemy, hunter.position)
            t_enemy, _ = self._intercept(enemy, hunter, enemy_vel)
            if t_enemy < t_catch - 1.0:
                worst = min(worst, 0.35)
            elif t_enemy < t_catch:
                worst = min(worst, 0.6)
        return worst

    def _intercept_bid(self, hunter, target, remaining, runners, enemies, urgency):
        spec = self.specs[target.drone_type]
        margin = remaining - self._time_to_score(target)
        deny = spec.point_value * self._clamp(margin / SCORE_WINDOW, 0.0, 1.0) * urgency

        protect = 0.0
        if target.drone_type is DroneType.FAST:
            best = 0.0
            for runner in runners:
                t_catch = self._catch_time(target, runner)
                if t_catch < PROTECT_HORIZON:
                    chance = (1.0 - t_catch / PROTECT_HORIZON) * self._runner_score_potential(runner, remaining)
                    if chance > best:
                        best = chance
            protect = SLOW_VALUE * best

        value = deny + protect
        if value <= 0.05:
            return 0.0, None

        target_velocity = self._predicted_velocity(target, self.own_goal.center)
        t_catch, aim = self._intercept(hunter, target, target_velocity)
        if t_catch >= remaining or t_catch >= self._time_to_score(target) + 0.6:
            return 0.0, aim

        confidence = exp(-t_catch / DENY_HORIZON)
        if target.drone_type is DroneType.FAST:
            confidence *= 0.9
        if not self._clear(hunter.position, aim, 0.3):
            confidence *= 0.85
        confidence *= self._escort_danger(hunter, target, enemies, t_catch)
        if self.assignments.get(hunter.id) == target.id:
            confidence *= 1.15
        return value * confidence, aim

    def _defend_value(self, hunter, remaining, urgency):
        if remaining < ENDGAME_HORIZON:
            return self._score_value(hunter, remaining, urgency)
        return DEFEND_VALUE * urgency

    def _score_value(self, hunter, remaining, urgency):
        spec = self.specs[hunter.drone_type]
        travel = 1.15 * self._path_length(hunter.position) / spec.max_speed
        reachable = self._clamp((remaining - travel) / SCORE_WINDOW, 0.0, 1.0)
        return SCORE_WEIGHT * spec.point_value * reachable * urgency

    def _match_hunters(self, hunters, enemies, runners, remaining, urgency):
        if not hunters:
            return {}, {}
        count_hunters = len(hunters)
        count_enemies = len(enemies)
        columns = count_enemies + count_hunters
        cost = [[0.0] * columns for _ in range(count_hunters)]
        aims = [[None] * columns for _ in range(count_hunters)]
        for row, hunter in enumerate(hunters):
            for column, target in enumerate(enemies):
                bid, aim = self._intercept_bid(hunter, target, remaining, runners, enemies, urgency)
                cost[row][column] = -bid
                aims[row][column] = aim
            fallback = self._defend_value(hunter, remaining, urgency)
            for slot in range(count_hunters):
                cost[row][count_enemies + slot] = -fallback if slot == row else 1e9

        row_indices, column_indices = linear_sum_assignment(cost)
        targets, target_aims = {}, {}
        for row, column in zip(row_indices, column_indices, strict=True):
            if column < count_enemies:
                bid = -cost[row][column]
                if bid > max(0.03, self._defend_value(hunters[row], remaining, urgency)):
                    targets[hunters[row].id] = enemies[column].id
                    target_aims[hunters[row].id] = aims[row][column]
        return targets, target_aims

    # ------------------------------------------------------------------
    # hunter pursuit
    # ------------------------------------------------------------------

    def _pursue(self, hunter, target, aim):
        distance = self._distance(hunter.position, target.position)
        if distance < 2.0:
            aim = (
                target.position[0] + target.velocity[0] * 0.12,
                target.position[1] + target.velocity[1] * 0.12,
            )
        elif aim is None or not self._clear(hunter.position, aim, 0.3):
            aim = target.position
        direction = self._unit(hunter.position, aim)
        return self._drive(hunter, direction)

    # ------------------------------------------------------------------
    # control loop
    # ------------------------------------------------------------------

    def step(self, state):
        remaining = max(0.0, MATCH_DURATION - state.time)
        own = [drone for drone in state.own_drones if drone.status is DroneStatus.ACTIVE]
        enemies = [drone for drone in state.opponent_drones if drone.status is DroneStatus.ACTIVE]
        runners = [drone for drone in own if drone.drone_type is DroneType.SLOW]
        hunters = [drone for drone in own if drone.drone_type is DroneType.FAST]
        enemy_by_id = {drone.id: drone for drone in enemies}

        urgency = self._clamp(1.0 + 0.09 * (state.opponent_score - state.own_score), 0.65, 1.7)
        targets, aims = self._match_hunters(hunters, enemies, runners, remaining, urgency)
        self.assignments = targets

        actions = {}
        for drone in runners:
            actions[drone.id] = self._run(drone, enemies)
        for drone in hunters:
            target = enemy_by_id.get(targets.get(drone.id))
            if target is None:
                if remaining < ENDGAME_HORIZON:
                    actions[drone.id] = self._run(drone, enemies)
                else:
                    actions[drone.id] = self._defend(drone, enemies)
            else:
                actions[drone.id] = self._pursue(drone, target, aims.get(drone.id))
        return actions
