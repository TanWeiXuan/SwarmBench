"""SwarmBench submission: visibility-graph routing with Hungarian auction.

Model: Qwen 3.6 35B A3B
Author: renj1ete0

Design
------
Three core ideas:

1. *Visibility-graph routing.*  ``initialize`` inflates obstacles into convex
   polygons and builds a visibility graph over their vertices plus the goal
   mouth.  A single Dijkstra pass from the goal labels every vertex with its
   true shortest path length.  Each control tick does a local visibility scan
   from the drone's current position to pick the visible vertex that minimises
   "hop here then walk the precomputed shortest remainder" — exact shortest
   paths with no grid resolution error.

2. *Cover-seeking evasion.*  A pursued runner does not just dodge in velocity
   space: among visibility-graph vertices that keep it making reasonable
   progress, it favours ones an approaching hunter cannot see, so a detour
   behind an obstacle is chosen over one that stays in the open.

3. *Hungarian auction for interception.*  Every enemy is priced by its point
   value damped by scoring urgency and raid threat, then discounted by how
   confidently a given FAST drone can reach it.  Each hunter also gets a
   private "attack" column for rushing the goal, so hunters convert to
   scorers exactly when denial stops paying for itself.

Borrowed elements
-----------------
- Visibility-graph routing with Dijkstra precomputation: inspired by the
  approach in submissions/renj1ete0/claude_sonnet_5_max.py, but re-implemented
  independently with my own constants and integration.
- Hungarian auction with deny/raid values: inspired by the general approach
  in submissions/renj1ete0/claude_sonnet_5_max.py, but with my own cost
  formula and bidding logic.
- Cover-seeking evasion: inspired by the concept in submissions/renj1ete0/claude_sonnet_5_max.py,
  but re-implemented independently.
- Corner braking and predicted-velocity interception: standard control theory
  concepts, independently implemented.
"""

from __future__ import annotations

from heapq import heappop, heappush
from math import cos, exp, hypot, inf, pi, sin, sqrt

from scipy.optimize import linear_sum_assignment

from swarmbench import BaseSwarmController, CircleObstacle, DroneStatus, DroneType


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MATCH_DURATION = 90.0
CLEARANCE = 0.62
"""Drone radius (0.25) plus planning margin."""

CIRCLE_SIDES = 8
"""Vertices for circumscribing circular obstacles."""

GAIN_VELOCITY = 3.3
GAIN_ACCELERATION = 0.20
CORNER_BRAKE_ANGLE = 0.55
"""Radians of heading change beyond which speed is throttled."""

WALL_MARGIN = 1.5
AVOID_SKIN = 0.5

DODGE_HORIZON = 3.2
DODGE_RADIUS = 2.0
DODGE_REPULSION_RADIUS = 9.0
COVER_DETOUR_TOLERANCE = 1.35
"""A shielded vertex is accepted if its route cost is within this factor."""

DENY_HORIZON = 6.0
RAID_VALUE = 5.0
RAID_HORIZON = 11.0
SCORE_WEIGHT = 2.0


# ---------------------------------------------------------------------------
# Swarm controller
# ---------------------------------------------------------------------------

class SwarmController(BaseSwarmController):
    """Runners follow exact visibility-graph routes; hunters bid on threats."""

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
        self.aims = {}

        self.circles = tuple(
            (ob.center[0], ob.center[1], ob.radius)
            for ob in game_info.obstacles
            if isinstance(ob, CircleObstacle)
        )
        self.rectangles = tuple(
            (ob.x_min, ob.x_max, ob.y_min, ob.y_max)
            for ob in game_info.obstacles
            if not isinstance(ob, CircleObstacle)
        )

        # Goal mouth point for routing
        if self.goal.x_min <= 1e-6:
            self.goal_mouth = (self.goal.x_max, self.goal.center[1])
        else:
            self.goal_mouth = (self.goal.x_min, self.goal.center[1])

        self._build_visibility_graph()

        # Lane spread for goal approach
        ordered = sorted(game_info.own_initial_drones, key=lambda d: (d.position[1], d.id))
        low = self.goal.y_min + 0.9
        high = self.goal.y_max - 0.9
        span = max(0.0, high - low)
        count = max(1, len(ordered))
        self.lanes = {
            d.id: low + span * (rank + 0.5) / count
            for rank, d in enumerate(ordered)
        }

    # ------------------------------------------------------------------
    # visibility graph
    # ------------------------------------------------------------------

    def _build_visibility_graph(self):
        """Build visibility graph over obstacle vertices + goal mouth."""
        nodes = [self.goal_mouth]
        for cx, cy, r in self.circles:
            reach = r / cos(pi / CIRCLE_SIDES) + CLEARANCE
            for i in range(CIRCLE_SIDES):
                angle = 2.0 * pi * i / CIRCLE_SIDES
                nodes.append((cx + reach * cos(angle), cy + reach * sin(angle)))
        for x0, x1, y0, y1 in self.rectangles:
            nodes.append((x0 - CLEARANCE, y0 - CLEARANCE))
            nodes.append((x1 + CLEARANCE, y0 - CLEARANCE))
            nodes.append((x1 + CLEARANCE, y1 + CLEARANCE))
            nodes.append((x0 - CLEARANCE, y1 + CLEARANCE))

        # Keep nodes inside arena
        nodes = [
            n for n in nodes
            if -1.0 <= n[0] <= self.width + 1.0 and -1.0 <= n[1] <= self.height + 1.0
        ]
        self.nodes = nodes

        # Build adjacency
        count = len(nodes)
        adj = [[] for _ in range(count)]
        for i in range(count):
            for j in range(i + 1, count):
                if self._clear(nodes[i], nodes[j], CLEARANCE):
                    w = hypot(nodes[i][0] - nodes[j][0], nodes[i][1] - nodes[j][1])
                    adj[i].append((j, w))
                    adj[j].append((i, w))
        self.adj = adj

        # Dijkstra from goal mouth (node 0)
        dist = [inf] * count
        dist[0] = 0.0
        frontier = [(0.0, 0)]
        while frontier:
            c, node = heappop(frontier)
            if c > dist[node]:
                continue
            for nb, w in adj[node]:
                cand = c + w
                if cand < dist[nb]:
                    dist[nb] = cand
                    heappush(frontier, (cand, nb))
        self.goal_dist = dist

    # ------------------------------------------------------------------
    # geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unit(origin, target):
        dx = target[0] - origin[0]
        dy = target[1] - origin[1]
        m = hypot(dx, dy)
        if m < 1e-9:
            return 0.0, 0.0
        return dx / m, dy / m

    @staticmethod
    def _clamp(v, lo, hi):
        return lo if v < lo else (hi if v > hi else v)

    @staticmethod
    def _seg_hits_circle(start, end, cx, cy, r):
        ox = start[0] - cx
        oy = start[1] - cy
        if ox * ox + oy * oy <= r * r:
            return True
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        q = dx * dx + dy * dy
        if q < 1e-12:
            return False
        lin = 2.0 * (ox * dx + oy * dy)
        con = ox * ox + oy * oy - r * r
        disc = lin * lin - 4.0 * q * con
        if disc < 0.0:
            return False
        root = sqrt(disc)
        near = (-lin - root) / (2.0 * q)
        far = (-lin + root) / (2.0 * q)
        return near <= 1.0 and far >= 0.0

    @staticmethod
    def _seg_hits_box(start, end, x0, x1, y0, y1):
        t0, t1 = 0.0, 1.0
        for o, d, lo, hi in ((start[0], end[0] - start[0], x0, x1),
                              (start[1], end[1] - start[1], y0, y1)):
            if abs(d) < 1e-12:
                if o < lo or o > hi:
                    return False
                continue
            f = (lo - o) / d
            s = (hi - o) / d
            if f > s:
                f, s = s, f
            t0 = max(t0, f)
            t1 = min(t1, s)
            if t0 > t1:
                return False
        return t1 >= 0.0 and t0 <= 1.0

    def _clear(self, start, end, margin):
        for cx, cy, r in self.circles:
            if self._seg_hits_circle(start, end, cx, cy, r + margin):
                return False
        for x0, x1, y0, y1 in self.rectangles:
            if self._seg_hits_box(start, end, x0 - margin, x1 + margin, y0 - margin, y1 + margin):
                return False
        return True

    def _nearest_blocker(self, pos, direction, lookahead):
        """Find nearest obstacle center along a ray."""
        end = (pos[0] + direction[0] * lookahead, pos[1] + direction[1] * lookahead)
        best = None
        for cx, cy, r in self.circles:
            reach = r + AVOID_SKIN
            if self._seg_hits_circle(pos, end, cx, cy, reach):
                d = hypot(cx - pos[0], cy - pos[1])
                if best is None or d < best[0]:
                    best = (d, cx, cy, reach)
        for x0, x1, y0, y1 in self.rectangles:
            if self._seg_hits_box(pos, end, x0 - AVOID_SKIN, x1 + AVOID_SKIN, y0 - AVOID_SKIN, y1 + AVOID_SKIN):
                cx = (x0 + x1) / 2.0
                cy = (y0 + y1) / 2.0
                reach = 0.5 * hypot(x1 - x0, y1 - y0) + AVOID_SKIN
                d = hypot(cx - pos[0], cy - pos[1])
                if best is None or d < best[0]:
                    best = (d, cx, cy, reach)
        return best

    # ------------------------------------------------------------------
    # routing
    # ------------------------------------------------------------------

    def _route_options(self, pos, goal, margin):
        """Every reachable aim point from here, cheapest-total-route first.

        Returns list of (route_cost, point).
        """
        opts = []
        if self._clear(pos, goal, margin):
            opts.append((hypot(goal[0] - pos[0], goal[1] - pos[1]), goal))
        for i, node in enumerate(self.nodes):
            if self.goal_dist[i] == inf:
                continue
            if self._clear(pos, node, margin):
                cost = hypot(node[0] - pos[0], node[1] - pos[1]) + self.goal_dist[i]
                opts.append((cost, node))
        opts.sort(key=lambda x: x[0])
        return opts

    def _route_target(self, pos, goal, threats):
        """Next aim point, biased away from watching enemies."""
        opts = self._route_options(pos, goal, CLEARANCE)
        if not opts:
            return goal
        best_cost = opts[0][0]

        if threats:
            for cost, pt in opts:
                if cost > best_cost * COVER_DETOUR_TOLERANCE:
                    break
                if all(not self._clear(t, pt, 0.0) for t in threats):
                    return pt

        return opts[0][1]

    def _path_length(self, pos, goal):
        opts = self._route_options(pos, goal, CLEARANCE)
        if opts:
            return opts[0][0]
        return hypot(pos[0] - goal[0], pos[1] - goal[1])

    # ------------------------------------------------------------------
    # low-level control
    # ------------------------------------------------------------------

    def _avoid(self, pos, direction, speed):
        """Obstacle avoidance with tangent escape."""
        if direction[0] == 0.0 and direction[1] == 0.0:
            return direction
        lookahead = 1.4 + 0.75 * speed
        blocker = self._nearest_blocker(pos, direction, lookahead)
        if blocker is None:
            return direction
        _, cx, cy, r = blocker
        tx = cx - pos[0]
        ty = cy - pos[1]
        d = hypot(tx, ty)
        if d < 1e-6:
            return direction
        tx /= d
        ty /= d
        if d <= r:
            tan = (-ty, tx)
            if tan[0] * direction[0] + tan[1] * direction[1] < 0.0:
                tan = (ty, -tx)
            esc = (tan[0] - 0.7 * tx, tan[1] - 0.7 * ty)
            m = hypot(esc[0], esc[1])
            return (esc[0] / m, esc[1] / m) if m > 1e-9 else direction
        sine = self._clamp(r / d, -1.0, 1.0)
        cosine = sqrt(max(0.0, 1.0 - sine * sine))
        left = (tx * cosine - ty * sine, tx * sine + ty * cosine)
        right = (tx * cosine + ty * sine, -tx * sine + ty * cosine)
        if left[0] * direction[0] + left[1] * direction[1] >= right[0] * direction[0] + right[1] * direction[1]:
            return left
        return right

    def _drive(self, drone, aim, threats=None):
        """Compute acceleration toward aim point."""
        spec = self.specs[drone.drone_type]
        pos = drone.position
        speed = hypot(drone.velocity[0], drone.velocity[1])
        direction = self._unit(pos, aim)
        if threats:
            direction = self._kinematic_dodge(drone, direction, threats)
        direction = self._avoid(pos, direction, speed)

        # Corner braking
        target_speed = spec.max_speed
        if speed > 0.4:
            heading = (drone.velocity[0] / speed, drone.velocity[1] / speed)
            turn = heading[0] * direction[0] + heading[1] * direction[1]
            if turn < 1.0 - CORNER_BRAKE_ANGLE:
                target_speed *= max(0.35, 0.5 + 0.5 * turn)

        desired_x = direction[0] * target_speed
        desired_y = direction[1] * target_speed
        cmd_x = GAIN_VELOCITY * (desired_x - drone.velocity[0]) - GAIN_ACCELERATION * drone.acceleration[0]
        cmd_y = GAIN_VELOCITY * (desired_y - drone.velocity[1]) - GAIN_ACCELERATION * drone.acceleration[1]

        # Wall margins
        if pos[1] < WALL_MARGIN:
            cmd_y += spec.max_acceleration * (WALL_MARGIN - pos[1]) / WALL_MARGIN
        elif pos[1] > self.height - WALL_MARGIN:
            cmd_y -= spec.max_acceleration * (pos[1] - self.height + WALL_MARGIN) / WALL_MARGIN
        if pos[0] < WALL_MARGIN:
            cmd_x += spec.max_acceleration * (WALL_MARGIN - pos[0]) / WALL_MARGIN
        elif pos[0] > self.width - WALL_MARGIN:
            cmd_x -= spec.max_acceleration * (pos[0] - self.width + WALL_MARGIN) / WALL_MARGIN

        # Clamp to max acceleration
        mag = hypot(cmd_x, cmd_y)
        if mag > spec.max_acceleration:
            s = spec.max_acceleration / mag
            cmd_x *= s
            cmd_y *= s
        return cmd_x, cmd_y

    # ------------------------------------------------------------------
    # threat model
    # ------------------------------------------------------------------

    def _predicted_velocity(self, drone, goal):
        spec = self.specs[drone.drone_type]
        tx, ty = self._unit(drone.position, goal)
        return (
            0.6 * drone.velocity[0] + 0.4 * tx * spec.max_speed,
            0.6 * drone.velocity[1] + 0.4 * ty * spec.max_speed,
        )

    def _intercept_time(self, hunter, target_pos, target_vel):
        """Solve for time to intercept. Returns (time, intercept_point)."""
        speed = self.specs[hunter.drone_type].max_speed
        ox = target_pos[0] - hunter.position[0]
        oy = target_pos[1] - hunter.position[1]
        vx = target_vel[0]
        vy = target_vel[1]
        q = vx * vx + vy * vy - speed * speed
        lin = 2.0 * (ox * vx + oy * vy)
        con = ox * ox + oy * oy

        t = None
        if abs(q) < 1e-9:
            if lin < -1e-9:
                t = -con / lin
        else:
            disc = lin * lin - 4.0 * q * con
            if disc >= 0.0:
                root = sqrt(disc)
                roots = [v for v in ((-lin - root) / (2.0 * q), (-lin + root) / (2.0 * q)) if v >= 0.0]
                if roots:
                    t = min(roots)
        if t is None or t > 30.0:
            t = sqrt(con) / max(0.5, speed)
        t += 0.15
        px = self._clamp(target_pos[0] + vx * t, 0.3, self.width - 0.3)
        py = self._clamp(target_pos[1] + vy * t, 0.3, self.height - 0.3)
        return t, (px, py)

    def _time_to_own_goal(self, enemy):
        d = hypot(enemy.position[0] - self.own_goal.center[0],
                  enemy.position[1] - self.own_goal.center[1])
        return 1.12 * d / self.specs[enemy.drone_type].max_speed

    def _raid_chance(self, enemy, runners):
        if not runners or enemy.drone_type is DroneType.SLOW:
            return 0.0
        best = inf
        for runner in runners:
            vel = self._predicted_velocity(runner, self.goal_mouth)
            t, _ = self._intercept_time(enemy, runner.position, vel)
            if t < best:
                best = t
        if best == inf:
            return 0.0
        return self._clamp(1.0 - best / RAID_HORIZON, 0.0, 1.0)

    # ------------------------------------------------------------------
    # interception auction
    # ------------------------------------------------------------------

    def _deny_value(self, enemy, remaining, runners):
        """Value of intercepting this enemy."""
        spec = self.specs[enemy.drone_type]
        margin = remaining - self._time_to_own_goal(enemy)
        scoring = self._clamp(margin / 2.5, 0.0, 1.0)
        return spec.point_value * scoring + RAID_VALUE * self._raid_chance(enemy, runners)

    def _intercept_bid(self, hunter, enemy, remaining, runners):
        """Bid for this hunter-target pair. Returns (bid_value, aim_point)."""
        value = self._deny_value(enemy, remaining, runners)
        if value <= 0.05:
            return 0.0, None
        vel = self._predicted_velocity(enemy, self.own_goal.center)
        t, pt = self._intercept_time(hunter, enemy.position, vel)
        if t >= remaining:
            return 0.0, pt
        confidence = exp(-t / DENY_HORIZON)
        if t > self._time_to_own_goal(enemy) + 0.6:
            confidence *= 0.12
        if not self._clear(hunter.position, pt, 0.3):
            confidence *= 0.85
        if self.assignments.get(hunter.id) == enemy.id:
            confidence *= 1.15
        return value * confidence, pt

    def _attack_bid(self, hunter, remaining, urgency):
        """Bid for rushing the goal instead of intercepting."""
        spec = self.specs[hunter.drone_type]
        travel = 1.15 * self._path_length(hunter.position, self.goal_mouth) / spec.max_speed
        reachable = self._clamp((remaining - travel) / 2.5, 0.0, 1.0)
        return SCORE_WEIGHT * spec.point_value * reachable * urgency

    def _clear_the_board(self, hunters, enemies, runners, remaining, urgency):
        """Hungarian auction over hunters x (enemies + attack slots)."""
        if not hunters:
            return {}, {}
        rows = len(hunters)
        ec = len(enemies)
        cols = ec + rows
        cost = [[0.0] * cols for _ in range(rows)]
        points = [[None] * cols for _ in range(rows)]

        for row, hunter in enumerate(hunters):
            for col, enemy in enumerate(enemies):
                bid, pt = self._intercept_bid(hunter, enemy, remaining, runners)
                cost[row][col] = -bid
                points[row][col] = pt
            fallback = self._attack_bid(hunter, remaining, urgency)
            for slot in range(rows):
                cost[row][ec + slot] = -fallback if slot == row else 1e6

        ri, ci = linear_sum_assignment(cost)
        targets, aims = {}, {}
        for row, col in zip(ri, ci, strict=True):
            if col < ec and -cost[row][col] > max(0.03, self._attack_bid(hunters[row], remaining, urgency)):
                targets[hunters[row].id] = enemies[col].id
                aims[hunters[row].id] = points[row][col]
        return targets, aims

    # ------------------------------------------------------------------
    # behaviours
    # ------------------------------------------------------------------

    def _goal_point(self, drone):
        lane = self.lanes.get(drone.id, self.goal_mouth[1])
        return (self.goal_mouth[0], self._clamp(lane, self.goal.y_min + 0.5, self.goal.y_max - 0.5))

    def _threats(self, drone, enemies):
        """Find enemies that are closing on this drone within dodge horizon."""
        found = []
        for enemy in enemies:
            ox = enemy.position[0] - drone.position[0]
            oy = enemy.position[1] - drone.position[1]
            if abs(ox) > 14.0 or abs(oy) > 14.0:
                continue
            cx = enemy.velocity[0] - drone.velocity[0]
            cy = enemy.velocity[1] - drone.velocity[1]
            cl = cx * cx + cy * cy
            if cl < 1e-6:
                continue
            t = -(ox * cx + oy * cy) / cl
            if t <= 0.0 or t > DODGE_HORIZON:
                continue
            miss = hypot(ox + cx * t, oy + cy * t)
            if miss <= DODGE_RADIUS:
                found.append((t, enemy))
        found.sort(key=lambda x: x[0])
        return [e.position for _, e in found[:2]]

    def _kinematic_dodge(self, drone, direction, threats):
        """Blend direction away from nearest threat."""
        if not threats:
            return direction
        closest = min(threats, key=lambda p: hypot(p[0] - drone.position[0], p[1] - drone.position[1]))
        ox = drone.position[0] - closest[0]
        oy = drone.position[1] - closest[1]
        d = hypot(ox, oy)
        if d < 1e-6 or d > DODGE_REPULSION_RADIUS:
            return direction
        w = self._clamp(1.0 - d / DODGE_REPULSION_RADIUS, 0.0, 1.0) * 0.8
        bx = direction[0] + w * ox / d
        by = direction[1] + w * oy / d
        m = hypot(bx, by)
        return (bx / m, by / m) if m > 1e-9 else direction

    def _run(self, drone, enemies):
        """Runner behaviour: route to goal with cover-seeking evasion."""
        goal = self._goal_point(drone)
        threats = self._threats(drone, enemies)
        aim = self._route_target(drone.position, goal, threats)
        return self._drive(drone, aim, threats)

    def _pursue(self, hunter, enemy, aim):
        """Hunter behaviour: drive toward intercept point."""
        d = hypot(enemy.position[0] - hunter.position[0],
                  enemy.position[1] - hunter.position[1])
        if d < 2.0:
            aim = (enemy.position[0] + enemy.velocity[0] * 0.12,
                   enemy.position[1] + enemy.velocity[1] * 0.12)
        elif aim is None:
            aim = enemy.position
        return self._drive(hunter, aim)

    # ------------------------------------------------------------------
    # main step
    # ------------------------------------------------------------------

    def step(self, state):
        time = state.time
        remaining = max(0.0, MATCH_DURATION - time)
        urgency = remaining / MATCH_DURATION

        enemies = [d for d in state.opponent_drones if d.status is DroneStatus.ACTIVE]
        own = [d for d in state.own_drones if d.status is DroneStatus.ACTIVE]
        fast = [d for d in own if d.drone_type is DroneType.FAST]
        slow = [d for d in own if d.drone_type is DroneType.SLOW]

        enemy_by_id = {d.id: d for d in enemies}

        # --- Auction: match FAST drones to enemies ---
        self.assignments = {}
        self.aims = {}
        if fast and enemies:
            targets, aims = self._clear_the_board(fast, enemies, slow, remaining, urgency)
            self.assignments = targets
            self.aims = aims

        # --- Compute actions ---
        actions = {}
        for drone in own:
            if drone.drone_type is DroneType.FAST and drone.id in self.assignments:
                # Hunter
                enemy = enemy_by_id[self.assignments[drone.id]]
                aim = self.aims.get(drone.id)
                actions[drone.id] = self._pursue(drone, enemy, aim)
            else:
                # Runner
                actions[drone.id] = self._run(drone, enemies)
        return actions


SwarmController = SwarmController
