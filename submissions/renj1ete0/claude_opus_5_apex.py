"""Apex - a value-driven swarm controller for SwarmBench.

Author: renj1ete0.  Written by the coding agent **Claude Opus 5** (Anthropic).
The strategy below was derived from the engine source and the game spec; no
strategy, algorithm, structure, constants or code was taken from another
controller.

Why the controller is shaped this way
-------------------------------------
Every elimination in this game is mutual, so the final score difference is just
the value each side fails to deliver.  Enumerating the four possible trades
settles the whole design:

    our FAST  x their SLOW   ->  +4      (we spend 1 to deny 5)
    our SLOW  x their FAST   ->  -4
    FAST x FAST, SLOW x SLOW ->   0      exactly neutral

So only two events matter, and everything else - obstacle deaths and drones
still idling at the whistle - is pure leakage.  Two consequences drive the code:

1. *Escorting beats hunting.*  A hunter always spends itself; an escort is an
   option that is only exercised when a hunter actually arrives, and otherwise
   still banks its own point at the goal.  FAST drones therefore default to
   shielding a SLOW drone and peel off to intercept only when a specific enemy
   becomes the better buy.
2. *Geometry decides who wins a trade.*  A FAST drone catches a SLOW drone with
   certainty (2x speed) but a FAST tail-chase of another FAST never converges,
   so enemy SLOW drones are *pursued* while enemy FAST drones are *interposed*:
   the shield sits on the segment between the hunter and the point its victim
   will occupy when it arrives, where the head-on closure makes the trade
   reliable.  A SLOW drone that cannot outrun its hunter maximises time to
   contact and drags the hunter onto a friendly FAST drone - stalling is free
   because the crossing only needs half the match.

Each control step prices every (own drone, task) pair in those differential
points - engage an enemy, shield a SLOW drone, or run for the goal - and solves
one small linear-sum assignment over the result, with hysteresis so assignments
do not chatter.

Navigation and safety
---------------------
Obstacles kill outright, and the generator keeps every obstacle inside
x in [20, 80], y in [2, 58].  Routing uses a geodesic distance field (Dijkstra
over a 0.5 m grid weighted by clearance) built once in initialize(), string
pulled against exact line-of-sight tests.  Under that sits a tangent-steering
layer plus an emergency brake that no tactical term can override; steering
always answers an obstacle with a full-speed heading change rather than braking,
because slowing down only shortens the lever arm the lateral acceleration has.

Nothing may be wasted: a drone that can no longer reach the goal in time is
re-tasked as an interceptor, and a drone that is about to be intercepted anyway
rams the most valuable enemy still in reach.

Every stage degrades instead of raising - a failed field build, a missing SciPy,
or an unexpected state all fall back to simpler behaviour - because an exception
or a 5 s stall forfeits the match outright.  A step is O(n^2) over at most 40
drones and measures well under 40 ms against the 500 ms soft deadline.
"""

from __future__ import annotations

import math

from swarmbench import (
    BaseSwarmController,
    CircleObstacle,
    DroneStatus,
    DroneType,
)

# ---------------------------------------------------------------------------
# Engine constants (mirrored from swarmbench.api.types so the controller never
# depends on values it is not handed).
# ---------------------------------------------------------------------------
_ARENA_W = 100.0
_ARENA_H = 60.0
_DRONE_R = 0.25
_KILL_R = 0.75
_MATCH_T = 90.0

_FAST_SPEED, _FAST_ACC, _FAST_VALUE = 5.0, 4.0, 1.0
_SLOW_SPEED, _SLOW_ACC, _SLOW_VALUE = 2.5, 2.0, 5.0

# Planning margins.  The generator guarantees a 0.25 + 0.35 m corridor, so the
# routing layer blocks a little less than that and prices the rest as cost.
_GRID = 0.5
_BLOCK_PAD = 0.45          # grid cells this close to a surface are unusable
_ROUTE_PAD = 0.50          # line-of-sight inflation for waypoint smoothing
_STEER_PAD = 0.95          # tangent steering keeps this much off the surface
_PANIC_PAD = 0.70          # hard "get out" band
_LOOK_BASE = 3.0           # tangent-steering lookahead, metres
_LOOK_RATE = 1.9           # ... plus this many seconds of travel
_BRAKE_CAP = 2.6           # longest track the emergency layer looks along
_BRAKE_PAD = 0.60          # reaction slack added to the stopping time

# Tactical weights.  Values are expressed in "score differential" points: a
# FAST(1) trading itself for a SLOW(5) is worth _DIFF * 5.
_DIFF = 0.8
_THREAT_TAU = 12.0         # how fast confidence in a distant threat decays
_ENGAGE_TAU = 20.0         # discount on our own travel time to an engagement
_ESCORT_W = 1.3            # relative appetite for holding FAST drones in reserve
_ESCORT_GAP = 1.4          # how far in front of its charge an escort rides
_SHIELD_RANGE = 14.0       # threat horizon that turns an escort into a shield
_SHIELD_LEAD = 4.0         # seconds of charge motion a shield aims off by
_PANIC_T = 3.5             # a SLOW under this time-to-contact stops attacking
_PANIC_RESERVE = 10.0      # ... but only while it can still spare the seconds
_PANIC_PULL = 0.75         # how hard a fleeing SLOW runs toward a friendly FAST
_BLOCK_COMMIT = 2.4        # seconds-to-contact at which a shield charges in
_SNAP_KILL = 2.0           # escorts break off for a SLOW they can reach this fast
_SECOND_WAVE = 0.6         # value of the follow-up attacker on a shielded SLOW
_MIN_CONNECT = 0.35        # floor on the odds of reaching a shielded SLOW
_INTERPOSE_FRAC = 0.5      # where a blocker sits between hunter and victim
_INTERPOSE_CAP = 7.0
_COMMIT_RANGE = 6.0        # blockers switch to lead pursuit inside this range
_STICKY_BONUS = 0.35       # hysteresis on the assignment
_TRACK_GAIN = 2.6          # velocity-tracking proportional gain
_NO_TASK = 1.0e5           # cost of an impossible (row, column) pairing


def _clamp(value: float, low: float, high: float) -> float:
    return low if value < low else (high if value > high else value)


def _ramp(value: float, low: float, high: float) -> float:
    if value <= low:
        return 0.0
    if value >= high:
        return 1.0
    return (value - low) / (high - low)


def _wrap(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def _catch_time(px, py, speed, tx, ty, tvx, tvy, radius=_KILL_R):
    """Earliest time a pursuer at constant `speed` closes to `radius` of a target
    drifting at (tvx, tvy).  ``None`` when the quadratic has no positive root."""
    dx, dy = tx - px, ty - py
    gap = math.hypot(dx, dy)
    if gap <= radius:
        return 0.0
    scale = (gap - radius) / gap
    dx *= scale
    dy *= scale
    a = tvx * tvx + tvy * tvy - speed * speed
    b = 2.0 * (dx * tvx + dy * tvy)
    c = dx * dx + dy * dy
    if -1e-9 < a < 1e-9:
        if b > -1e-9:
            return None
        return -c / b
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return None
    root = math.sqrt(disc)
    best = None
    for value in ((-b - root) / (2.0 * a), (-b + root) / (2.0 * a)):
        if value > 0.0 and (best is None or value < best):
            best = value
    return best


class _Obstacle:
    """Circle or axis-aligned rectangle with the queries the controller needs."""

    __slots__ = ("circle", "cx", "cy", "r", "x0", "x1", "y0", "y1")

    def __init__(self, raw) -> None:
        if isinstance(raw, CircleObstacle):
            self.circle = True
            self.cx = float(raw.center[0])
            self.cy = float(raw.center[1])
            self.r = float(raw.radius)
            self.x0, self.x1 = self.cx - self.r, self.cx + self.r
            self.y0, self.y1 = self.cy - self.r, self.cy + self.r
        else:
            self.circle = False
            self.x0, self.x1 = float(raw.x_min), float(raw.x_max)
            self.y0, self.y1 = float(raw.y_min), float(raw.y_max)
            self.cx = 0.5 * (self.x0 + self.x1)
            self.cy = 0.5 * (self.y0 + self.y1)
            self.r = 0.5 * math.hypot(self.x1 - self.x0, self.y1 - self.y0)

    def distance(self, x: float, y: float) -> float:
        """Signed distance to the surface, negative inside."""
        if self.circle:
            return math.hypot(x - self.cx, y - self.cy) - self.r
        dx = self.x0 - x
        if x - self.x1 > dx:
            dx = x - self.x1
        dy = self.y0 - y
        if y - self.y1 > dy:
            dy = y - self.y1
        if dx > 0.0 or dy > 0.0:
            return math.hypot(max(dx, 0.0), max(dy, 0.0))
        return -min(x - self.x0, self.x1 - x, y - self.y0, self.y1 - y)

    def grid_distance(self, np, gx, gy):
        if self.circle:
            return np.hypot(gx - self.cx, gy - self.cy) - self.r
        dx = np.maximum(np.maximum(self.x0 - gx, gx - self.x1), 0.0)
        dy = np.maximum(np.maximum(self.y0 - gy, gy - self.y1), 0.0)
        outside = np.hypot(dx, dy)
        inside = -np.minimum(
            np.minimum(gx - self.x0, self.x1 - gx),
            np.minimum(gy - self.y0, self.y1 - gy),
        )
        return np.where((dx > 0.0) | (dy > 0.0), outside, inside)

    def hits_segment(self, ax, ay, bx, by, pad) -> bool:
        if self.circle:
            radius = self.r + pad
            ex, ey = bx - ax, by - ay
            fx, fy = ax - self.cx, ay - self.cy
            c = fx * fx + fy * fy - radius * radius
            if c <= 0.0:
                return True
            a = ex * ex + ey * ey
            if a <= 1e-12:
                return False
            b = 2.0 * (fx * ex + fy * ey)
            disc = b * b - 4.0 * a * c
            if disc < 0.0:
                return False
            t = (-b - math.sqrt(disc)) / (2.0 * a)
            return 0.0 <= t <= 1.0
        lo_x, hi_x = self.x0 - pad, self.x1 + pad
        lo_y, hi_y = self.y0 - pad, self.y1 + pad
        enter, leave = 0.0, 1.0
        for origin, delta, low, high in (
            (ax, bx - ax, lo_x, hi_x),
            (ay, by - ay, lo_y, hi_y),
        ):
            if -1e-12 < delta < 1e-12:
                if origin < low or origin > high:
                    return False
                continue
            t0 = (low - origin) / delta
            t1 = (high - origin) / delta
            if t0 > t1:
                t0, t1 = t1, t0
            if t0 > enter:
                enter = t0
            if t1 < leave:
                leave = t1
            if enter > leave:
                return False
        return enter <= 1.0

    def cone(self, x: float, y: float, pad: float):
        """Bearing to the centre plus the angular span the padded body occupies."""
        base = math.atan2(self.cy - y, self.cx - x)
        if self.circle:
            gap = math.hypot(self.cx - x, self.cy - y)
            radius = self.r + pad
            if gap <= radius + 1e-6:
                return base, -1.4, 1.4
            return base, -math.asin(min(0.999, radius / gap)), math.asin(min(0.999, radius / gap))
        lo_x, hi_x = self.x0 - pad, self.x1 + pad
        lo_y, hi_y = self.y0 - pad, self.y1 + pad
        if lo_x <= x <= hi_x and lo_y <= y <= hi_y:
            return base, -1.4, 1.4
        low = 0.0
        high = 0.0
        for corner_x, corner_y in ((lo_x, lo_y), (lo_x, hi_y), (hi_x, lo_y), (hi_x, hi_y)):
            delta = _wrap(math.atan2(corner_y - y, corner_x - x) - base)
            if delta < low:
                low = delta
            if delta > high:
                high = delta
        return base, low, high


class _Field:
    """Geodesic distance-to-goal field over a coarse grid of the arena."""

    def __init__(self, obstacles, goal, np, csr_matrix, dijkstra) -> None:
        res = _GRID
        nx = int(round(_ARENA_W / res)) + 1
        ny = int(round(_ARENA_H / res)) + 1
        self.nx, self.ny, self.res = nx, ny, res
        gx = (np.arange(nx) * res)[:, None] * np.ones((1, ny))
        gy = np.ones((nx, 1)) * (np.arange(ny) * res)[None, :]
        clear = np.full((nx, ny), 60.0)
        for obstacle in obstacles:
            clear = np.minimum(clear, obstacle.grid_distance(np, gx, gy))
        blocked = clear < _BLOCK_PAD
        blocked[:, 0] = True
        blocked[:, ny - 1] = True
        weight = 1.0 + 6.0 * np.clip((1.4 - clear) / 1.4, 0.0, 1.0) ** 2

        index = np.arange(nx * ny).reshape(nx, ny)
        rows, cols, data = [], [], []
        for di, dj in ((1, 0), (0, 1), (1, 1), (1, -1)):
            src_i = slice(max(0, -di), nx - max(0, di))
            src_j = slice(max(0, -dj), ny - max(0, dj))
            dst_i = slice(max(0, di), nx - max(0, -di))
            dst_j = slice(max(0, dj), ny - max(0, -dj))
            length = res * math.hypot(di, dj)
            usable = (~blocked[src_i, src_j]).ravel() & (~blocked[dst_i, dst_j]).ravel()
            cost = (length * 0.5 * (weight[src_i, src_j] + weight[dst_i, dst_j])).ravel()
            rows.append(index[src_i, src_j].ravel()[usable])
            cols.append(index[dst_i, dst_j].ravel()[usable])
            data.append(cost[usable])
        graph = csr_matrix(
            (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
            shape=(nx * ny, nx * ny),
        )
        inside = (
            (gx >= goal.x_min - 1e-6)
            & (gx <= goal.x_max + 1e-6)
            & (gy >= goal.y_min - 1e-6)
            & (gy <= goal.y_max + 1e-6)
        )
        sources = index[inside & (~blocked)]
        if sources.size == 0:
            sources = index[inside]
        distances = dijkstra(graph, directed=False, indices=sources, min_only=True)
        distances = np.where(np.isfinite(distances), distances, 1.0e6)
        self.dist = distances.tolist()
        self.blocked = blocked.ravel().tolist()

    def _cell(self, x: float, y: float):
        i = int(x / self.res + 0.5)
        j = int(y / self.res + 0.5)
        if i < 0:
            i = 0
        elif i >= self.nx:
            i = self.nx - 1
        if j < 0:
            j = 0
        elif j >= self.ny:
            j = self.ny - 1
        return i, j

    def open_cell(self, x: float, y: float):
        i, j = self._cell(x, y)
        if not self.blocked[i * self.ny + j]:
            return i, j
        for radius in (1, 2, 3, 4, 5, 6, 8):
            best = None
            best_dist = 1.0e17
            for di in range(-radius, radius + 1):
                for dj in range(-radius, radius + 1):
                    if max(abs(di), abs(dj)) != radius:
                        continue
                    ii, jj = i + di, j + dj
                    if 0 <= ii < self.nx and 0 <= jj < self.ny:
                        key = ii * self.ny + jj
                        if not self.blocked[key] and self.dist[key] < best_dist:
                            best_dist = self.dist[key]
                            best = (ii, jj)
            if best is not None:
                return best
        return i, j

    def value(self, x: float, y: float) -> float:
        i, j = self.open_cell(x, y)
        return self.dist[i * self.ny + j]

    def descend(self, x: float, y: float, limit: int = 44):
        i, j = self.open_cell(x, y)
        ny, nx = self.ny, self.nx
        dist, blocked = self.dist, self.blocked
        current = dist[i * ny + j]
        points = []
        for _ in range(limit):
            best_i = -1
            best_j = -1
            best = current
            for di in (-1, 0, 1):
                ii = i + di
                if ii < 0 or ii >= nx:
                    continue
                base = ii * ny
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    jj = j + dj
                    if jj < 0 or jj >= ny:
                        continue
                    key = base + jj
                    if not blocked[key] and dist[key] < best:
                        best = dist[key]
                        best_i, best_j = ii, jj
            if best_i < 0:
                break
            i, j, current = best_i, best_j, best
            points.append((i * self.res, j * self.res))
            if current <= 0.0:
                break
        return points


class SwarmController(BaseSwarmController):
    """Assignment-driven attack, interposition and denial."""

    # -- lifecycle ---------------------------------------------------------
    def initialize(self, game_info) -> None:
        # Defaults first: a controller that raises here forfeits the match, so
        # every later stage is allowed to fail without taking the team with it.
        self.obstacles = []
        self.duration = _MATCH_T
        self.forward = 1.0
        self.attack_x, self.attack_y0, self.attack_y1 = _ARENA_W - 1.6, 0.0, _ARENA_H
        self.home_x, self.home_y0, self.home_y1 = 1.4, 0.0, _ARENA_H
        self.attack_field = None
        self.defend_field = None
        self.solver = None
        self.prev_target: dict[int, tuple[str, int]] = {}
        self.spec: dict[int, tuple[float, float, float]] = {}
        try:
            goal, home = game_info.target_goal, game_info.own_goal
            self.forward = 1.0 if goal.center[0] > 50.0 else -1.0
            self.attack_x = goal.x_min + 1.6 if self.forward > 0.0 else goal.x_max - 1.6
            self.attack_y0, self.attack_y1 = goal.y_min, goal.y_max
            self.home_x = home.x_max - 1.4 if self.forward > 0.0 else home.x_min + 1.4
            self.home_y0, self.home_y1 = home.y_min, home.y_max
            self.obstacles = [_Obstacle(item) for item in game_info.obstacles]
            for drone in tuple(game_info.own_initial_drones) + tuple(game_info.opponent_initial_drones):
                if drone.drone_type is DroneType.FAST:
                    self.spec[drone.id] = (_FAST_SPEED, _FAST_ACC, _FAST_VALUE)
                else:
                    self.spec[drone.id] = (_SLOW_SPEED, _SLOW_ACC, _SLOW_VALUE)
        except Exception:
            return
        try:
            import numpy as np
            from scipy.optimize import linear_sum_assignment
            from scipy.sparse import csr_matrix
            from scipy.sparse.csgraph import dijkstra

            self.solver = linear_sum_assignment
            self.attack_field = _Field(self.obstacles, goal, np, csr_matrix, dijkstra)
            self.defend_field = _Field(self.obstacles, home, np, csr_matrix, dijkstra)
        except Exception:
            self.attack_field = None
            self.defend_field = None

    # -- public entry point -------------------------------------------------
    def step(self, state):
        try:
            return self._think(state)
        except Exception:
            try:
                return self._simple(state)
            except Exception:
                return {}

    # -- geometry helpers ---------------------------------------------------
    def _clear_line(self, ax, ay, bx, by, pad=_ROUTE_PAD) -> bool:
        for obstacle in self.obstacles:
            if obstacle.hits_segment(ax, ay, bx, by, pad):
                return False
        return True

    def _spec_of(self, drone):
        found = self.spec.get(drone.id)
        if found is not None:
            return found
        if drone.drone_type is DroneType.FAST:
            return (_FAST_SPEED, _FAST_ACC, _FAST_VALUE)
        return (_SLOW_SPEED, _SLOW_ACC, _SLOW_VALUE)

    def _goal_aim(self, y: float):
        return (self.attack_x, _clamp(y, self.attack_y0 + 1.5, self.attack_y1 - 1.5))

    def _home_aim(self, y: float):
        return (self.home_x, _clamp(y, self.home_y0 + 1.0, self.home_y1 - 1.0))

    def _route_to_goal(self, drone):
        """Waypoint toward the target goal, around obstacles when necessary."""
        px, py = drone.position
        aim = self._goal_aim(py)
        if self._clear_line(px, py, aim[0], aim[1]):
            return aim
        if self.attack_field is None:
            return aim
        points = self.attack_field.descend(px, py)
        if not points:
            return aim
        for index in range(len(points) - 1, -1, -2):
            candidate = points[index]
            if self._clear_line(px, py, candidate[0], candidate[1]):
                return candidate
        return points[0]

    def _lead_point(self, foe, horizon: float):
        horizon = _clamp(horizon, 0.0, 3.2)
        speed = self._spec_of(foe)[0]
        vx, vy = foe.velocity
        ax, ay = foe.acceleration
        dx = vx * horizon + 0.30 * ax * horizon * horizon
        dy = vy * horizon + 0.30 * ay * horizon * horizon
        reach = speed * horizon + 0.4
        travel = math.hypot(dx, dy)
        if travel > reach and travel > 1e-9:
            scale = reach / travel
            dx *= scale
            dy *= scale
        return (foe.position[0] + dx, foe.position[1] + dy)

    # -- steering -----------------------------------------------------------
    def _avoid(self, px, py, ux, uy, speed):
        """Rotate a unit heading onto the nearest tangent of any blocking body.

        Turning beats braking here: the command is always a full-speed heading
        change, because trading speed for time only shortens the lever arm the
        lateral acceleration has to work with.
        """
        if not self.obstacles:
            return ux, uy
        look = _LOOK_BASE + _LOOK_RATE * speed
        nearby = []
        for obstacle in self.obstacles:
            gap = obstacle.distance(px, py)
            if gap < look:
                nearby.append((gap, obstacle))
        if not nearby:
            return ux, uy
        nearby.sort(key=lambda item: item[0])
        heading = math.atan2(uy, ux)
        for _ in range(2):
            changed = False
            for gap, obstacle in nearby:
                if gap < _PANIC_PAD:
                    heading = math.atan2(py - obstacle.cy, px - obstacle.cx)
                    changed = True
                    continue
                base, low, high = obstacle.cone(px, py, _STEER_PAD)
                delta = _wrap(heading - base)
                if low < delta < high:
                    # only dodge bodies that are actually in front of us
                    if math.cos(delta) * gap > look:
                        continue
                    heading = base + (low if (delta - low) < (high - delta) else high)
                    changed = True
            if not changed:
                break
        return math.cos(heading), math.sin(heading)

    def _move(self, drone, aim, arrive=False, bias=None, follow=None):
        px, py = drone.position
        vx, vy = drone.velocity
        speed_max, acc_max, _ = self._spec_of(drone)
        dx, dy = aim[0] - px, aim[1] - py
        gap = math.hypot(dx, dy)
        if gap < 1e-6:
            ux, uy = 0.0, 0.0
        else:
            ux, uy = dx / gap, dy / gap
        if bias is not None and (bias[0] or bias[1]):
            ux += bias[0]
            uy += bias[1]
            norm = math.hypot(ux, uy)
            if norm > 1e-9:
                ux, uy = ux / norm, uy / norm
        speed = math.hypot(vx, vy)
        if gap > 1e-6 or (bias is not None and (bias[0] or bias[1])):
            ux, uy = self._avoid(px, py, ux, uy, speed)
        # keep away from the top and bottom edges of the arena
        if py < 1.1 and uy < 0.0:
            uy = 0.0
        elif py > _ARENA_H - 1.1 and uy > 0.0:
            uy = 0.0
        norm = math.hypot(ux, uy)
        if norm > 1e-9:
            ux, uy = ux / norm, uy / norm
        target_speed = speed_max
        if arrive:
            reach = max(0.0, gap - 0.25)
            target_speed = min(speed_max, math.sqrt(2.0 * acc_max * reach) if reach > 0.0 else 0.0)
            target_speed = min(target_speed, reach / 0.30 if reach > 0.0 else 0.0)
        want_x = ux * target_speed
        want_y = uy * target_speed
        if follow is not None:
            # ride along with a moving charge instead of chasing its wake
            want_x += follow[0]
            want_y += follow[1]
            norm = math.hypot(want_x, want_y)
            if norm > speed_max and norm > 1e-9:
                scale = speed_max / norm
                want_x *= scale
                want_y *= scale
        ax = _TRACK_GAIN * (want_x - vx)
        ay = _TRACK_GAIN * (want_y - vy)
        return self._safety(drone, ax, ay)

    def _safety(self, drone, ax, ay):
        """Final guard: brake and push out if the current track ends in a body."""
        px, py = drone.position
        vx, vy = drone.velocity
        speed_max, acc_max, _ = self._spec_of(drone)
        speed = math.hypot(vx, vy)
        if speed > 0.35 and self.obstacles:
            horizon = min(_BRAKE_CAP, speed / acc_max + _BRAKE_PAD)
            ex, ey = px + vx * horizon, py + vy * horizon
            for obstacle in self.obstacles:
                if obstacle.hits_segment(px, py, ex, ey, _DRONE_R + 0.22):
                    gap = max(0.05, obstacle.distance(px, py))
                    span = 2.2 + speed
                    urgency = _clamp((span - gap) / span, 0.0, 1.0)
                    away_x, away_y = px - obstacle.cx, py - obstacle.cy
                    norm = math.hypot(away_x, away_y)
                    if norm > 1e-9:
                        away_x, away_y = away_x / norm, away_y / norm
                    # brake along the track and push out of the body
                    ax = ax * (1.0 - 0.75 * urgency) - vx / max(0.25, speed) * acc_max * urgency
                    ay = ay * (1.0 - 0.75 * urgency) - vy / max(0.25, speed) * acc_max * urgency
                    ax += away_x * acc_max * urgency
                    ay += away_y * acc_max * urgency
                    break
        norm = math.hypot(ax, ay)
        if norm > acc_max and norm > 1e-9:
            scale = acc_max / norm
            ax *= scale
            ay *= scale
        if not (math.isfinite(ax) and math.isfinite(ay)):
            return (0.0, 0.0)
        return (ax, ay)

    # -- tactical core ------------------------------------------------------
    def _think(self, state):
        if state.time >= self.duration:
            # a longer-than-default match: keep planning instead of writing the
            # whole team off as unable to reach the goal in time
            self.duration = state.time + 30.0
        left = max(0.0, self.duration - state.time)
        own = [d for d in state.own_drones if d.status is DroneStatus.ACTIVE]
        foes = [d for d in state.opponent_drones if d.status is DroneStatus.ACTIVE]
        if not own:
            return {}

        margin = 0.0
        if state.own_score > state.opponent_score:
            margin = 1.0
        elif state.own_score < state.opponent_score:
            margin = -1.0

        # --- how long until each drone can score, and how likely that is ----
        own_reach = {}
        own_pscore = {}
        for drone in own:
            speed = self._spec_of(drone)[0]
            if self.attack_field is not None:
                travel = self.attack_field.value(drone.position[0], drone.position[1])
            else:
                aim = self._goal_aim(drone.position[1])
                travel = math.hypot(aim[0] - drone.position[0], aim[1] - drone.position[1])
            reach = travel / speed + 0.8
            own_reach[drone.id] = reach
            own_pscore[drone.id] = _ramp(left - reach, 0.0, 3.0)

        foe_reach = {}
        foe_pscore = {}
        for foe in foes:
            speed = self._spec_of(foe)[0]
            if self.defend_field is not None:
                travel = self.defend_field.value(foe.position[0], foe.position[1])
            else:
                aim = self._home_aim(foe.position[1])
                travel = math.hypot(aim[0] - foe.position[0], aim[1] - foe.position[1])
            reach = travel / speed + 0.8
            foe_reach[foe.id] = reach
            foe_pscore[foe.id] = _ramp(left - reach, 0.0, 3.0)

        # --- what each enemy threatens --------------------------------------
        # Only two exchanges move the score differential: our FAST killing their
        # SLOW (+4) and their FAST killing our SLOW (-4).  FAST-for-FAST and
        # SLOW-for-SLOW trades are exactly neutral.  Everything below prices
        # tasks in those units.
        doom = {}
        threat = {}
        for foe in foes:
            foe_speed = self._spec_of(foe)[0]
            best_gain = 0.0
            best_victim = None
            best_time = None
            for drone in own:
                own_speed = self._spec_of(drone)[0]
                catch = _catch_time(
                    foe.position[0], foe.position[1], foe_speed,
                    drone.position[0], drone.position[1],
                    drone.velocity[0], drone.velocity[1],
                )
                if catch is None or catch > left:
                    continue
                if foe_speed <= own_speed + 1e-6 and catch > 6.0:
                    # equal or slower pursuers only matter in a head-on closure
                    continue
                gain = (
                    self._spec_of(drone)[2]
                    * own_pscore[drone.id]
                    * (1.0 / (1.0 + catch / _THREAT_TAU))
                )
                if gain > best_gain:
                    best_gain = gain
                    best_victim = drone
                    best_time = catch
                previous = doom.get(drone.id)
                if previous is None or catch < previous[0]:
                    doom[drone.id] = (catch, self._spec_of(foe)[2], foe.id)
            threat[foe.id] = (best_gain, best_victim, best_time)

        own_slow = [d for d in own if d.drone_type is DroneType.SLOW and own_pscore[d.id] > 0.0]
        foe_fast = sum(1 for f in foes if f.drone_type is DroneType.FAST)
        pressure = _clamp(foe_fast / float(len(own_slow)), 0.0, 1.0) if own_slow else 0.0

        # --- price every enemy: interpose value versus outright kill value ---
        kill_value = {}
        aim_plan = {}
        shield_gap = {}
        for foe in foes:
            gain, victim, catch = threat[foe.id]
            block = _DIFF * gain
            shield = 40.0
            for guard in foes:
                if guard.id == foe.id or guard.drone_type is not DroneType.FAST:
                    continue
                shield = min(shield, math.hypot(guard.position[0] - foe.position[0],
                                                guard.position[1] - foe.position[1]))
            shield_gap[foe.id] = shield
            if foe.drone_type is DroneType.SLOW:
                connect = _clamp((shield - 2.0) / 10.0, _MIN_CONNECT, 1.0)
                hunt = _DIFF * _SLOW_VALUE * foe_pscore[foe.id] * connect
            else:
                hunt = _DIFF * _FAST_VALUE * foe_pscore[foe.id]
            if margin > 0.0 and left < 30.0:
                hunt *= 1.15
                block *= 1.15
            kill_value[foe.id] = max(block, hunt)
            if victim is not None and block >= hunt:
                aim_plan[foe.id] = (victim.position, catch if catch is not None else 4.0)
            else:
                aim_plan[foe.id] = (self._home_aim(foe.position[1]), foe_reach[foe.id])

        # --- escorting keeps a FAST in reserve: it only spends itself when a
        # hunter actually arrives, and otherwise still banks its own point.
        escort_value = {}
        for slow in own_slow:
            escort_value[slow.id] = (
                _DIFF * _SLOW_VALUE * pressure + _FAST_VALUE * (1.0 - pressure)
            ) * own_pscore[slow.id] * _ESCORT_W

        # --- one column per task; escorted enemy SLOW drones get two, because
        # the first attacker only strips the shield and the second collects.
        columns = []
        for foe in foes:
            columns.append(("k", foe, 1.0))
            if foe.drone_type is DroneType.SLOW and shield_gap[foe.id] < 9.0:
                columns.append(("k", foe, _SECOND_WAVE))
        for slow in own_slow:
            columns.append(("e", slow, 1.0))
        rows = len(own)
        base = len(columns)
        cols = base + rows
        matrix = [[_NO_TASK] * cols for _ in range(rows)]
        for i, drone in enumerate(own):
            speed, _, value = self._spec_of(drone)
            is_fast = drone.drone_type is DroneType.FAST
            px, py = drone.position
            for j, (kind, other, scale) in enumerate(columns):
                if kind == "k":
                    foe_speed = self._spec_of(other)[0]
                    objective, objective_time = aim_plan[other.id]
                    station = self._station(other, objective, speed > foe_speed + 0.1)
                    travel = math.hypot(station[0] - px, station[1] - py)
                    arrive = travel / speed + 0.7
                    feasible = 1.0
                    if arrive > objective_time + 1.0:
                        feasible *= math.exp(-(arrive - objective_time - 1.0) / 2.5)
                    if speed < foe_speed - 0.1:
                        catch = _catch_time(px, py, speed, other.position[0], other.position[1],
                                            other.velocity[0], other.velocity[1])
                        feasible *= 0.0 if catch is None else math.exp(-catch / 1.6)
                    if arrive > left:
                        feasible = 0.0
                    utility = scale * kill_value[other.id] * feasible * math.exp(-arrive / _ENGAGE_TAU)
                    if self.prev_target.get(drone.id) == ("k", other.id):
                        utility += _STICKY_BONUS
                    matrix[i][j] = -utility
                elif is_fast and other.id != drone.id:
                    travel = math.hypot(other.position[0] - px, other.position[1] - py)
                    utility = escort_value[other.id] * math.exp(-(travel / speed) / _ENGAGE_TAU)
                    if self.prev_target.get(drone.id) == ("e", other.id):
                        utility += _STICKY_BONUS
                    matrix[i][j] = -utility
            score_utility = value * own_pscore[drone.id]
            if margin < 0.0 and left < 30.0:
                score_utility *= 1.2
            matrix[i][base + i] = -(score_utility + 0.03)

        pairing = self._assign(matrix, rows, cols)

        # --- convert the assignment into motion -----------------------------
        actions = {}
        new_targets = {}
        for i, drone in enumerate(own):
            column = pairing[i]
            task = None
            if column < base:
                kind, other, _ = columns[column]
                task = (kind, other)
            override = self._last_stand(drone, foes, doom, own_pscore)
            if override is not None:
                task = ("k", override)
            if task is None:
                actions[drone.id] = self._attack(drone, own, foes, left - own_reach[drone.id])
            elif task[0] == "k":
                new_targets[drone.id] = ("k", task[1].id)
                actions[drone.id] = self._engage(drone, task[1], aim_plan[task[1].id])
            else:
                new_targets[drone.id] = ("e", task[1].id)
                actions[drone.id] = self._escort(drone, task[1], foes, doom)
        self.prev_target = new_targets
        return actions

    def _escort(self, drone, charge, foes, doom):
        """Shadow a SLOW drone from the side any hunter must come through.

        The shield is placed on the segment between the hunter and the point the
        charge will actually occupy when the hunter arrives - aiming at the
        charge's current position leaves a gap the hunter's own lead angle walks
        straight through.
        """
        speed = self._spec_of(drone)[0]
        px, py = drone.position
        cx, cy = charge.position

        # a stray enemy SLOW inside easy reach is worth +4 outright
        best_catch = None
        best_foe = None
        for foe in foes:
            if foe.drone_type is not DroneType.SLOW:
                continue
            catch = _catch_time(px, py, speed, foe.position[0], foe.position[1],
                                foe.velocity[0], foe.velocity[1])
            if catch is not None and catch < _SNAP_KILL and (best_catch is None or catch < best_catch):
                best_catch = catch
                best_foe = foe
        if best_foe is not None:
            return self._move(drone, self._lead_point(best_foe, best_catch), arrive=False)

        entry = doom.get(charge.id)
        hunter = None
        hit = None
        if entry is not None and entry[0] < _SHIELD_RANGE:
            hit = entry[0]
            for foe in foes:
                if foe.id == entry[2]:
                    hunter = foe
                    break
        if hunter is None:
            station = (
                _clamp(cx + self.forward * _ESCORT_GAP, 0.5, _ARENA_W - 0.5),
                _clamp(cy, 1.0, _ARENA_H - 1.0),
            )
            return self._move(drone, station, arrive=True, follow=charge.velocity)

        # commit once the trade is on: either we take the hunter or it breaks off
        own_catch = _catch_time(px, py, speed, hunter.position[0], hunter.position[1],
                                hunter.velocity[0], hunter.velocity[1])
        if own_catch is not None and own_catch < _BLOCK_COMMIT and own_catch <= hit + 0.5:
            return self._move(drone, self._lead_point(hunter, own_catch), arrive=False)

        horizon = _clamp(hit, 0.0, _SHIELD_LEAD)
        mx = cx + charge.velocity[0] * horizon
        my = cy + charge.velocity[1] * horizon
        bx, by = hunter.position[0] - mx, hunter.position[1] - my
        norm = math.hypot(bx, by)
        if norm < 1e-6:
            bx, by, norm = self.forward, 0.0, 1.0
        gap = min(_ESCORT_GAP, 0.45 * norm)
        station = (
            _clamp(mx + bx / norm * gap, 0.5, _ARENA_W - 0.5),
            _clamp(my + by / norm * gap, 1.0, _ARENA_H - 1.0),
        )
        return self._move(drone, station, arrive=True, follow=charge.velocity)

    def _assign(self, matrix, rows, cols):
        if self.solver is not None:
            try:
                row_index, col_index = self.solver(matrix)
                pairing = [cols - 1] * rows
                for r, c in zip(row_index, col_index):
                    pairing[int(r)] = int(c)
                return pairing
            except Exception:
                pass
        # greedy fallback
        pairing = [-1] * rows
        order = []
        for i in range(rows):
            for j in range(cols):
                order.append((matrix[i][j], i, j))
        order.sort()
        used_rows = set()
        used_cols = set()
        for _, i, j in order:
            if i in used_rows or j in used_cols:
                continue
            used_rows.add(i)
            used_cols.add(j)
            pairing[i] = j
            if len(used_rows) == rows:
                break
        for i in range(rows):
            if pairing[i] < 0:
                pairing[i] = len(matrix[0]) - rows + i
        return pairing

    def _station(self, foe, objective, can_pursue):
        """Where an interceptor should be to take `foe` down."""
        if can_pursue:
            return foe.position
        ox, oy = objective
        dx, dy = foe.position[0] - ox, foe.position[1] - oy
        gap = math.hypot(dx, dy)
        if gap < 1e-6:
            return foe.position
        offset = _clamp(_INTERPOSE_FRAC * gap, 1.2, _INTERPOSE_CAP)
        return (ox + dx / gap * offset, oy + dy / gap * offset)

    def _engage(self, drone, foe, plan):
        speed = self._spec_of(drone)[0]
        foe_speed = self._spec_of(foe)[0]
        px, py = drone.position
        separation = math.hypot(foe.position[0] - px, foe.position[1] - py)
        catch = _catch_time(px, py, speed, foe.position[0], foe.position[1],
                            foe.velocity[0], foe.velocity[1])
        objective = plan[0]
        objective_gap = math.hypot(foe.position[0] - objective[0], foe.position[1] - objective[1])
        pursue = speed > foe_speed + 0.1
        if not pursue:
            close = separation < _COMMIT_RANGE or objective_gap < 5.0
            if not close and (catch is None or catch > 2.4):
                station = self._station(foe, objective, False)
                return self._move(drone, station, arrive=True)
        horizon = catch if catch is not None else separation / max(1.0, speed)
        return self._move(drone, self._lead_point(foe, horizon), arrive=False)

    def _last_stand(self, drone, foes, doom, own_pscore):
        """A drone that is about to die takes the most valuable enemy with it."""
        entry = doom.get(drone.id)
        if entry is None:
            return None
        death, killer_value, killer_id = entry
        if death > 2.2:
            return None
        speed = self._spec_of(drone)[0]
        if own_pscore[drone.id] > 0.0 and death > 1.3:
            return None
        best = None
        best_value = killer_value
        for foe in foes:
            foe_value = self._spec_of(foe)[2]
            if foe_value <= best_value or foe.id == killer_id:
                continue
            catch = _catch_time(drone.position[0], drone.position[1], speed,
                                foe.position[0], foe.position[1],
                                foe.velocity[0], foe.velocity[1])
            if catch is None or catch > death + 0.35:
                continue
            best_value = foe_value
            best = foe
        return best

    def _attack(self, drone, own, foes, slack=99.0):
        """Run for the goal while shaping the path away from live threats.

        `slack` is the spare time this drone still has over its trip to the
        goal.  Evasion is only ever paid for out of that slack: a drone that
        flees for the rest of the match converts a possible five points into a
        certain zero, which is strictly worse than pressing on and being caught.
        """
        px, py = drone.position
        timid = slack > _PANIC_RESERVE
        speed_self = self._spec_of(drone)[0]
        aim = self._route_to_goal(drone)
        dx, dy = aim[0] - px, aim[1] - py
        gap = math.hypot(dx, dy)
        if gap > 1e-6:
            nav_x, nav_y = dx / gap, dy / gap
        else:
            nav_x, nav_y = self.forward, 0.0

        push_x = 0.0
        push_y = 0.0
        worst = None
        worst_foe = None
        for foe in foes:
            foe_speed = self._spec_of(foe)[0]
            fx, fy = foe.position
            offset = math.hypot(fx - px, fy - py)
            if offset > 26.0:
                continue
            catch = _catch_time(fx, fy, foe_speed, px, py, drone.velocity[0], drone.velocity[1])
            if catch is None or catch > 7.0:
                continue
            if foe_speed > speed_self + 0.1 and (worst is None or catch < worst):
                worst = catch
                worst_foe = foe
            weight = (7.0 - catch) / 7.0
            weight *= weight
            ahead = self._lead_point(foe, min(catch, 1.5))
            away_x, away_y = px - ahead[0], py - ahead[1]
            norm = math.hypot(away_x, away_y)
            if norm < 1e-6:
                continue
            push_x += away_x / norm * weight
            push_y += away_y / norm * weight

        # Running out the clock is free - we finish the crossing with seconds to
        # spare - so a drone that cannot outrun its hunter maximises the time to
        # contact and drags the hunter onto a friendly FAST drone instead.
        if timid and worst is not None and worst < _PANIC_T:
            ahead = self._lead_point(worst_foe, worst)
            run_x, run_y = px - ahead[0], py - ahead[1]
            norm = math.hypot(run_x, run_y)
            if norm > 1e-6:
                run_x, run_y = run_x / norm, run_y / norm
                helper = None
                helper_gap = 20.0
                for mate in own:
                    if mate.id == drone.id or mate.drone_type is not DroneType.FAST:
                        continue
                    span = math.hypot(mate.position[0] - px, mate.position[1] - py)
                    if span < helper_gap:
                        helper_gap = span
                        helper = mate
                if helper is not None and helper_gap > 1e-6:
                    run_x += (helper.position[0] - px) / helper_gap * _PANIC_PULL
                    run_y += (helper.position[1] - py) / helper_gap * _PANIC_PULL
                    norm = math.hypot(run_x, run_y)
                    if norm > 1e-6:
                        run_x, run_y = run_x / norm, run_y / norm
                return self._move(drone, (px + run_x * 12.0, py + run_y * 12.0), arrive=False)
        for mate in own:
            if mate.id == drone.id:
                continue
            ox, oy = px - mate.position[0], py - mate.position[1]
            offset = math.hypot(ox, oy)
            if 1e-6 < offset < 1.6:
                push_x += ox / offset * 0.22 * (1.6 - offset)
                push_y += oy / offset * 0.22 * (1.6 - offset)

        # keep the forward component: sidestepping is useful, retreating is not
        along = push_x * nav_x + push_y * nav_y
        lat_x = push_x - along * nav_x
        lat_y = push_y - along * nav_y
        along = _clamp(along, -0.55 if timid else 0.0, 0.35)
        bias_x = lat_x + along * nav_x
        bias_y = lat_y + along * nav_y
        norm = math.hypot(bias_x, bias_y)
        limit = 1.25
        if norm > limit and norm > 1e-9:
            bias_x *= limit / norm
            bias_y *= limit / norm
        return self._move(drone, aim, arrive=False, bias=(bias_x, bias_y))

    # -- fallback -----------------------------------------------------------
    def _simple(self, state):
        actions = {}
        for drone in state.own_drones:
            if drone.status is not DroneStatus.ACTIVE:
                continue
            aim = self._goal_aim(drone.position[1])
            actions[drone.id] = self._move(drone, aim, arrive=False)
        return actions
