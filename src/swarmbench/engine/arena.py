"""Seeded v1 arena generation and deterministic traversability checks."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import hypot
from random import Random

from swarmbench.api import (
    ARENA_HEIGHT,
    ARENA_WIDTH,
    DRONE_RADIUS,
    CircleObstacle,
    DroneSnapshot,
    DroneType,
    GoalZone,
    Obstacle,
    RectangleObstacle,
    Team,
    Vec2,
)
from swarmbench.version import SCENARIO_GENERATOR_VERSION

GRID_RESOLUTION = 0.5
PLANNING_CLEARANCE = 0.35
SPAWN_MIN_SPACING = 0.8
MAX_GENERATION_ATTEMPTS = 100


@dataclass(frozen=True, slots=True)
class Scenario:
    seed: int
    generator_version: int
    width: float
    height: float
    goal_for_a: GoalZone
    goal_for_b: GoalZone
    obstacles: tuple[Obstacle, ...]
    drones: tuple[DroneSnapshot, ...]

    def target_goal(self, team: Team) -> GoalZone:
        return self.goal_for_a if team is Team.A else self.goal_for_b

    def own_goal(self, team: Team) -> GoalZone:
        return self.goal_for_b if team is Team.A else self.goal_for_a

    def team_drones(self, team: Team) -> tuple[DroneSnapshot, ...]:
        return tuple(drone for drone in self.drones if drone.team is team)


def _point_blocked(point: Vec2, obstacle: Obstacle, clearance: float) -> bool:
    x, y = point
    if isinstance(obstacle, CircleObstacle):
        return hypot(x - obstacle.center[0], y - obstacle.center[1]) <= obstacle.radius + clearance
    return (
        obstacle.x_min - clearance <= x <= obstacle.x_max + clearance
        and obstacle.y_min - clearance <= y <= obstacle.y_max + clearance
    )


def point_blocked(point: Vec2, obstacles: tuple[Obstacle, ...], clearance: float = DRONE_RADIUS) -> bool:
    return any(_point_blocked(point, obstacle, clearance) for obstacle in obstacles)


def _obstacle_bounds(obstacle: Obstacle, padding: float = 0.0) -> tuple[float, float, float, float]:
    if isinstance(obstacle, CircleObstacle):
        x, y = obstacle.center
        radius = obstacle.radius + padding
        return (x - radius, x + radius, y - radius, y + radius)
    return (
        obstacle.x_min - padding,
        obstacle.x_max + padding,
        obstacle.y_min - padding,
        obstacle.y_max + padding,
    )


def _obstacles_overlap(left: Obstacle, right: Obstacle, padding: float = 0.5) -> bool:
    lx0, lx1, ly0, ly1 = _obstacle_bounds(left, padding)
    rx0, rx1, ry0, ry1 = _obstacle_bounds(right, padding)
    return lx0 <= rx1 and rx0 <= lx1 and ly0 <= ry1 and ry0 <= ly1


def _generate_obstacles(rng: Random) -> tuple[Obstacle, ...]:
    obstacles: list[Obstacle] = []
    target_count = rng.randint(6, 12)
    draws = 0
    while len(obstacles) < target_count and draws < 500:
        draws += 1
        if rng.random() < 0.5:
            radius = rng.uniform(1.5, 3.5)
            candidate: Obstacle = CircleObstacle(
                (rng.uniform(20.0 + radius, 80.0 - radius), rng.uniform(2.0 + radius, 58.0 - radius)),
                radius,
            )
        else:
            width = rng.uniform(2.0, 7.0)
            height = rng.uniform(2.0, 7.0)
            center_x = rng.uniform(20.0 + width / 2, 80.0 - width / 2)
            center_y = rng.uniform(2.0 + height / 2, 58.0 - height / 2)
            candidate = RectangleObstacle(
                center_x - width / 2,
                center_x + width / 2,
                center_y - height / 2,
                center_y + height / 2,
            )
        if not any(_obstacles_overlap(candidate, existing) for existing in obstacles):
            obstacles.append(candidate)
    if len(obstacles) != target_count:
        raise RuntimeError("could not place requested obstacle count")
    return tuple(obstacles)


def _generate_team_spawns(rng: Random, team: Team, obstacles: tuple[Obstacle, ...], first_id: int) -> tuple[DroneSnapshot, ...]:
    positions: list[Vec2] = []
    x_range = (5.0, 14.0) if team is Team.A else (86.0, 95.0)
    for _ in range(20):
        for _attempt in range(1_000):
            point = (rng.uniform(*x_range), rng.uniform(3.0, 57.0))
            if point_blocked(point, obstacles, DRONE_RADIUS):
                continue
            if all(hypot(point[0] - other[0], point[1] - other[1]) >= SPAWN_MIN_SPACING for other in positions):
                positions.append(point)
                break
        else:
            raise RuntimeError("could not place team spawn")

    types = [DroneType.FAST] * 10 + [DroneType.SLOW] * 10
    rng.shuffle(types)
    return tuple(
        DroneSnapshot(first_id + index, team, types[index], point)
        for index, point in enumerate(positions)
    )


def _reachable_from_goal(scenario: Scenario, goal: GoalZone) -> set[tuple[int, int]]:
    resolution = GRID_RESOLUTION
    nx = round(scenario.width / resolution) + 1
    ny = round(scenario.height / resolution) + 1
    clearance = DRONE_RADIUS + PLANNING_CLEARANCE

    def open_cell(cell: tuple[int, int]) -> bool:
        i, j = cell
        point = (i * resolution, j * resolution)
        return not point_blocked(point, scenario.obstacles, clearance)

    starts = [
        (i, j)
        for i in range(nx)
        for j in range(ny)
        if goal.contains((i * resolution, j * resolution)) and open_cell((i, j))
    ]
    reached = set(starts)
    queue = deque(starts)
    while queue:
        i, j = queue.popleft()
        for neighbor in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
            ni, nj = neighbor
            if 0 <= ni < nx and 0 <= nj < ny and neighbor not in reached and open_cell(neighbor):
                reached.add(neighbor)
                queue.append(neighbor)
    return reached


def scenario_is_traversable(scenario: Scenario) -> bool:
    reachable_a = _reachable_from_goal(scenario, scenario.goal_for_a)
    reachable_b = _reachable_from_goal(scenario, scenario.goal_for_b)
    for drone in scenario.drones:
        cell = (round(drone.position[0] / GRID_RESOLUTION), round(drone.position[1] / GRID_RESOLUTION))
        if cell not in (reachable_a if drone.team is Team.A else reachable_b):
            return False
    return True


def validate_scenario(scenario: Scenario, *, check_traversability: bool = True) -> None:
    if scenario.generator_version != SCENARIO_GENERATOR_VERSION:
        raise ValueError("unsupported scenario generator version")
    if len(scenario.drones) != 40:
        raise ValueError("scenario must contain 40 drones")
    for team in Team:
        drones = scenario.team_drones(team)
        if len(drones) != 20 or sum(d.drone_type is DroneType.FAST for d in drones) != 10:
            raise ValueError("each team must contain 10 FAST and 10 SLOW drones")
        for index, drone in enumerate(drones):
            x, y = drone.position
            if not (DRONE_RADIUS <= x <= scenario.width - DRONE_RADIUS and DRONE_RADIUS <= y <= scenario.height - DRONE_RADIUS):
                raise ValueError("spawn lies outside arena")
            if point_blocked(drone.position, scenario.obstacles, DRONE_RADIUS):
                raise ValueError("spawn overlaps obstacle")
            if any(hypot(x - other.position[0], y - other.position[1]) < SPAWN_MIN_SPACING for other in drones[index + 1 :]):
                raise ValueError("spawn spacing is too small")
    for obstacle in scenario.obstacles:
        x0, x1, y0, y1 = _obstacle_bounds(obstacle)
        if x0 < 18.0 or x1 > 82.0 or y0 < 0.0 or y1 > scenario.height:
            raise ValueError("obstacle overlaps a protected region")
    if any(_obstacles_overlap(left, right, 0.0) for index, left in enumerate(scenario.obstacles) for right in scenario.obstacles[index + 1 :]):
        raise ValueError("obstacles overlap")
    if check_traversability and not scenario_is_traversable(scenario):
        raise ValueError("scenario is not traversable")


def generate_scenario(seed: int, generator_version: int = SCENARIO_GENERATOR_VERSION) -> Scenario:
    if generator_version != SCENARIO_GENERATOR_VERSION:
        raise ValueError(f"unsupported generator version: {generator_version}")
    rng = Random(seed)
    for _ in range(MAX_GENERATION_ATTEMPTS):
        goal_for_a_center = rng.uniform(10.0, 50.0)
        goal_for_b_center = rng.uniform(10.0, 50.0)
        goal_for_a = GoalZone(97.0, 100.0, goal_for_a_center - 7.0, goal_for_a_center + 7.0)
        goal_for_b = GoalZone(0.0, 3.0, goal_for_b_center - 7.0, goal_for_b_center + 7.0)
        try:
            obstacles = _generate_obstacles(rng)
            drones = _generate_team_spawns(rng, Team.A, obstacles, 0) + _generate_team_spawns(rng, Team.B, obstacles, 20)
        except RuntimeError:
            continue
        scenario = Scenario(seed, generator_version, ARENA_WIDTH, ARENA_HEIGHT, goal_for_a, goal_for_b, obstacles, drones)
        if scenario_is_traversable(scenario):
            return scenario
    raise RuntimeError(f"could not generate traversable arena after {MAX_GENERATION_ATTEMPTS} attempts")

