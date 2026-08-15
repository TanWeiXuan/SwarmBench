"""Small deterministic steering helpers shared by the readable baselines."""

from __future__ import annotations

from math import hypot

from swarmbench import CircleObstacle, DroneSnapshot, DroneSpec, GoalZone, RectangleObstacle


def goal_target(goal: GoalZone, drone: DroneSnapshot) -> tuple[float, float]:
    y = min(goal.y_max - 0.75, max(goal.y_min + 0.75, drone.position[1]))
    return (goal.center[0], y)


def distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return hypot(left[0] - right[0], left[1] - right[1])


def _obstacle_center_radius(obstacle: CircleObstacle | RectangleObstacle) -> tuple[tuple[float, float], float]:
    if isinstance(obstacle, CircleObstacle):
        return obstacle.center, obstacle.radius
    center = ((obstacle.x_min + obstacle.x_max) / 2, (obstacle.y_min + obstacle.y_max) / 2)
    radius = hypot(obstacle.x_max - obstacle.x_min, obstacle.y_max - obstacle.y_min) / 2
    return center, radius


def steer(
    drone: DroneSnapshot,
    target: tuple[float, float],
    spec: DroneSpec,
    obstacles: tuple[CircleObstacle | RectangleObstacle, ...],
    *,
    repulsion: float = 1.0,
) -> tuple[float, float]:
    """Velocity tracking plus short-range deterministic obstacle steering."""
    dx, dy = target[0] - drone.position[0], target[1] - drone.position[1]
    remaining = hypot(dx, dy)
    if remaining < 1e-9:
        desired_velocity = (0.0, 0.0)
    else:
        desired_speed = min(spec.max_speed, (2.0 * spec.max_acceleration * remaining) ** 0.5)
        desired_velocity = (dx / remaining * desired_speed, dy / remaining * desired_speed)

    ax = 2.2 * (desired_velocity[0] - drone.velocity[0])
    ay = 2.2 * (desired_velocity[1] - drone.velocity[1])
    if remaining > 0:
        forward = (dx / remaining, dy / remaining)
        for obstacle in obstacles:
            center, radius = _obstacle_center_radius(obstacle)
            ox, oy = center[0] - drone.position[0], center[1] - drone.position[1]
            projection = ox * forward[0] + oy * forward[1]
            lateral = ox * (-forward[1]) + oy * forward[0]
            safe_radius = radius + 1.1
            if -0.5 < projection < 8.0 and abs(lateral) < safe_radius:
                side = -1.0 if lateral > 0 else 1.0
                if abs(lateral) < 1e-9:
                    side = 1.0 if drone.id % 2 == 0 else -1.0
                strength = repulsion * spec.max_acceleration * (1.0 - max(0.0, projection) / 8.0)
                ax += -forward[1] * side * strength
                ay += forward[0] * side * strength
            center_distance = hypot(ox, oy)
            surface_distance = center_distance - radius
            if 0 < surface_distance < 3.0:
                strength = repulsion * spec.max_acceleration * (3.0 - surface_distance) / 3.0
                ax -= ox / center_distance * strength
                ay -= oy / center_distance * strength
    return (ax, ay)

