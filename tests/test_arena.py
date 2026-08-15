from math import hypot

from swarmbench.api import DRONE_RADIUS, DroneType, Team
from swarmbench.engine.arena import (
    SPAWN_MIN_SPACING,
    generate_scenario,
    point_blocked,
    scenario_is_traversable,
    validate_scenario,
)


def test_generation_is_deterministic_and_varies_by_seed() -> None:
    assert generate_scenario(12345) == generate_scenario(12345)
    assert generate_scenario(12345) != generate_scenario(12346)


def test_generated_scenario_has_valid_goals_spawns_and_obstacles() -> None:
    scenario = generate_scenario(7)
    validate_scenario(scenario)
    assert 6 <= len(scenario.obstacles) <= 12
    assert scenario.goal_for_a.x_max == scenario.width
    assert scenario.goal_for_b.x_min == 0.0
    for team in Team:
        drones = scenario.team_drones(team)
        assert len(drones) == 20
        assert sum(drone.drone_type is DroneType.FAST for drone in drones) == 10
        assert sum(drone.drone_type is DroneType.SLOW for drone in drones) == 10
        for index, drone in enumerate(drones):
            assert not point_blocked(drone.position, scenario.obstacles, DRONE_RADIUS)
            assert all(
                hypot(drone.position[0] - other.position[0], drone.position[1] - other.position[1]) >= SPAWN_MIN_SPACING
                for other in drones[index + 1 :]
            )


def test_hundreds_of_seeded_arenas_are_valid_and_reachable() -> None:
    for seed in range(200):
        scenario = generate_scenario(seed)
        validate_scenario(scenario, check_traversability=False)
        assert scenario_is_traversable(scenario)

