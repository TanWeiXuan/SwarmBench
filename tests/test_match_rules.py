from dataclasses import replace

from swarmbench.api import CircleObstacle, DroneSnapshot, DroneStatus, DroneType, GoalZone, Team
from swarmbench.engine.arena import Scenario
from swarmbench.engine.events import EventType
from swarmbench.engine.match import Simulator


def scenario_with(*drones: DroneSnapshot, obstacles=()) -> Scenario:
    return Scenario(1, 1, 100.0, 60.0, GoalZone(97.0, 100.0, 20.0, 40.0), GoalZone(0.0, 3.0, 20.0, 40.0), obstacles, drones)


def moving(drone_id: int, team: Team, position: tuple[float, float], velocity: tuple[float, float], kind: DroneType = DroneType.FAST) -> DroneSnapshot:
    return DroneSnapshot(drone_id, team, kind, position, velocity)


def test_friendly_drones_do_not_collide() -> None:
    simulator = Simulator(scenario_with(moving(0, Team.A, (50.0, 30.0), (5.0, 0.0)), moving(1, Team.A, (50.5, 30.0), (-5.0, 0.0))))
    assert simulator.step() == ()
    assert all(drone.status is DroneStatus.ACTIVE for drone in simulator.snapshots())


def test_one_for_one_rule_prevents_multi_elimination() -> None:
    simulator = Simulator(
        scenario_with(
            moving(0, Team.A, (50.0, 30.0), (0.0, 0.0)),
            moving(20, Team.B, (50.5, 30.0), (0.0, 0.0)),
            moving(21, Team.B, (50.6, 30.0), (0.0, 0.0)),
        )
    )
    events = simulator.step()
    assert len(events) == 1
    assert events[0].drone_ids == (0, 20)
    assert simulator.drones[21].snapshot.status is DroneStatus.ACTIVE


def test_goal_scores_and_removes_drone() -> None:
    drone = moving(0, Team.A, (96.9, 30.0), (5.0, 0.0), DroneType.SLOW)
    simulator = Simulator(scenario_with(drone))
    events = simulator.step()
    assert events[0].event_type is EventType.GOAL
    assert simulator.scores[Team.A] == 5
    assert simulator.drones[0].snapshot.status is DroneStatus.SCORED


def test_swept_obstacle_crash_eliminates_drone() -> None:
    drone = moving(0, Team.A, (49.0, 30.0), (30.0, 0.0))
    simulator = Simulator(scenario_with(drone, obstacles=(CircleObstacle((50.0, 30.0), 0.5),)), dt=0.1)
    events = simulator.step()
    assert events[0].event_type is EventType.OBSTACLE_CRASH
    assert simulator.drones[0].snapshot.status is DroneStatus.ELIMINATED


def test_exact_tie_interception_precedes_goal() -> None:
    a = moving(0, Team.A, (97.0, 30.0), (0.0, 0.0))
    b = moving(20, Team.B, (97.5, 30.0), (0.0, 0.0))
    simulator = Simulator(scenario_with(a, b))
    events = simulator.step()
    assert events[0].event_type is EventType.INTERCEPTION
    assert simulator.scores[Team.A] == 0
    assert simulator.drones[0].snapshot.status is DroneStatus.ELIMINATED


def test_simultaneous_tie_break_uses_drone_ids() -> None:
    a = moving(0, Team.A, (50.0, 30.0), (0.0, 0.0))
    b_low = moving(20, Team.B, (50.5, 30.0), (0.0, 0.0))
    b_high = replace(b_low, id=21)
    simulator = Simulator(scenario_with(a, b_high, b_low))
    event = simulator.step()[0]
    assert event.drone_ids == (0, 20)
