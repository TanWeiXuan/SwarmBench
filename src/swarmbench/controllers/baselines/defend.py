"""SLOW attackers supported by FAST defenders."""

from swarmbench import BaseSwarmController, DroneStatus, DroneType, Team

from swarmbench.controllers.baselines.common import distance, goal_target, steer


class DefendController(BaseSwarmController):
    def initialize(self, game_info):
        self.team = game_info.team
        self.goal = game_info.target_goal
        self.own_goal = game_info.own_goal
        self.obstacles = game_info.obstacles
        self.specs = dict(game_info.drone_specs)

    def step(self, state):
        enemies = [drone for drone in state.opponent_drones if drone.status is DroneStatus.ACTIVE]
        actions = {}
        for drone in state.own_drones:
            if drone.status is not DroneStatus.ACTIVE:
                continue
            target = goal_target(self.goal, drone)
            if drone.drone_type is DroneType.FAST and enemies:
                threats = sorted(
                    enemies,
                    key=lambda enemy: (
                        distance(enemy.position, self.own_goal.center) + 0.35 * distance(drone.position, enemy.position),
                        enemy.id,
                    ),
                )
                threat = threats[0]
                in_defensive_half = threat.position[0] < 55.0 if self.team is Team.A else threat.position[0] > 45.0
                if in_defensive_half:
                    target = threat.position
            actions[drone.id] = steer(drone, target, self.specs[drone.drone_type], self.obstacles)
        return actions


SwarmController = DefendController
