"""Direct goal-seeking baseline with local obstacle steering."""

from swarmbench import BaseSwarmController, DroneStatus

from swarmbench.controllers.baselines.common import goal_target, steer


class RushController(BaseSwarmController):
    def initialize(self, game_info):
        self.goal = game_info.target_goal
        self.obstacles = game_info.obstacles
        self.specs = dict(game_info.drone_specs)

    def step(self, state):
        return {
            drone.id: steer(drone, goal_target(self.goal, drone), self.specs[drone.drone_type], self.obstacles)
            for drone in state.own_drones
            if drone.status is DroneStatus.ACTIVE
        }


SwarmController = RushController
