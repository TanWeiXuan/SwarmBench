# Controller API v1

Controllers import the public surface from `swarmbench` and define exactly one `SwarmController(BaseSwarmController)`.

```python
from swarmbench import BaseSwarmController, DroneStatus


class SwarmController(BaseSwarmController):
    def initialize(self, game_info):
        self.goal = game_info.target_goal
        self.history = []

    def step(self, state):
        self.history.append(state.time)
        return {
            drone.id: (2.0, 0.0)
            for drone in state.own_drones
            if drone.status is DroneStatus.ACTIVE
        }
```

`initialize(game_info)` is called once with team/side, arena size, target and own goals, immutable obstacles, FAST/SLOW specifications, both initial teams, scenario identity, controller RNG seed, and API version. Initialization has a separate 10 s watchdog. Load a PyTorch model or precompute paths here rather than at module scope.

`step(state)` is called at 10 Hz simulation time. State contains timestamp, complete immutable own/opponent snapshots, and both scores. A drone snapshot has ID, team, type, position, velocity, acceleration, status, and optional destruction reason.

Return a dictionary mapping own integer IDs to two finite numeric acceleration components. The engine clips over-limit finite vectors by norm. Omitted active drones keep their previous command. A malformed individual command also keeps its previous value. Unknown IDs are ignored and counted. Initial commands are `(0, 0)`.

The same Python object handles every call in one match, and a fresh subprocess/object is created for the next match. Internal state on `self` is unrestricted within sandbox/resource constraints.

Both sides run concurrently. The 500 ms step limit is soft: late actions are discarded, earlier commands remain, and state mutations made by the completed call persist. The 5 s watchdog is hard and forfeits. Stay comfortably below 500 ms; CI warns when p95 exceeds 400 ms or maximum exceeds 450 ms.

## PyTorch pattern

PyTorch CPU is in the fixed environment. A useful policy can embed small weights directly in the single source file:

```python
class SwarmController(BaseSwarmController):
    def initialize(self, game_info):
        import torch
        self.torch = torch
        self.weight = torch.tensor([[0.8, 0.0], [0.0, 0.8]])

    def step(self, state):
        actions = {}
        for drone in state.own_drones:
            delta = self.torch.tensor(self.target) - self.torch.tensor(drone.position)
            actions[drone.id] = tuple((self.weight @ delta).tolist())
        return actions
```

Official inference is CPU-only with one-thread defaults and deterministic algorithms enabled where PyTorch supports them. External native libraries may still introduce platform variation.

