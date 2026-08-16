# Built-in controllers

SwarmBench includes five deterministic baseline controllers. They use the same
public `BaseSwarmController` API as community submissions and expose a
`SwarmController` alias so the normal controller runner can load them.

These baselines are intentionally understandable reference strategies, not
competition-grade agents:

- **Rush** (`rush.py`) sends every active drone toward its target goal using
  velocity-aware steering and local obstacle avoidance.
- **Defend** (`defend.py`) advances SLOW drones while FAST drones intercept the
  most urgent enemy in their defensive half. FAST drones advance when there is
  no useful defensive target.
- **Greedy value** (`greedy_value.py`) greedily gives FAST defenders unique
  enemy targets, favoring nearby high-value SLOW drones. Unassigned drones
  continue toward the goal.
- **Assignment** (`assignment.py`) recomputes a global minimum-cost assignment
  from FAST defenders to active enemies every five controller steps using
  SciPy's linear-sum assignment solver. Its cost considers distance, enemy
  point value, and progress toward the defended goal.
- **Potential field** (`potential_field.py`) combines goal attraction with
  obstacle avoidance, friendly separation, and short-range attraction from
  FAST drones toward valuable enemies.

Shared goal targeting, braking, speed alignment, and obstacle steering live in
`baselines/common.py`. All five strategies retain state for one match and are
re-created for the next match, just like submitted controllers.

Use a built-in controller by its short name:

```bash
swarmbench match --controller-a rush --controller-b assignment --seed 42
```

See [`docs/CONTROLLER_API.md`](../../../docs/CONTROLLER_API.md) for the public
API and a minimal controller example.
