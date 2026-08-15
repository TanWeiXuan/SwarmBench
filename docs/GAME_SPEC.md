# SwarmBench v1 game specification

## Versions and units

The v1 constants are `ENGINE_VERSION=1.0.0`, controller API 1, scenario generator 1, and replay format 1. Coordinates, time, velocity, acceleration, and jerk use SI units. x increases left-to-right and y bottom-to-top.

## Arena and teams

The arena is 100 m × 60 m. Team A spawns near the left boundary and targets the right goal; Team B is mirrored. Each team has exactly ten FAST and ten SLOW drones. FAST specifications are 5 m/s speed, 4 m/s² acceleration, 16 m/s³ jerk, and 1 point. SLOW specifications are 2.5 m/s, 2 m/s², 8 m/s³, and 5 points. The physical radius is 0.25 m; the abstract enemy interception radius is 0.75 m. Friendly drones do not collide.

Each generated rectangular goal is 3 m deep, 14 m high, attached to its arena boundary, and independently placed in y with margins. Six through twelve small non-overlapping circle/axis-aligned-rectangle obstacles are drawn from the seeded generator. Protected spawn/goal-side strips contain no obstacles. Spawn positions have at least 0.8 m spacing and are assigned a deterministically shuffled type order.

Candidate arenas are checked on a 0.5 m grid after inflating obstacles by the 0.25 m drone radius plus 0.35 m planning clearance. Flood fill from each target goal must reach every corresponding spawn. Failed candidates are rejected while continuing the same RNG stream, up to 100 attempts.

## Dynamics

Physics runs at 20 Hz (`dt=0.05 s`) and controls at 10 Hz. A desired acceleration is clipped by vector norm to the type limit. At each physics tick:

```text
delta = desired_acceleration - actual_acceleration
actual_acceleration += norm_clip(delta, max_jerk * dt)
```

The engine then integrates `dp/dt=v`, `dv/dt=a` with RK4 while acceleration is constant over the tick. Velocity and acceleration are norm-clipped after integration. A retained command applies for two physics ticks and remains in force until a valid replacement arrives.

## Events and ordering

The engine performs swept continuous tests from each tick's old to new positions. Drone–drone interception uses relative swept points at 0.75 m. Obstacle tests inflate geometry by 0.25 m. Goal entry uses swept segment/AABB entry.

Candidates sort by normalized contact time, priority, then involved integer drone IDs. Exact-time priorities are obstacle crash, enemy interception, and goal. Once a drone resolves, later candidates involving it are ignored. This enforces one-for-one interceptions. Obstacle/interception sets `ELIMINATED` with its reason; goal entry immediately awards the drone value and sets `SCORED`. Resolved drones leave active play.

## Match result

The default duration is 90 simulation seconds. The higher score wins; equal scores draw. A controller exception or hard timeout forfeits immediately. Glicko-2 observes only win/draw/loss, while scores and timing remain separate statistics.

Both controllers are evaluated concurrently at each control instant and simulation time does not advance while they run. A result after 500 ms but before 5 s is discarded, preserving the controller's internal state and prior commands. At 5 s the process is terminated and forfeits.

## Determinism

The scenario seed and generator version completely determine arena geometry/spawns. Team controller seeds are SHA-256 derivations of match seed and side. All event iteration and tie-breaks are stable. A replay is authoritative metadata plus initial scenario, controller hashes, accepted command changes, important events, and final result; rendering is never authoritative.

