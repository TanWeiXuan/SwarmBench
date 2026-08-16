"""Matplotlib replay renderer kept separate from authoritative simulation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from swarmbench.api import PHYSICS_DT, CircleObstacle, DroneStatus, DroneType, RectangleObstacle, Team

from .format import Replay, ReplayFrame, reconstruct_frames


def explosion_events_at(replay: Replay, timestamp: float, lifetime: float = 0.25) -> list[dict[str, Any]]:
    return [
        event
        for event in replay.events
        if event.get("type") in {"INTERCEPTION", "OBSTACLE_CRASH"}
        and 0.0 <= timestamp - float(event["time"]) <= lifetime
    ]


def _draw_frame(axis: Any, replay: Replay, frame: ReplayFrame, trails: dict[int, list[tuple[float, float]]]) -> None:
    from matplotlib.patches import Circle, Rectangle

    axis.clear()
    scenario = replay.scenario
    axis.set(xlim=(0, scenario.width), ylim=(0, scenario.height), aspect="equal")
    axis.set_facecolor("#f7f7f2")
    axis.add_patch(Rectangle((0, 0), scenario.width, scenario.height, fill=False, edgecolor="#222222", linewidth=1.5))
    for goal, color in ((scenario.goal_for_a, "#4f8dd6"), (scenario.goal_for_b, "#dc5a5a")):
        axis.add_patch(
            Rectangle((goal.x_min, goal.y_min), goal.x_max - goal.x_min, goal.y_max - goal.y_min, color=color, alpha=0.20)
        )
    for obstacle in scenario.obstacles:
        if isinstance(obstacle, CircleObstacle):
            axis.add_patch(Circle(obstacle.center, obstacle.radius, color="#555555"))
        elif isinstance(obstacle, RectangleObstacle):
            axis.add_patch(
                Rectangle(
                    (obstacle.x_min, obstacle.y_min),
                    obstacle.x_max - obstacle.x_min,
                    obstacle.y_max - obstacle.y_min,
                    color="#555555",
                )
            )
    for drone in frame.drones:
        if drone.status is not DroneStatus.ACTIVE:
            continue
        trails.setdefault(drone.id, []).append(drone.position)
        trails[drone.id] = trails[drone.id][-20:]
        color = "#1769aa" if drone.team is Team.A else "#c62828"
        marker = "o" if drone.drone_type is DroneType.FAST else "s"
        if len(trails[drone.id]) > 1:
            xs, ys = zip(*trails[drone.id])
            axis.plot(xs, ys, color=color, alpha=0.18, linewidth=0.7)
        axis.scatter(*drone.position, color=color, marker=marker, s=22 if marker == "o" else 30, zorder=5)
    for event in explosion_events_at(replay, frame.time):
        age = frame.time - float(event["time"])
        size = 40 + 260 * age / 0.25
        axis.scatter(*event["position"], marker="*", s=size, color="#ff8f00", alpha=max(0.1, 1 - age / 0.25), zorder=10)
    axis.set_title(f"SwarmBench  t={frame.time:05.2f}s   A {frame.scores[0]} — {frame.scores[1]} B")
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")


def render_replay(
    replay: Replay,
    output: str | Path | None = None,
    *,
    fps: int = 10,
    quality: str = "low",
) -> Path | None:
    import matplotlib.pyplot as plt
    from matplotlib import animation

    source_fps = round(1 / PHYSICS_DT)
    if fps not in {5, 10, source_fps}:
        raise ValueError(f"fps must be one of 5, 10, or {source_fps}")
    if quality == "low":
        figure_size, dpi, bitrate = (8, 4.8), 80, 1000
    elif quality == "high":
        figure_size, dpi, bitrate = (10, 6), 140, 1800
    else:
        raise ValueError("quality must be 'low' or 'high'")

    print("Reconstructing replay frames...", flush=True)
    frames = list(reconstruct_frames(replay, every_ticks=source_fps // fps))
    print(f"Prepared {len(frames)} frames at {fps} FPS ({quality} quality).", flush=True)
    figure, axis = plt.subplots(figsize=figure_size, dpi=dpi, constrained_layout=True)
    trails: dict[int, list[tuple[float, float]]] = {}
    destination = Path(output) if output is not None else None
    if destination is not None and destination.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        print(f"Rendering final frame to {destination}...", flush=True)
        _draw_frame(axis, replay, frames[-1], trails)
        figure.savefig(destination, dpi=dpi)
        plt.close(figure)
        print(f"Finished rendering {destination}.", flush=True)
        return destination

    def update(index: int):
        _draw_frame(axis, replay, frames[index], trails)
        return ()

    movie = animation.FuncAnimation(figure, update, frames=len(frames), interval=1000 / fps, blit=False)
    if destination is None:
        print("Opening interactive renderer...", flush=True)
        plt.show()
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix.lower() == ".mp4" and animation.writers.is_available("ffmpeg"):
        writer = animation.FFMpegWriter(fps=fps, bitrate=bitrate)
    else:
        if destination.suffix.lower() != ".gif":
            destination = destination.with_suffix(".gif")
        writer = animation.PillowWriter(fps=fps)

    print(f"Encoding {len(frames)} frames to {destination}...", flush=True)
    last_reported = -1

    def report_progress(current_frame: int, total_frames: int) -> None:
        nonlocal last_reported
        total = total_frames or len(frames)
        percent = min(100, (current_frame + 1) * 100 // total)
        milestone = percent // 10 * 10
        if milestone > last_reported:
            last_reported = milestone
            print(f"Rendering progress: {milestone}%", flush=True)

    movie.save(destination, writer=writer, dpi=dpi, progress_callback=report_progress)
    plt.close(figure)
    print(f"Finished rendering {destination}.", flush=True)
    return destination


def render_arena(scenario, output: str | Path) -> Path:
    replay = Replay(scenario, {"id": "none", "sha256": ""}, {"id": "none", "sha256": ""}, [], [], 0.0, {"A": 0, "B": 0}, "DRAW")
    return render_replay(replay, output)  # type: ignore[return-value]

