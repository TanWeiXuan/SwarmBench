"""Single-file submission validation, smoke testing, and calibration."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from swarmbench.api import CONTROLLER_PERIOD, DRONE_SPECS, PHYSICS_DT, DroneStatus, GameInfo, GameState, Team
from swarmbench.controller_runner import ControllerError, ControllerProcess
from swarmbench.controllers.baselines import BASELINE_NAMES, baseline_path
from swarmbench.engine import Scenario, Simulator, generate_scenario
from swarmbench.match import controller_seed, run_match
from swarmbench.version import CALIBRATION_VERSION, CONTROLLER_API_VERSION, ENGINE_VERSION

from .glicko2 import GlickoRating, update_rating

MAX_SUBMISSION_BYTES = 5 * 1024 * 1024
ALLOWED_THIRD_PARTY_IMPORTS = {"numpy", "scipy", "torch", "networkx", "swarmbench"}


class SubmissionValidationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ChangedFile:
    path: str
    size: int
    symlink: bool = False


def validate_pr_files(files: list[ChangedFile], author: str) -> str | None:
    submission_files = [item for item in files if PurePosixPath(item.path).parts[:1] == ("submissions",)]
    if not submission_files:
        return None
    if len(files) != 1 or len(submission_files) != 1:
        raise SubmissionValidationError("a controller PR must modify exactly one file")
    item = submission_files[0]
    parts = PurePosixPath(item.path).parts
    if len(parts) != 3 or parts[0] != "submissions" or not parts[2].endswith(".py"):
        raise SubmissionValidationError("controller must be submissions/<github-login>/<controller-name>.py")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", parts[1]) or not re.fullmatch(r"[A-Za-z0-9_.-]+\.py", parts[2]):
        raise SubmissionValidationError("submission path contains unsupported characters")
    if parts[1].casefold() != author.casefold():
        raise SubmissionValidationError("top-level submission folder must match the PR author")
    if item.symlink:
        raise SubmissionValidationError("submission may not be a symlink")
    if item.size > MAX_SUBMISSION_BYTES:
        raise SubmissionValidationError("submission exceeds the 5 MiB limit")
    return item.path


def validate_source(path: str | Path) -> None:
    source_path = Path(path)
    if source_path.is_symlink() or source_path.stat().st_size > MAX_SUBMISSION_BYTES:
        raise SubmissionValidationError("invalid submission file")
    try:
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    except (OSError, UnicodeDecodeError, SyntaxError) as error:
        raise SubmissionValidationError(f"invalid Python source: {error}") from error
    controller_classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SwarmController"]
    if len(controller_classes) != 1:
        raise SubmissionValidationError("define exactly one class named SwarmController")
    if not any(
        (isinstance(base, ast.Name) and base.id == "BaseSwarmController")
        or (isinstance(base, ast.Attribute) and base.attr == "BaseSwarmController")
        for base in controller_classes[0].bases
    ):
        raise SubmissionValidationError("SwarmController must inherit BaseSwarmController")
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module.split(".")[0]]
        for name in names:
            if name not in sys.stdlib_module_names and name not in ALLOWED_THIRD_PARTY_IMPORTS:
                raise SubmissionValidationError(f"unsupported import: {name}")


def _empty_scenario(seed: int, team: Team = Team.A) -> Scenario:
    generated = generate_scenario(seed)
    return Scenario(
        generated.seed,
        generated.generator_version,
        generated.width,
        generated.height,
        generated.goal_for_a,
        generated.goal_for_b,
        (),
        generated.team_drones(team),
    )


def _info(scenario: Scenario, team: Team) -> GameInfo:
    return GameInfo(
        team,
        scenario.width,
        scenario.height,
        scenario.target_goal(team),
        scenario.own_goal(team),
        scenario.obstacles,
        tuple(DRONE_SPECS.items()),
        scenario.team_drones(team),
        scenario.team_drones(team.opponent),
        scenario.seed,
        scenario.generator_version,
        controller_seed(scenario.seed, team),
        CONTROLLER_API_VERSION,
    )


def _state(simulator: Simulator, team: Team) -> GameState:
    return GameState(
        simulator.time,
        simulator.snapshots(team),
        simulator.snapshots(team.opponent),
        simulator.scores[team],
        simulator.scores[team.opponent],
    )


def smoke_test(
    path: str | Path,
    *,
    seed: int = 90210,
    duration: float = 90.0,
    backend: str = "local",
) -> dict[str, Any]:
    validate_source(path)
    scenario = _empty_scenario(seed)
    simulator = Simulator(scenario)
    with ControllerProcess(path, backend=backend) as controller:
        try:
            controller.initialize(_info(scenario, Team.A))
            for tick in range(round(duration / PHYSICS_DT)):
                if tick % round(CONTROLLER_PERIOD / PHYSICS_DT) == 0:
                    result = controller.step(_state(simulator, Team.A))
                    simulator.set_commands(Team.A, result.accepted_changes)
                simulator.step()
        except ControllerError as error:
            raise SubmissionValidationError(str(error)) from error
        timing = controller.stats.summary()
    score = simulator.scores[Team.A]
    if score <= 0:
        raise SubmissionValidationError("empty-arena smoke test scored zero")
    if timing["missed_updates"] or timing["hard_timeouts"] or timing["exceptions"]:
        raise SubmissionValidationError("controller exceeded timing or reliability limits")
    warnings = []
    if timing["p95"] > 0.4 or timing["max"] > 0.45:
        warnings.append("controller timing leaves little margin below the 500 ms soft deadline")
    return {
        "engine_version": ENGINE_VERSION,
        "controller_api_version": CONTROLLER_API_VERSION,
        "score": score,
        "maximum_score": 60,
        "timing": timing,
        "warnings": warnings,
    }


def calibration_seed(seed_index: int) -> int:
    return int.from_bytes(hashlib.sha256(f"calibration:{CALIBRATION_VERSION}:{seed_index}".encode()).digest()[:8], "big") % (2**63)


def calibrate_seed(
    path: str | Path,
    *,
    submission_id: str,
    head_sha: str,
    seed_index: int,
    duration: float = 90.0,
    backend: str = "local",
) -> dict[str, Any]:
    validate_source(path)
    seed = calibration_seed(seed_index)
    games = []
    subject_results: list[tuple[GlickoRating, float]] = []
    for opponent in BASELINE_NAMES:
        first = run_match(path, baseline_path(opponent), seed=seed, duration=duration, backend=backend)
        second = run_match(baseline_path(opponent), path, seed=seed, duration=duration, backend=backend)
        first_score = 0.5 if first.winner is None else (1.0 if first.winner is Team.A else 0.0)
        second_score = 0.5 if second.winner is None else (1.0 if second.winner is Team.B else 0.0)
        subject_results.extend(((GlickoRating(), first_score), (GlickoRating(), second_score)))
        games.extend(
            (
                {"opponent": opponent, "side": "A", "result": first_score, "score_for": first.score_a, "score_against": first.score_b, "timing": first.stats_a},
                {"opponent": opponent, "side": "B", "result": second_score, "score_for": second.score_b, "score_against": second.score_a, "timing": second.stats_b},
            )
        )
    provisional = update_rating(GlickoRating(), subject_results)
    return {
        "schema": "swarmbench-calibration-seed-v1",
        "engine_version": ENGINE_VERSION,
        "calibration_version": CALIBRATION_VERSION,
        "submission_id": submission_id,
        "submission_path": str(path).replace("\\", "/"),
        "head_sha": head_sha,
        "seed_index": seed_index,
        "scenario_seed": seed,
        "opponents": list(BASELINE_NAMES),
        "games": games,
        "provisional": {
            "rating": provisional.rating,
            "deviation": provisional.deviation,
            "volatility": provisional.volatility,
        },
    }


def aggregate_calibration(parts: list[dict[str, Any]], *, submission_id: str, submission_path: str, head_sha: str) -> dict[str, Any]:
    if not parts or len({part.get("seed_index") for part in parts}) != len(parts):
        raise SubmissionValidationError("missing or duplicate calibration seed results")
    games = []
    opponents = set()
    for part in parts:
        if (
            part.get("schema") != "swarmbench-calibration-seed-v1"
            or part.get("engine_version") != ENGINE_VERSION
            or part.get("calibration_version") != CALIBRATION_VERSION
            or part.get("submission_id") != submission_id
            or part.get("head_sha") != head_sha
        ):
            raise SubmissionValidationError("calibration identity mismatch")
        games.extend(part.get("games", []))
        opponents.update(part.get("opponents", []))
    results = [float(game["result"]) for game in games]
    if any(result not in {0.0, 0.5, 1.0} for result in results):
        raise SubmissionValidationError("invalid calibration game result")
    provisional = update_rating(GlickoRating(), [(GlickoRating(), result) for result in results])
    timings = [game["timing"] for game in games]
    artifact = {
        "schema": "swarmbench-calibration-v1",
        "engine_version": ENGINE_VERSION,
        "calibration_version": CALIBRATION_VERSION,
        "submission_id": submission_id,
        "submission_path": submission_path.replace("\\", "/"),
        "head_sha": head_sha,
        "opponents": sorted(opponents),
        "match_count": len(games),
        "wins": sum(result == 1.0 for result in results),
        "draws": sum(result == 0.5 for result in results),
        "losses": sum(result == 0.0 for result in results),
        "score_mean": sum(float(game["score_for"]) for game in games) / len(games),
        "timing": {
            "mean": sum(float(item["mean"]) for item in timings) / len(timings),
            "p95": max(float(item["p95"]) for item in timings),
            "max": max(float(item["max"]) for item in timings),
            "missed_updates": sum(int(item["missed_updates"]) for item in timings),
            "hard_timeouts": sum(int(item["hard_timeouts"]) for item in timings),
            "invalid_actions": sum(int(item["invalid_actions"]) for item in timings),
        },
        "provisional_rating": provisional.rating,
        "deviation": provisional.deviation,
        "volatility": provisional.volatility,
    }
    validate_calibration_artifact(artifact)
    return artifact


def validate_calibration_artifact(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict) or data.get("schema") != "swarmbench-calibration-v1":
        raise SubmissionValidationError("invalid calibration schema")
    required_strings = ("engine_version", "submission_id", "submission_path", "head_sha")
    if any(not isinstance(data.get(key), str) or not data[key] for key in required_strings):
        raise SubmissionValidationError("invalid calibration identity")
    if data["engine_version"] != ENGINE_VERSION or data.get("calibration_version") != CALIBRATION_VERSION:
        raise SubmissionValidationError("unsupported calibration version")
    if not re.fullmatch(r"[0-9a-fA-F]{40,64}", data["head_sha"]):
        raise SubmissionValidationError("invalid head SHA")
    integer_fields = ("match_count", "wins", "draws", "losses")
    if any(not isinstance(data.get(key), int) or data[key] < 0 for key in integer_fields):
        raise SubmissionValidationError("invalid calibration counts")
    if data["match_count"] != data["wins"] + data["draws"] + data["losses"]:
        raise SubmissionValidationError("calibration counts do not add up")
    numeric_ranges = {"score_mean": (0, 60), "provisional_rating": (0, 4000), "deviation": (0, 350), "volatility": (0.001, 1)}
    for key, (low, high) in numeric_ranges.items():
        value = data.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
            raise SubmissionValidationError(f"invalid {key}")
    timing = data.get("timing")
    if not isinstance(timing, dict) or any(not isinstance(value, (int, float)) or value < 0 for value in timing.values()):
        raise SubmissionValidationError("invalid timing metrics")
    if not isinstance(data.get("opponents"), list) or any(not isinstance(value, str) for value in data["opponents"]):
        raise SubmissionValidationError("invalid opponent list")
    return data


def _changed_files(base_ref: str) -> list[ChangedFile]:
    completed = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=ACMR", f"{base_ref}...HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    files = []
    for name in completed.stdout.splitlines():
        path = Path(name)
        files.append(ChangedFile(name.replace("\\", "/"), path.lstat().st_size, path.is_symlink()))
    return files


def validate_controller_cli(path: Path) -> int:
    try:
        result = smoke_test(path, backend=os.environ.get("SWARMBENCH_BACKEND", "local"))
    except SubmissionValidationError as error:
        print(f"validation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    structure = subparsers.add_parser("structure")
    structure.add_argument("--base-ref", required=True)
    structure.add_argument("--author", required=True)
    structure.add_argument("--output", type=Path, required=True)
    source = subparsers.add_parser("source")
    source.add_argument("path", type=Path)
    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("path", type=Path)
    smoke.add_argument("--output", type=Path, required=True)
    calibration = subparsers.add_parser("calibrate")
    calibration.add_argument("path", type=Path)
    calibration.add_argument("--submission-id", required=True)
    calibration.add_argument("--head-sha", required=True)
    calibration.add_argument("--seed-index", type=int, required=True)
    calibration.add_argument("--output", type=Path, required=True)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("directory", type=Path)
    aggregate.add_argument("--submission-id", required=True)
    aggregate.add_argument("--submission-path", required=True)
    aggregate.add_argument("--head-sha", required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "structure":
            path = validate_pr_files(_changed_files(args.base_ref), args.author)
            args.output.write_text(json.dumps({"submission_path": path}) + "\n", encoding="utf-8")
        elif args.command == "source":
            validate_source(args.path)
        elif args.command == "smoke":
            result = smoke_test(args.path, backend=os.environ.get("SWARMBENCH_BACKEND", "local"))
            args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        elif args.command == "calibrate":
            result = calibrate_seed(
                args.path,
                submission_id=args.submission_id,
                head_sha=args.head_sha,
                seed_index=args.seed_index,
                backend=os.environ.get("SWARMBENCH_BACKEND", "local"),
            )
            args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        else:
            parts = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(args.directory.rglob("calibration-seed-*.json"))]
            result = aggregate_calibration(parts, submission_id=args.submission_id, submission_path=args.submission_path, head_sha=args.head_sha)
            args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    except SubmissionValidationError as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
