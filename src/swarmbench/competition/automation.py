"""Trusted tournament workflow preparation, reporting, and publication."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from swarmbench.controllers.baselines import BASELINE_NAMES, baseline_path
from swarmbench.version import CONTROLLER_API_VERSION, ENGINE_VERSION, SCENARIO_GENERATOR_VERSION, TOURNAMENT_FORMAT_VERSION

from .matchmaking import ScheduledGame
from .publisher import update_readme_leaderboard
from .ratings import RatingRecord, load_ratings, ratings_to_dict, save_ratings
from .tournament import TournamentPlan, aggregate_batches, create_plan, execute_batch, validate_batch


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def resolve_seed(value: str | None, run_id: str) -> int:
    if value:
        try:
            return int(value) % (2**63)
        except ValueError:
            payload = value.encode()
    else:
        payload = f"github-run:{run_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63)


def _record_dict(record: RatingRecord) -> dict[str, Any]:
    return asdict(record)


def prepare_plan(
    ratings: dict[str, RatingRecord],
    *,
    seed: int,
    mode: str,
    size: str,
    run_id: str,
    repository: str,
) -> dict[str, Any]:
    plan = create_plan(ratings, seed, mode=mode, size=size)
    return {
        "format_version": TOURNAMENT_FORMAT_VERSION,
        "engine_version": ENGINE_VERSION,
        "controller_api_version": CONTROLLER_API_VERSION,
        "generator_version": SCENARIO_GENERATOR_VERSION,
        "tournament_id": run_id,
        "run_url": f"https://github.com/{repository}/actions/runs/{run_id}",
        "started_at": _now(),
        "seed": plan.seed,
        "mode": plan.mode,
        "size": size,
        "ratings": [_record_dict(ratings[key]) for key in sorted(ratings)],
        "pairings": [list(pair) for pair in plan.pairings],
        "games": [asdict(game) for game in plan.games],
        "batches": [list(batch) for batch in plan.batches],
    }


def validate_plan(data: Any) -> tuple[TournamentPlan, dict[str, RatingRecord]]:
    if not isinstance(data, dict) or data.get("format_version") != TOURNAMENT_FORMAT_VERSION or data.get("engine_version") != ENGINE_VERSION:
        raise ValueError("invalid tournament plan identity")
    try:
        records = {}
        for item in data["ratings"]:
            record = RatingRecord(**item)
            if record.controller_id in records:
                raise ValueError("duplicate controller")
            records[record.controller_id] = record
        pairings = tuple(tuple(pair) for pair in data["pairings"])
        games = tuple(ScheduledGame(**game) for game in data["games"])
        batches = tuple(tuple(batch) for batch in data["batches"])
        plan = TournamentPlan(int(data["seed"]), str(data["mode"]), pairings, games, batches)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid tournament plan: {error}") from error
    if len(batches) != 5 or set(game.game_id for game in games) != set().union(*map(set, batches)):
        raise ValueError("plan batch coverage mismatch")
    if any(game.controller_a not in records or game.controller_b not in records for game in games):
        raise ValueError("plan references unknown controller")
    return plan, records


def controller_paths(records: dict[str, RatingRecord], root: Path) -> dict[str, Path]:
    paths = {}
    for controller_id, record in records.items():
        if record.built_in:
            if controller_id not in BASELINE_NAMES:
                raise ValueError(f"unknown built-in controller: {controller_id}")
            paths[controller_id] = baseline_path(controller_id)
        else:
            parts = controller_id.split("/", 1)
            if len(parts) != 2:
                raise ValueError("invalid community controller ID")
            paths[controller_id] = root / "submissions" / parts[0] / f"{parts[1]}.py"
            if not paths[controller_id].is_file():
                raise ValueError(f"controller file is missing: {controller_id}")
    return paths


def _gh_graphql(query: str, **fields: str) -> dict[str, Any]:
    command = ["gh", "api", "graphql", "-f", f"query={query}"]
    for key, value in fields.items():
        command.extend(("-f", f"{key}={value}"))
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def _discussion_category(owner: str, name: str) -> tuple[str, str]:
    query = "query($owner:String!,$name:String!){repository(owner:$owner,name:$name){id discussionCategories(first:100){nodes{id name}}}}"
    data = _gh_graphql(query, owner=owner, name=name)["data"]["repository"]
    category = next((item for item in data["discussionCategories"]["nodes"] if item["name"] == "Tournament Results"), None)
    if category is None:
        raise RuntimeError("Tournament Results Discussion category is missing")
    return data["id"], category["id"]


def initial_discussion_body(data: dict[str, Any]) -> str:
    return "\n".join(
        (
            "## Status: RUNNING",
            "",
            f"- Tournament ID: `{data['tournament_id']}`",
            f"- Mode: **{data['mode']}**",
            f"- [GitHub Actions run]({data['run_url']})",
            f"- Engine/API/generator: `{data['engine_version']}` / `{data['controller_api_version']}` / `{data['generator_version']}`",
            f"- Tournament seed: `{data['seed']}`",
            f"- Controllers: {len(data['ratings'])}",
            f"- Planned pairings: {len(data['pairings'])}",
            f"- Planned games: {len(data['games'])}",
            f"- Started: {data['started_at']}",
            "",
            "Provisional progress will be posted at approximately 20% increments. Ratings change only after every batch validates.",
        )
    )


def create_discussion(data: dict[str, Any], repository: str) -> dict[str, str]:
    owner, name = repository.split("/", 1)
    repository_id, category_id = _discussion_category(owner, name)
    kind = "Ranking" if data["mode"] == "official" else "Exhibition"
    title = f"{kind} Tournament #{data['tournament_id']} — {datetime.now(UTC).date().isoformat()}"
    mutation = "mutation($repositoryId:ID!,$categoryId:ID!,$title:String!,$body:String!){createDiscussion(input:{repositoryId:$repositoryId,categoryId:$categoryId,title:$title,body:$body}){discussion{id url}}}"
    result = _gh_graphql(
        mutation,
        repositoryId=repository_id,
        categoryId=category_id,
        title=title,
        body=initial_discussion_body(data),
    )
    discussion = result["data"]["createDiscussion"]["discussion"]
    return {"id": discussion["id"], "url": discussion["url"], "title": title}


def _add_comment(discussion_id: str, body: str) -> None:
    mutation = "mutation($discussionId:ID!,$body:String!){addDiscussionComment(input:{discussionId:$discussionId,body:$body}){comment{id}}}"
    _gh_graphql(mutation, discussionId=discussion_id, body=body)


def _update_discussion(discussion_id: str, body: str) -> None:
    mutation = "mutation($discussionId:ID!,$body:String!){updateDiscussion(input:{discussionId:$discussionId,body:$body}){discussion{id}}}"
    _gh_graphql(mutation, discussionId=discussion_id, body=body)


def _load_batches(directory: Path) -> list[dict[str, Any]]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(directory.rglob("batch-*.json"))]


def progress_summary(data: dict[str, Any], batches: list[dict[str, Any]], completed_index: int) -> str:
    plan, _ = validate_plan(data)
    games = []
    for index, batch in enumerate(batches):
        games.extend(validate_batch(plan, batch, index))
    expected_completed = sum(len(plan.batches[index]) for index in range(completed_index + 1))
    if len(games) != expected_completed:
        raise ValueError("progress result count mismatch")
    hard = sum(int(game[side].get("hard_timeouts", 0)) for game in games for side in ("stats_a", "stats_b"))
    soft = sum(int(game[side].get("missed_updates", 0)) for game in games for side in ("stats_a", "stats_b"))
    exceptions = sum(int(game[side].get("exceptions", 0)) for game in games for side in ("stats_a", "stats_b"))
    percent = round(100 * len(games) / len(plan.games)) if plan.games else 100
    pairings = len({game["pairing_id"] for game in games})
    return "\n".join(
        (
            f"### Progress — {percent}%",
            "",
            f"Completed: {len(games)} / {len(plan.games)} games",
            f"Pairings touched: {pairings} / {len(plan.pairings)}",
            f"Controller exceptions: {exceptions}",
            f"Hard timeouts: {hard}",
            f"Soft-deadline misses: {soft}",
            "",
            "Current results are provisional; Glicko-2 updates are applied only when the full rating period completes.",
        )
    )


def _game_totals(games: tuple[dict[str, Any], ...]) -> tuple[int, int, int]:
    wins = sum(game["result_a"] == 1.0 for game in games)
    draws = sum(game["result_a"] == 0.5 for game in games)
    losses = len(games) - wins - draws
    return wins, draws, losses


def final_report(data: dict[str, Any], outcome) -> str:
    plan, _ = validate_plan(data)
    wins, draws, losses = _game_totals(outcome.games)
    hard = sum(int(game[side].get("hard_timeouts", 0)) for game in outcome.games for side in ("stats_a", "stats_b"))
    soft = sum(int(game[side].get("missed_updates", 0)) for game in outcome.games for side in ("stats_a", "stats_b"))
    exceptions = sum(int(game[side].get("exceptions", 0)) for game in outcome.games for side in ("stats_a", "stats_b"))
    aggregate = {controller_id: [0, 0, 0] for controller_id in outcome.ratings_before}
    for game in outcome.games:
        left, right, score = game["controller_a"], game["controller_b"], game["result_a"]
        if score == 1.0:
            aggregate[left][0] += 1
            aggregate[right][2] += 1
        elif score == 0.0:
            aggregate[right][0] += 1
            aggregate[left][2] += 1
        else:
            aggregate[left][1] += 1
            aggregate[right][1] += 1
    deltas = sorted(
        ((key, outcome.ratings_after[key].rating - record.rating) for key, record in outcome.ratings_before.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    closest = min(outcome.games, key=lambda game: (abs(game["score_a"] - game["score_b"]), game["game_id"])) if outcome.games else None
    upsets = []
    timing_means = []
    timing_p95 = []
    timing_maximums = []
    for game in outcome.games:
        for side in ("stats_a", "stats_b"):
            timing_means.append(float(game[side].get("mean", 0.0)))
            timing_p95.append(float(game[side].get("p95", 0.0)))
            timing_maximums.append(float(game[side].get("max", 0.0)))
        if game["result_a"] in {0.0, 1.0}:
            winner = game["controller_a"] if game["result_a"] == 1.0 else game["controller_b"]
            loser = game["controller_b"] if game["result_a"] == 1.0 else game["controller_a"]
            gap = outcome.ratings_before[loser].rating - outcome.ratings_before[winner].rating
            if gap > 0:
                upsets.append((gap, game["game_id"], winner, loser))
    lines = [
        "## Status: COMPLETE",
        "",
        f"- Tournament ID: `{data['tournament_id']}`",
        f"- Mode: **{plan.mode}**",
        f"- Started / ended: {data['started_at']} / {_now()}",
        f"- [GitHub Actions run]({data['run_url']})",
        f"- Engine/API/generator: `{data['engine_version']}` / `{data['controller_api_version']}` / `{data['generator_version']}`",
        f"- Tournament seed: `{plan.seed}`",
        f"- Controllers / pairings / games: {len(outcome.ratings_before)} / {len(plan.pairings)} / {len(outcome.games)}",
        f"- Side-A W/D/L: {wins} / {draws} / {losses}",
        f"- Exceptions / hard timeouts / soft misses: {exceptions} / {hard} / {soft}",
        f"- Timing mean / worst p95 / maximum: {(sum(timing_means) / len(timing_means) if timing_means else 0.0) * 1000:.2f} / {max(timing_p95, default=0.0) * 1000:.2f} / {max(timing_maximums, default=0.0) * 1000:.2f} ms",
        "",
        "### Rating period",
        "",
        "| Controller | Before | After | Delta | W-D-L |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for controller_id in sorted(outcome.ratings_before):
        before, after = outcome.ratings_before[controller_id], outcome.ratings_after[controller_id]
        w, d, loss = aggregate[controller_id]
        lines.append(f"| {controller_id} | {before.rating:.1f} | {after.rating:.1f} | {after.rating - before.rating:+.1f} | {w}-{d}-{loss} |")
    if deltas:
        lines.extend(("", f"Biggest gain: `{deltas[0][0]}` ({deltas[0][1]:+.1f}); biggest loss: `{deltas[-1][0]}` ({deltas[-1][1]:+.1f})."))
    if closest:
        lines.append(f"Closest game: `{closest['game_id']}` — {closest['score_a']}-{closest['score_b']}.")
    if upsets:
        gap, game_id, winner, loser = max(upsets)
        lines.append(f"Notable upset: `{winner}` defeated `{loser}` in `{game_id}` despite a {gap:.1f}-point pre-period rating gap.")
    if plan.mode == "official":
        top = sorted((record for record in outcome.ratings_after.values() if not record.built_in), key=lambda record: (-record.rating, record.controller_id))[:10]
        lines.extend(("", "### Community top 10", ""))
        if top:
            lines.extend(f"{index}. `{record.controller_id}` — {record.rating:.1f} ± {record.deviation:.1f}" for index, record in enumerate(top, 1))
        else:
            lines.append("No community controllers have been accepted yet.")
    lines.extend(("", "### Pairing details", ""))
    for pairing in plan.pairings:
        selected = [game for game in outcome.games if {game["controller_a"], game["controller_b"]} == set(pairing)]
        first_wins = sum((game["result_a"] if game["controller_a"] == pairing[0] else 1 - game["result_a"]) == 1.0 for game in selected)
        pairing_draws = sum(game["result_a"] == 0.5 for game in selected)
        lines.append(f"- `{pairing[0]}` vs `{pairing[1]}`: {first_wins} wins / {pairing_draws} draws / {len(selected) - first_wins - pairing_draws} losses for `{pairing[0]}`")
    lines.extend(("", "Selected compact replay JSON files are attached to this workflow run's `tournament-replays` artifact."))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--ratings", type=Path, required=True)
    prepare.add_argument("--mode", required=True)
    prepare.add_argument("--size", required=True)
    prepare.add_argument("--seed")
    prepare.add_argument("--run-id", required=True)
    prepare.add_argument("--repository", required=True)
    prepare.add_argument("--output", type=Path, required=True)
    discussion = subparsers.add_parser("discussion-start")
    discussion.add_argument("--plan", type=Path, required=True)
    discussion.add_argument("--repository", required=True)
    discussion.add_argument("--output", type=Path, required=True)
    compute = subparsers.add_parser("compute")
    compute.add_argument("--plan", type=Path, required=True)
    compute.add_argument("--batch", type=int, required=True)
    compute.add_argument("--output", type=Path, required=True)
    compute.add_argument("--replay-dir", type=Path, required=True)
    compute.add_argument("--duration", type=float, default=90.0)
    report = subparsers.add_parser("report")
    report.add_argument("--plan", type=Path, required=True)
    report.add_argument("--discussion", type=Path, required=True)
    report.add_argument("--batches", type=Path, required=True)
    report.add_argument("--completed-index", type=int, required=True)
    final = subparsers.add_parser("final")
    final.add_argument("--plan", type=Path, required=True)
    final.add_argument("--discussion", type=Path, required=True)
    final.add_argument("--batches", type=Path, required=True)
    final.add_argument("--ratings", type=Path, required=True)
    final.add_argument("--readme", type=Path, required=True)
    final.add_argument("--output", type=Path, required=True)
    failure = subparsers.add_parser("failure")
    failure.add_argument("--plan", type=Path, required=True)
    failure.add_argument("--discussion", type=Path, required=True)
    failure.add_argument("--stage", required=True)
    args = parser.parse_args(argv)

    if args.command == "prepare":
        seed = resolve_seed(args.seed, args.run_id)
        data = prepare_plan(load_ratings(args.ratings), seed=seed, mode=args.mode, size=args.size, run_id=args.run_id, repository=args.repository)
        args.output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    elif args.command == "discussion-start":
        data = json.loads(args.plan.read_text(encoding="utf-8"))
        validate_plan(data)
        args.output.write_text(json.dumps(create_discussion(data, args.repository), indent=2) + "\n", encoding="utf-8")
    elif args.command == "compute":
        data = json.loads(args.plan.read_text(encoding="utf-8"))
        plan, records = validate_plan(data)
        result = execute_batch(
            plan,
            args.batch,
            controller_paths(records, Path.cwd()),
            duration=args.duration,
            backend=os.environ.get("SWARMBENCH_BACKEND", "local"),
            replay_dir=args.replay_dir,
        )
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    elif args.command == "report":
        data = json.loads(args.plan.read_text(encoding="utf-8"))
        discussion_info = json.loads(args.discussion.read_text(encoding="utf-8"))
        batches = _load_batches(args.batches)
        summary = progress_summary(data, batches, args.completed_index)
        _update_discussion(discussion_info["id"], initial_discussion_body(data) + "\n\n" + summary)
        _add_comment(discussion_info["id"], summary)
    elif args.command == "final":
        data = json.loads(args.plan.read_text(encoding="utf-8"))
        discussion_info = json.loads(args.discussion.read_text(encoding="utf-8"))
        plan, records = validate_plan(data)
        outcome = aggregate_batches(plan, _load_batches(args.batches), records)
        report_body = final_report(data, outcome)
        if plan.mode == "official":
            save_ratings(outcome.ratings_after, args.ratings)
            update_readme_leaderboard(args.readme, outcome.ratings_after)
        _update_discussion(discussion_info["id"], report_body)
        _add_comment(discussion_info["id"], "### Final result\n\n" + report_body)
        args.output.write_text(
            json.dumps(
                {
                    "format_version": TOURNAMENT_FORMAT_VERSION,
                    "mode": plan.mode,
                    "game_count": len(outcome.games),
                    "ratings_before": ratings_to_dict(outcome.ratings_before),
                    "ratings_after": ratings_to_dict(outcome.ratings_after),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    else:
        data = json.loads(args.plan.read_text(encoding="utf-8"))
        discussion_info = json.loads(args.discussion.read_text(encoding="utf-8"))
        body = initial_discussion_body(data).replace("## Status: RUNNING", "## Status: FAILED")
        _update_discussion(discussion_info["id"], body)
        _add_comment(discussion_info["id"], f"### Tournament failed\n\nThe workflow stopped during `{args.stage}`. Official ratings were not partially updated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
