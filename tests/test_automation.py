import json
from pathlib import Path

import pytest

from swarmbench.competition.automation import prepare_plan, progress_summary, resolve_seed, validate_plan
from swarmbench.competition.publisher import leaderboard_markdown, update_readme_leaderboard
from swarmbench.competition.ratings import RatingRecord
from swarmbench.version import ENGINE_VERSION, TOURNAMENT_FORMAT_VERSION


def records() -> dict[str, RatingRecord]:
    return {
        f"c{index}": RatingRecord(f"c{index}", f"C{index}", "alice", rating=1400 + index * 50)
        for index in range(5)
    }


def test_seed_resolution_handles_integer_string_and_empty() -> None:
    assert resolve_seed("42", "10") == 42
    assert resolve_seed("named-seed", "10") == resolve_seed("named-seed", "99")
    assert resolve_seed(None, "10") == resolve_seed(None, "10")


def test_automation_plan_round_trip_and_batch_coverage() -> None:
    data = prepare_plan(records(), seed=42, mode="official", size="small", run_id="123", repository="owner/repo")
    plan, restored = validate_plan(json.loads(json.dumps(data)))
    assert restored == records()
    assert len(plan.batches) == 5
    assert set(game.game_id for game in plan.games) == set().union(*map(set, plan.batches))


def test_plan_tampering_is_rejected() -> None:
    data = prepare_plan(records(), seed=42, mode="official", size="small", run_id="123", repository="owner/repo")
    data["engine_version"] = "tampered"
    with pytest.raises(ValueError):
        validate_plan(data)


def test_progress_summary_validates_each_completed_batch() -> None:
    data = prepare_plan(records(), seed=42, mode="official", size="small", run_id="123", repository="owner/repo")
    plan, _ = validate_plan(data)
    batch = {
        "format_version": TOURNAMENT_FORMAT_VERSION,
        "engine_version": ENGINE_VERSION,
        "tournament_seed": plan.seed,
        "batch_index": 0,
        "expected_game_ids": sorted(plan.batches[0]),
        "games": [],
    }
    games = {game.game_id: game for game in plan.games}
    for game_id in plan.batches[0]:
        game = games[game_id]
        batch["games"].append(
            {
                "game_id": game.game_id,
                "pairing_id": game.pairing_id,
                "controller_a": game.controller_a,
                "controller_b": game.controller_b,
                "scenario_seed": game.scenario_seed,
                "score_a": 0,
                "score_b": 0,
                "result_a": 0.5,
                "stats_a": {},
                "stats_b": {},
            }
        )
    summary = progress_summary(data, [batch], 0)
    assert "Completed:" in summary and "provisional" in summary


def test_readme_leaderboard_contains_community_only(tmp_path: Path) -> None:
    state = {
        "rush": RatingRecord("rush", "Rush", "SwarmBench", 1800, built_in=True),
        "alice/a": RatingRecord("alice/a", "A", "alice", 1700, wins=2, games=2),
    }
    block = leaderboard_markdown(state)
    assert "alice" in block and "Rush" not in block
    readme = tmp_path / "README.md"
    readme.write_text("before\n<!-- LEADERBOARD_START -->\nold\n<!-- LEADERBOARD_END -->\nafter\n", encoding="utf-8")
    update_readme_leaderboard(readme, state)
    assert "| 1 | A | alice |" in readme.read_text(encoding="utf-8")
