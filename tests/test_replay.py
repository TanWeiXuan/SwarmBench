from pathlib import Path
from copy import deepcopy

import pytest

from swarmbench.api import Team
from swarmbench.controllers.baselines import baseline_path
from swarmbench.match import run_match
from swarmbench.replay import ReplayValidationError, load_replay, reconstruct_frames, save_replay, validate_replay, verify_reconstruction
from swarmbench.replay.renderer import explosion_events_at, render_replay


@pytest.fixture(scope="module")
def short_match():
    return run_match(baseline_path("rush"), baseline_path("defend"), seed=12, duration=1.0)


def test_replay_save_load_round_trip(short_match, tmp_path: Path) -> None:
    destination = save_replay(short_match.replay, tmp_path / "match.json")
    loaded = load_replay(destination)
    assert loaded.to_dict() == short_match.replay.to_dict()
    verify_reconstruction(loaded)


def test_compressed_replay_round_trip(short_match, tmp_path: Path) -> None:
    destination = save_replay(short_match.replay, tmp_path / "match.json.gz")
    assert load_replay(destination).to_dict() == short_match.replay.to_dict()


def test_replay_validation_rejects_wrong_version(short_match) -> None:
    data = short_match.replay.to_dict()
    data["replay_version"] = 999
    with pytest.raises(ReplayValidationError):
        validate_replay(data)


def test_reconstruction_is_deterministic(short_match) -> None:
    first = list(reconstruct_frames(short_match.replay))
    second = list(reconstruct_frames(short_match.replay))
    assert first == second


def test_renderer_creates_image(short_match, tmp_path: Path, capsys) -> None:
    output = render_replay(short_match.replay, tmp_path / "match.png")
    assert output is not None and output.stat().st_size > 1_000
    messages = capsys.readouterr().out
    assert "Reconstructing replay frames..." in messages
    assert "Prepared" in messages
    assert "Rendering final frame" in messages
    assert "Finished rendering" in messages


def test_animation_renderer_reports_progress(short_match, tmp_path: Path, capsys) -> None:
    output = render_replay(short_match.replay, tmp_path / "match.gif", fps=10)
    assert output is not None and output.stat().st_size > 1_000
    messages = capsys.readouterr().out
    assert "Encoding" in messages
    assert "Rendering progress: 100%" in messages
    assert "Finished rendering" in messages


def test_explosion_effect_is_included_in_rendered_frame(short_match, tmp_path: Path) -> None:
    replay = deepcopy(short_match.replay)
    replay.events.append(
        {"time": 0.9, "type": "INTERCEPTION", "drone_ids": [0, 20], "position": [50.0, 30.0], "team": None, "points": 0}
    )
    assert explosion_events_at(replay, 1.0)
    output = render_replay(replay, tmp_path / "explosion.png")
    assert output is not None and output.stat().st_size > 1_000
    assert not explosion_events_at(replay, 1.3)


def test_match_metadata_and_result_are_complete(short_match) -> None:
    assert short_match.replay.scenario.seed == 12
    assert len(short_match.replay.controller_a["sha256"]) == 64
    assert short_match.replay.result in {"A", "B", "DRAW"}
    assert short_match.winner in {Team.A, Team.B, None}


def test_match_simulation_is_repeatable_ignoring_wall_clock_metrics(short_match) -> None:
    repeated = run_match(baseline_path("rush"), baseline_path("defend"), seed=12, duration=1.0)
    assert repeated.replay.to_dict() == short_match.replay.to_dict()
