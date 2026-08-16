from swarmbench import cli
from swarmbench.cli import main


def test_arena_cli(capsys) -> None:
    assert main(["arena", "--seed", "42"]) == 0
    assert "seed=42" in capsys.readouterr().out


def test_benchmark_cli(capsys) -> None:
    assert main(["benchmark", "--duration", "0.1", "--seed", "42"]) == 0
    assert "real-time" in capsys.readouterr().out


def test_render_cli_announces_replay_loading(monkeypatch, capsys, tmp_path) -> None:
    replay = object()
    output = tmp_path / "match.png"
    monkeypatch.setattr(cli, "load_replay", lambda _path: replay)
    monkeypatch.setattr(cli, "render_replay", lambda value, destination: destination)

    assert main(["render", "match.json", "--output", str(output)]) == 0
    messages = capsys.readouterr().out
    assert "Loading replay from match.json..." in messages
    assert f"rendered {output}" in messages
