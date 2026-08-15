from swarmbench.cli import main


def test_arena_cli(capsys) -> None:
    assert main(["arena", "--seed", "42"]) == 0
    assert "seed=42" in capsys.readouterr().out


def test_benchmark_cli(capsys) -> None:
    assert main(["benchmark", "--duration", "0.1", "--seed", "42"]) == 0
    assert "real-time" in capsys.readouterr().out
