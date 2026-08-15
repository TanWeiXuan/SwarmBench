from swarmbench.cli import main


def test_arena_cli(capsys) -> None:
    assert main(["arena", "--seed", "42"]) == 0
    assert "seed=42" in capsys.readouterr().out
