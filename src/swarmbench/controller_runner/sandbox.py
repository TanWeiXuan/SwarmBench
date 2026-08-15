"""Docker command construction for authoritative untrusted execution."""

from __future__ import annotations

from pathlib import Path


def docker_run_command(image: str, controller: Path, request_dir: Path) -> list[str]:
    """Build a networkless, read-only, resource-limited controller command."""
    return [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--read-only",
        "--cpus",
        "1",
        "--memory",
        "2g",
        "--pids-limit",
        "128",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=128m",
        "--mount",
        f"type=bind,src={controller.resolve()},dst=/submission/controller.py,readonly",
        "--mount",
        f"type=bind,src={request_dir.resolve()},dst=/requests,readonly",
        image,
        "python",
        "-m",
        "swarmbench.controller_runner.worker",
        "/submission/controller.py",
    ]

