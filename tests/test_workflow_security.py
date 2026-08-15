from pathlib import Path


ROOT = Path(__file__).parents[1]


def workflow(name: str) -> str:
    return (ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")


def test_untrusted_submission_workflow_is_read_only() -> None:
    text = workflow("submission-validation.yml")
    assert "pull_request_target" not in text
    assert "permissions:\n  contents: read" in text
    assert "pull-requests: write" not in text
    assert "discussions: write" not in text
    assert "SWARMBENCH_BACKEND: docker" in text


def test_privileged_reporter_never_checks_out_or_runs_controller_code() -> None:
    text = workflow("submission-reporter.yml")
    assert "pull-requests: write" in text
    assert "actions/checkout" not in text
    assert "python" not in text.lower()
    assert "submission.py" not in text


def test_acceptance_workflow_checks_out_only_merged_main() -> None:
    text = workflow("submission-accepted.yml")
    assert "pull_request_target" in text
    assert "ref: main" in text
    assert "ref: ${{ github.event.pull_request.head" not in text
    assert "competition.publisher" in text


def test_controller_dockerfile_drops_root() -> None:
    text = (ROOT / "Dockerfile.controller").read_text(encoding="utf-8")
    assert "USER 65534:65534" in text
