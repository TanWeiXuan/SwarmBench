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
    assert "pull_request_target:" not in text
    assert "contents: write" in text
    assert "actions: write" in text
    assert "pull-requests: write" in text
    assert "actions/checkout" not in text
    assert "python" not in text.lower()
    assert "submission.py" not in text
    assert "markPullRequestReadyForReview" in text
    assert "github.rest.pulls.merge" in text
    assert "submission-accepted.yml" in text


def test_acceptance_workflow_checks_out_only_merged_main() -> None:
    text = workflow("submission-accepted.yml")
    assert "pull_request_target" in text
    assert "workflow_dispatch:" in text
    assert "actions: write" in text
    assert "ref: main" in text
    assert "ref: ${{ github.event.pull_request.head" not in text
    assert "competition.publisher" in text
    assert "gh workflow run tests.yml" in text
    assert "gh workflow run submission-validation.yml" in text


def test_controller_dockerfile_drops_root() -> None:
    text = (ROOT / "Dockerfile.controller").read_text(encoding="utf-8")
    assert "COPY requirements-controller.txt pyproject.toml README.md ./" in text
    assert "USER 65534:65534" in text


def test_tournament_jobs_install_automation_dependencies() -> None:
    installs = [line for line in workflow("tournament.yml").splitlines() if "pip install -e" in line]
    assert installs
    assert all('".[competition]"' in line for line in installs)


def test_tournament_publisher_dispatches_required_bot_pr_checks() -> None:
    tournament = workflow("tournament.yml")
    assert "actions: write" in tournament.split("  final:", 1)[1]
    assert "gh workflow run tests.yml" in tournament
    assert "gh workflow run submission-validation.yml" in tournament
    assert "workflow_dispatch:" in workflow("tests.yml")
    assert "workflow_dispatch:" in workflow("submission-validation.yml")


def test_submission_jobs_install_validation_dependencies() -> None:
    installs = [line for line in workflow("submission-validation.yml").splitlines() if "pip install -e" in line]
    assert installs
    assert all('".[competition]"' in line for line in installs)


def test_tournament_compute_and_report_permissions_are_separated() -> None:
    text = workflow("tournament.yml")
    assert 'cron: "17 */6 * * *"' in text
    assert text.count("SWARMBENCH_BACKEND: docker") == 5
    assert text.count("Maintain live tournament Discussion") == 1
    for index in range(5):
        compute = text.split(f"  compute-{index}:", 1)[1]
        compute = compute.split(f"  compute-{index + 1}:", 1)[0] if index < 4 else compute.split("  final:", 1)[0]
        assert "contents: read" in compute
        assert "discussions: write" not in compute
        assert "pull-requests: write" not in compute

    reporter = text.split("  reporter:", 1)[1].split("  compute-0:", 1)[0]
    assert "contents: read" in reporter
    assert "actions: read" in reporter
    assert "discussions: write" in reporter
    assert "automation live-report" in reporter
    assert "automation compute" not in reporter

    final = text.split("  final:", 1)[1]
    assert "contents: write" in final
    assert "discussions: write" not in final
    assert "automation final" in final
    assert "automation compute" not in final
