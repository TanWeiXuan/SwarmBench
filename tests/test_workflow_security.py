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
    assert "COPY requirements-controller.txt pyproject.toml README.md ./" in text
    assert "USER 65534:65534" in text


def test_tournament_jobs_install_automation_dependencies() -> None:
    installs = [line for line in workflow("tournament.yml").splitlines() if "pip install -e" in line]
    assert installs
    assert all('".[competition]"' in line for line in installs)


def test_submission_jobs_install_validation_dependencies() -> None:
    installs = [line for line in workflow("submission-validation.yml").splitlines() if "pip install -e" in line]
    assert installs
    assert all('".[competition]"' in line for line in installs)


def test_tournament_compute_and_report_permissions_are_separated() -> None:
    text = workflow("tournament.yml")
    assert 'cron: "17 */6 * * *"' in text
    assert text.count("SWARMBENCH_BACKEND: docker") == 5
    assert text.count("Report progress") == 5
    for index in range(5):
        compute = text.split(f"  compute-{index}:", 1)[1].split(f"  report-{index}:", 1)[0]
        assert "contents: read" in compute
        assert "discussions: write" not in compute
        assert "pull-requests: write" not in compute
        reporter = text.split(f"  report-{index}:", 1)[1]
        reporter = reporter.split(f"  compute-{index + 1}:", 1)[0] if index < 4 else reporter.split("  final:", 1)[0]
        assert "contents: read" in reporter
        assert "discussions: write" in reporter
        assert "automation compute" not in reporter

    finalizer = text.split("  failure-finalizer:", 1)[1]
    assert "contents: read" in finalizer
    assert "discussions: write" in finalizer
    assert "automation compute" not in finalizer
