import json
from pathlib import Path

from typer.testing import CliRunner

from llmrouter.cli import app


def invoke(tmp_path: Path, *args: str):
    config = Path(__file__).resolve().parents[1] / "config" / "router.example.yaml"
    return CliRunner().invoke(
        app, ["--config", str(config), "--db", str(tmp_path / "cli.db"), *args]
    )


def test_cli_status_models_report_and_cache_stats(tmp_path: Path) -> None:
    for command in (
        ("status",), ("models",), ("report",), ("cache", "stats"),
    ):
        result = invoke(tmp_path, *command)
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload is not None


def test_cli_cache_clear_and_replay_missing_request(tmp_path: Path) -> None:
    clear = invoke(tmp_path, "cache", "clear", "--task", "example_task")
    assert clear.exit_code == 0
    assert "Deleted 0 cache entries" in clear.output

    replay = invoke(tmp_path, "replay", "missing-request")
    assert replay.exit_code == 1
    assert "No attempts found" in replay.output


def test_cli_report_rejects_invalid_month(tmp_path: Path) -> None:
    result = invoke(tmp_path, "report", "--month", "October")
    assert result.exit_code != 0


def test_call_command_exposes_no_cache_flag() -> None:
    result = CliRunner().invoke(app, ["call", "--help"])
    assert result.exit_code == 0
    assert "--no-cache" in result.output
