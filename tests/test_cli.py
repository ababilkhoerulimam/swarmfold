import json
from pathlib import Path

from typer.testing import CliRunner

from swarmfold.cli import app

FIXTURE = Path(__file__).parent / "fixtures" / "demo_classification.csv"


def test_run_demo_command(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "run-demo",
            "--dataset",
            str(FIXTURE),
            "--output-dir",
            str(tmp_path),
            "--max-experiments",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert '"status": "ACCEPTED"' in result.stdout
    assert '"packet_type": "candidate"' in result.stdout


def test_live_planner_requires_model(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "run-demo",
            "--dataset",
            str(FIXTURE),
            "--output-dir",
            str(tmp_path),
            "--planner",
            "live",
        ],
        env={"SWARMFOLD_MODEL": ""},
    )

    assert result.exit_code == 2
    assert "Live planning requires --model or SWARMFOLD_MODEL" in result.stderr


def test_validate_packet_command(tmp_path: Path) -> None:
    run_result = CliRunner().invoke(
        app,
        [
            "run-demo",
            "--dataset",
            str(FIXTURE),
            "--output-dir",
            str(tmp_path),
            "--max-experiments",
            "1",
        ],
    )
    assert run_result.exit_code == 0
    payload = json.loads(run_result.stdout)

    validation_result = CliRunner().invoke(
        app,
        ["validate-packet", payload["packet"]["packet_path"]],
    )

    assert validation_result.exit_code == 0
    assert '"packet_type": "candidate"' in validation_result.stdout
