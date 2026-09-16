import csv
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from swarmfold.artifacts import sha256_file
from swarmfold.graph import run_experiment
from swarmfold.packets import validate_terminal_packet
from swarmfold.planner import LangChainPlanner, PlannerSuggestion

FIXTURE = Path(__file__).parent / "fixtures" / "demo_classification.csv"
BUNDLE_ROOT = Path(__file__).parent / "fixtures"


class FakeStructuredModel:
    def invoke(self, prompt: str) -> dict[str, str]:
        assert "Do not claim that an experiment has already run" in prompt
        return {"hypothesis": "Scaling numeric features should stabilize the linear baseline."}


class FakeChatModel:
    def with_structured_output(self, schema):
        assert schema is PlannerSuggestion
        return FakeStructuredModel()


class InvalidStructuredModel:
    def invoke(self, prompt: str) -> dict[str, str]:
        return {
            "hypothesis": "This otherwise valid hypothesis references a fabricated model API.",
            "model_api": "sklearn.linear_model.ImaginaryClassifier",
        }


class InvalidChatModel:
    def with_structured_output(self, schema):
        return InvalidStructuredModel()


def test_experiment_graph_publishes_verified_candidate(tmp_path: Path) -> None:
    first = run_experiment(FIXTURE, seed=42, output_dir=tmp_path / "first", max_experiments=1)
    second = run_experiment(FIXTURE, seed=42, output_dir=tmp_path / "second", max_experiments=1)

    assert first["status"] == "ACCEPTED"
    assert first["verification"]["accepted"] is True
    assert first["run_result"]["metrics"] == second["run_result"]["metrics"]
    assert first["run_result"]["train_rows"] == 18
    assert first["run_result"]["validation_rows"] == 6
    assert first["run_result"]["tracking_run_id"]
    assert len(first["run_result"]["artifacts"]) == 3
    assert first["contracts"]["contract_sha256"]
    assert first["proposal"]["data_version"] == first["contracts"]["data"]["data_version"]
    assert first["proposal"]["split_id"] == first["contracts"]["validation"]["split_id"]
    assert first["technique_plan"]["model_family"] == first["proposal"]["model_family"]
    assert first["technique_plan"]["is_executable"] is True
    assert first["retrieval_provenance"]["runtime_mapping"]["model_family"] == (
        first["proposal"]["model_family"]
    )

    packet_path = Path(first["packet"]["packet_path"])
    assert first["packet"]["packet_type"] == "candidate"
    assert (packet_path / "COMPLETED.json").is_file()
    assert (packet_path / "submission.csv").is_file()
    assert (packet_path / "technique_plan.json").is_file()
    assert (packet_path / "retrieval-provenance.json").is_file()
    assert sha256_file(packet_path / "artifact-manifest.json") == first["packet"]["manifest_sha256"]
    assert validate_terminal_packet(packet_path).packet_type == "candidate"


def test_experiment_graph_publishes_failure_packet(tmp_path: Path) -> None:
    result = run_experiment(
        FIXTURE,
        test_size=0.01,
        output_dir=tmp_path,
        max_experiments=1,
    )

    assert result["status"] == "REJECTED"
    assert result["run_result"]["status"] == "FAILED"
    assert "test_size" in result["verification"]["reasons"][0]
    assert result["packet"]["packet_type"] == "failure"
    packet_path = Path(result["packet"]["packet_path"])
    assert (packet_path / "COMPLETED.json").is_file()
    assert (packet_path / "failure-manifest.json").is_file()
    assert not (packet_path / "submission.csv").exists()
    assert validate_terminal_packet(packet_path).packet_type == "failure"


def test_experiment_uses_profiled_bundle_and_submission_template(tmp_path: Path) -> None:
    result = run_experiment(
        BUNDLE_ROOT / "bundle_learn.csv",
        target_column="outcome",
        test_dataset_path=BUNDLE_ROOT / "bundle_score.csv",
        sample_submission_path=BUNDLE_ROOT / "bundle_template.csv",
        project_path=BUNDLE_ROOT / "bundle_brief.md",
        output_dir=tmp_path,
        max_experiments=1,
    )

    assert result["status"] == "ACCEPTED"
    assert result["run_result"]["test_rows"] == 3
    assert len(result["run_result"]["artifacts"]) == 4
    packet_path = Path(result["packet"]["packet_path"])
    submission_lines = (packet_path / "submission.csv").read_text(encoding="utf-8").splitlines()
    assert submission_lines[0] == "record_id,outcome"
    assert [line.split(",", 1)[0] for line in submission_lines[1:]] == ["101", "102", "103"]
    assert (packet_path / "project.md").is_file()
    notebook = json.loads((packet_path / "end-to-end.ipynb").read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(code_cells) == 8
    assert all(cell["source"][0].startswith("# Cell ") for cell in code_cells)
    assert all(cell["execution_count"] is None and cell["outputs"] == [] for cell in code_cells)
    for index, cell in enumerate(code_cells, start=1):
        compile("".join(cell["source"]), f"notebook-cell-{index}", "exec")
    assert notebook["metadata"]["swarmfold"]["evidence_status"] == ("recorded_notebook_unexecuted")
    assert "Modeling quality gates" in json.dumps(notebook)
    assert validate_terminal_packet(packet_path).packet_type == "candidate"


def test_mixed_categorical_bundle_uses_roc_auc_probabilities(tmp_path: Path) -> None:
    bundle = tmp_path / "mixed-bundle"
    bundle.mkdir()
    train_path = bundle / "learn.csv"
    train_path.write_text(
        "record_id,age,city,charger,target\n"
        + "".join(
            f"{index},{20 + index},{'Urban' if index % 2 else 'Rural'},"
            f"{'Yes' if index % 3 else 'No'},{'Yes' if index % 2 else 'No'}\n"
            for index in range(1, 41)
        ),
        encoding="utf-8",
    )
    test_path = bundle / "score.csv"
    test_path.write_text(
        "record_id,age,city,charger\n101,31,Urban,Yes\n102,45,Suburban,No\n103,52,Rural,Yes\n",
        encoding="utf-8",
    )
    template_path = bundle / "submission.csv"
    template_path.write_text("record_id,target\n101,0.5\n102,0.5\n103,0.5\n", encoding="utf-8")
    brief_path = bundle / "project.md"
    brief_path.write_text("target_column: target\nmetric: ROC-AUC\n", encoding="utf-8")

    result = run_experiment(
        train_path,
        target_column="target",
        metric_name="roc_auc",
        test_dataset_path=test_path,
        sample_submission_path=template_path,
        project_path=brief_path,
        output_dir=tmp_path / "output",
        max_experiments=1,
    )

    assert result["status"] == "ACCEPTED"
    assert 0.0 <= result["run_result"]["metrics"]["roc_auc"] <= 1.0
    assert result["run_result"]["positive_class"] == "Yes"
    packet_path = Path(result["packet"]["packet_path"])
    submission = list(csv.DictReader((packet_path / "submission.csv").open(encoding="utf-8")))
    probabilities = [float(row["target"]) for row in submission]
    assert all(0.0 <= value <= 1.0 for value in probabilities)
    oof = pq.read_table(packet_path / "oof_predictions.parquet").to_pydict()
    assert all(0.0 <= float(value) <= 1.0 for value in oof["prediction"])
    notebook_text = (packet_path / "end-to-end.ipynb").read_text(encoding="utf-8")
    assert "roc_auc" in notebook_text
    assert "OneHotEncoder" in notebook_text


def test_graph_executes_multiclass_and_regression_tasks(tmp_path: Path) -> None:
    multiclass_path = tmp_path / "multiclass.csv"
    multiclass_path.write_text(
        "row,numeric,category,target\n"
        + "".join(
            f"{index},{index % 7},{'left' if index % 2 else 'right'},"
            f"{['cat', 'dog', 'bird'][index % 3]}\n"
            for index in range(45)
        ),
        encoding="utf-8",
    )
    multiclass = run_experiment(
        multiclass_path,
        metric_name="accuracy",
        output_dir=tmp_path / "multiclass-output",
        max_experiments=1,
    )

    assert multiclass["status"] == "ACCEPTED"
    assert multiclass["contracts"]["competition"]["task_type"] == "multiclass_classification"
    assert multiclass["contracts"]["metric"]["metric_name"] == "accuracy"

    regression_path = tmp_path / "regression.csv"
    regression_path.write_text(
        "row,numeric,category,target\n"
        + "".join(
            f"{index},{index % 11},{'left' if index % 2 else 'right'},"
            f"{(index * 2.75) + ((index % 4) * 0.25)}\n"
            for index in range(40)
        ),
        encoding="utf-8",
    )
    regression = run_experiment(
        regression_path,
        output_dir=tmp_path / "regression-output",
        max_experiments=1,
    )

    assert regression["status"] == "ACCEPTED"
    assert regression["contracts"]["competition"]["task_type"] == "regression"
    assert regression["contracts"]["metric"]["metric_name"] == "rmse"
    assert regression["run_result"]["metrics"]["rmse"] >= 0.0


def test_live_planner_uses_validated_structured_output(tmp_path: Path) -> None:
    planner = LangChainPlanner("fake:model", model_factory=lambda _: FakeChatModel())

    result = run_experiment(FIXTURE, output_dir=tmp_path, planner=planner, max_experiments=1)

    assert result["status"] == "ACCEPTED"
    assert result["proposal"]["hypothesis"].startswith("Scaling numeric features")
    manifest = json.loads(
        Path(result["packet"]["packet_path"], "artifact-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["tracking_run_id"] == result["run_result"]["tracking_run_id"]


def test_live_planner_requires_provider_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    planner = LangChainPlanner("openai:example-model")

    with pytest.raises(RuntimeError, match="requires OPENAI_API_KEY"):
        planner.propose(
            {
                "dataset_path": str(FIXTURE),
                "target_column": "target",
                "seed": 42,
                "test_size": 0.25,
                "status": "PENDING",
            }
        )


def test_invalid_live_proposal_is_rejected_before_training(tmp_path: Path) -> None:
    planner = LangChainPlanner("fake:model", model_factory=lambda _: InvalidChatModel())

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        run_experiment(FIXTURE, output_dir=tmp_path, planner=planner, max_experiments=1)

    assert not (tmp_path / "runs").exists()
    assert not (tmp_path / "mlflow.db").exists()


def test_packet_validator_rejects_tampering(tmp_path: Path) -> None:
    result = run_experiment(FIXTURE, output_dir=tmp_path, max_experiments=1)
    packet_path = Path(result["packet"]["packet_path"])
    (packet_path / "submission.csv").write_text("row,prediction\n0,tampered\n", encoding="utf-8")

    with pytest.raises(ValueError, match="integrity verification"):
        validate_terminal_packet(packet_path)
