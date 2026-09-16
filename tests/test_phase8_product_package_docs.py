from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from typer.testing import CliRunner

from swarmfold.capability_matrix import (
    build_capability_matrix,
    export_capability_matrix_file,
    format_json_matrix,
    format_markdown_matrix,
)
from swarmfold.cli import app
from swarmfold.package_verifier import verify_portfolio_package
from swarmfold.portfolio import build_full_package

runner = CliRunner()


def _make_candidate_packet(
    run_directory: Path,
    lane_id: str,
    score: float,
    oof_preds: list[float],
    test_preds: list[float],
) -> dict[str, object]:
    packet = run_directory / "source-packets" / lane_id
    packet.mkdir(parents=True, exist_ok=True)
    targets = ["0", "0", "1", "1"]
    pq.write_table(
        pa.table(
            {
                "row": [str(i) for i in range(4)],
                "prediction": [str(v) for v in oof_preds],
                "target": targets,
            }
        ),
        packet / "oof_predictions.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "row": [str(i) for i in range(2)],
                "prediction": [str(v) for v in test_preds],
            }
        ),
        packet / "test_predictions.parquet",
    )
    with (packet / "submission.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", "prediction"])
        writer.writeheader()
        for idx, val in enumerate(test_preds):
            writer.writerow({"id": str(idx), "prediction": f"{val:.4f}"})

    (packet / "contracts.json").write_text(
        json.dumps({"competition": {"prediction_column": "prediction"}}),
        encoding="utf-8",
    )
    (packet / "end-to-end.ipynb").write_text(
        json.dumps({"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 2}),
        encoding="utf-8",
    )
    (packet / "experiment-ledger.jsonl").write_text(
        json.dumps({"lane_id": lane_id, "experiment_id": f"{lane_id}-1", "model_family": "rf"})
        + "\n",
        encoding="utf-8",
    )
    (packet / "findings.jsonl").write_text(
        json.dumps(
            {
                "lane_id": lane_id,
                "experiment_id": f"{lane_id}-1",
                "verdict": "ACCEPTED",
                "recommendation": "Use RF model",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (packet / "modeling-quality-evidence.json").write_text(
        json.dumps({"validation_audit": {"leakage_detected": False, "folds": 5}}),
        encoding="utf-8",
    )
    (packet / "technique_plan.json").write_text(
        json.dumps({"prioritized_techniques": ["target_encoding"]}),
        encoding="utf-8",
    )
    (packet / "proposal.json").write_text(
        json.dumps({"dataset_path": str((run_directory / "private" / "train.csv").resolve())}),
        encoding="utf-8",
    )
    return {
        "status": "ACCEPTED",
        "packet": {
            "packet_type": "candidate",
            "packet_path": str(packet.resolve()),
        },
        "completed_results": [
            {
                "experiment_id": f"{lane_id}-1",
                "model_family": "rf",
                "metrics": {"roc_auc": score},
            }
        ],
    }


def _build_test_package(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    bundle = run_dir / "bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / "project.md").write_text("Test Competition", encoding="utf-8")

    lane_results = {
        "ai-a": _make_candidate_packet(run_dir, "ai-a", 0.70, [0.1, 0.2, 0.6, 0.7], [0.2, 0.8]),
        "ai-b": _make_candidate_packet(run_dir, "ai-b", 0.75, [0.2, 0.8, 0.4, 0.9], [0.4, 0.6]),
    }
    record = {
        "run_id": "a" * 32,
        "iteration_id": "iter-phase8",
        "profile": {
            "task_type": "binary_classification",
            "metric": "roc_auc",
            "target_column": "target",
        },
        "max_experiments": 4,
        "lanes": [
            {
                "lane_id": lid,
                "name": lid.upper(),
                "enabled": lid in lane_results,
                "status": "ACCEPTED" if lid in lane_results else "DISABLED",
                "max_experiments": 2,
            }
            for lid in ("ai-a", "ai-b", "ai-c", "ai-d", "ai-e")
        ],
    }
    build_full_package(run_dir, record, lane_results)
    return run_dir / "full-package.zip"


# =====================================================================
# Capability Matrix Tests
# =====================================================================


def test_capability_matrix_generation_and_export(tmp_path: Path) -> None:
    matrix = build_capability_matrix()
    assert len(matrix) >= 30

    categories = {entry.category for entry in matrix}
    expected_categories = {
        "model_family",
        "feature_strategy",
        "validation_strategy",
        "task_adapter",
        "metric",
    }
    assert expected_categories.issubset(categories)

    names = {entry.name for entry in matrix}
    assert "logistic_regression" in names
    assert "lightgbm" in names
    assert "stratified_kfold" in names
    assert "binary_classification" in names
    assert "roc_auc" in names

    for entry in matrix:
        assert entry.name
        assert entry.description
        assert entry.status_summary

    by_name = {entry.name: entry for entry in matrix}
    assert by_name["multilabel_classification"].executable is False
    assert by_name["time_series_forecasting"].scheduled is False
    assert by_name["pca_projection"].executable is False
    assert by_name["polynomial_features"].tested is False

    # Markdown export
    md = format_markdown_matrix()
    assert "# SwarmFold Capability Matrix" in md
    assert "## Model Families" in md
    assert "## Feature Engineering Strategies" in md
    assert "## Validation Splitters" in md
    assert "## Task Adapters" in md
    assert "| Component | Status | Declared |" in md

    # JSON export
    data = format_json_matrix()
    assert data["schema_version"] == "capability-matrix-v1"
    assert data["total_capabilities"] == len(matrix)
    assert data["operational_count"] > 0
    assert "categories" in data

    # Export file
    target = tmp_path / "test_matrix.md"
    export_capability_matrix_file(target)
    assert target.is_file()
    assert target.read_text(encoding="utf-8") == md


def test_cli_capability_matrix(tmp_path: Path) -> None:
    res = runner.invoke(app, ["capability-matrix"])
    assert res.exit_code == 0
    assert "# SwarmFold Capability Matrix" in res.stdout
    assert "logistic_regression" in res.stdout

    res_json = runner.invoke(app, ["capability-matrix", "--format", "json"])
    assert res_json.exit_code == 0
    parsed = json.loads(res_json.stdout)
    assert parsed["schema_version"] == "capability-matrix-v1"

    out_file = tmp_path / "cli_matrix.md"
    res_out = runner.invoke(app, ["capability-matrix", "--output", str(out_file)])
    assert res_out.exit_code == 0
    assert out_file.is_file()
    assert "Capability matrix written to" in res_out.stdout


# =====================================================================
# Package Verifier Tests
# =====================================================================


def test_verify_portfolio_package_valid(tmp_path: Path) -> None:
    pkg_zip = _build_test_package(tmp_path)
    assert pkg_zip.is_file()

    # Test ZIP verification
    report = verify_portfolio_package(pkg_zip)
    assert report["valid"] is True
    assert report["errors"] == []
    assert report["missing_files"] == []
    assert report["checksum_mismatches"] == []
    assert report["secret_violations"] == []
    assert report["portability_violations"] == []
    assert report["file_count"] >= 10

    with zipfile.ZipFile(pkg_zip) as archive:
        proposal = json.loads(
            archive.read("experiments/ai-a/packet/proposal.json").decode("utf-8")
        )
    assert proposal["dataset_path"] == "artifacts/train.csv"

    # Test uncompressed directory verification
    uncompressed_dir = tmp_path / "run" / "full-package"
    report_dir = verify_portfolio_package(uncompressed_dir)
    assert report_dir["valid"] is True
    assert report_dir["errors"] == []


def test_verify_portfolio_package_tampering(tmp_path: Path) -> None:
    pkg_zip = _build_test_package(tmp_path)

    # Corrupt a file inside the zip without updating manifest
    tampered_zip = tmp_path / "tampered.zip"
    with zipfile.ZipFile(pkg_zip, "r") as zin, zipfile.ZipFile(tampered_zip, "w") as zout:
        for item in zin.infolist():
            content = zin.read(item.filename)
            if item.filename == "submission.csv":
                content = b"tampered,corrupted,data\n1,0.999\n"
            zout.writestr(item, content)

    report = verify_portfolio_package(tampered_zip)
    assert report["valid"] is False
    assert len(report["checksum_mismatches"]) == 1
    assert report["checksum_mismatches"][0]["file"] == "submission.csv"
    assert any("Checksum mismatch for submission.csv" in err for err in report["errors"])


def test_verify_portfolio_package_missing_required(tmp_path: Path) -> None:
    pkg_zip = _build_test_package(tmp_path)

    missing_zip = tmp_path / "missing_file.zip"
    with zipfile.ZipFile(pkg_zip, "r") as zin, zipfile.ZipFile(missing_zip, "w") as zout:
        for item in zin.infolist():
            if item.filename != "submission.csv":
                zout.writestr(item, zin.read(item.filename))

    report = verify_portfolio_package(missing_zip)
    assert report["valid"] is False
    assert "submission.csv" in report["missing_files"]
    assert any("Required artifact 'submission.csv' is missing" in err for err in report["errors"])


def test_verify_portfolio_package_secret_leak(tmp_path: Path) -> None:
    pkg_zip = _build_test_package(tmp_path)

    leak_zip = tmp_path / "leaked_secret.zip"
    with zipfile.ZipFile(pkg_zip, "r") as zin, zipfile.ZipFile(leak_zip, "w") as zout:
        for item in zin.infolist():
            content = zin.read(item.filename)
            if item.filename == "manifest.json":
                # Inject a leaked API key into manifest
                data = json.loads(content.decode("utf-8"))
                data["api_key"] = "sk-live-abcdef1234567890abcdef"
                content = json.dumps(data).encode("utf-8")
            zout.writestr(item, content)

    report = verify_portfolio_package(leak_zip)
    assert report["valid"] is False
    assert len(report["secret_violations"]) >= 1
    assert any("Possible secret found" in err for err in report["errors"])


def test_verify_portfolio_package_absolute_path_violation(tmp_path: Path) -> None:
    pkg_zip = _build_test_package(tmp_path)

    bad_zip = tmp_path / "abs_path.zip"
    with zipfile.ZipFile(pkg_zip, "r") as zin, zipfile.ZipFile(bad_zip, "w") as zout:
        for item in zin.infolist():
            content = zin.read(item.filename)
            if item.filename == "manifest.json":
                # Inject a Windows drive path into manifest metadata
                data = json.loads(content.decode("utf-8"))
                data["local_machine_path"] = "C:\\Users\\SecretUser\\dataset.csv"
                content = json.dumps(data).encode("utf-8")
            zout.writestr(item, content)

    report = verify_portfolio_package(bad_zip)
    assert report["valid"] is False
    assert len(report["portability_violations"]) >= 1
    assert any("Non-portable absolute path" in err for err in report["errors"])


def test_verify_portfolio_package_rejects_unmanifested_file(tmp_path: Path) -> None:
    _build_test_package(tmp_path)
    package_dir = tmp_path / "run" / "full-package"
    (package_dir / "rogue.txt").write_text("not declared", encoding="utf-8")

    report = verify_portfolio_package(package_dir)

    assert report["valid"] is False
    assert "File is missing from manifest: rogue.txt" in report["errors"]


def test_cli_verify_package(tmp_path: Path) -> None:
    pkg_zip = _build_test_package(tmp_path)

    # Valid package verification via CLI
    res = runner.invoke(app, ["verify-package", str(pkg_zip)])
    assert res.exit_code == 0
    parsed = json.loads(res.stdout)
    assert parsed["valid"] is True

    # Tampered package via CLI exits with code 1
    tampered_zip = tmp_path / "cli_tampered.zip"
    with zipfile.ZipFile(pkg_zip, "r") as zin, zipfile.ZipFile(tampered_zip, "w") as zout:
        for item in zin.infolist():
            content = zin.read(item.filename)
            if item.filename == "submission.csv":
                content = b"tampered content\n"
            zout.writestr(item, content)

    res_tampered = runner.invoke(app, ["verify-package", str(tampered_zip)])
    assert res_tampered.exit_code == 1
    assert "Checksum mismatch" in res_tampered.stdout


# =====================================================================
# Dashboard HTML & Panels Tests
# =====================================================================


def test_dashboard_html_contains_phase8_panels() -> None:
    html_path = (
        Path(__file__).parent.parent / "src" / "swarmfold" / "web" / "dashboard.html"
    )
    assert html_path.is_file()
    content = html_path.read_text(encoding="utf-8")

    # Required tab buttons
    for tab_id in (
        "tab-overview",
        "tab-contracts",
        "tab-folds",
        "tab-events",
        "tab-ensemble",
        "tab-reflection",
        "tab-literature",
    ):
        assert f'data-tab="{tab_id}"' in content, f"Missing tab button: {tab_id}"

    # Required panels
    for panel_id in (
        "tab-overview",
        "tab-contracts",
        "tab-folds",
        "tab-events",
        "tab-ensemble",
        "tab-reflection",
        "tab-literature",
    ):
        assert f'id="{panel_id}"' in content, f"Missing panel: {panel_id}"

    assert 'id="execution-dag-panel"' in content

    # Required rendering functions
    for func in (
        "renderExecutionDAG",
        "renderContractsPanel",
        "renderFoldsPanel",
        "renderEventsPanel",
        "renderEnsemblePanel",
        "renderReflectionPanel",
        "renderTechniquesPanel",
    ):
        assert func in content, f"Missing rendering function: {func}"


def test_dashboard_confirm_records_contracts(tmp_path: Path) -> None:
    from swarmfold.dashboard import DashboardRunManager
    from swarmfold.intake import CompetitionProfile, FileUnderstanding

    profile = CompetitionProfile(
        summary="Binary classification task",
        task_type="binary_classification",
        metric="roc_auc",
        training_file="train.csv",
        inference_files=["test.csv"],
        submission_template="sample.csv",
        brief_file="brief.md",
        target_column="target",
        prediction_columns=["target"],
        id_columns=["id"],
        validation_strategy="stratified_kfold",
        files=[
            FileUnderstanding(filename="train.csv", role="training_data", reasoning="train"),
            FileUnderstanding(filename="test.csv", role="inference_data", reasoning="test"),
            FileUnderstanding(
                filename="sample.csv",
                role="submission_template",
                reasoning="sample",
            ),
            FileUnderstanding(filename="brief.md", role="competition_brief", reasoning="brief"),
        ],
        execution_ready=True,
        blockers=[],
        questions=[],
        confidence=0.99,
    )

    manager = DashboardRunManager(tmp_path / "runs", runner=lambda *args, **kwargs: {})
    # Initialize run directory
    run_id = "0" * 32
    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True)
    bundle_dir = run_dir / "bundle"
    bundle_dir.mkdir()
    # Write train, test, sample, and brief files
    train_lines = ["id,target,feat1"] + [f"{i},{i % 2},{i * 10}" for i in range(12)]
    (bundle_dir / "train.csv").write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    test_lines = ["id,feat1"] + [f"{i + 20},{i * 10}" for i in range(4)]
    (bundle_dir / "test.csv").write_text("\n".join(test_lines) + "\n", encoding="utf-8")
    sample_lines = ["id,target"] + [f"{i + 20},0" for i in range(4)]
    (bundle_dir / "sample.csv").write_text("\n".join(sample_lines) + "\n", encoding="utf-8")
    (bundle_dir / "brief.md").write_text("Competition brief", encoding="utf-8")

    initial_record = {
        "run_id": run_id,
        "iteration_id": "iter-1",
        "bundle_files": ["train.csv", "test.csv", "sample.csv", "brief.md"],
        "profile": profile.model_dump(),
        "status": "AWAITING_CONFIRMATION",
        "lanes": [],
    }
    (run_dir / "status.json").write_text(json.dumps(initial_record), encoding="utf-8")

    # Call confirm()
    confirmed = manager.confirm(
        run_id=run_id,
        profile_value=profile,
    )
    assert confirmed["status"] == "READY_TO_TRAIN"
    contracts = confirmed.get("contracts")
    assert isinstance(contracts, dict)
    assert contracts["task_type"] == "binary_classification"
    assert contracts["target_column"] == "target"
    assert contracts["metric"] == "roc_auc"
    assert contracts["train_rows"] == 12
    assert contracts["test_rows"] == 4
    assert "feat1" in contracts["feature_columns"]
    assert "id" in contracts["id_columns"]
