import csv
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import TypeAdapter

from swarmfold.artifacts import sha256_file, write_json
from swarmfold.contracts import contract_bundle_sha256
from swarmfold.notebook import write_end_to_end_notebook
from swarmfold.schemas import (
    ArtifactRecord,
    CandidateManifest,
    CompletionRecord,
    ContractBundle,
    ExperimentProposal,
    ExperimentState,
    FailureManifest,
    PacketFileRecord,
    PacketManifest,
    PacketResult,
    RunResult,
    VerificationResult,
)
from swarmfold.versioning import migrate_record

_MANIFEST_ADAPTER = TypeAdapter(PacketManifest)


def publish_terminal_packet(state: ExperimentState) -> ExperimentState:
    proposal = ExperimentProposal.model_validate(state["proposal"])
    result = RunResult.model_validate(state["run_result"])
    verification = VerificationResult.model_validate(state["verification"])
    contracts = ContractBundle.model_validate(state["contracts"])
    proposals = {
        item.experiment_id: item
        for item in (
            ExperimentProposal.model_validate(value) for value in state.get("experiment_queue", [])
        )
    }
    packet_type = "candidate" if verification.accepted else "failure"
    packet_parent = Path(state["output_dir"]) / "packets" / result.run_id
    packet_directory = packet_parent / f"{packet_type}-packet"
    staging_directory = packet_parent / f".{packet_type}-packet.staging"
    staging_directory.mkdir(parents=True, exist_ok=False)

    proposal_path = write_json(staging_directory / "proposal.json", proposal.model_dump())
    result_path = write_json(staging_directory / "run-result.json", result.model_dump())
    verification_path = write_json(
        staging_directory / "verification.json", verification.model_dump()
    )
    contracts_path = staging_directory / "contracts.json"
    shutil.copy2(state["contracts_path"], contracts_path)
    ledger_records = []
    for value in state.get("completed_results", []):
        completed = RunResult.model_validate(value)
        proposal_record = proposals.get(completed.experiment_id)
        ledger_records.append(
            {
                **completed.model_dump(),
                "hypothesis": proposal_record.hypothesis if proposal_record else None,
                "feature_strategy": (proposal_record.feature_strategy if proposal_record else None),
                "parameters": proposal_record.parameters if proposal_record else {},
            }
        )
    ledger_path = _write_jsonl(staging_directory / "experiment-ledger.jsonl", ledger_records)
    findings_path = _write_jsonl(staging_directory / "findings.jsonl", state.get("critiques", []))
    decision_path = staging_directory / "decision.md"
    decision_path.write_text(
        _decision_text(packet_type, result, verification, external_test=bool(result.test_rows)),
        encoding="utf-8",
    )
    portable_artifacts = _copy_portable_artifacts(staging_directory, result)

    packet_files = [
        proposal_path,
        result_path,
        verification_path,
        contracts_path,
        ledger_path,
        findings_path,
        decision_path,
    ]
    literature_review = state.get("literature_research")
    if literature_review:
        literature_path = write_json(
            staging_directory / "literature-review.json", literature_review
        )
        packet_files.append(literature_path)
    reflection = state.get("iteration_reflection")
    if reflection:
        reflection_json_path = write_json(
            staging_directory / "iteration-reflection.json", reflection
        )
        packet_files.append(reflection_json_path)
        reflection_source = Path(state.get("reflection_path", ""))
        reflection_markdown_source = reflection_source.with_suffix(".md")
        if reflection_markdown_source.is_file():
            reflection_markdown_path = staging_directory / "iteration-reflection.md"
            shutil.copy2(reflection_markdown_source, reflection_markdown_path)
            packet_files.append(reflection_markdown_path)
    quality_evidence = {
        name: state[name]
        for name in (
            "problem_formulation",
            "validation_audit",
            "domain_decomposition",
            "ensemble_assessment",
            "robustness_assessment",
            "scheduler_decisions",
            "iteration_reflection",
        )
        if name in state
    }
    if quality_evidence:
        quality_path = write_json(
            staging_directory / "modeling-quality-evidence.json", quality_evidence
        )
        packet_files.append(quality_path)
    if state.get("model_registry"):
        registry_path = write_json(
            staging_directory / "model-capability-registry.json", state["model_registry"]
        )
        packet_files.append(registry_path)
    if state.get("technique_plan"):
        technique_plan_path = write_json(
            staging_directory / "technique_plan.json", state["technique_plan"]
        )
        packet_files.append(technique_plan_path)
    if state.get("retrieval_provenance"):
        retrieval_path = write_json(
            staging_directory / "retrieval-provenance.json",
            state["retrieval_provenance"],
        )
        packet_files.append(retrieval_path)
    if contracts.data.test_dataset_path:
        successful = [
            RunResult.model_validate(value)
            for value in state.get("completed_results", [])
            if RunResult.model_validate(value).status == "SUCCEEDED"
        ]
        for index, experiment_result in enumerate(successful, start=1):
            experiment_proposal = proposals.get(experiment_result.experiment_id)
            if experiment_proposal is None:
                continue
            notebook_path = write_end_to_end_notebook(
                staging_directory / f"experiment-{index:03d}.ipynb",
                proposal=experiment_proposal,
                result=experiment_result,
                contracts=contracts,
                literature_review=literature_review,
                quality_evidence=quality_evidence,
            )
            packet_files.append(notebook_path)
    if state.get("project_path"):
        project_path = staging_directory / "project.md"
        shutil.copy2(state["project_path"], project_path)
        packet_files.append(project_path)
    if packet_type == "candidate":
        submission_path = _write_submission(staging_directory / "submission.csv", result, state)
        oof_path = _write_prediction_parquet(
            staging_directory / "oof_predictions.parquet", result, include_target=True
        )
        test_path = _write_prediction_parquet(
            staging_directory / "test_predictions.parquet", result, include_target=False
        )
        packet_files.extend([submission_path, oof_path, test_path])
        if contracts.data.test_dataset_path:
            notebook_path = write_end_to_end_notebook(
                staging_directory / "end-to-end.ipynb",
                proposal=proposal,
                result=result,
                contracts=contracts,
                literature_review=literature_review,
                quality_evidence=quality_evidence,
            )
            packet_files.append(notebook_path)
        manifest: CandidateManifest | FailureManifest = CandidateManifest(
            schema_version="0.2",
            packet_type="candidate",
            iteration_id=state["iteration_id"],
            lane_id=state["lane_id"],
            experiment_id=proposal.experiment_id,
            run_id=result.run_id,
            tracking_run_id=_required(result.tracking_run_id, "tracking_run_id"),
            contract_sha256=_required(contracts.contract_sha256, "contract_sha256"),
            data_version=proposal.data_version,
            metric_version=proposal.metric_version,
            split_id=proposal.split_id,
            dataset_sha256=_required(result.dataset_sha256, "dataset_sha256"),
            metrics=result.metrics,
            submission_rows=result.test_rows or result.validation_rows,
            artifacts=portable_artifacts,
            resource_usage=result.resource_usage,
            packet_files=[_packet_file_record(path) for path in packet_files],
        )
        manifest_name = "artifact-manifest.json"
    else:
        manifest = FailureManifest(
            schema_version="0.2",
            packet_type="failure",
            iteration_id=state["iteration_id"],
            lane_id=state["lane_id"],
            experiment_id=proposal.experiment_id,
            run_id=result.run_id,
            contract_sha256=_required(contracts.contract_sha256, "contract_sha256"),
            terminal_reason="; ".join(verification.reasons) or "Evidence verification failed",
            completed_run_ids=[
                RunResult.model_validate(item).run_id
                for item in state.get("completed_results", [])
                if RunResult.model_validate(item).status == "SUCCEEDED"
            ],
            missing_deliverables=["submission.csv"],
            retry_useful=result.status == "FAILED",
            artifacts=portable_artifacts,
            resource_usage=result.resource_usage,
            packet_files=[_packet_file_record(path) for path in packet_files],
        )
        manifest_name = "failure-manifest.json"

    manifest_path = write_json(staging_directory / manifest_name, manifest.model_dump())
    completion = CompletionRecord(
        schema_version="0.2",
        packet_type=packet_type,
        iteration_id=state["iteration_id"],
        lane_id=state["lane_id"],
        run_id=result.run_id,
        manifest_name=manifest_name,
        manifest_sha256=sha256_file(manifest_path),
        published_at=datetime.now(UTC).isoformat(),
    )
    write_json(staging_directory / "COMPLETED.json", completion.model_dump())
    validate_terminal_packet(staging_directory)
    staging_directory.rename(packet_directory)
    packet = validate_terminal_packet(packet_directory)
    return {"packet": packet.model_dump()}


def validate_terminal_packet(packet_path: str | Path) -> PacketResult:
    packet_directory = Path(packet_path).resolve()
    completion_path = packet_directory / "COMPLETED.json"
    if not completion_path.is_file():
        raise ValueError("Packet is incomplete: COMPLETED.json is missing")
    completion = CompletionRecord.model_validate(
        migrate_record("packet_completion", _read_object(completion_path))
    )
    manifest_path = packet_directory / completion.manifest_name
    if not manifest_path.is_file():
        raise ValueError(f"Packet manifest is missing: {completion.manifest_name}")
    if sha256_file(manifest_path) != completion.manifest_sha256:
        raise ValueError("Packet manifest hash mismatch")

    manifest = _MANIFEST_ADAPTER.validate_python(
        migrate_record("packet_manifest", _read_object(manifest_path))
    )
    if (
        manifest.packet_type != completion.packet_type
        or manifest.run_id != completion.run_id
        or manifest.iteration_id != completion.iteration_id
        or manifest.lane_id != completion.lane_id
    ):
        raise ValueError("Completion record does not identify the packet manifest")
    for record in manifest.packet_files:
        if Path(record.name).name != record.name:
            raise ValueError(f"Unsafe packet file name: {record.name}")
        path = packet_directory / record.name
        if not path.is_file():
            raise ValueError(f"Packet file is missing: {record.name}")
        if path.stat().st_size != record.size_bytes or sha256_file(path) != record.sha256:
            raise ValueError(f"Packet file failed integrity verification: {record.name}")
    for artifact in getattr(manifest, "artifacts", []):
        path = Path(artifact.path)
        if not path.is_absolute():
            path = (packet_directory / path).resolve()
            if packet_directory not in path.parents:
                raise ValueError(f"Unsafe run artifact path: {artifact.path}")
        if not path.is_file() or path.stat().st_size != artifact.size_bytes:
            raise ValueError(f"Run artifact is missing or has the wrong size: {path.name}")
        if sha256_file(path) != artifact.sha256:
            raise ValueError(f"Run artifact hash mismatch: {path.name}")
    contracts = ContractBundle.model_validate(_read_json(packet_directory / "contracts.json"))
    contract_sha256 = contract_bundle_sha256(contracts)
    if contract_sha256 != manifest.contract_sha256 or contracts.contract_sha256 != contract_sha256:
        raise ValueError("Packet contracts do not match the manifest")
    if isinstance(manifest, CandidateManifest):
        _validate_submission(
            packet_directory / "submission.csv",
            manifest.submission_rows,
            list(contracts.competition.submission_columns),
            contracts.competition.prediction_column,
        )

    return PacketResult(
        packet_type=manifest.packet_type,
        iteration_id=manifest.iteration_id,
        lane_id=manifest.lane_id,
        packet_path=str(packet_directory),
        manifest_sha256=completion.manifest_sha256,
    )


def read_packet_manifest(packet_path: str | Path) -> PacketManifest:
    packet_directory = Path(packet_path).resolve()
    validate_terminal_packet(packet_directory)
    completion = CompletionRecord.model_validate(
        migrate_record(
            "packet_completion",
            _read_object(packet_directory / "COMPLETED.json"),
        )
    )
    return _MANIFEST_ADAPTER.validate_python(
        migrate_record(
            "packet_manifest",
            _read_object(packet_directory / completion.manifest_name),
        )
    )


def _write_submission(path: Path, result: RunResult, state: ExperimentState) -> Path:
    if state.get("sample_submission_path"):
        prediction_record = next(
            (
                artifact
                for artifact in result.artifacts
                if Path(artifact.path).name == "test_predictions.csv"
            ),
            None,
        )
        if prediction_record is None:
            raise ValueError("Accepted run has no external test predictions")
        with Path(prediction_record.path).open(newline="", encoding="utf-8") as source:
            predictions = [row["predicted"] for row in csv.DictReader(source)]
        with Path(state["sample_submission_path"]).open(
            newline="", encoding="utf-8-sig"
        ) as template:
            reader = csv.DictReader(template)
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)
        contracts = ContractBundle.model_validate(state["contracts"])
        prediction_column = contracts.competition.prediction_column
        if len(rows) != len(predictions):
            raise ValueError("Test predictions do not match the submission template row count")
        for row, prediction in zip(rows, predictions, strict=True):
            row[prediction_column] = str(prediction)
        with path.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(destination, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return path
    source_record = next(
        (
            artifact
            for artifact in result.artifacts
            if Path(artifact.path).name == "validation_predictions.csv"
        ),
        None,
    )
    if source_record is None:
        raise ValueError("Accepted run has no validation predictions")
    with Path(source_record.path).open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=["row", "prediction"])
        writer.writeheader()
        writer.writerows({"row": row["row"], "prediction": row["predicted"]} for row in rows)
    return path


def _write_prediction_parquet(path: Path, result: RunResult, *, include_target: bool) -> Path:
    wanted_name = "validation_predictions.csv" if include_target else "test_predictions.csv"
    source_record = next(
        (artifact for artifact in result.artifacts if Path(artifact.path).name == wanted_name),
        None,
    )
    if source_record is None and not include_target:
        return _write_prediction_parquet(path, result, include_target=True)
    if source_record is None:
        raise ValueError("Accepted run has no validation predictions")
    with Path(source_record.path).open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    predictions = [
        row["score"] if include_target and row.get("score") else row["predicted"] for row in rows
    ]
    values: dict[str, list[str]] = {
        "row": [row["row"] for row in rows],
        "prediction": predictions,
    }
    if include_target:
        values["target"] = [row["actual"] for row in rows]
    pq.write_table(pa.table(values), path)
    return path


def _packet_file_record(path: Path) -> PacketFileRecord:
    return PacketFileRecord(
        name=path.name,
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
    )


def _copy_portable_artifacts(packet_directory: Path, result: RunResult) -> list[ArtifactRecord]:
    portable: list[ArtifactRecord] = []
    artifact_directory = packet_directory / "artifacts"
    artifact_directory.mkdir()
    for record in result.artifacts:
        source = Path(record.path)
        destination = artifact_directory / source.name
        shutil.copy2(source, destination)
        portable.append(
            ArtifactRecord(
                path=destination.relative_to(packet_directory).as_posix(),
                sha256=sha256_file(destination),
                size_bytes=destination.stat().st_size,
            )
        )
    return portable


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> Path:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
    return path


def _validate_submission(
    path: Path,
    expected_rows: int,
    expected_columns: list[str] | None = None,
    prediction_column: str = "prediction",
) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = expected_columns or ["row", "prediction"]
        if reader.fieldnames != columns:
            raise ValueError("Submission columns do not match the frozen contract")
        rows = list(reader)
    if len(rows) != expected_rows:
        raise ValueError("Submission row count does not match the source run")
    if columns == ["row", "prediction"]:
        row_ids = [row["row"] for row in rows]
        if len(set(row_ids)) != len(row_ids):
            raise ValueError("Submission row identifiers are not unique")
    if any(not row[prediction_column] for row in rows):
        raise ValueError("Submission contains a missing prediction")


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_object(path: Path) -> dict[str, object]:
    value = _read_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path.name}")
    return value


def _required(value: str | None, field: str) -> str:
    if value is None:
        raise ValueError(f"Accepted run is missing {field}")
    return value


def _decision_text(
    packet_type: str,
    result: RunResult,
    verification: VerificationResult,
    *,
    external_test: bool = False,
) -> str:
    if packet_type == "candidate":
        metric_name = next(iter(result.metrics))
        metric_value = result.metrics[metric_name]
        return (
            "# Candidate decision\n\n"
            f"Run `{result.run_id}` was accepted with validation {metric_name} "
            f"`{metric_value:.6f}`.\n\n"
            + (
                "The exported submission follows the uploaded template and has not been uploaded.\n"
                if external_test
                else (
                    "The exported submission is for the frozen fixture holdout and has not "
                    "been uploaded.\n"
                )
            )
        )
    reasons = "\n".join(f"- {reason}" for reason in verification.reasons)
    return f"# Failure decision\n\nRun `{result.run_id}` was rejected.\n\n{reasons}\n"
