import csv
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal

import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from swarmfold.artifacts import sha256_file, write_json
from swarmfold.contracts import contract_bundle_sha256
from swarmfold.packets import validate_terminal_packet
from swarmfold.schemas import (
    ArtifactRecord,
    CandidateManifest,
    CompletionRecord,
    ContractBundle,
    FailureManifest,
    PacketFileRecord,
    PacketResult,
    WaiverManifest,
)


class HumanCandidateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    packet_type: Literal["candidate"]
    iteration_id: str
    lane_id: str = Field(pattern=r"^human-[123]$")
    candidate_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]+$")
    run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]+$")
    metrics: dict[str, float]
    resource_usage: dict[str, float] = Field(default_factory=dict)
    decision: str = Field(min_length=10)


class HumanFailureRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    packet_type: Literal["failure"]
    iteration_id: str
    lane_id: str = Field(pattern=r"^human-[123]$")
    experiment_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]+$")
    run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]+$")
    terminal_reason: str = Field(min_length=10)
    completed_run_ids: list[str] = Field(default_factory=list)
    missing_deliverables: list[str] = Field(min_length=1)
    retry_useful: bool
    resource_usage: dict[str, float] = Field(default_factory=dict)
    decision: str = Field(min_length=10)


class HumanWaiverRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    packet_type: Literal["waiver"]
    iteration_id: str
    lane_id: str = Field(pattern=r"^human-[123]$")
    run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]+$")
    reason: str = Field(min_length=10)
    approver: str = Field(min_length=2)
    incomplete_work: list[str] = Field(default_factory=list)
    preserved_artifacts: bool = False
    decision: str = Field(min_length=10)


HumanPacketRequest = Annotated[
    HumanCandidateRequest | HumanFailureRequest | HumanWaiverRequest,
    Field(discriminator="packet_type"),
]
HUMAN_PACKET_ADAPTER = TypeAdapter(HumanPacketRequest)


def publish_human_packet(
    request: HumanPacketRequest,
    contracts: ContractBundle,
    output_dir: str | Path,
    *,
    submission: str | Path | None = None,
    oof_predictions: str | Path | None = None,
    test_predictions: str | Path | None = None,
    experiment_ledger: str | Path | None = None,
    findings: str | Path | None = None,
    artifacts: list[str | Path] | None = None,
) -> PacketResult:
    if contracts.contract_sha256 != contract_bundle_sha256(contracts):
        raise ValueError("Contract bundle hash is invalid")
    packet_parent = Path(output_dir) / "packets" / request.run_id
    packet_directory = packet_parent / f"{request.packet_type}-packet"
    staging = packet_parent / f".{request.packet_type}-packet.staging"
    if packet_directory.exists() or staging.exists():
        raise ValueError("Human lane packet already exists")
    staging.mkdir(parents=True)
    try:
        _build_human_packet(
            request,
            contracts,
            staging,
            submission=submission,
            oof_predictions=oof_predictions,
            test_predictions=test_predictions,
            experiment_ledger=experiment_ledger,
            findings=findings,
            artifacts=artifacts or [],
        )
        validate_terminal_packet(staging)
        staging.rename(packet_directory)
        return validate_terminal_packet(packet_directory)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _build_human_packet(
    request: HumanPacketRequest,
    contracts: ContractBundle,
    staging: Path,
    **inputs: object,
) -> PacketResult:
    contracts_path = write_json(staging / "contracts.json", contracts.model_dump())
    decision_path = staging / "decision.md"
    decision_path.write_text(request.decision.rstrip() + "\n", encoding="utf-8")
    packet_files = [contracts_path, decision_path]
    portable_artifacts = _copy_artifacts(staging, inputs["artifacts"])

    if isinstance(request, HumanCandidateRequest):
        named = {
            "submission.csv": inputs["submission"],
            "oof_predictions.parquet": inputs["oof_predictions"],
            "test_predictions.parquet": inputs["test_predictions"],
            "experiment-ledger.jsonl": inputs["experiment_ledger"],
            "findings.jsonl": inputs["findings"],
        }
        packet_files.extend(_copy_required_files(staging, named))
        submission_rows = _submission_row_count(staging / "submission.csv")
        _validate_prediction_parquet(
            staging / "oof_predictions.parquet", submission_rows, require_target=True
        )
        _validate_prediction_parquet(
            staging / "test_predictions.parquet", submission_rows, require_target=False
        )
        manifest = CandidateManifest(
            schema_version="0.2",
            packet_type="candidate",
            iteration_id=request.iteration_id,
            lane_id=request.lane_id,
            source_kind="human",
            experiment_id=request.candidate_id,
            run_id=request.run_id,
            contract_sha256=contracts.contract_sha256,
            data_version=contracts.data.data_version,
            metric_version=contracts.metric.metric_version,
            split_id=contracts.validation.split_id,
            dataset_sha256=contracts.data.dataset_sha256,
            metrics=request.metrics,
            submission_rows=submission_rows,
            artifacts=portable_artifacts,
            resource_usage=request.resource_usage,
            packet_files=[_record(path) for path in packet_files],
        )
        manifest_name = "artifact-manifest.json"
    elif isinstance(request, HumanFailureRequest):
        named = {
            "experiment-ledger.jsonl": inputs["experiment_ledger"],
            "findings.jsonl": inputs["findings"],
        }
        packet_files.extend(_copy_required_files(staging, named))
        manifest = FailureManifest(
            schema_version="0.2",
            packet_type="failure",
            iteration_id=request.iteration_id,
            lane_id=request.lane_id,
            source_kind="human",
            experiment_id=request.experiment_id,
            run_id=request.run_id,
            contract_sha256=contracts.contract_sha256,
            terminal_reason=request.terminal_reason,
            completed_run_ids=request.completed_run_ids,
            missing_deliverables=request.missing_deliverables,
            retry_useful=request.retry_useful,
            artifacts=portable_artifacts,
            resource_usage=request.resource_usage,
            packet_files=[_record(path) for path in packet_files],
        )
        manifest_name = "failure-manifest.json"
    else:
        manifest = WaiverManifest(
            schema_version="0.2",
            packet_type="waiver",
            iteration_id=request.iteration_id,
            lane_id=request.lane_id,
            run_id=request.run_id,
            contract_sha256=contracts.contract_sha256,
            reason=request.reason,
            approver=request.approver,
            incomplete_work=request.incomplete_work,
            preserved_artifacts=request.preserved_artifacts,
            packet_files=[_record(path) for path in packet_files],
        )
        manifest_name = "waiver-manifest.json"

    manifest_path = write_json(staging / manifest_name, manifest.model_dump())
    completion = CompletionRecord(
        schema_version="0.2",
        packet_type=request.packet_type,
        iteration_id=request.iteration_id,
        lane_id=request.lane_id,
        run_id=request.run_id,
        manifest_name=manifest_name,
        manifest_sha256=sha256_file(manifest_path),
        published_at=datetime.now(UTC).isoformat(),
    )
    write_json(staging / "COMPLETED.json", completion.model_dump())
    return PacketResult(
        packet_type=request.packet_type,
        iteration_id=request.iteration_id,
        lane_id=request.lane_id,
        packet_path=str(staging.resolve()),
        manifest_sha256=completion.manifest_sha256,
    )


def _copy_required_files(staging: Path, inputs: dict[str, object]) -> list[Path]:
    copied = []
    for name, raw_path in inputs.items():
        if raw_path is None:
            raise ValueError(f"Human packet requires {name}")
        source = Path(raw_path)
        if not source.is_file():
            raise ValueError(f"Human packet input is missing: {source}")
        destination = staging / name
        shutil.copy2(source, destination)
        copied.append(destination)
    return copied


def _copy_artifacts(staging: Path, raw_paths: object) -> list[ArtifactRecord]:
    records = []
    if not isinstance(raw_paths, list):
        raise ValueError("Human packet artifacts must be a list")
    paths = [Path(path) for path in raw_paths]
    if not paths:
        return records
    artifact_dir = staging / "artifacts"
    artifact_dir.mkdir()
    names: set[str] = set()
    for source in paths:
        if not source.is_file():
            raise ValueError(f"Human packet artifact is missing: {source}")
        if source.name in names:
            raise ValueError(f"Human packet artifact names must be unique: {source.name}")
        names.add(source.name)
        destination = artifact_dir / source.name
        shutil.copy2(source, destination)
        records.append(
            ArtifactRecord(
                path=destination.relative_to(staging).as_posix(),
                sha256=sha256_file(destination),
                size_bytes=destination.stat().st_size,
            )
        )
    return records


def _submission_row_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["row", "prediction"]:
            raise ValueError("Submission columns do not match the frozen contract")
        return sum(1 for _ in reader)


def _validate_prediction_parquet(path: Path, expected_rows: int, *, require_target: bool) -> None:
    table = pq.read_table(path)
    required = {"row", "prediction"}
    if require_target:
        required.add("target")
    if not required.issubset(table.column_names):
        raise ValueError(f"Prediction parquet is missing columns: {', '.join(sorted(required))}")
    if table.num_rows != expected_rows:
        raise ValueError("Prediction parquet row count does not match submission")
    row_ids = [str(value) for value in table.column("row").to_pylist()]
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("Prediction parquet row identifiers are not unique")
    if any(value is None for value in table.column("prediction").to_pylist()):
        raise ValueError("Prediction parquet contains missing predictions")


def _record(path: Path) -> PacketFileRecord:
    return PacketFileRecord(
        name=path.name,
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
    )


def load_human_packet_request(path: str | Path) -> HumanPacketRequest:
    return HUMAN_PACKET_ADAPTER.validate_python(json.loads(Path(path).read_text(encoding="utf-8")))
