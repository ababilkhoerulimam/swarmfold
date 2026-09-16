import json
import shutil
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq
from pydantic import ValidationError

from swarmfold.artifacts import sha256_file, write_json
from swarmfold.contracts import contract_bundle_sha256
from swarmfold.packets import read_packet_manifest, validate_terminal_packet
from swarmfold.schemas import (
    CandidateManifest,
    ContractBundle,
    FailureManifest,
    KnowledgeFinding,
    WaiverManifest,
)
from swarmfold.versioning import migrate_record


def merge_knowledge(
    *,
    iteration: dict[str, object],
    packet_records: list[dict[str, object]],
    decisions: list[dict[str, object]],
    leaderboard_results: list[dict[str, object]],
    snapshot_id: str,
    output_root: str | Path,
) -> dict[str, object]:
    iteration_id = str(iteration["iteration_id"])
    raw_charters = iteration.get("lane_charters")
    if not isinstance(raw_charters, list):
        raise ValueError("Iteration lane charters are missing")
    expected_lanes = sorted(str(charter["lane_id"]) for charter in raw_charters)
    if sorted(str(record["lane_id"]) for record in packet_records) != expected_lanes:
        raise ValueError("Knowledge merge requires one verified packet from every lane")
    contracts = ContractBundle.model_validate(iteration["contracts"])
    if contracts.contract_sha256 != contract_bundle_sha256(contracts):
        raise ValueError("Iteration contract bundle hash is invalid")

    snapshot_dir = Path(output_root).resolve() / snapshot_id
    if snapshot_dir.exists():
        existing = validate_knowledge_snapshot(snapshot_dir)
        if existing["source_iteration_id"] != iteration_id:
            raise ValueError("Knowledge snapshot version is already assigned")
        return {**existing, "idempotent_replay": True}
    staging = snapshot_dir.parent / f".{snapshot_id}.staging"
    if staging.exists():
        raise ValueError("Knowledge snapshot publication is already in progress")
    staging.mkdir(parents=True)
    try:
        result = _build_snapshot(
            staging=staging,
            iteration=iteration,
            packet_records=packet_records,
            decisions=decisions,
            leaderboard_results=leaderboard_results,
            contracts=contracts,
            snapshot_id=snapshot_id,
        )
        staging.rename(snapshot_dir)
        validated = validate_knowledge_snapshot(snapshot_dir)
        return {**validated, "idempotent_replay": result["idempotent_replay"]}
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def validate_knowledge_snapshot(snapshot_path: str | Path) -> dict[str, object]:
    snapshot_dir = Path(snapshot_path).resolve()
    completion_path = snapshot_dir / "COMPLETED.json"
    if not completion_path.is_file():
        raise ValueError("Knowledge snapshot is incomplete")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    manifest_path = snapshot_dir / "snapshot-manifest.json"
    if not manifest_path.is_file() or sha256_file(manifest_path) != completion.get(
        "manifest_sha256"
    ):
        raise ValueError("Knowledge snapshot manifest hash mismatch")
    raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw_manifest, dict):
        raise ValueError("Knowledge snapshot manifest must be an object")
    manifest = migrate_record("knowledge_snapshot", raw_manifest)
    if manifest.get("snapshot_id") != completion.get("snapshot_id") or manifest.get(
        "source_iteration_id"
    ) != completion.get("source_iteration_id"):
        raise ValueError("Knowledge snapshot completion identity mismatch")
    for record in manifest.get("files", []):
        name = record.get("name", "")
        if Path(name).name != name:
            raise ValueError("Knowledge snapshot contains an unsafe file name")
        path = snapshot_dir / name
        if (
            not path.is_file()
            or path.stat().st_size != record.get("size_bytes")
            or sha256_file(path) != record.get("sha256")
        ):
            raise ValueError(f"Knowledge snapshot file failed verification: {name}")
    return {
        "snapshot_id": manifest["snapshot_id"],
        "source_iteration_id": manifest["source_iteration_id"],
        "manifest_sha256": completion["manifest_sha256"],
        "artifact_path": str(snapshot_dir),
        "idempotent_replay": False,
    }


def read_knowledge_snapshot(snapshot_path: str | Path) -> dict[str, object]:
    snapshot_dir = Path(snapshot_path).resolve()
    identity = validate_knowledge_snapshot(snapshot_dir)
    contents: dict[str, object] = {}
    for path in sorted(snapshot_dir.iterdir()):
        if path.suffix == ".json" and path.name not in {"COMPLETED.json", "snapshot-manifest.json"}:
            contents[path.name] = json.loads(path.read_text(encoding="utf-8"))
        elif path.suffix == ".jsonl":
            contents[path.name] = _read_jsonl(path)
        elif path.suffix == ".md":
            contents[path.name] = path.read_text(encoding="utf-8")
    return {**identity, "contents": contents}


def _build_snapshot(
    *,
    staging: Path,
    iteration: dict[str, object],
    packet_records: list[dict[str, object]],
    decisions: list[dict[str, object]],
    leaderboard_results: list[dict[str, object]],
    contracts: ContractBundle,
    snapshot_id: str,
) -> dict[str, object]:
    iteration_id = str(iteration["iteration_id"])
    decision_by_lane = {str(item["lane_id"]): item for item in decisions}
    leaderboard_by_lane = {str(item["lane_id"]): item for item in leaderboard_results}
    findings: list[KnowledgeFinding] = []
    history = []
    lane_performance = []
    predictions: dict[str, dict[str, str]] = {}

    for record in packet_records:
        packet_path = Path(str(record["artifact_path"]))
        validate_terminal_packet(packet_path)
        manifest = read_packet_manifest(packet_path)
        lane_id = manifest.lane_id
        if (
            manifest.iteration_id != iteration_id
            or manifest.lane_id != record["lane_id"]
            or manifest.contract_sha256 != contracts.contract_sha256
        ):
            raise ValueError("Packet identity or contracts do not match the iteration")
        local_score = (
            manifest.metrics.get("accuracy") if isinstance(manifest, CandidateManifest) else None
        )
        decision = decision_by_lane.get(lane_id)
        leaderboard = leaderboard_by_lane.get(lane_id)
        history.append(
            {
                "iteration_id": iteration_id,
                "lane_id": lane_id,
                "packet_type": manifest.packet_type,
                "run_id": manifest.run_id,
                "packet_archive_sha256": record["packet_sha256"],
                "local_score": local_score,
            }
        )
        lane_performance.append(
            {
                "lane_id": lane_id,
                "packet_type": manifest.packet_type,
                "local_score": local_score,
                "submission_decision": decision["decision"] if decision else None,
                "leaderboard_score": leaderboard["score"] if leaderboard else None,
            }
        )
        findings.extend(_packet_findings(packet_path, manifest, contracts, iteration_id))
        if isinstance(manifest, CandidateManifest):
            predictions[lane_id] = _read_predictions(packet_path / "oof_predictions.parquet")

    findings.extend(
        _leaderboard_findings(leaderboard_results, contracts=contracts, iteration_id=iteration_id)
    )
    accepted, rejected, conflicts = _reconcile_findings(findings)
    champion_board = _champion_board(lane_performance)
    diversity = _prediction_diversity(predictions)
    backlog = _hypothesis_backlog(findings, conflicts)

    files = [
        write_json(staging / "competition-contract.json", contracts.competition.model_dump()),
        write_json(staging / "data-contract.json", contracts.data.model_dump()),
        write_json(staging / "metric-contract.json", contracts.metric.model_dump()),
        write_json(staging / "validation-contract.json", contracts.validation.model_dump()),
        write_json(staging / "champion-board.json", champion_board),
        _write_jsonl(staging / "experiment-history.jsonl", history),
        _write_jsonl(staging / "accepted-findings.jsonl", [item.model_dump() for item in accepted]),
        _write_jsonl(staging / "rejected-findings.jsonl", [item.model_dump() for item in rejected]),
        _write_jsonl(staging / "conflicts.jsonl", [item.model_dump() for item in conflicts]),
        write_json(staging / "prediction-diversity.json", diversity),
        write_json(staging / "hypothesis-backlog.json", backlog),
        write_json(staging / "lane-performance.json", lane_performance),
    ]
    summary = staging / "iteration-summary.md"
    summary.write_text(
        _summary_text(
            iteration_id,
            snapshot_id,
            lane_performance,
            accepted,
            rejected,
            conflicts,
            champion_board,
        ),
        encoding="utf-8",
    )
    files.append(summary)
    manifest = {
        "schema_version": "0.2",
        "snapshot_id": snapshot_id,
        "source_iteration_id": iteration_id,
        "source_snapshot_id": iteration.get("knowledge_snapshot_id"),
        "contract_sha256": contracts.contract_sha256,
        "schema_versions": {
            "packet": "0.2",
            "finding": "evidence-v1",
            "prompt": "planner-v1",
        },
        "created_at": datetime.now(UTC).isoformat(),
        "files": [
            {"name": path.name, "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for path in files
        ],
    }
    manifest_path = write_json(staging / "snapshot-manifest.json", manifest)
    completion = {
        "schema_version": "0.1",
        "snapshot_id": snapshot_id,
        "source_iteration_id": iteration_id,
        "manifest_sha256": sha256_file(manifest_path),
        "published_at": datetime.now(UTC).isoformat(),
    }
    write_json(staging / "COMPLETED.json", completion)
    validate_knowledge_snapshot(staging)
    return {
        "snapshot_id": snapshot_id,
        "source_iteration_id": iteration_id,
        "manifest_sha256": completion["manifest_sha256"],
        "artifact_path": str(staging),
        "idempotent_replay": False,
    }


def _packet_findings(
    packet_path: Path,
    manifest: CandidateManifest | FailureManifest | WaiverManifest,
    contracts: ContractBundle,
    iteration_id: str,
) -> list[KnowledgeFinding]:
    parsed = []
    findings_path = packet_path / "findings.jsonl"
    if findings_path.is_file():
        for index, value in enumerate(_read_jsonl(findings_path), start=1):
            try:
                finding = KnowledgeFinding.model_validate(value)
            except ValidationError:
                finding = _legacy_finding(value, manifest, contracts, iteration_id, index)
            _validate_finding_provenance(finding, manifest, contracts, iteration_id)
            parsed.append(finding)
    if parsed:
        return parsed
    if isinstance(manifest, CandidateManifest):
        score = manifest.metrics.get("accuracy")
        return [
            _finding(
                manifest,
                contracts,
                iteration_id,
                "VALIDATED",
                f"Lane {manifest.lane_id} produced a verified candidate",
                "A reproducible candidate passed packet and contract verification",
                local_delta=score,
            )
        ]
    if isinstance(manifest, FailureManifest):
        return [
            _finding(
                manifest,
                contracts,
                iteration_id,
                "REJECTED",
                manifest.terminal_reason,
                "The lane ended with a verified failure packet",
                limitations=manifest.missing_deliverables,
            )
        ]
    return [
        _finding(
            manifest,
            contracts,
            iteration_id,
            "UNKNOWN",
            manifest.reason,
            "A human approved a lane waiver without candidate evidence",
            limitations=manifest.incomplete_work,
        )
    ]


def _legacy_finding(
    value: dict[str, object],
    manifest: CandidateManifest | FailureManifest | WaiverManifest,
    contracts: ContractBundle,
    iteration_id: str,
    index: int,
) -> KnowledgeFinding:
    verdict = str(value.get("verdict", "UNKNOWN"))
    status = "VALIDATED" if verdict in {"IMPROVED", "TIED"} else "REJECTED"
    claim = str(value.get("recommendation") or f"Recorded lane outcome {index}")
    return _finding(
        manifest,
        contracts,
        iteration_id,
        status,
        claim,
        f"Legacy result critique verdict: {verdict}",
        suffix=str(index),
        local_delta=value.get("accuracy") if isinstance(value.get("accuracy"), float) else None,
    )


def _finding(
    manifest: CandidateManifest | FailureManifest | WaiverManifest,
    contracts: ContractBundle,
    iteration_id: str,
    status: str,
    claim: str,
    mechanism: str,
    *,
    suffix: str = "outcome",
    local_delta: float | None = None,
    limitations: list[str] | None = None,
) -> KnowledgeFinding:
    return KnowledgeFinding(
        finding_id=f"{iteration_id}-{manifest.lane_id}-{suffix}",
        iteration_id=iteration_id,
        source_lane=manifest.lane_id,
        status=status,
        claim=claim,
        mechanism=mechanism,
        source_runs=[manifest.run_id],
        data_version=contracts.data.data_version,
        split_id=contracts.validation.split_id,
        metric_name=contracts.metric.metric_name,
        local_delta=local_delta,
        limitations=limitations or [],
    )


def _validate_finding_provenance(
    finding: KnowledgeFinding,
    manifest: CandidateManifest | FailureManifest | WaiverManifest,
    contracts: ContractBundle,
    iteration_id: str,
) -> None:
    if finding.iteration_id != iteration_id or finding.source_lane != manifest.lane_id:
        raise ValueError("Finding identity does not match its packet")
    if (
        finding.data_version != contracts.data.data_version
        or finding.split_id != contracts.validation.split_id
        or finding.metric_name != contracts.metric.metric_name
    ):
        raise ValueError("Finding provenance does not match the frozen contracts")
    allowed_runs = {manifest.run_id}
    if isinstance(manifest, FailureManifest):
        allowed_runs.update(manifest.completed_run_ids)
    if not set(finding.source_runs).issubset(allowed_runs):
        raise ValueError("Finding references a run outside its verified packet")
    allowed_artifacts = {artifact.path for artifact in getattr(manifest, "artifacts", [])}
    if not set(finding.artifact_uris).issubset(allowed_artifacts):
        raise ValueError("Finding references an artifact outside its verified packet")


def _leaderboard_findings(
    rows: list[dict[str, object]],
    *,
    contracts: ContractBundle,
    iteration_id: str,
) -> list[KnowledgeFinding]:
    findings = []
    for row in rows:
        if row["local_supported"]:
            continue
        lane_id = str(row["lane_id"])
        findings.append(
            KnowledgeFinding(
                finding_id=f"{iteration_id}-{lane_id}-leaderboard",
                iteration_id=iteration_id,
                source_lane=lane_id,
                status="LEADERBOARD_ONLY",
                claim=f"Lane {lane_id} recorded leaderboard score {row['score']}",
                mechanism="Leaderboard movement lacks declared local support",
                source_runs=[],
                data_version=contracts.data.data_version,
                split_id=contracts.validation.split_id,
                metric_name=contracts.metric.metric_name,
                leaderboard_delta=float(row["score"]),
                limitations=["Leaderboard evidence is not local validation evidence"],
            )
        )
    return findings


def _reconcile_findings(
    findings: list[KnowledgeFinding],
) -> tuple[list[KnowledgeFinding], list[KnowledgeFinding], list[KnowledgeFinding]]:
    accepted = [
        item
        for item in findings
        if item.status in {"OBSERVED", "VALIDATED", "CROSS_LANE", "LEADERBOARD_ONLY"}
    ]
    rejected = [item for item in findings if item.status == "REJECTED"]
    conflicts = [item for item in findings if item.status == "CONFLICT"]
    groups: dict[str, list[KnowledgeFinding]] = defaultdict(list)
    for item in findings:
        groups[" ".join(item.claim.casefold().split())].append(item)
    for group in groups.values():
        lanes = {item.source_lane for item in group}
        supported = [
            item for item in group if item.status in {"OBSERVED", "VALIDATED", "CROSS_LANE"}
        ]
        opposed = [item for item in group if item.status == "REJECTED"]
        if supported and opposed:
            first = group[0]
            conflicts.append(
                first.model_copy(
                    update={
                        "finding_id": f"{first.iteration_id}-conflict-{len(conflicts) + 1}",
                        "source_lane": first.source_lane,
                        "status": "CONFLICT",
                        "mechanism": "Credible verified lanes reported opposing evidence",
                        "source_runs": sorted({run for item in group for run in item.source_runs}),
                        "limitations": [f"Conflicting lanes: {', '.join(sorted(lanes))}"],
                    }
                )
            )
        elif len(lanes) > 1 and supported:
            first = supported[0]
            accepted.append(
                first.model_copy(
                    update={
                        "finding_id": f"{first.iteration_id}-cross-lane-{len(accepted) + 1}",
                        "status": "CROSS_LANE",
                        "source_runs": sorted(
                            {run for item in supported for run in item.source_runs}
                        ),
                        "limitations": [f"Supporting lanes: {', '.join(sorted(lanes))}"],
                    }
                )
            )
    return accepted, rejected, conflicts


def _champion_board(lanes: list[dict[str, object]]) -> list[dict[str, object]]:
    candidates = [item for item in lanes if item["packet_type"] == "candidate"]
    return sorted(
        candidates,
        key=lambda item: (
            item["leaderboard_score"] is not None,
            item["leaderboard_score"] if item["leaderboard_score"] is not None else float("-inf"),
            item["local_score"] if item["local_score"] is not None else float("-inf"),
        ),
        reverse=True,
    )


def _read_predictions(path: Path) -> dict[str, str]:
    table = pq.read_table(path, columns=["row", "prediction"])
    rows = table.column("row").to_pylist()
    predictions = table.column("prediction").to_pylist()
    return {str(row): str(prediction) for row, prediction in zip(rows, predictions, strict=True)}


def _prediction_diversity(predictions: dict[str, dict[str, str]]) -> dict[str, object]:
    pairs = []
    lanes = sorted(predictions)
    for index, left in enumerate(lanes):
        for right in lanes[index + 1 :]:
            if predictions[left].keys() != predictions[right].keys():
                raise ValueError("Candidate prediction rows are not aligned across lanes")
            rows = predictions[left]
            agreement = sum(
                predictions[left][row] == predictions[right][row] for row in rows
            ) / len(rows)
            pairs.append({"left_lane": left, "right_lane": right, "agreement": agreement})
    return {"candidate_lanes": lanes, "pairwise_agreement": pairs}


def _hypothesis_backlog(
    findings: list[KnowledgeFinding], conflicts: list[KnowledgeFinding]
) -> list[dict[str, object]]:
    unresolved = [item for item in findings if item.status in {"HYPOTHESIS", "UNKNOWN"}]
    return [
        {
            "claim": item.claim,
            "source_lane": item.source_lane,
            "reason": "conflict" if item.status == "CONFLICT" else "unresolved",
        }
        for item in [*conflicts, *unresolved]
    ]


def _summary_text(
    iteration_id: str,
    snapshot_id: str,
    lanes: list[dict[str, object]],
    accepted: list[KnowledgeFinding],
    rejected: list[KnowledgeFinding],
    conflicts: list[KnowledgeFinding],
    champion_board: list[dict[str, object]],
) -> str:
    champion = champion_board[0]["lane_id"] if champion_board else "none"
    return (
        f"# Iteration {iteration_id}\n\n"
        f"Snapshot: `{snapshot_id}`\n\n"
        f"Terminal lanes: {len(lanes)}\n\n"
        f"Champion lane: `{champion}`\n\n"
        f"Accepted findings: {len(accepted)}; rejected findings: {len(rejected)}; "
        f"conflicts: {len(conflicts)}.\n"
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL records must be objects: {path.name}")
            values.append(value)
    return values


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> Path:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
    return path
