import json
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from swarmfold.artifacts import write_json
from swarmfold.knowledge import read_knowledge_snapshot
from swarmfold.packets import read_packet_manifest, validate_terminal_packet
from swarmfold.schemas import CandidateManifest, FailureManifest


def evaluate_swarm_effectiveness(
    *,
    iteration_id: str,
    packet_records: list[dict[str, object]],
    snapshot_path: str | Path,
) -> dict[str, object]:
    snapshot = read_knowledge_snapshot(snapshot_path)
    contents = snapshot["contents"]
    if not isinstance(contents, dict):
        raise ValueError("Knowledge snapshot contents are invalid")
    history = _records(contents, "experiment-history.jsonl")
    rejected = _records(contents, "rejected-findings.jsonl")
    lane_performance = _records(contents, "lane-performance.json")
    diversity = contents.get("prediction-diversity.json", {})

    experiment_ids = []
    total_seconds = 0.0
    candidate_count = 0
    reproducible_candidates = 0
    mechanisms = set()
    fabricated_failures = 0
    failed_lanes = 0
    recovered_lanes = 0
    for record in packet_records:
        packet_path = Path(str(record["artifact_path"]))
        validate_terminal_packet(packet_path)
        manifest = read_packet_manifest(packet_path)
        experiment_ids.append(manifest.run_id)
        resource_usage = getattr(manifest, "resource_usage", {})
        total_seconds += float(resource_usage.get("wall_seconds", 0.0))
        if isinstance(manifest, CandidateManifest):
            candidate_count += 1
            if manifest.packet_files:
                reproducible_candidates += 1
        if isinstance(manifest, FailureManifest):
            failed_lanes += 1
            if manifest.completed_run_ids:
                recovered_lanes += 1
            reason = manifest.terminal_reason.casefold()
            if "fabricated" in reason or "unsupported" in reason or " api" in reason:
                fabricated_failures += 1
        proposal_path = packet_path / "proposal.json"
        if proposal_path.is_file():
            proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
            mechanisms.add(str(proposal.get("model_family", "unknown")))
        elif isinstance(manifest, CandidateManifest):
            mechanisms.add("human_manual")

    counts = Counter(experiment_ids)
    duplicate_count = sum(count - 1 for count in counts.values() if count > 1)
    experiment_count = len(experiment_ids)
    mean_oof_diversity = _mean_oof_diversity(diversity)
    rank_consistency = _local_leaderboard_rank_consistency(lane_performance)
    repeated_rejections = _repeated_rejection_rate(rejected)
    valid_per_hour = candidate_count / (total_seconds / 3600) if total_seconds > 0 else None
    source_snapshot_id = _snapshot_source_id(snapshot_path)

    return {
        "schema_version": "0.1",
        "iteration_id": iteration_id,
        "snapshot_id": snapshot["snapshot_id"],
        "generated_at": datetime.now(UTC).isoformat(),
        "metrics": {
            "valid_experiments_per_compute_hour": valid_per_hour,
            "duplicate_experiment_rate": duplicate_count / experiment_count
            if experiment_count
            else 0.0,
            "fabricated_api_rejection_rate": fabricated_failures / failed_lanes
            if failed_lanes
            else 0.0,
            "failed_run_recovery_rate": recovered_lanes / failed_lanes if failed_lanes else 0.0,
            "reproducible_candidate_share": reproducible_candidates / candidate_count
            if candidate_count
            else 0.0,
            "model_mechanisms_tested": len(mechanisms),
            "mean_oof_diversity": mean_oof_diversity,
            "local_to_leaderboard_rank_consistency": rank_consistency,
            "knowledge_snapshot_reused": source_snapshot_id is not None,
            "repeated_rejected_hypothesis_rate": repeated_rejections,
            "terminal_lane_count": len(history),
        },
        "unavailable": {
            "human_resolution_minutes": "No time-tracking evidence was supplied.",
            "score_improvement_same_compute": "No matched non-swarm control run was supplied.",
        },
    }


def write_effectiveness_report(
    *,
    iteration_id: str,
    packet_records: list[dict[str, object]],
    snapshot_path: str | Path,
    output_path: str | Path,
) -> Path:
    report = evaluate_swarm_effectiveness(
        iteration_id=iteration_id,
        packet_records=packet_records,
        snapshot_path=snapshot_path,
    )
    return write_json(output_path, report)


def _records(contents: dict[str, object], name: str) -> list[dict[str, object]]:
    value = contents.get(name, [])
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"Knowledge snapshot record set is invalid: {name}")
    return value


def _mean_oof_diversity(value: object) -> float | None:
    if not isinstance(value, dict):
        return None
    pairs = value.get("pairwise_agreement", [])
    if not isinstance(pairs, list) or not pairs:
        return None
    agreements = [float(pair["agreement"]) for pair in pairs if isinstance(pair, dict)]
    return 1.0 - sum(agreements) / len(agreements) if agreements else None


def _local_leaderboard_rank_consistency(lanes: list[dict[str, object]]) -> float | None:
    comparable = [
        lane
        for lane in lanes
        if lane.get("local_score") is not None and lane.get("leaderboard_score") is not None
    ]
    if len(comparable) < 2:
        return None
    local = _ranks([float(lane["local_score"]) for lane in comparable])
    leaderboard = _ranks([float(lane["leaderboard_score"]) for lane in comparable])
    count = len(comparable)
    squared_difference = sum(
        (left - right) ** 2 for left, right in zip(local, leaderboard, strict=True)
    )
    return 1.0 - (6.0 * squared_difference) / (count * (count**2 - 1))


def _ranks(values: list[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=lambda index: values[index], reverse=True)
    ranks = [0.0] * len(values)
    position = 0
    while position < len(ordered):
        end = position + 1
        while end < len(ordered) and math.isclose(values[ordered[end]], values[ordered[position]]):
            end += 1
        rank = (position + 1 + end) / 2
        for index in ordered[position:end]:
            ranks[index] = rank
        position = end
    return ranks


def _repeated_rejection_rate(rejected: list[dict[str, object]]) -> float:
    claims = [" ".join(str(item.get("claim", "")).casefold().split()) for item in rejected]
    claims = [claim for claim in claims if claim]
    if not claims:
        return 0.0
    counts = Counter(claims)
    return sum(count - 1 for count in counts.values() if count > 1) / len(claims)


def _snapshot_source_id(snapshot_path: str | Path) -> str | None:
    manifest = json.loads(
        (Path(snapshot_path) / "snapshot-manifest.json").read_text(encoding="utf-8")
    )
    value = manifest.get("source_snapshot_id")
    return str(value) if value is not None else None
