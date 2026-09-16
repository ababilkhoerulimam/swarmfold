import contextlib
import csv
import json
import platform
import re
import shutil
import stat
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Literal

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from swarmfold.artifacts import sha256_file, write_json
from swarmfold.capability_matrix import format_markdown_matrix
from swarmfold.ensemble import PredictionRegistry, UnifiedEnsembleEngine
from swarmfold.metrics import get_metric_spec

AILaneId = Literal["ai-a", "ai-b", "ai-c", "ai-d", "ai-e"]

LANE_DEFINITIONS: dict[str, dict[str, str]] = {
    "ai-a": {
        "name": "Conservative Scientist",
        "focus": "Stable baselines, sanity checks, and low variance candidates.",
    },
    "ai-b": {
        "name": "Boosting Specialist",
        "focus": "Tree ensembles, nonlinear interactions, and bounded tuning.",
    },
    "ai-c": {
        "name": "Feature Scientist",
        "focus": "Representations, feature selection, and preprocessing hypotheses.",
    },
    "ai-d": {
        "name": "Alternative Explorer",
        "focus": "Nonstandard but compatible model families and diverse predictions.",
    },
    "ai-e": {
        "name": "Robustness Challenger",
        "focus": "Imbalance, regularization, stability, and ensemble readiness.",
    },
}


class LaneConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    lane_id: AILaneId
    enabled: bool = True
    max_experiments: int = Field(default=5, ge=5, le=100)

    @property
    def name(self) -> str:
        return LANE_DEFINITIONS[self.lane_id]["name"]

    @property
    def focus(self) -> str:
        return LANE_DEFINITIONS[self.lane_id]["focus"]

    def public_dict(self) -> dict[str, object]:
        return {**self.model_dump(), "name": self.name, "focus": self.focus}


class PortfolioConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    lanes: list[LaneConfig]
    max_concurrent_lanes: int = Field(default=2, ge=1, le=5)
    ensemble_enabled: bool = True
    max_ensemble_trials: int | None = Field(default=100, ge=1)

    @model_validator(mode="after")
    def validate_lane_set(self) -> "PortfolioConfig":
        lane_ids = [lane.lane_id for lane in self.lanes]
        if len(lane_ids) != len(set(lane_ids)):
            raise ValueError("Lane configurations must have unique lane IDs")
        if set(lane_ids) != set(LANE_DEFINITIONS):
            raise ValueError("Portfolio configuration must include AI A through AI E")
        if not self.enabled_lanes:
            raise ValueError("Enable at least one AI lane")
        return self

    @property
    def enabled_lanes(self) -> list[LaneConfig]:
        return [lane for lane in self.lanes if lane.enabled]

    @property
    def total_experiment_budget(self) -> int:
        return sum(lane.max_experiments for lane in self.enabled_lanes)

    def enforce_total_limit(self, maximum: int) -> None:
        if self.total_experiment_budget > maximum:
            raise ValueError(
                f"Portfolio requests {self.total_experiment_budget} experiments; "
                f"the server limit is {maximum}"
            )

    def public_lanes(self) -> list[dict[str, object]]:
        return [
            {**lane.public_dict(), "status": "PLANNED" if lane.enabled else "DISABLED"}
            for lane in self.lanes
        ]


def default_portfolio() -> PortfolioConfig:
    return PortfolioConfig(
        lanes=[LaneConfig(lane_id=lane_id) for lane_id in LANE_DEFINITIONS],
        max_concurrent_lanes=2,
    )


def single_lane_portfolio(lane_id: str, max_experiments: int) -> PortfolioConfig:
    if lane_id not in LANE_DEFINITIONS:
        raise ValueError(f"Unknown AI lane: {lane_id}")
    return PortfolioConfig(
        lanes=[
            LaneConfig(
                lane_id=value,
                enabled=value == lane_id,
                max_experiments=max_experiments if value == lane_id else 5,
            )
            for value in LANE_DEFINITIONS
        ],
        max_concurrent_lanes=1,
    )


def inspect_prior_packages(files: list[Path]) -> list[dict[str, object]]:
    memories = []
    for path in files:
        if path.suffix.casefold() != ".zip":
            continue
        try:
            with zipfile.ZipFile(path) as archive:
                names = set(archive.namelist())
                if "experiment-knowledge.md" not in names or "manifest.json" not in names:
                    continue
                knowledge_info = archive.getinfo("experiment-knowledge.md")
                manifest_info = archive.getinfo("manifest.json")
                if knowledge_info.file_size > 2_000_000 or manifest_info.file_size > 2_000_000:
                    raise ValueError(f"Prior package metadata is too large: {path.name}")
                knowledge = archive.read(knowledge_info).decode("utf-8")
                manifest = json.loads(archive.read(manifest_info).decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
            raise ValueError(f"Cannot read prior experiment package {path.name}: {exc}") from exc
        memories.append(
            {
                "source_file": path.name,
                "source_run_id": manifest.get("run_id"),
                "source_iteration_id": manifest.get("iteration_id"),
                "source_profile": manifest.get("profile"),
                "knowledge": knowledge,
                "transfer_status": "CONTEXT_ONLY",
            }
        )
    return memories


def build_full_package(
    run_directory: Path,
    record: dict[str, object],
    lane_results: dict[str, dict[str, object]],
) -> dict[str, object]:
    package_directory = run_directory / "full-package"
    staging_directory = run_directory / ".full-package.staging"
    if staging_directory.exists():
        shutil.rmtree(staging_directory)
    staging_directory.mkdir()

    bundle_source = run_directory / "bundle"
    if bundle_source.is_dir():
        shutil.copytree(bundle_source, staging_directory / "competition-bundle")

    experiment_root = staging_directory / "experiments"
    findings_root = staging_directory / "findings"
    experiment_root.mkdir()
    findings_root.mkdir()

    combined_ledger: list[dict[str, object]] = []
    combined_findings: list[dict[str, object]] = []
    literature: list[dict[str, object]] = []
    candidate_packets: list[tuple[str, Path, dict[str, object]]] = []

    for lane_id, lane_result in sorted(lane_results.items()):
        lane_directory = experiment_root / lane_id
        lane_directory.mkdir()
        write_json(lane_directory / "lane-result.json", _portable_lane_result(lane_result))
        packet = lane_result.get("packet")
        if not isinstance(packet, dict) or not isinstance(packet.get("packet_path"), str):
            continue
        packet_path = Path(packet["packet_path"]).resolve()
        packet_copy = lane_directory / "packet"
        shutil.copytree(packet_path, packet_copy)
        _make_packet_portable(packet_copy)
        if packet.get("packet_type") == "candidate":
            candidate_packets.append((lane_id, packet_copy, lane_result))
        combined_ledger.extend(
            _read_jsonl_with_lane(packet_copy / "experiment-ledger.jsonl", lane_id)
        )
        lane_findings = _read_jsonl_with_lane(packet_copy / "findings.jsonl", lane_id)
        combined_findings.extend(lane_findings)
        _write_jsonl(findings_root / f"{lane_id}.jsonl", lane_findings)
        review_path = packet_copy / "literature-review.json"
        if review_path.is_file():
            literature.append(
                {"lane_id": lane_id, "review": json.loads(review_path.read_text(encoding="utf-8"))}
            )

    _write_jsonl(staging_directory / "experiment-ledger.jsonl", combined_ledger)
    _write_jsonl(staging_directory / "findings.jsonl", combined_findings)
    write_json(staging_directory / "literature-review.json", {"lanes": literature})

    profile = record.get("profile") if isinstance(record.get("profile"), dict) else {}
    if record.get("ensemble_enabled", True):
        ensemble = _build_ensemble(
            candidate_packets,
            staging_directory,
            task_type=str(profile.get("task_type", "binary_classification")),
        )
    else:
        ensemble = {
            "status": "DISABLED",
            "adopted": False,
            "method": "none",
            "reason": "Agent F ensemble stage was disabled by portfolio configuration",
        }
    champion = ensemble.get("champion_lane")
    contracts_written = False
    if isinstance(champion, str):
        champion_packet = next(path for lane, path, _ in candidate_packets if lane == champion)
        for name in ("end-to-end.ipynb", "submission.csv", "contracts.json"):
            source = champion_packet / name
            destination = staging_directory / name
            if source.is_file() and not destination.exists():
                shutil.copy2(source, destination)
                if name == "contracts.json":
                    contracts_written = True

    if not contracts_written and not (staging_directory / "contracts.json").is_file():
        write_json(staging_directory / "contracts.json", record.get("profile") or {})

    validation_audit: dict[str, object] = {}
    if isinstance(champion, str):
        champion_packet = next(path for lane, path, _ in candidate_packets if lane == champion)
        evidence_p = champion_packet / "modeling-quality-evidence.json"
        if evidence_p.is_file():
            with contextlib.suppress(Exception):
                ev = json.loads(evidence_p.read_text(encoding="utf-8"))
                validation_audit = ev.get("validation_audit", {})
    write_json(staging_directory / "validation-audit.json", validation_audit)

    fingerprints: list[dict[str, object]] = []
    if bundle_source.is_dir():
        for bf in sorted(bundle_source.iterdir()):
            if bf.is_file():
                fingerprints.append(
                    {
                        "filename": bf.name,
                        "sha256": sha256_file(bf),
                        "size_bytes": bf.stat().st_size,
                    }
                )
    write_json(staging_directory / "data-fingerprints.json", {"bundle_files": fingerprints})

    env_lock: dict[str, object] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "packages": {
            "pandas": getattr(__import__("pandas", None, None, [], 0), "__version__", "unknown"),
            "numpy": getattr(__import__("numpy", None, None, [], 0), "__version__", "unknown"),
            "scikit_learn": getattr(
                __import__("sklearn", None, None, [], 0), "__version__", "unknown"
            ),
        },
    }
    uv_lock_path = Path("uv.lock")
    if uv_lock_path.is_file():
        env_lock["uv_lock_sha256"] = sha256_file(uv_lock_path)
    write_json(staging_directory / "environment-lock.json", env_lock)

    (staging_directory / "capability-matrix.md").write_text(
        format_markdown_matrix(), encoding="utf-8"
    )

    technique_evidence: list[dict[str, object]] = []
    for lane_id, packet_copy, _ in candidate_packets:
        plan_path = packet_copy / "technique_plan.json"
        if plan_path.is_file():
            with contextlib.suppress(Exception):
                technique_evidence.append(
                    {
                        "lane_id": lane_id,
                        "plan": json.loads(plan_path.read_text(encoding="utf-8")),
                    }
                )
    write_json(staging_directory / "technique-evidence.json", {"lanes": technique_evidence})

    write_json(staging_directory / "ensemble-assessment.json", ensemble)
    write_json(staging_directory / "ensemble_spec.json", ensemble)
    write_json(
        staging_directory / "agent-f.json",
        {
            "agent": "Agent F",
            "role": "portfolio-level ensemble specialist",
            "stage": "ensemble",
            "status": ensemble.get("status", "SKIPPED"),
            "method": ensemble.get("method", "none"),
            "adopted": ensemble.get("adopted", False),
            "member_count": len(ensemble.get("members", [])),
            "reason": ensemble.get("reason", ""),
            "diagnostics": ensemble.get("diagnostics", []),
        },
    )
    knowledge_path = staging_directory / "experiment-knowledge.md"
    knowledge_path.write_text(
        _knowledge_markdown(record, lane_results, combined_ledger, combined_findings, ensemble),
        encoding="utf-8",
    )

    manifest = {
        "schema_version": "portfolio-package-v1",
        "run_id": record["run_id"],
        "iteration_id": record["iteration_id"],
        "profile": record.get("profile"),
        "lanes": record.get("lanes", []),
        "total_experiment_budget": record.get("max_experiments"),
        "executed_experiments": len(combined_ledger),
        "ensemble": ensemble,
        "files": [],
    }
    manifest["files"] = [
        {
            "path": path.relative_to(staging_directory).as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(staging_directory.rglob("*"))
        if path.is_file()
    ]
    manifest_path = write_json(staging_directory / "manifest.json", manifest)

    if package_directory.exists():
        shutil.rmtree(package_directory)
    staging_directory.rename(package_directory)
    archive_path = run_directory / "full-package.zip"
    _archive_directory(package_directory, archive_path)
    return {
        "path": str(archive_path.resolve()),
        "sha256": sha256_file(archive_path),
        "manifest_sha256": sha256_file(package_directory / manifest_path.name),
        "champion_lane": champion,
        "ensemble_adopted": ensemble.get("adopted", False),
    }


def _portable_lane_result(result: dict[str, object]) -> dict[str, object]:
    portable = {
        key: value
        for key, value in result.items()
        if key not in {"completed_results", "critiques", "packet"}
    }
    packet = result.get("packet")
    if isinstance(packet, dict):
        portable["packet"] = {key: value for key, value in packet.items() if key != "packet_path"}
    return portable


def _portable_text(value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return f"artifacts/{path.name}"
    windows_path = re.compile(r"[A-Za-z]:[\\/][^'\"\n\r]+")
    unix_path = re.compile(r"/(?:Users|home|tmp|var)/[^'\"\n\r]+")
    value = windows_path.sub(lambda match: f"artifacts/{Path(match.group()).name}", value)
    return unix_path.sub(lambda match: f"artifacts/{Path(match.group()).name}", value)


def _portable_value(value):
    if isinstance(value, dict):
        return {key: _portable_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_portable_value(item) for item in value]
    if isinstance(value, str):
        return _portable_text(value)
    return value


def _make_packet_portable(packet_directory: Path) -> None:
    for path in sorted(packet_directory.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.casefold() in {".json", ".ipynb"}:
            with contextlib.suppress(json.JSONDecodeError, UnicodeDecodeError):
                payload = json.loads(path.read_text(encoding="utf-8"))
                write_json(path, _portable_value(payload))
        elif path.suffix.casefold() == ".jsonl":
            records = []
            with contextlib.suppress(json.JSONDecodeError, UnicodeDecodeError):
                records = [
                    _portable_value(json.loads(line))
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            if records:
                _write_jsonl(path, records)

    completion_path = packet_directory / "COMPLETED.json"
    if not completion_path.is_file():
        return
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    manifest_name = str(completion.get("manifest_name", "artifact-manifest.json"))
    manifest_path = packet_directory / manifest_name
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for record in manifest.get("packet_files", []):
        packet_file = packet_directory / str(record.get("name", ""))
        if packet_file.is_file():
            record["sha256"] = sha256_file(packet_file)
            record["size_bytes"] = packet_file.stat().st_size
    write_json(manifest_path, manifest)
    completion["manifest_sha256"] = sha256_file(manifest_path)
    write_json(completion_path, completion)


def _read_jsonl_with_lane(path: Path, lane_id: str) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                records.append({"lane_id": lane_id, **value})
    return records


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")


def _build_ensemble(
    candidates: list[tuple[str, Path, dict[str, object]]],
    package_directory: Path,
    *,
    task_type: str = "binary_classification",
) -> dict[str, object]:
    if not candidates:
        return {"status": "SKIPPED", "adopted": False, "reason": "No candidate packet"}
    metric_name = _metric_name(candidates[0][2])
    metric_spec = get_metric_spec(metric_name)
    ranked = sorted(
        candidates,
        key=lambda item: _best_score(item[2], metric_name),
        reverse=metric_spec.higher_is_better,
    )
    champion_lane, champion_packet, champion_result = ranked[0]
    base = {
        "status": "ASSESSED",
        "adopted": False,
        "champion_lane": champion_lane,
        "champion_score": _best_score(champion_result, metric_name),
        "members": [lane_id for lane_id, _, _ in ranked],
    }
    try:
        registry = PredictionRegistry()
        member_lanes, has_full_experiment_registry = _register_experiment_predictions(
            registry,
            ranked,
            metric_name,
            package_directory,
        )
        registry.validate_alignment()

        if not has_full_experiment_registry:
            methods = (
                ["equal_average", "pair_blend"]
                if metric_name in {"roc_auc", "log_loss", "rmse", "mae", "r2"}
                else ["majority_vote"]
            )
        elif metric_name in {"roc_auc", "log_loss"}:
            methods = [
                "equal_average",
                "pair_blend",
                "rank_average",
                "constrained_weighted_blend",
                "caruana",
                "stacking",
            ]
        elif metric_name in {"rmse", "mae", "r2"}:
            methods = [
                "equal_average",
                "pair_blend",
                "constrained_weighted_blend",
                "caruana",
                "stacking",
            ]
        else:
            methods = ["majority_vote", "stacking"]
        engine = UnifiedEnsembleEngine(
            metric_name=metric_name,
            task_type=task_type,
            methods=methods,
            artifact_directory=package_directory / "ensemble",
        )
        decision, test_predictions = engine.evaluate_registry(registry)

        if decision.adopted and test_predictions is not None:
            _write_ensemble_submission(
                champion_packet / "submission.csv",
                champion_packet / "contracts.json",
                package_directory / "submission.csv",
                test_predictions.tolist()
                if hasattr(test_predictions, "tolist")
                else list(test_predictions),
            )
        weights_dict = dict(decision.weights)
        corr_matrix: dict[str, dict[str, float]] = {}
        try:
            corr_df = registry.correlation_matrix(metric_name)
            corr_matrix = {
                str(col): {str(row): float(corr_df.loc[row, col]) for row in corr_df.index}
                for col in corr_df.columns
            }
        except Exception:
            pass

        resolved_champion_lane = member_lanes.get(
            decision.champion_lane or "", champion_lane
        )
        return {
            **base,
            "champion_lane": resolved_champion_lane,
            "champion_member": decision.champion_lane,
            "members": list(registry.members),
            "status": decision.status,
            "adopted": decision.adopted,
            "improved_over_best_single": decision.improved_over_best_single,
            "ensemble_score": decision.ensemble_score,
            "metric_name": metric_name,
            "method": decision.method,
            "rule": decision.rule,
            "weights": weights_dict,
            "selected_members": decision.selected_members,
            "correlation_matrix": corr_matrix,
            "reason": (
                "Cross-lane OOF ensemble improved the best single model"
                if decision.adopted
                else "Cross-lane OOF ensemble did not beat the best single model"
            ),
            "reconstruction_spec": decision.reconstruction_spec,
            "diagnostics": decision.diagnostics,
        }
    except (KeyError, OSError, TypeError, ValueError) as exc:
        return {**base, "status": "SKIPPED", "reason": str(exc)}


def _prediction_table(path: Path) -> dict[str, list[str]]:
    table = pq.read_table(path).to_pydict()
    return {name: [str(value) for value in values] for name, values in table.items()}


def _register_experiment_predictions(
    registry: PredictionRegistry,
    candidates: list[tuple[str, Path, dict[str, object]]],
    metric_name: str,
    package_directory: Path,
) -> tuple[dict[str, str], bool]:
    member_lanes: dict[str, str] = {}
    seen_prediction_hashes: set[str] = set()
    used_run_artifacts = False
    for lane_id, packet_path, lane_result in candidates:
        successful = [
            result
            for result in lane_result.get("completed_results", [])
            if isinstance(result, dict) and result.get("status", "SUCCEEDED") == "SUCCEEDED"
        ]
        registered = 0
        for result in successful:
            artifacts = result.get("artifacts", [])
            validation_path = _artifact_path(artifacts, "validation_predictions.csv")
            test_path = _artifact_path(artifacts, "test_predictions.csv")
            if validation_path is None or test_path is None:
                continue
            validation_rows = _read_prediction_csv(validation_path)
            test_rows = _read_prediction_csv(test_path)
            if not validation_rows or not test_rows:
                continue
            experiment_id = str(result.get("experiment_id", f"experiment-{registered + 1}"))
            member_id = lane_id if len(successful) == 1 else f"{lane_id}--{experiment_id}"
            prediction_key = "score" if metric_name in {"roc_auc", "log_loss"} else "predicted"
            predictions = [row.get(prediction_key) or row["predicted"] for row in validation_rows]
            score = float(result.get("metrics", {}).get(metric_name, 0.0))
            registry.add_member(
                member_id=member_id,
                row_ids=[row["row"] for row in validation_rows],
                predictions=predictions,
                targets=[row["actual"] for row in validation_rows],
                model_family=str(result.get("model_family", "unknown")),
                metric_score=score,
                test_predictions=[row["predicted"] for row in test_rows],
                experiment_id=experiment_id,
            )
            prediction_hash = registry.members[member_id].prediction_hash
            if prediction_hash in seen_prediction_hashes:
                del registry.members[member_id]
                continue
            seen_prediction_hashes.add(prediction_hash)
            member_lanes[member_id] = lane_id
            used_run_artifacts = True
            _write_member_prediction_artifacts(
                package_directory,
                member_id,
                validation_rows,
                test_rows,
                prediction_key,
            )
            registered += 1

        if registered:
            continue
        oof_table = _prediction_table(packet_path / "oof_predictions.parquet")
        test_table = _prediction_table(packet_path / "test_predictions.parquet")
        registry.add_member(
            member_id=lane_id,
            row_ids=oof_table["row"],
            predictions=oof_table["prediction"],
            targets=oof_table["target"],
            metric_score=_best_score(lane_result, metric_name),
            test_predictions=test_table["prediction"],
            experiment_id=lane_id,
        )
        prediction_hash = registry.members[lane_id].prediction_hash
        if prediction_hash in seen_prediction_hashes:
            del registry.members[lane_id]
            continue
        seen_prediction_hashes.add(prediction_hash)
        member_lanes[lane_id] = lane_id
    return member_lanes, used_run_artifacts


def _artifact_path(artifacts: object, filename: str) -> Path | None:
    if not isinstance(artifacts, list):
        return None
    for artifact in artifacts:
        if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str):
            continue
        path = Path(artifact["path"])
        if path.name == filename and path.is_file():
            return path
    return None


def _read_prediction_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_member_prediction_artifacts(
    package_directory: Path,
    member_id: str,
    validation_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    prediction_key: str,
) -> None:
    member_directory = package_directory / "ensemble-members" / member_id
    member_directory.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "row": [row["row"] for row in validation_rows],
                "prediction": [
                    row.get(prediction_key) or row["predicted"] for row in validation_rows
                ],
                "target": [row["actual"] for row in validation_rows],
            }
        ),
        member_directory / "oof_predictions.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "row": [row["row"] for row in test_rows],
                "prediction": [row["predicted"] for row in test_rows],
            }
        ),
        member_directory / "test_predictions.parquet",
    )


def _weighted_average(columns: list[list[str]], weights: list[float]) -> list[float]:
    return [
        sum(float(column[index]) * weight for column, weight in zip(columns, weights, strict=True))
        for index in range(len(columns[0]))
    ]


def _blend_strategies(member_count: int) -> list[tuple[str, list[float]]]:
    equal = [1.0 / member_count] * member_count
    strategies = [("equal-weight average", equal)]
    for challenger in range(1, member_count):
        for champion_weight in (0.5, 0.65, 0.8, 0.9, 0.95):
            weights = [0.0] * member_count
            weights[0] = champion_weight
            weights[challenger] = 1.0 - champion_weight
            strategies.append(
                (
                    f"OOF-selected pair blend ({champion_weight:.2f}/{1 - champion_weight:.2f})",
                    weights,
                )
            )
    return strategies


def _majority_vote(columns: list[list[str]]) -> list[str]:
    return [
        Counter(column[index] for column in columns).most_common(1)[0][0]
        for index in range(len(columns[0]))
    ]


def _positive_label(values: list[str]) -> str:
    for value in sorted(set(values)):
        if value.casefold() in {"yes", "true", "1", "positive"}:
            return value
    return sorted(set(values))[-1]


def _metric_name(result: dict[str, object]) -> str:
    completed = result.get("completed_results", [])
    if isinstance(completed, list):
        for item in completed:
            if isinstance(item, dict) and isinstance(item.get("metrics"), dict):
                return str(next(iter(item["metrics"]), "accuracy"))
    return "accuracy"


def _best_score(result: dict[str, object], metric_name: str | None = None) -> float:
    resolved_metric = metric_name or _metric_name(result)
    metric_spec = get_metric_spec(resolved_metric)
    scores = []
    completed = result.get("completed_results", [])
    if isinstance(completed, list):
        for item in completed:
            if isinstance(item, dict) and isinstance(item.get("metrics"), dict):
                value = item["metrics"].get(resolved_metric)
                if isinstance(value, int | float):
                    scores.append(float(value))
    if scores:
        return metric_spec.best(scores)
    return float("-inf") if metric_spec.higher_is_better else float("inf")


def _score_blend(metric_name: str, targets: list[str], predictions: list[float]) -> float:
    if metric_name == "roc_auc":
        positive = _positive_label(targets)
        return float(roc_auc_score([value == positive for value in targets], predictions))
    if metric_name == "log_loss":
        positive = _positive_label(targets)
        return float(log_loss([value == positive for value in targets], predictions))
    numeric_targets = [float(value) for value in targets]
    if metric_name == "rmse":
        return float(mean_squared_error(numeric_targets, predictions) ** 0.5)
    if metric_name == "mae":
        return float(mean_absolute_error(numeric_targets, predictions))
    if metric_name == "r2":
        return float(r2_score(numeric_targets, predictions))
    raise ValueError(f"Metric '{metric_name}' does not support numeric blending")


def _score_labels(metric_name: str, targets: list[str], predictions: list[str]) -> float:
    if metric_name == "accuracy":
        return float(accuracy_score(targets, predictions))
    if metric_name == "f1_macro":
        return float(f1_score(targets, predictions, average="macro"))
    raise ValueError(f"Metric '{metric_name}' does not support label voting")


def _write_ensemble_submission(
    template_path: Path,
    contracts_path: Path,
    output_path: Path,
    predictions: list[object],
) -> None:
    contracts = json.loads(contracts_path.read_text(encoding="utf-8"))
    prediction_column = contracts["competition"]["prediction_column"]
    with template_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if len(rows) != len(predictions):
        raise ValueError("Ensemble predictions do not match submission rows")
    for row, prediction in zip(rows, predictions, strict=True):
        row[prediction_column] = str(prediction)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _knowledge_markdown(
    record: dict[str, object],
    lane_results: dict[str, dict[str, object]],
    ledger: list[dict[str, object]],
    findings: list[dict[str, object]],
    ensemble: dict[str, object],
) -> str:
    profile = record.get("profile") if isinstance(record.get("profile"), dict) else {}
    lines = [
        "# Experiment knowledge",
        "",
        f"Run: `{record['run_id']}`",
        f"Iteration: `{record['iteration_id']}`",
        f"Task: `{profile.get('task_type', 'unknown')}`",
        f"Metric: `{profile.get('metric', 'unknown')}`",
        f"Target: `{profile.get('target_column', 'unknown')}`",
        "",
        "## Lane outcomes",
        "",
        "| Lane | Behavior | Status | Budget | Executed | Best score |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    lane_configs = {
        str(item["lane_id"]): item
        for item in record.get("lanes", [])
        if isinstance(item, dict) and item.get("lane_id")
    }
    for lane_id, config in lane_configs.items():
        result = lane_results.get(lane_id, {})
        executed = len(result.get("completed_results", [])) if isinstance(result, dict) else 0
        score = _best_score(result) if result else float("-inf")
        score_text = "n/a" if score == float("-inf") else f"{score:.6f}"
        lines.append(
            f"| {lane_id} | {config.get('name', '')} | {config.get('status', 'DISABLED')} | "
            f"{config.get('max_experiments', 0)} | {executed} | {score_text} |"
        )
    lines.extend(
        [
            "",
            "## Ensemble decision",
            "",
            f"Status: `{ensemble.get('status', 'unknown')}`",
            f"Adopted: `{ensemble.get('adopted', False)}`",
            f"Reason: {ensemble.get('reason', 'No assessment available')}",
            "",
            "## Experiment evidence",
            "",
        ]
    )
    for item in ledger:
        metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
        lines.append(
            f"- `{item.get('lane_id')}/{item.get('experiment_id')}`: "
            f"{item.get('model_family')} with metrics `{json.dumps(metrics, sort_keys=True)}`."
        )
    lines.extend(["", "## Findings", ""])
    if findings:
        for item in findings:
            lines.append(
                f"- `{item.get('lane_id')}/{item.get('experiment_id')}`: "
                f"{item.get('verdict', item.get('status', 'recorded'))}. "
                f"{item.get('recommendation', '')}".rstrip()
            )
    else:
        lines.append("No structured findings were produced.")
    lines.extend(
        [
            "",
            "## Guidance for the next iteration",
            "",
            "Reuse this package only when the competition contract, target, metric, "
            "and data schema are compatible.",
            "Reuse the verified champion as the anchor baseline instead of repeating it blindly.",
            "Prioritize rejected hypotheses only when new evidence changes their "
            "mechanism or validation risk.",
            "Keep ensemble members only when aligned OOF predictions demonstrate useful diversity.",
            "",
        ]
    )
    return "\n".join(lines)


def _archive_directory(source: Path, destination: Path) -> None:
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_symlink() or stat.S_ISLNK(path.stat().st_mode):
                raise ValueError("Full packages cannot contain symbolic links")
            if path.is_file():
                archive.write(path, arcname=path.relative_to(source).as_posix())
