import contextlib
import json
import os
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any

from swarmfold.competition_bundle import inspect_competition_bundle
from swarmfold.intake import (
    CompetitionProfile,
    CompetitionProfiler,
    LLMCompetitionProfiler,
    detect_unsupported_modalities,
)
from swarmfold.model_registry import model_capability_registry
from swarmfold.portfolio import (
    PortfolioConfig,
    build_full_package,
    inspect_prior_packages,
    single_lane_portfolio,
)

RunCallable = Callable[..., dict[str, Any]]


class DashboardRunManager:
    def __init__(
        self,
        root: Path,
        *,
        runner: RunCallable | None = None,
        profiler: CompetitionProfiler | None = None,
        tracking_uri: str | None = None,
    ) -> None:
        if runner is None:
            from swarmfold.graph import run_experiment

            runner = run_experiment
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.runner = runner
        self.profiler = profiler or LLMCompetitionProfiler()
        self.tracking_uri = tracking_uri or os.getenv(
            "SWARMFOLD_UI_TRACKING_URI", "http://mlflow:5000"
        )
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="swarmfold-ui")
        self._lock = Lock()
        self._futures: dict[str, Future[None]] = {}

    def run_directory(self, run_id: str) -> Path:
        if len(run_id) != 32 or any(character not in "0123456789abcdef" for character in run_id):
            raise ValueError("Invalid run ID")
        destination = (self.root / run_id).resolve()
        if self.root not in destination.parents:
            raise ValueError("Invalid run path")
        return destination

    def prepare(self, run_id: str) -> Path:
        directory = self.run_directory(run_id)
        directory.mkdir(parents=True, exist_ok=False)
        return directory

    def start(
        self,
        run_id: str,
        *,
        dataset_path: Path,
        original_filename: str,
        target_column: str,
        max_experiments: int,
        max_training_seconds: float | None,
        iteration_id: str,
        lane_id: str,
    ) -> dict[str, Any]:
        portfolio = single_lane_portfolio(lane_id, max_experiments)
        record = {
            "run_id": run_id,
            "status": "QUEUED",
            "created_at": _now(),
            "updated_at": _now(),
            "dataset_name": original_filename,
            "target_column": target_column,
            "iteration_id": iteration_id,
            "lane_id": lane_id,
            "lanes": portfolio.public_lanes(),
            "model": os.getenv("SWARMFOLD_MODEL", "deterministic mock planner"),
            "max_experiments": max_experiments,
            "max_training_seconds": max_training_seconds,
            "experiments": [],
            "packet": None,
            "error": None,
        }
        self._write_record(run_id, record)
        future = self._executor.submit(
            self._execute_portfolio,
            run_id,
            dataset_path,
            target_column,
            max_training_seconds,
            iteration_id,
            portfolio,
        )
        with self._lock:
            self._futures[run_id] = future
        return record

    def create_intake(
        self,
        run_id: str,
        *,
        filenames: list[str],
        max_training_seconds: float | None,
        iteration_id: str,
        portfolio: PortfolioConfig,
    ) -> dict[str, Any]:
        maximum = int(os.getenv("SWARMFOLD_MAX_PORTFOLIO_EXPERIMENTS", "200"))
        portfolio.enforce_total_limit(maximum)
        first_enabled = portfolio.enabled_lanes[0]
        record = {
            "run_id": run_id,
            "status": "PROFILING",
            "created_at": _now(),
            "updated_at": _now(),
            "dataset_name": f"{len(filenames)} uploaded files",
            "bundle_files": filenames,
            "target_column": None,
            "iteration_id": iteration_id,
            "lane_id": first_enabled.lane_id,
            "lanes": portfolio.public_lanes(),
            "max_concurrent_lanes": portfolio.max_concurrent_lanes,
            "model": os.getenv("SWARMFOLD_MODEL", ""),
            "max_experiments": portfolio.total_experiment_budget,
            "max_training_seconds": max_training_seconds,
            "ensemble_enabled": portfolio.ensemble_enabled,
            "max_ensemble_trials": portfolio.max_ensemble_trials,
            "agent_f": {
                "name": "Agent F",
                "role": "Portfolio Ensemble Specialist",
                "status": "WAITING_FOR_MEMBERS" if portfolio.ensemble_enabled else "DISABLED",
                "enabled": portfolio.ensemble_enabled,
                "max_ensemble_trials": portfolio.max_ensemble_trials,
            },
            "profile": None,
            "experiments": [],
            "packet": None,
            "full_package": None,
            "error": None,
        }
        self._write_record(run_id, record)
        future = self._executor.submit(self._profile, run_id)
        with self._lock:
            self._futures[run_id] = future
        return record

    def confirm(self, run_id: str, profile_value: dict[str, Any]) -> dict[str, Any]:
        record = self.get(run_id)
        if record["status"] != "AWAITING_CONFIRMATION":
            raise ValueError("Run is not awaiting profile confirmation")
        profile = CompetitionProfile.model_validate(profile_value)
        original_profile = CompetitionProfile.model_validate(record["profile"])
        original_issues = [*original_profile.blockers, *original_profile.questions]
        submitted_issues = [*profile.blockers, *profile.questions]
        required_issues = list(dict.fromkeys([*original_issues, *submitted_issues]))
        resolved_issues = {
            resolution.issue
            for resolution in profile.resolutions
            if resolution.status == "resolved" and resolution.response.strip()
        }
        unresolved = [issue for issue in required_issues if issue not in resolved_issues]
        if unresolved:
            raise ValueError(
                "Add a resolution or answer for every blocker and question before training"
            )
        profile = profile.model_copy(update={"execution_ready": True})
        available = set(record["bundle_files"])
        referenced = {
            profile.training_file,
            profile.submission_template,
            profile.brief_file,
            *profile.inference_files,
            *(item.filename for item in profile.files),
        }
        if any(name and name not in available for name in referenced):
            raise ValueError("Confirmed profile references a file outside the uploaded bundle")
        if (
            not profile.training_file
            or len(profile.inference_files) != 1
            or not profile.submission_template
            or not profile.target_column
        ):
            raise ValueError(
                "The built-in adapter needs one training file, one inference file, "
                "one submission template, and a target column"
            )
        bundle_directory = self.run_directory(run_id) / "bundle"
        bundle_files = sorted(path for path in bundle_directory.iterdir() if path.is_file())
        unsupported = detect_unsupported_modalities(bundle_files)
        if unsupported:
            names = ", ".join(
                sorted({p.name for p in bundle_files if detect_unsupported_modalities([p])})
            )
            raise ValueError(
                f"Cannot confirm run: bundle contains unsupported modality files: {names}"
            )
        markdown_names = {
            path.name for path in bundle_files if path.suffix.casefold() == ".md"
        }
        brief_role_tokens = ("brief", "rule", "readme", "documentation", "context")
        profiled_brief_names = {
            item.filename
            for item in profile.files
            if item.filename in markdown_names
            and any(token in item.role.casefold() for token in brief_role_tokens)
        }
        if profile.brief_file:
            project_candidates = [profile.brief_file]
        else:
            project_candidates = sorted(profiled_brief_names or markdown_names)
        if not project_candidates:
            raise ValueError("The uploaded bundle must include a Markdown competition brief")
        if len(project_candidates) != 1:
            raise ValueError(
                "The confirmed profile must identify exactly one Markdown competition brief"
            )
        if project_candidates[0] not in markdown_names:
            raise ValueError("The selected competition brief must be an uploaded Markdown file")
        train_path = bundle_directory / profile.training_file
        test_path = bundle_directory / profile.inference_files[0]
        submission_path = bundle_directory / profile.submission_template
        project_path = bundle_directory / project_candidates[0]
        spec = inspect_competition_bundle(
            train_path,
            test_path,
            submission_path,
            project_path,
            target_column=profile.target_column,
        )
        executable_tasks = {
            "binary_classification",
            "multiclass_classification",
            "regression",
            "multilabel_classification",
            "time_series_forecasting",
        }
        if profile.task_type not in executable_tasks:
            raise ValueError(
                f"Task '{profile.task_type}' has a declared research adapter but no "
                "end-to-end training runtime. Confirmation is blocked."
            )
        norm_metric = _normalize_metric(profile.metric, profile.task_type)
        record.update(
            status="READY_TO_TRAIN",
            updated_at=_now(),
            profile=profile.model_dump(),
            target_column=spec.target_column,
            metric_name=norm_metric,
            contracts={
                "task_type": profile.task_type,
                "target_column": spec.target_column,
                "metric": norm_metric,
                "feature_columns": spec.feature_columns,
                "prediction_column": spec.prediction_column,
                "id_columns": spec.id_columns,
                "train_rows": spec.train_rows,
                "test_rows": spec.test_rows,
            },
            execution_inputs={
                "training_file": profile.training_file,
                "inference_file": profile.inference_files[0],
                "submission_file": profile.submission_template,
                "brief_file": project_candidates[0],
            },
        )
        self._write_record(run_id, record)
        return record

    def start_training(self, run_id: str) -> dict[str, Any]:
        record = self.get(run_id)
        if record["status"] != "READY_TO_TRAIN":
            raise ValueError("Run is not ready to start training")
        portfolio = PortfolioConfig(
            lanes=[
                {
                    "lane_id": item["lane_id"],
                    "enabled": item["enabled"],
                    "max_experiments": item["max_experiments"],
                }
                for item in record["lanes"]
            ],
            max_concurrent_lanes=int(record["max_concurrent_lanes"]),
            ensemble_enabled=bool(record.get("ensemble_enabled", True)),
            max_ensemble_trials=record.get("max_ensemble_trials"),
        )
        inputs = record["execution_inputs"]
        bundle_directory = self.run_directory(run_id) / "bundle"
        record.update(status="QUEUED", updated_at=_now())
        for lane in record["lanes"]:
            if lane["enabled"]:
                lane["status"] = "QUEUED"
        self._write_record(run_id, record)
        future = self._executor.submit(
            self._execute_portfolio,
            run_id,
            bundle_directory / inputs["training_file"],
            str(record["target_column"]),
            record["max_training_seconds"],
            str(record["iteration_id"]),
            portfolio,
            bundle_directory / inputs["inference_file"],
            bundle_directory / inputs["submission_file"],
            bundle_directory / inputs["brief_file"],
            str(record["metric_name"]),
            str(record["profile"]["task_type"]),
            str(record["profile"].get("validation_strategy") or ""),
        )
        with self._lock:
            self._futures[run_id] = future
        return record

    def get(self, run_id: str) -> dict[str, Any]:
        path = self.run_directory(run_id) / "status.json"
        if not path.is_file():
            raise FileNotFoundError("Dashboard run not found")
        return json.loads(path.read_text(encoding="utf-8"))

    def list(self, limit: int = 20) -> list[dict[str, Any]]:
        records = []
        for path in self.root.glob("*/status.json"):
            try:
                records.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        records.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
        return records[:limit]

    def wait(self, run_id: str, timeout: float | None = None) -> dict[str, Any]:
        with self._lock:
            future = self._futures.get(run_id)
        if future is None:
            raise FileNotFoundError("Dashboard run future not found")
        future.result(timeout=timeout)
        return self.get(run_id)

    def artifact(self, run_id: str, artifact_name: str) -> Path:
        allowed = {
            "full-package.zip": self.run_directory(run_id) / "full-package.zip",
        }
        record = self.get(run_id)
        packet = record.get("packet")
        if isinstance(packet, dict) and isinstance(packet.get("packet_path"), str):
            packet_path = Path(packet["packet_path"]).resolve()
            for name in (
                "submission.csv",
                "decision.md",
                "experiment-ledger.jsonl",
                "artifact-manifest.json",
                "end-to-end.ipynb",
                "literature-review.json",
                "modeling-quality-evidence.json",
            ):
                allowed[name] = packet_path / name
        path = allowed.get(artifact_name)
        if path is None or not path.is_file():
            raise FileNotFoundError("Run artifact not found")
        return path

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=False)

    def _execute_portfolio(
        self,
        run_id: str,
        dataset_path: Path,
        target_column: str,
        max_training_seconds: float | None,
        iteration_id: str,
        portfolio: PortfolioConfig,
        test_dataset_path: Path | None = None,
        sample_submission_path: Path | None = None,
        project_path: Path | None = None,
        metric_name: str = "accuracy",
        task_type: str | None = None,
        validation_strategy: str | None = None,
    ) -> None:
        record = self.get(run_id)
        record.update(status="RUNNING", updated_at=_now())
        self._write_record(run_id, record)
        lane_results: dict[str, dict[str, Any]] = {}
        workers = min(portfolio.max_concurrent_lanes, len(portfolio.enabled_lanes))
        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix=f"portfolio-{run_id[:8]}"
        ) as pool:
            futures = {
                pool.submit(
                    self._execute_lane,
                    run_id,
                    lane,
                    dataset_path,
                    target_column,
                    max_training_seconds,
                    iteration_id,
                    test_dataset_path,
                    sample_submission_path,
                    project_path,
                    metric_name,
                    task_type,
                    validation_strategy,
                ): lane.lane_id
                for lane in portfolio.enabled_lanes
            }
            for future in as_completed(futures):
                lane_id = futures[future]
                try:
                    lane_results[lane_id] = future.result()
                except Exception as exc:
                    lane_results[lane_id] = {
                        "status": "FAILED",
                        "error": f"{type(exc).__name__}: {exc}",
                        "completed_results": [],
                        "critiques": [],
                        "packet": None,
                    }
                self._record_lane_result(run_id, lane_id, lane_results[lane_id])
        record = self.get(run_id)
        try:
            package = build_full_package(self.run_directory(run_id), record, lane_results)
            accepted = [
                value for value in lane_results.values() if value.get("status") == "ACCEPTED"
            ]
            terminal_status = "ACCEPTED" if accepted else "REJECTED"
            champion_lane = package.get("champion_lane")
            champion = lane_results.get(str(champion_lane), {})
            ensemble_assessment: dict[str, Any] = {}
            ensemble_file = (
                self.run_directory(run_id) / "full-package" / "ensemble-assessment.json"
            )
            if ensemble_file.is_file():
                with contextlib.suppress(Exception):
                    ensemble_assessment = json.loads(ensemble_file.read_text(encoding="utf-8"))

            agent_f: dict[str, Any] = record.get("agent_f", {})
            agent_file = self.run_directory(run_id) / "full-package" / "agent-f.json"
            if agent_file.is_file():
                with contextlib.suppress(Exception):
                    agent_f = json.loads(agent_file.read_text(encoding="utf-8"))

            val_audit: dict[str, Any] = {}
            audit_file = self.run_directory(run_id) / "full-package" / "validation-audit.json"
            if audit_file.is_file():
                with contextlib.suppress(Exception):
                    val_audit = json.loads(audit_file.read_text(encoding="utf-8"))

            tech_evidence: list[dict[str, Any]] = []
            tech_file = (
                self.run_directory(run_id) / "full-package" / "technique-evidence.json"
            )
            if tech_file.is_file():
                with contextlib.suppress(Exception):
                    tech_evidence = json.loads(
                        tech_file.read_text(encoding="utf-8")
                    ).get("lanes", [])

            reflection_text = ""
            ref_file = (
                self.run_directory(run_id) / "full-package" / "experiment-knowledge.md"
            )
            if ref_file.is_file():
                with contextlib.suppress(Exception):
                    reflection_text = ref_file.read_text(encoding="utf-8")

            prune_and_repair_events: list[dict[str, Any]] = []
            for lane_id, result in sorted(lane_results.items()):
                for item in result.get("completed_results", []):
                    if not isinstance(item, dict):
                        continue
                    if item.get("repair_history"):
                        prune_and_repair_events.append(
                            {
                                "lane_id": lane_id,
                                "event_type": "SELF_HEALING_REPAIR",
                                "experiment_id": item.get("experiment_id"),
                                "original_experiment_id": item.get("original_experiment_id"),
                                "details": item.get("repair_history"),
                            }
                        )
                    if item.get("failure_record"):
                        prune_and_repair_events.append(
                            {
                                "lane_id": lane_id,
                                "event_type": "FAILURE",
                                "experiment_id": item.get("experiment_id"),
                                "details": item.get("failure_record"),
                            }
                        )
                for critique in result.get("critiques", []):
                    if isinstance(critique, dict) and critique.get("verdict") == "PRUNED":
                        prune_and_repair_events.append(
                            {
                                "lane_id": lane_id,
                                "event_type": "PRUNED",
                                "experiment_id": critique.get("experiment_id"),
                                "details": critique.get(
                                    "reason", "Trial pruned by early stopping"
                                ),
                            }
                        )

            record.update(
                status=terminal_status,
                updated_at=_now(),
                experiments=[
                    {"lane_id": lane_id, **summary}
                    for lane_id, result in sorted(lane_results.items())
                    for summary in _experiment_summary(result)
                ],
                packet=champion.get("packet"),
                full_package=package,
                ensemble_assessment=ensemble_assessment,
                agent_f=agent_f,
                validation_audit=val_audit,
                technique_evidence=tech_evidence,
                reflection_text=reflection_text,
                prune_and_repair_events=prune_and_repair_events,
                diversity=ensemble_assessment.get("correlation_matrix", {}),
                stop_reason="portfolio completed",
                error=None,
            )
        except Exception as exc:
            record.update(
                status="FAILED",
                updated_at=_now(),
                error=f"{type(exc).__name__}: {exc}",
            )
        self._write_record(run_id, record)

    def _execute_lane(
        self,
        run_id: str,
        lane,
        dataset_path: Path,
        target_column: str,
        max_training_seconds: float | None,
        iteration_id: str,
        test_dataset_path: Path | None,
        sample_submission_path: Path | None,
        project_path: Path | None,
        metric_name: str,
        task_type: str | None,
        validation_strategy: str | None,
    ) -> dict[str, Any]:
        self._set_lane_status(run_id, lane.lane_id, "RUNNING")
        prior_knowledge = self.get(run_id).get("prior_knowledge", [])
        return self.runner(
            dataset_path,
            target_column=target_column,
            output_dir=self.run_directory(run_id) / "lanes" / lane.lane_id,
            tracking_uri=self.tracking_uri,
            max_experiments=lane.max_experiments,
            max_training_seconds=max_training_seconds,
            thread_id=f"{run_id}-{lane.lane_id}",
            iteration_id=iteration_id,
            lane_id=lane.lane_id,
            test_dataset_path=test_dataset_path,
            sample_submission_path=sample_submission_path,
            project_path=project_path,
            metric_name=metric_name,
            task_type=task_type,
            validation_strategy=validation_strategy or None,
            prior_knowledge=prior_knowledge,
        )

    def _set_lane_status(self, run_id: str, lane_id: str, status: str) -> None:
        with self._lock:
            record = self._read_record(run_id)
            for lane in record["lanes"]:
                if lane["lane_id"] == lane_id:
                    lane["status"] = status
                    break
            record["updated_at"] = _now()
            self._write_record_unlocked(run_id, record)

    def _record_lane_result(self, run_id: str, lane_id: str, result: dict[str, Any]) -> None:
        with self._lock:
            record = self._read_record(run_id)
            for lane in record["lanes"]:
                if lane["lane_id"] == lane_id:
                    lane.update(
                        status=str(result.get("status", "FAILED")),
                        executed_experiments=len(result.get("completed_results", [])),
                        experiments=_experiment_summary(result),
                        stop_reason=result.get("stop_reason"),
                        error=result.get("error"),
                    )
                    break
            record["experiments"] = [
                {"lane_id": lane["lane_id"], **item}
                for lane in record["lanes"]
                for item in lane.get("experiments", [])
            ]
            record["updated_at"] = _now()
            self._write_record_unlocked(run_id, record)

    def _profile(self, run_id: str) -> None:
        record = self.get(run_id)
        try:
            bundle_directory = self.run_directory(run_id) / "bundle"
            files = sorted(path for path in bundle_directory.iterdir() if path.is_file())
            profile = self.profiler.profile(files)
            unsupported = detect_unsupported_modalities(files)
            if unsupported:
                all_blockers = list(profile.blockers)
                for b in unsupported:
                    if b not in all_blockers:
                        all_blockers.append(b)
                profile = profile.model_copy(
                    update={"execution_ready": False, "blockers": all_blockers}
                )
            prior_knowledge = inspect_prior_packages(files)
            uploaded = set(record["bundle_files"])
            described = {item.filename for item in profile.files}
            if described != uploaded:
                missing = sorted(uploaded - described)
                unknown = sorted(described - uploaded)
                raise ValueError(
                    f"Profiler file inventory mismatch; missing={missing}, unknown={unknown}"
                )
            record.update(
                status="AWAITING_CONFIRMATION",
                updated_at=_now(),
                profile=profile.model_dump(),
                prior_knowledge=prior_knowledge,
                error=None,
            )
        except Exception as exc:
            record.update(
                status="PROFILE_FAILED",
                updated_at=_now(),
                error=f"{type(exc).__name__}: {exc}",
            )
        self._write_record(run_id, record)

    def _write_record(self, run_id: str, value: dict[str, Any]) -> None:
        with self._lock:
            self._write_record_unlocked(run_id, value)

    def _read_record(self, run_id: str) -> dict[str, Any]:
        path = self.run_directory(run_id) / "status.json"
        if not path.is_file():
            raise FileNotFoundError("Dashboard run not found")
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_record_unlocked(self, run_id: str, value: dict[str, Any]) -> None:
        path = self.run_directory(run_id) / "status.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)


def dashboard_html() -> str:
    return (Path(__file__).parent / "web" / "dashboard.html").read_text(encoding="utf-8")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _experiment_summary(result: dict[str, Any]) -> list[dict[str, Any]]:
    verdicts = {
        str(item.get("experiment_id")): item.get("verdict")
        for item in result.get("critiques", [])
        if isinstance(item, dict)
    }
    proposals = {
        str(item.get("experiment_id")): item
        for item in result.get("experiment_queue", [])
        if isinstance(item, dict)
    }
    estimators = {
        str(item["model_family"]): str(item["estimator"])
        for item in model_capability_registry()
        if "estimator" in item
    }
    summaries = []
    for item in result.get("completed_results", []):
        if not isinstance(item, dict):
            continue
        metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
        metric_name = next(iter(metrics), "accuracy")
        proposal = proposals.get(str(item.get("experiment_id")), {})
        model_family = str(item.get("model_family"))
        summaries.append(
            {
                "experiment_id": item.get("experiment_id"),
                "model_family": model_family,
                "estimator": estimators.get(model_family, model_family),
                "feature_strategy": proposal.get("feature_strategy"),
                "parameters": proposal.get("parameters", {}),
                "hypothesis": proposal.get("hypothesis"),
                "status": item.get("status"),
                "accuracy": metrics.get("accuracy"),
                "metric_name": metric_name,
                "score": metrics.get(metric_name),
                "duration_seconds": item.get("duration_seconds"),
                "resource_usage": item.get("resource_usage", {}),
                "verdict": verdicts.get(str(item.get("experiment_id"))),
                "fold_metrics": item.get("fold_metrics", []),
                "failure_record": item.get("failure_record"),
                "repair_history": item.get("repair_history", []),
                "original_experiment_id": item.get("original_experiment_id"),
                "resumed_from_checkpoint": item.get("resumed_from_checkpoint", False),
            }
        )
    return summaries


def _normalize_metric(value: str | None, task_type: str = "binary_classification") -> str:
    default_metric = "rmse" if task_type == "regression" else "accuracy"
    normalized = (value or default_metric).casefold().replace("-", "_").replace(" ", "_")
    aliases = {
        "accuracy": "accuracy",
        "roc_auc": "roc_auc",
        "rocauc": "roc_auc",
        "auc": "roc_auc",
        "log_loss": "log_loss",
        "logloss": "log_loss",
        "f1_macro": "f1_macro",
        "macro_f1": "f1_macro",
        "rmse": "rmse",
        "root_mean_squared_error": "rmse",
        "mae": "mae",
        "mean_absolute_error": "mae",
        "r2": "r2",
        "r_squared": "r2",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported execution metric: {value}") from exc
