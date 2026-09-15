import json
import os
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any

from swarmfold.competition_bundle import inspect_competition_bundle
from swarmfold.intake import CompetitionProfile, CompetitionProfiler, LLMCompetitionProfiler
from swarmfold.packet_archive import create_packet_archive

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
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="swarmfold-ui")
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
        max_training_seconds: float,
        iteration_id: str,
        lane_id: str,
    ) -> dict[str, Any]:
        record = {
            "run_id": run_id,
            "status": "QUEUED",
            "created_at": _now(),
            "updated_at": _now(),
            "dataset_name": original_filename,
            "target_column": target_column,
            "iteration_id": iteration_id,
            "lane_id": lane_id,
            "model": os.getenv("SWARMFOLD_MODEL", "deterministic mock planner"),
            "max_experiments": max_experiments,
            "max_training_seconds": max_training_seconds,
            "experiments": [],
            "packet": None,
            "error": None,
        }
        self._write_record(run_id, record)
        future = self._executor.submit(
            self._execute,
            run_id,
            dataset_path,
            target_column,
            max_experiments,
            max_training_seconds,
            iteration_id,
            lane_id,
        )
        with self._lock:
            self._futures[run_id] = future
        return record

    def create_intake(
        self,
        run_id: str,
        *,
        filenames: list[str],
        max_experiments: int,
        max_training_seconds: float,
        iteration_id: str,
        lane_id: str,
    ) -> dict[str, Any]:
        record = {
            "run_id": run_id,
            "status": "PROFILING",
            "created_at": _now(),
            "updated_at": _now(),
            "dataset_name": f"{len(filenames)} uploaded files",
            "bundle_files": filenames,
            "target_column": None,
            "iteration_id": iteration_id,
            "lane_id": lane_id,
            "model": os.getenv("SWARMFOLD_MODEL", ""),
            "max_experiments": max_experiments,
            "max_training_seconds": max_training_seconds,
            "profile": None,
            "experiments": [],
            "packet": None,
            "error": None,
        }
        self._write_record(run_id, record)
        future = self._executor.submit(self._profile, run_id)
        with self._lock:
            self._futures[run_id] = future
        return record

    def confirm(
        self, run_id: str, profile_value: dict[str, Any]
    ) -> dict[str, Any]:
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
        markdown_names = {
            path.name
            for path in bundle_directory.iterdir()
            if path.suffix.casefold() == ".md"
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
        record.update(
            status="QUEUED",
            updated_at=_now(),
            profile=profile.model_dump(),
            target_column=spec.target_column,
            metric_name=_normalize_metric(profile.metric),
        )
        self._write_record(run_id, record)
        future = self._executor.submit(
            self._execute,
            run_id,
            train_path,
            spec.target_column,
            int(record["max_experiments"]),
            float(record["max_training_seconds"]),
            str(record["iteration_id"]),
            str(record["lane_id"]),
            test_path,
            submission_path,
            project_path,
            _normalize_metric(profile.metric),
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
            "packet.zip": self.run_directory(run_id) / "packet.zip",
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

    def _execute(
        self,
        run_id: str,
        dataset_path: Path,
        target_column: str,
        max_experiments: int,
        max_training_seconds: float,
        iteration_id: str,
        lane_id: str,
        test_dataset_path: Path | None = None,
        sample_submission_path: Path | None = None,
        project_path: Path | None = None,
        metric_name: str = "accuracy",
    ) -> None:
        record = self.get(run_id)
        record.update(status="RUNNING", updated_at=_now())
        self._write_record(run_id, record)
        try:
            result = self.runner(
                dataset_path,
                target_column=target_column,
                output_dir=self.run_directory(run_id) / "output",
                tracking_uri=self.tracking_uri,
                max_experiments=max_experiments,
                max_training_seconds=max_training_seconds,
                thread_id=run_id,
                iteration_id=iteration_id,
                lane_id=lane_id,
                test_dataset_path=test_dataset_path,
                sample_submission_path=sample_submission_path,
                project_path=project_path,
                metric_name=metric_name,
            )
            packet = result.get("packet")
            if isinstance(packet, dict) and isinstance(packet.get("packet_path"), str):
                create_packet_archive(
                    Path(packet["packet_path"]), self.run_directory(run_id) / "packet.zip"
                )
            record.update(
                status=str(result.get("status", "FAILED")),
                updated_at=_now(),
                experiments=_experiment_summary(result),
                packet=packet,
                stop_reason=result.get("stop_reason"),
                resource_usage=result.get("resource_usage", {}),
                verification=result.get("verification"),
                literature_research=result.get("literature_research"),
                modeling_quality={
                    name: result.get(name)
                    for name in (
                        "problem_formulation",
                        "validation_audit",
                        "domain_decomposition",
                        "ensemble_assessment",
                        "robustness_assessment",
                    )
                },
                error=None,
            )
        except Exception as exc:
            record.update(
                status="FAILED",
                updated_at=_now(),
                error=f"{type(exc).__name__}: {exc}",
            )
        self._write_record(run_id, record)

    def _profile(self, run_id: str) -> None:
        record = self.get(run_id)
        try:
            bundle_directory = self.run_directory(run_id) / "bundle"
            files = sorted(path for path in bundle_directory.iterdir() if path.is_file())
            profile = self.profiler.profile(files)
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
        path = self.run_directory(run_id) / "status.json"
        temporary = path.with_suffix(".json.tmp")
        with self._lock:
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
    summaries = []
    for item in result.get("completed_results", []):
        if not isinstance(item, dict):
            continue
        metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
        metric_name = next(iter(metrics), "accuracy")
        summaries.append(
            {
                "experiment_id": item.get("experiment_id"),
                "model_family": item.get("model_family"),
                "status": item.get("status"),
                "accuracy": metrics.get("accuracy"),
                "metric_name": metric_name,
                "score": metrics.get(metric_name),
                "duration_seconds": item.get("duration_seconds"),
                "resource_usage": item.get("resource_usage", {}),
                "verdict": verdicts.get(str(item.get("experiment_id"))),
            }
        )
    return summaries


def _normalize_metric(value: str | None) -> str:
    normalized = (value or "accuracy").casefold().replace("-", "_").replace(" ", "_")
    aliases = {
        "accuracy": "accuracy",
        "roc_auc": "roc_auc",
        "rocauc": "roc_auc",
        "auc": "roc_auc",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported execution metric: {value}") from exc
