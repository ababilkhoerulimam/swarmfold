import hashlib
import hmac
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

import httpx
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from swarmfold.artifacts import sha256_file
from swarmfold.contracts import contract_bundle_sha256
from swarmfold.coordinator_models import (
    BarrierStatus,
    IterationCreate,
    KnowledgeMergeRequest,
    KnowledgeSnapshotReceipt,
    LeaderboardResultCreate,
    PacketReceipt,
    PodHeartbeat,
    PodRegistration,
    SubmissionDecisionCreate,
    SubmissionGateStatus,
)
from swarmfold.coordinator_store import CoordinatorStore, StoreConflict, StoreNotFound
from swarmfold.dashboard import DashboardRunManager
from swarmfold.effectiveness import write_effectiveness_report
from swarmfold.intake import CompetitionProfile
from swarmfold.knowledge import merge_knowledge, read_knowledge_snapshot
from swarmfold.packet_archive import MAX_ARCHIVE_BYTES, extract_and_validate_archive

_BEARER = HTTPBearer(auto_error=False)
_MAX_UI_FILE_BYTES = 512 * 1024 * 1024
_MAX_UI_BUNDLE_BYTES = 2 * 1024 * 1024 * 1024
_MAX_UI_FILES = 200


@dataclass(frozen=True)
class CoordinatorSettings:
    database_url: str
    artifact_root: Path
    admin_token: str
    pod_tokens: dict[str, str]
    auditor_token: str | None = None

    @classmethod
    def from_env(cls) -> "CoordinatorSettings":
        raw_pod_tokens = os.getenv("SWARMFOLD_POD_TOKENS")
        if raw_pod_tokens is None:
            raise ValueError("SWARMFOLD_POD_TOKENS is required")
        parsed = json.loads(raw_pod_tokens)
        if not isinstance(parsed, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in parsed.items()
        ):
            raise ValueError("SWARMFOLD_POD_TOKENS must be a JSON object of string tokens")
        admin_token = os.getenv("SWARMFOLD_COORDINATOR_TOKEN")
        if admin_token is None:
            raise ValueError("SWARMFOLD_COORDINATOR_TOKEN is required")
        return cls(
            database_url=os.getenv(
                "SWARMFOLD_DATABASE_URL", "sqlite:///./.swarmfold/coordinator.db"
            ),
            artifact_root=Path(
                os.getenv("SWARMFOLD_ARTIFACT_ROOT", ".swarmfold/coordinator-artifacts")
            ).resolve(),
            admin_token=admin_token,
            pod_tokens=parsed,
            auditor_token=os.getenv("SWARMFOLD_AUDITOR_TOKEN"),
        )

    def __post_init__(self) -> None:
        if len(self.admin_token) < 16:
            raise ValueError("Coordinator token must contain at least 16 characters")
        if len(self.pod_tokens) != 3 or any(len(token) < 16 for token in self.pod_tokens.values()):
            raise ValueError("Exactly three pod tokens of at least 16 characters are required")
        if self.auditor_token is not None and len(self.auditor_token) < 16:
            raise ValueError("Auditor token must contain at least 16 characters")
        tokens = [self.admin_token, *self.pod_tokens.values()]
        if self.auditor_token is not None:
            tokens.append(self.auditor_token)
        if len(set(tokens)) != len(tokens):
            raise ValueError("Coordinator, auditor, and pod tokens must be distinct")


@dataclass(frozen=True)
class Principal:
    role: Literal["admin", "pod", "auditor"]
    pod_id: str | None = None


def create_coordinator_app(
    settings: CoordinatorSettings,
    run_manager: DashboardRunManager | None = None,
) -> FastAPI:
    settings.artifact_root.mkdir(parents=True, exist_ok=True)
    app = FastAPI(title="SwarmFold Coordinator", version="0.1.0")
    app.state.settings = settings
    app.state.store = CoordinatorStore(settings.database_url)
    app.state.run_manager = run_manager or DashboardRunManager(
        settings.artifact_root / "ui-runs"
    )
    app.router.add_event_handler("shutdown", app.state.run_manager.close)

    @app.middleware("http")
    async def audit_access(request: Request, call_next):
        request_id = uuid4().hex
        request.state.audit_request_id = request_id
        request.state.principal = None
        try:
            response = await call_next(request)
        except Exception:
            if request.url.path.startswith("/v1/"):
                _record_access(app, request, request_id, 500)
            raise
        if request.url.path.startswith("/v1/"):
            _record_access(app, request, request_id, response.status_code)
        response.headers["X-Request-ID"] = request_id
        return response

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def dashboard() -> HTMLResponse:
        page = Path(__file__).with_name("web") / "dashboard.html"
        return HTMLResponse(
            page.read_text(encoding="utf-8"),
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/v1/ui/config")
    def dashboard_config(
        _: Annotated[Principal, Depends(require_admin)],
    ) -> dict[str, object]:
        return {
            "planner": os.getenv("SWARMFOLD_PLANNER", "mock"),
            "model": os.getenv("SWARMFOLD_MODEL", "not configured"),
            "base_url": os.getenv("OPENAI_BASE_URL")
            or os.getenv("OPENAI_API_BASE")
            or "https://api.openai.com/v1",
            "api_key_configured": bool(os.getenv("OPENAI_API_KEY")),
        }

    @app.post("/v1/ui/provider-check")
    def dashboard_provider_check(
        _: Annotated[Principal, Depends(require_admin)],
    ) -> dict[str, object]:
        base_url = (
            os.getenv("OPENAI_BASE_URL")
            or os.getenv("OPENAI_API_BASE")
            or "https://api.openai.com/v1"
        ).rstrip("/")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=409, detail="Provider API key is not configured")
        try:
            with httpx.Client(timeout=15.0, trust_env=False) as client:
                response = client.get(
                    f"{base_url}/models",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Provider connection failed: {type(exc).__name__}",
            ) from exc
        models = sorted(
            str(item["id"])
            for item in payload.get("data", [])
            if isinstance(item, dict) and item.get("id")
        )
        configured = os.getenv("SWARMFOLD_MODEL", "")
        selected = configured.partition(":")[2] or configured
        return {
            "model_count": len(models),
            "selected_model": selected,
            "selected_model_available": selected in models,
        }

    @app.post("/v1/ui/runs", status_code=status.HTTP_202_ACCEPTED)
    async def create_dashboard_run(
        _: Annotated[Principal, Depends(require_admin)],
        files: Annotated[list[UploadFile], File()],
        max_experiments: Annotated[int, Form(ge=1, le=20)] = 4,
        max_training_seconds: Annotated[float, Form(gt=0, le=86400)] = 120.0,
        iteration_id: Annotated[str, Form(min_length=1, max_length=128)] = "local-iteration",
        lane_id: Annotated[Literal["ai-a", "ai-b", "ai-c"], Form()] = "ai-a",
    ) -> dict[str, object]:
        if not files or len(files) > _MAX_UI_FILES:
            raise HTTPException(status_code=422, detail="Upload between 1 and 200 files")
        filenames = [Path(upload.filename or "").name for upload in files]
        if any(not name or name in {".", ".."} for name in filenames):
            raise HTTPException(status_code=422, detail="Every uploaded file needs a safe name")
        folded_names = [name.casefold() for name in filenames]
        if len(folded_names) != len(set(folded_names)):
            raise HTTPException(status_code=422, detail="Uploaded filenames must be unique")
        if not any(Path(name).suffix.casefold() == ".md" for name in filenames):
            raise HTTPException(
                status_code=422,
                detail="Bundle must include a Markdown competition brief",
            )
        manager = _run_manager(app)
        run_id = uuid4().hex
        directory = manager.prepare(run_id)
        bundle_directory = directory / "bundle"
        bundle_directory.mkdir()
        total_size = 0
        try:
            for upload, filename in zip(files, filenames, strict=True):
                file_size = 0
                with (bundle_directory / filename).open("wb") as handle:
                    while chunk := await upload.read(1024 * 1024):
                        file_size += len(chunk)
                        total_size += len(chunk)
                        if file_size > _MAX_UI_FILE_BYTES:
                            raise HTTPException(
                                status_code=413,
                                detail=f"{filename} exceeds the 512 MB per-file limit",
                            )
                        if total_size > _MAX_UI_BUNDLE_BYTES:
                            raise HTTPException(
                                status_code=413,
                                detail="Competition bundle exceeds the 2 GB limit",
                            )
                        handle.write(chunk)
            return manager.create_intake(
                run_id,
                filenames=filenames,
                max_experiments=max_experiments,
                max_training_seconds=max_training_seconds,
                iteration_id=iteration_id,
                lane_id=lane_id,
            )
        except HTTPException:
            shutil.rmtree(directory, ignore_errors=True)
            raise
        except (OSError, ValueError) as exc:
            shutil.rmtree(directory, ignore_errors=True)
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/ui/runs/{run_id}/confirm", status_code=status.HTTP_202_ACCEPTED)
    def confirm_dashboard_profile(
        run_id: str,
        profile: CompetitionProfile,
        _: Annotated[Principal, Depends(require_admin)],
    ) -> dict[str, object]:
        try:
            return _run_manager(app).confirm(run_id, profile.model_dump())
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Dashboard run not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/v1/ui/runs")
    def dashboard_runs(
        _: Annotated[Principal, Depends(require_admin)],
    ) -> list[dict[str, object]]:
        return _run_manager(app).list()

    @app.get("/v1/ui/runs/{run_id}")
    def dashboard_run(
        run_id: str,
        _: Annotated[Principal, Depends(require_admin)],
    ) -> dict[str, object]:
        try:
            return _run_manager(app).get(run_id)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="Dashboard run not found") from exc

    @app.get("/v1/ui/runs/{run_id}/artifacts/{artifact_name}")
    def dashboard_artifact(
        run_id: str,
        artifact_name: str,
        _: Annotated[Principal, Depends(require_admin)],
    ) -> FileResponse:
        try:
            path = _run_manager(app).artifact(run_id, artifact_name)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="Run artifact not found") from exc
        return FileResponse(path, filename=artifact_name)

    @app.post("/v1/iterations", status_code=status.HTTP_201_CREATED)
    def create_iteration(
        value: IterationCreate,
        _: Annotated[Principal, Depends(require_admin)],
    ) -> dict[str, object]:
        _validate_contract_hash(value)
        created = _store(app).create_iteration(value)
        return {"iteration_id": value.iteration_id, "created": created}

    @app.get("/v1/iterations/{iteration_id}")
    def get_iteration(
        iteration_id: str,
        _: Annotated[Principal, Depends(require_operational_reader)],
    ) -> dict[str, object]:
        store = _store(app)
        result = store.get_iteration(iteration_id)
        snapshot_id = result.get("knowledge_snapshot_id")
        if isinstance(snapshot_id, str):
            snapshot = store.knowledge_snapshot(snapshot_id)
            result["knowledge_snapshot"] = read_knowledge_snapshot(
                str(snapshot["artifact_path"])
            )
        else:
            result["knowledge_snapshot"] = None
        return result

    @app.post("/v1/pods/register")
    def register_pod(
        value: PodRegistration,
        principal: Annotated[Principal, Depends(require_pod)],
    ) -> dict[str, str]:
        if principal.pod_id != value.pod_id:
            raise HTTPException(status_code=403, detail="Pod token cannot register another pod")
        return _store(app).register_pod(value)

    @app.post("/v1/pods/{pod_id}/heartbeat")
    def heartbeat(
        pod_id: str,
        value: PodHeartbeat,
        principal: Annotated[Principal, Depends(require_pod)],
    ) -> dict[str, str]:
        if principal.pod_id != pod_id:
            raise HTTPException(status_code=403, detail="Pod token cannot update another pod")
        return _store(app).heartbeat(pod_id, value.status)

    @app.post("/v1/iterations/{iteration_id}/lanes/{lane_id}/packet")
    async def upload_packet(
        iteration_id: str,
        lane_id: str,
        packet_sha256: Annotated[str, Form(pattern=r"^[a-f0-9]{64}$")],
        packet: Annotated[UploadFile, File()],
        principal: Annotated[Principal, Depends(require_pod)],
    ) -> PacketReceipt:
        pod = _store(app).pod(_required_pod_id(principal))
        if pod["iteration_id"] != iteration_id or pod["lane_id"] != lane_id:
            raise HTTPException(
                status_code=403, detail="Pod is not assigned to this iteration lane"
            )
        receipt = await _receive_packet(
            app,
            iteration_id=iteration_id,
            lane_id=lane_id,
            packet_sha256=packet_sha256,
            upload=packet,
        )
        _store(app).heartbeat(_required_pod_id(principal), "PACKET_READY")
        return receipt

    @app.post("/v1/iterations/{iteration_id}/human-lanes/{lane_id}/packet")
    async def upload_human_packet(
        iteration_id: str,
        lane_id: str,
        packet_sha256: Annotated[str, Form(pattern=r"^[a-f0-9]{64}$")],
        packet: Annotated[UploadFile, File()],
        _: Annotated[Principal, Depends(require_admin)],
    ) -> PacketReceipt:
        if lane_id not in {"human-1", "human-2", "human-3"}:
            raise HTTPException(status_code=422, detail="Human packet requires a human lane")
        return await _receive_packet(
            app,
            iteration_id=iteration_id,
            lane_id=lane_id,
            packet_sha256=packet_sha256,
            upload=packet,
        )

    @app.get("/v1/iterations/{iteration_id}/barrier")
    def barrier(
        iteration_id: str,
        _: Annotated[Principal, Depends(require_operational_reader)],
    ) -> BarrierStatus:
        received, missing = _store(app).barrier(iteration_id)
        return BarrierStatus(
            iteration_id=iteration_id,
            status="READY" if not missing else "PENDING",
            received_lanes=received,
            missing_lanes=missing,
        )

    @app.get("/v1/iterations/{iteration_id}/submission-gate")
    def submission_gate(
        iteration_id: str,
        _: Annotated[Principal, Depends(require_operational_reader)],
    ) -> SubmissionGateStatus:
        return SubmissionGateStatus.model_validate(_store(app).submission_gate(iteration_id))

    @app.post("/v1/iterations/{iteration_id}/submission-decisions")
    def decide_submission(
        iteration_id: str,
        value: SubmissionDecisionCreate,
        _: Annotated[Principal, Depends(require_admin)],
    ) -> dict[str, object]:
        created = _store(app).record_submission_decision(iteration_id, value)
        return {"iteration_id": iteration_id, "lane_id": value.lane_id, "created": created}

    @app.post("/v1/iterations/{iteration_id}/leaderboard-results")
    def record_leaderboard_result(
        iteration_id: str,
        value: LeaderboardResultCreate,
        _: Annotated[Principal, Depends(require_admin)],
    ) -> dict[str, object]:
        created = _store(app).record_leaderboard_result(iteration_id, value)
        return {"iteration_id": iteration_id, "lane_id": value.lane_id, "created": created}

    @app.post("/v1/iterations/{iteration_id}/leaderboard/close")
    def close_leaderboard(
        iteration_id: str,
        _: Annotated[Principal, Depends(require_admin)],
    ) -> dict[str, object]:
        closed = _store(app).close_leaderboard(iteration_id)
        return {"iteration_id": iteration_id, "closed": closed}

    @app.post("/v1/iterations/{iteration_id}/knowledge-snapshots")
    def publish_knowledge_snapshot(
        iteration_id: str,
        value: KnowledgeMergeRequest,
        _: Annotated[Principal, Depends(require_admin)],
    ) -> KnowledgeSnapshotReceipt:
        store = _store(app)
        if not store.leaderboard_ready(iteration_id):
            raise StoreConflict(
                "Knowledge merge requires complete leaderboard results or a closed intake"
            )
        existing = store.knowledge_snapshot_for_iteration(iteration_id)
        if existing is not None and existing["snapshot_id"] != value.snapshot_id:
            raise StoreConflict("Iteration already published a different knowledge snapshot")
        result = merge_knowledge(
            iteration=store.get_iteration(iteration_id),
            packet_records=store.packet_records(iteration_id),
            decisions=store.submission_decision_records(iteration_id),
            leaderboard_results=store.leaderboard_result_records(iteration_id),
            snapshot_id=value.snapshot_id,
            output_root=app.state.settings.artifact_root / "knowledge",
        )
        created = store.record_knowledge_snapshot(
            snapshot_id=value.snapshot_id,
            source_iteration_id=iteration_id,
            manifest_sha256=str(result["manifest_sha256"]),
            artifact_path=str(result["artifact_path"]),
        )
        return KnowledgeSnapshotReceipt(
            snapshot_id=value.snapshot_id,
            source_iteration_id=iteration_id,
            manifest_sha256=str(result["manifest_sha256"]),
            artifact_path=str(result["artifact_path"]),
            idempotent_replay=not created,
        )

    @app.get("/v1/knowledge-snapshots/{snapshot_id}")
    def get_knowledge_snapshot(
        snapshot_id: str,
        _: Annotated[Principal, Depends(require_operational_reader)],
    ) -> dict[str, object]:
        record = _store(app).knowledge_snapshot(snapshot_id)
        return read_knowledge_snapshot(str(record["artifact_path"]))

    @app.get("/v1/audit-events")
    def audit_events(
        _: Annotated[Principal, Depends(require_auditor)],
        limit: int = 100,
    ) -> dict[str, object]:
        if not 1 <= limit <= 1000:
            raise HTTPException(status_code=422, detail="limit must be between 1 and 1000")
        store = _store(app)
        return {"chain_valid": store.verify_audit_chain(), "events": store.audit_log(limit)}

    @app.post("/v1/iterations/{iteration_id}/effectiveness")
    def create_effectiveness_report(
        iteration_id: str,
        _: Annotated[Principal, Depends(require_admin)],
    ) -> dict[str, object]:
        store = _store(app)
        snapshot = store.knowledge_snapshot_for_iteration(iteration_id)
        if snapshot is None:
            raise StoreConflict("Effectiveness evaluation requires a knowledge snapshot")
        path = write_effectiveness_report(
            iteration_id=iteration_id,
            packet_records=store.packet_records(iteration_id),
            snapshot_path=str(snapshot["artifact_path"]),
            output_path=(
                app.state.settings.artifact_root / "effectiveness" / f"{iteration_id}.json"
            ),
        )
        return json.loads(path.read_text(encoding="utf-8"))

    @app.get("/v1/iterations/{iteration_id}/effectiveness")
    def get_effectiveness_report(
        iteration_id: str,
        _: Annotated[Principal, Depends(require_auditor)],
    ) -> dict[str, object]:
        path = app.state.settings.artifact_root / "effectiveness" / f"{iteration_id}.json"
        if not path.is_file():
            raise StoreNotFound("Effectiveness report not found")
        return json.loads(path.read_text(encoding="utf-8"))

    @app.exception_handler(StoreNotFound)
    async def not_found(_: Request, exc: StoreNotFound):
        return _error_response(404, str(exc))

    @app.exception_handler(StoreConflict)
    async def conflict(_: Request, exc: StoreConflict):
        return _error_response(409, str(exc))

    return app


def authenticate(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_BEARER)],
) -> Principal:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Bearer token required")
    settings: CoordinatorSettings = request.app.state.settings
    supplied = _token_hash(credentials.credentials)
    if hmac.compare_digest(supplied, _token_hash(settings.admin_token)):
        principal = Principal(role="admin")
        request.state.principal = principal
        return principal
    if settings.auditor_token and hmac.compare_digest(
        supplied, _token_hash(settings.auditor_token)
    ):
        principal = Principal(role="auditor")
        request.state.principal = principal
        return principal
    for pod_id, token in settings.pod_tokens.items():
        if hmac.compare_digest(supplied, _token_hash(token)):
            principal = Principal(role="pod", pod_id=pod_id)
            request.state.principal = principal
            return principal
    raise HTTPException(status_code=401, detail="Invalid bearer token")


def require_admin(principal: Annotated[Principal, Depends(authenticate)]) -> Principal:
    if principal.role != "admin":
        raise HTTPException(status_code=403, detail="Coordinator role required")
    return principal


def require_pod(principal: Annotated[Principal, Depends(authenticate)]) -> Principal:
    if principal.role != "pod":
        raise HTTPException(status_code=403, detail="Pod role required")
    return principal


def require_auditor(principal: Annotated[Principal, Depends(authenticate)]) -> Principal:
    if principal.role not in {"admin", "auditor"}:
        raise HTTPException(status_code=403, detail="Audit reader role required")
    return principal


def require_operational_reader(
    principal: Annotated[Principal, Depends(authenticate)],
) -> Principal:
    if principal.role not in {"admin", "pod"}:
        raise HTTPException(status_code=403, detail="Operational reader role required")
    return principal


async def _receive_packet(
    app: FastAPI,
    *,
    iteration_id: str,
    lane_id: str,
    packet_sha256: str,
    upload: UploadFile,
) -> PacketReceipt:
    settings: CoordinatorSettings = app.state.settings
    incoming_root = settings.artifact_root / ".incoming"
    incoming_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=incoming_root) as temporary:
        temporary_path = Path(temporary)
        archive_path = temporary_path / "packet.zip"
        size = 0
        with archive_path.open("wb") as handle:
            while chunk := await upload.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_ARCHIVE_BYTES:
                    raise HTTPException(
                        status_code=413, detail="Packet archive exceeds upload limit"
                    )
                handle.write(chunk)
        if sha256_file(archive_path) != packet_sha256:
            raise HTTPException(status_code=422, detail="Packet archive hash mismatch")
        extracted = temporary_path / "extracted"
        try:
            packet_result = extract_and_validate_archive(archive_path, extracted)
        except (OSError, ValueError) as exc:
            raise HTTPException(
                status_code=422, detail=f"Packet verification failed: {exc}"
            ) from exc
        if packet_result.iteration_id != iteration_id or packet_result.lane_id != lane_id:
            raise HTTPException(
                status_code=422, detail="Packet identity does not match upload lane"
            )
        destination = settings.artifact_root / "iterations" / iteration_id / lane_id / packet_sha256
        created_artifact = not destination.exists()
        if created_artifact:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(extracted / "packet"), destination)
        try:
            created_record = _store(app).record_packet(
                iteration_id=iteration_id,
                lane_id=lane_id,
                packet_type=packet_result.packet_type,
                packet_sha256=packet_sha256,
                artifact_path=str(destination.resolve()),
            )
        except Exception:
            if created_artifact:
                shutil.rmtree(destination)
            raise
    return PacketReceipt(
        iteration_id=iteration_id,
        lane_id=lane_id,
        packet_type=packet_result.packet_type,
        packet_sha256=packet_sha256,
        artifact_path=str(destination.resolve()),
        idempotent_replay=not created_record,
    )


def _validate_contract_hash(value: IterationCreate) -> None:
    expected = contract_bundle_sha256(value.contracts)
    if value.contracts.contract_sha256 != expected:
        raise HTTPException(status_code=422, detail="Contract bundle hash is invalid")


def _required_pod_id(principal: Principal) -> str:
    if principal.pod_id is None:
        raise HTTPException(status_code=403, detail="Pod identity required")
    return principal.pod_id


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _store(app: FastAPI) -> CoordinatorStore:
    return app.state.store


def _run_manager(app: FastAPI) -> DashboardRunManager:
    return app.state.run_manager


def _error_response(status_code: int, detail: str):
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=status_code, content={"detail": detail})


def _record_access(app: FastAPI, request: Request, request_id: str, status_code: int) -> None:
    principal = getattr(request.state, "principal", None)
    actor_role = principal.role if isinstance(principal, Principal) else "anonymous"
    actor_id = principal.pod_id if isinstance(principal, Principal) else None
    _store(app).record_audit_event(
        request_id=request_id,
        actor_role=actor_role,
        actor_id=actor_id,
        action=request.method,
        resource=request.url.path,
        status_code=status_code,
    )
