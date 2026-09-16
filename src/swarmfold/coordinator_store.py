import hashlib
import json
from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    Column,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    create_engine,
    delete,
    insert,
    select,
    update,
)
from sqlalchemy.engine import Engine

from swarmfold.coordinator_models import (
    IterationCreate,
    LeaderboardResultCreate,
    PodRegistration,
    SubmissionDecisionCreate,
)

metadata = MetaData()

iterations = Table(
    "iterations",
    metadata,
    Column("iteration_id", String(128), primary_key=True),
    Column("contract_sha256", String(64), nullable=False),
    Column("contract_json", Text, nullable=False),
    Column("expected_lanes_json", Text, nullable=False),
    Column("created_at", String(40), nullable=False),
)

lane_charters = Table(
    "lane_charters",
    metadata,
    Column("iteration_id", String(128), nullable=False),
    Column("lane_id", String(32), nullable=False),
    Column("charter_json", Text, nullable=False),
    UniqueConstraint("iteration_id", "lane_id"),
)

pods = Table(
    "pods",
    metadata,
    Column("pod_id", String(128), primary_key=True),
    Column("iteration_id", String(128), nullable=False),
    Column("lane_id", String(32), nullable=False),
    Column("status", String(32), nullable=False),
    Column("capabilities_json", Text, nullable=False),
    Column("registered_at", String(40), nullable=False),
    Column("last_seen_at", String(40), nullable=False),
)

packets = Table(
    "packets",
    metadata,
    Column("iteration_id", String(128), nullable=False),
    Column("lane_id", String(32), nullable=False),
    Column("packet_type", String(32), nullable=False),
    Column("packet_sha256", String(64), nullable=False),
    Column("artifact_path", Text, nullable=False),
    Column("uploaded_at", String(40), nullable=False),
    UniqueConstraint("iteration_id", "lane_id"),
)

iteration_settings = Table(
    "iteration_settings",
    metadata,
    Column("iteration_id", String(128), primary_key=True),
    Column("submission_quota", Integer, nullable=False),
    Column("knowledge_snapshot_id", String(128), nullable=True),
)

submission_decisions = Table(
    "submission_decisions",
    metadata,
    Column("iteration_id", String(128), nullable=False),
    Column("lane_id", String(32), nullable=False),
    Column("decision", String(16), nullable=False),
    Column("approver", String(128), nullable=False),
    Column("reason", Text, nullable=False),
    Column("decided_at", String(40), nullable=False),
    UniqueConstraint("iteration_id", "lane_id"),
)

leaderboard_results = Table(
    "leaderboard_results",
    metadata,
    Column("iteration_id", String(128), nullable=False),
    Column("lane_id", String(32), nullable=False),
    Column("score_json", Text, nullable=False),
    Column("rank", Integer, nullable=True),
    Column("recorded_by", String(128), nullable=False),
    Column("local_supported", Boolean, nullable=False),
    Column("notes", Text, nullable=False),
    Column("recorded_at", String(40), nullable=False),
    UniqueConstraint("iteration_id", "lane_id"),
)

leaderboard_states = Table(
    "leaderboard_states",
    metadata,
    Column("iteration_id", String(128), primary_key=True),
    Column("closed_at", String(40), nullable=False),
)

knowledge_snapshots = Table(
    "knowledge_snapshots",
    metadata,
    Column("snapshot_id", String(128), primary_key=True),
    Column("source_iteration_id", String(128), nullable=False, unique=True),
    Column("manifest_sha256", String(64), nullable=False),
    Column("artifact_path", Text, nullable=False),
    Column("created_at", String(40), nullable=False),
)

audit_events = Table(
    "audit_events",
    metadata,
    Column("sequence", Integer, primary_key=True, autoincrement=True),
    Column("event_id", String(32), nullable=False, unique=True),
    Column("request_id", String(32), nullable=False),
    Column("actor_role", String(32), nullable=False),
    Column("actor_id", String(128), nullable=True),
    Column("action", String(32), nullable=False),
    Column("resource", Text, nullable=False),
    Column("outcome", String(32), nullable=False),
    Column("status_code", Integer, nullable=False),
    Column("occurred_at", String(40), nullable=False),
    Column("previous_hash", String(64), nullable=True),
    Column("event_hash", String(64), nullable=False),
)

audit_chain_state = Table(
    "audit_chain_state",
    metadata,
    Column("chain_id", String(32), primary_key=True),
    Column("event_hash", String(64), nullable=True),
)


class StoreConflict(ValueError):
    pass


class StoreNotFound(ValueError):
    pass


class CoordinatorStore:
    def __init__(self, database_url: str) -> None:
        connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
        self.engine: Engine = create_engine(database_url, connect_args=connect_args)
        metadata.create_all(self.engine)
        with self.engine.begin() as connection:
            head = connection.execute(
                select(audit_chain_state.c.chain_id).where(audit_chain_state.c.chain_id == "global")
            ).scalar_one_or_none()
            if head is None:
                connection.execute(
                    insert(audit_chain_state).values(chain_id="global", event_hash=None)
                )

    def create_iteration(self, value: IterationCreate) -> bool:
        contract_sha256 = value.contracts.contract_sha256
        if contract_sha256 is None:
            raise ValueError("Iteration contracts must include contract_sha256")
        contract_json = _json(value.contracts.model_dump())
        charters = sorted(value.lane_charters, key=lambda charter: charter.lane_id)
        expected_lanes = [charter.lane_id for charter in charters]
        with self.engine.begin() as connection:
            if value.knowledge_snapshot_id is not None:
                snapshot = connection.execute(
                    select(knowledge_snapshots.c.snapshot_id).where(
                        knowledge_snapshots.c.snapshot_id == value.knowledge_snapshot_id
                    )
                ).scalar_one_or_none()
                if snapshot is None:
                    raise StoreNotFound("Knowledge snapshot not found")
            existing = (
                connection.execute(
                    select(iterations).where(iterations.c.iteration_id == value.iteration_id)
                )
                .mappings()
                .first()
            )
            if existing is not None:
                existing_charters = (
                    connection.execute(
                        select(lane_charters.c.charter_json)
                        .where(lane_charters.c.iteration_id == value.iteration_id)
                        .order_by(lane_charters.c.lane_id)
                    )
                    .scalars()
                    .all()
                )
                settings = (
                    connection.execute(
                        select(iteration_settings).where(
                            iteration_settings.c.iteration_id == value.iteration_id
                        )
                    )
                    .mappings()
                    .one()
                )
                same = (
                    existing["contract_sha256"] == contract_sha256
                    and existing["contract_json"] == contract_json
                    and existing["expected_lanes_json"] == _json(expected_lanes)
                    and list(existing_charters)
                    == [_json(charter.model_dump()) for charter in charters]
                    and settings["submission_quota"] == value.submission_quota
                    and settings["knowledge_snapshot_id"] == value.knowledge_snapshot_id
                )
                if not same:
                    raise StoreConflict("Iteration is immutable and already exists")
                return False
            now = _now()
            connection.execute(
                insert(iterations).values(
                    iteration_id=value.iteration_id,
                    contract_sha256=contract_sha256,
                    contract_json=contract_json,
                    expected_lanes_json=_json(expected_lanes),
                    created_at=now,
                )
            )
            connection.execute(
                insert(lane_charters),
                [
                    {
                        "iteration_id": value.iteration_id,
                        "lane_id": charter.lane_id,
                        "charter_json": _json(charter.model_dump()),
                    }
                    for charter in charters
                ],
            )
            connection.execute(
                insert(iteration_settings).values(
                    iteration_id=value.iteration_id,
                    submission_quota=value.submission_quota,
                    knowledge_snapshot_id=value.knowledge_snapshot_id,
                )
            )
        return True

    def get_iteration(self, iteration_id: str) -> dict[str, object]:
        with self.engine.connect() as connection:
            row = (
                connection.execute(
                    select(iterations).where(iterations.c.iteration_id == iteration_id)
                )
                .mappings()
                .first()
            )
            if row is None:
                raise StoreNotFound("Iteration not found")
            charter_rows = connection.execute(
                select(lane_charters.c.charter_json)
                .where(lane_charters.c.iteration_id == iteration_id)
                .order_by(lane_charters.c.lane_id)
            ).scalars()
            settings = (
                connection.execute(
                    select(iteration_settings).where(
                        iteration_settings.c.iteration_id == iteration_id
                    )
                )
                .mappings()
                .one()
            )
            return {
                "iteration_id": iteration_id,
                "contracts": json.loads(row["contract_json"]),
                "lane_charters": [json.loads(value) for value in charter_rows],
                "submission_quota": settings["submission_quota"],
                "knowledge_snapshot_id": settings["knowledge_snapshot_id"],
            }

    def register_pod(self, value: PodRegistration) -> dict[str, str]:
        iteration = self.get_iteration(value.iteration_id)
        expected_lanes = {item["lane_id"] for item in iteration["lane_charters"]}
        if value.lane_id not in expected_lanes:
            raise StoreConflict("Pod lane is not assigned in this iteration")
        now = _now()
        with self.engine.begin() as connection:
            existing = (
                connection.execute(select(pods).where(pods.c.pod_id == value.pod_id))
                .mappings()
                .first()
            )
            if existing is not None and (
                existing["iteration_id"] != value.iteration_id
                or existing["lane_id"] != value.lane_id
            ):
                raise StoreConflict("Pod ID is already registered to another lane")
            values = {
                "iteration_id": value.iteration_id,
                "lane_id": value.lane_id,
                "status": "IDLE",
                "capabilities_json": _json(value.capabilities),
                "last_seen_at": now,
            }
            if existing is None:
                connection.execute(
                    insert(pods).values(pod_id=value.pod_id, registered_at=now, **values)
                )
            else:
                connection.execute(
                    update(pods).where(pods.c.pod_id == value.pod_id).values(**values)
                )
        return {"pod_id": value.pod_id, "lane_id": value.lane_id, "last_seen_at": now}

    def heartbeat(self, pod_id: str, status: str) -> dict[str, str]:
        now = _now()
        with self.engine.begin() as connection:
            result = connection.execute(
                update(pods).where(pods.c.pod_id == pod_id).values(status=status, last_seen_at=now)
            )
            if result.rowcount != 1:
                raise StoreNotFound("Pod not registered")
        return {"pod_id": pod_id, "status": status, "last_seen_at": now}

    def pod(self, pod_id: str) -> dict[str, str]:
        with self.engine.connect() as connection:
            row = connection.execute(select(pods).where(pods.c.pod_id == pod_id)).mappings().first()
        if row is None:
            raise StoreNotFound("Pod not registered")
        return dict(row)

    def record_packet(
        self,
        *,
        iteration_id: str,
        lane_id: str,
        packet_type: str,
        packet_sha256: str,
        artifact_path: str,
    ) -> bool:
        iteration = self.get_iteration(iteration_id)
        expected_lanes = {item["lane_id"] for item in iteration["lane_charters"]}
        if lane_id not in expected_lanes:
            raise StoreConflict("Lane is not part of the iteration")
        with self.engine.begin() as connection:
            existing = (
                connection.execute(
                    select(packets).where(
                        packets.c.iteration_id == iteration_id,
                        packets.c.lane_id == lane_id,
                    )
                )
                .mappings()
                .first()
            )
            if existing is not None:
                if existing["packet_sha256"] != packet_sha256:
                    raise StoreConflict("Lane already published a different terminal packet")
                return False
            connection.execute(
                insert(packets).values(
                    iteration_id=iteration_id,
                    lane_id=lane_id,
                    packet_type=packet_type,
                    packet_sha256=packet_sha256,
                    artifact_path=artifact_path,
                    uploaded_at=_now(),
                )
            )
        return True

    def barrier(self, iteration_id: str) -> tuple[list[str], list[str]]:
        iteration = self.get_iteration(iteration_id)
        expected = sorted(item["lane_id"] for item in iteration["lane_charters"])
        with self.engine.connect() as connection:
            received = sorted(
                connection.execute(
                    select(packets.c.lane_id).where(packets.c.iteration_id == iteration_id)
                ).scalars()
            )
        return received, sorted(set(expected) - set(received))

    def packet_records(self, iteration_id: str) -> list[dict[str, object]]:
        self.get_iteration(iteration_id)
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(packets)
                .where(packets.c.iteration_id == iteration_id)
                .order_by(packets.c.lane_id)
            ).mappings()
            return [dict(row) for row in rows]

    def record_submission_decision(
        self,
        iteration_id: str,
        value: SubmissionDecisionCreate,
    ) -> bool:
        _, missing = self.barrier(iteration_id)
        if missing:
            raise StoreConflict("Submission decisions require a closed iteration barrier")
        iteration = self.get_iteration(iteration_id)
        with self.engine.begin() as connection:
            packet = (
                connection.execute(
                    select(packets).where(
                        packets.c.iteration_id == iteration_id,
                        packets.c.lane_id == value.lane_id,
                    )
                )
                .mappings()
                .first()
            )
            if packet is None or packet["packet_type"] != "candidate":
                raise StoreConflict("Only verified candidate packets can receive a decision")
            existing = (
                connection.execute(
                    select(submission_decisions).where(
                        submission_decisions.c.iteration_id == iteration_id,
                        submission_decisions.c.lane_id == value.lane_id,
                    )
                )
                .mappings()
                .first()
            )
            payload = value.model_dump()
            if existing is not None:
                if any(existing[key] != payload[key] for key in payload):
                    raise StoreConflict("Submission decision is immutable")
                return False
            if value.decision == "SUBMITTED":
                submitted_count = (
                    connection.execute(
                        select(submission_decisions.c.lane_id).where(
                            submission_decisions.c.iteration_id == iteration_id,
                            submission_decisions.c.decision == "SUBMITTED",
                        )
                    )
                    .scalars()
                    .all()
                )
                if len(submitted_count) >= int(iteration["submission_quota"]):
                    raise StoreConflict("Submission quota is exhausted")
            connection.execute(
                insert(submission_decisions).values(
                    iteration_id=iteration_id,
                    decided_at=_now(),
                    **payload,
                )
            )
        return True

    def submission_gate(self, iteration_id: str) -> dict[str, object]:
        _, missing = self.barrier(iteration_id)
        iteration = self.get_iteration(iteration_id)
        with self.engine.connect() as connection:
            candidate_lanes = sorted(
                connection.execute(
                    select(packets.c.lane_id).where(
                        packets.c.iteration_id == iteration_id,
                        packets.c.packet_type == "candidate",
                    )
                ).scalars()
            )
            rows = connection.execute(
                select(submission_decisions).where(
                    submission_decisions.c.iteration_id == iteration_id
                )
            ).mappings()
            decisions = {row["lane_id"]: row["decision"] for row in rows}
        submitted = sorted(lane for lane, decision in decisions.items() if decision == "SUBMITTED")
        withheld = sorted(lane for lane, decision in decisions.items() if decision == "WITHHELD")
        pending = sorted(set(candidate_lanes) - set(decisions))
        status = "BLOCKED" if missing else ("PENDING" if pending else "READY")
        return {
            "iteration_id": iteration_id,
            "status": status,
            "quota": iteration["submission_quota"],
            "candidate_lanes": candidate_lanes,
            "submitted_lanes": submitted,
            "withheld_lanes": withheld,
            "pending_lanes": pending,
        }

    def submission_decision_records(self, iteration_id: str) -> list[dict[str, object]]:
        self.get_iteration(iteration_id)
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(submission_decisions)
                .where(submission_decisions.c.iteration_id == iteration_id)
                .order_by(submission_decisions.c.lane_id)
            ).mappings()
            return [dict(row) for row in rows]

    def record_leaderboard_result(
        self,
        iteration_id: str,
        value: LeaderboardResultCreate,
    ) -> bool:
        gate = self.submission_gate(iteration_id)
        if gate["status"] != "READY":
            raise StoreConflict("Leaderboard intake requires a completed submission gate")
        if value.lane_id not in gate["submitted_lanes"]:
            raise StoreConflict("Leaderboard results require a submitted candidate")
        payload = value.model_dump()
        with self.engine.begin() as connection:
            existing = (
                connection.execute(
                    select(leaderboard_results).where(
                        leaderboard_results.c.iteration_id == iteration_id,
                        leaderboard_results.c.lane_id == value.lane_id,
                    )
                )
                .mappings()
                .first()
            )
            if existing is not None:
                same = (
                    json.loads(existing["score_json"]) == payload["score"]
                    and existing["rank"] == payload["rank"]
                    and existing["recorded_by"] == payload["recorded_by"]
                    and existing["local_supported"] == payload["local_supported"]
                    and existing["notes"] == payload["notes"]
                )
                if not same:
                    raise StoreConflict("Leaderboard result is immutable")
                return False
            connection.execute(
                insert(leaderboard_results).values(
                    iteration_id=iteration_id,
                    lane_id=value.lane_id,
                    score_json=_json(value.score),
                    rank=value.rank,
                    recorded_by=value.recorded_by,
                    local_supported=value.local_supported,
                    notes=value.notes,
                    recorded_at=_now(),
                )
            )
        return True

    def leaderboard_result_records(self, iteration_id: str) -> list[dict[str, object]]:
        self.get_iteration(iteration_id)
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(leaderboard_results)
                .where(leaderboard_results.c.iteration_id == iteration_id)
                .order_by(leaderboard_results.c.lane_id)
            ).mappings()
            return [{**dict(row), "score": json.loads(row["score_json"])} for row in rows]

    def close_leaderboard(self, iteration_id: str) -> bool:
        gate = self.submission_gate(iteration_id)
        if gate["status"] != "READY":
            raise StoreConflict("Leaderboard intake cannot close before submission decisions")
        with self.engine.begin() as connection:
            existing = connection.execute(
                select(leaderboard_states.c.iteration_id).where(
                    leaderboard_states.c.iteration_id == iteration_id
                )
            ).scalar_one_or_none()
            if existing is not None:
                return False
            connection.execute(
                insert(leaderboard_states).values(iteration_id=iteration_id, closed_at=_now())
            )
        return True

    def leaderboard_ready(self, iteration_id: str) -> bool:
        gate = self.submission_gate(iteration_id)
        if gate["status"] != "READY":
            return False
        with self.engine.connect() as connection:
            closed = connection.execute(
                select(leaderboard_states.c.iteration_id).where(
                    leaderboard_states.c.iteration_id == iteration_id
                )
            ).scalar_one_or_none()
        result_lanes = {row["lane_id"] for row in self.leaderboard_result_records(iteration_id)}
        return closed is not None or set(gate["submitted_lanes"]) == result_lanes

    def record_knowledge_snapshot(
        self,
        *,
        snapshot_id: str,
        source_iteration_id: str,
        manifest_sha256: str,
        artifact_path: str,
    ) -> bool:
        self.get_iteration(source_iteration_id)
        with self.engine.begin() as connection:
            existing = (
                connection.execute(
                    select(knowledge_snapshots).where(
                        (knowledge_snapshots.c.snapshot_id == snapshot_id)
                        | (knowledge_snapshots.c.source_iteration_id == source_iteration_id)
                    )
                )
                .mappings()
                .first()
            )
            if existing is not None:
                same = (
                    existing["snapshot_id"] == snapshot_id
                    and existing["source_iteration_id"] == source_iteration_id
                    and existing["manifest_sha256"] == manifest_sha256
                    and existing["artifact_path"] == artifact_path
                )
                if not same:
                    raise StoreConflict("Knowledge snapshot version or iteration is already used")
                return False
            connection.execute(
                insert(knowledge_snapshots).values(
                    snapshot_id=snapshot_id,
                    source_iteration_id=source_iteration_id,
                    manifest_sha256=manifest_sha256,
                    artifact_path=artifact_path,
                    created_at=_now(),
                )
            )
        return True

    def knowledge_snapshot(self, snapshot_id: str) -> dict[str, object]:
        with self.engine.connect() as connection:
            row = (
                connection.execute(
                    select(knowledge_snapshots).where(
                        knowledge_snapshots.c.snapshot_id == snapshot_id
                    )
                )
                .mappings()
                .first()
            )
        if row is None:
            raise StoreNotFound("Knowledge snapshot not found")
        return dict(row)

    def knowledge_snapshot_for_iteration(self, iteration_id: str) -> dict[str, object] | None:
        with self.engine.connect() as connection:
            row = (
                connection.execute(
                    select(knowledge_snapshots).where(
                        knowledge_snapshots.c.source_iteration_id == iteration_id
                    )
                )
                .mappings()
                .first()
            )
        return dict(row) if row is not None else None

    def record_audit_event(
        self,
        *,
        request_id: str,
        actor_role: str,
        actor_id: str | None,
        action: str,
        resource: str,
        status_code: int,
    ) -> dict[str, object]:
        occurred_at = _now()
        event_id = uuid4().hex
        outcome = "ALLOWED" if status_code < 400 else "DENIED"
        with self.engine.begin() as connection:
            previous_hash = connection.execute(
                select(audit_chain_state.c.event_hash)
                .where(audit_chain_state.c.chain_id == "global")
                .with_for_update()
            ).scalar_one()
            payload = {
                "event_id": event_id,
                "request_id": request_id,
                "actor_role": actor_role,
                "actor_id": actor_id,
                "action": action,
                "resource": resource,
                "outcome": outcome,
                "status_code": status_code,
                "occurred_at": occurred_at,
                "previous_hash": previous_hash,
            }
            event_hash = hashlib.sha256(_json(payload).encode("utf-8")).hexdigest()
            connection.execute(insert(audit_events).values(**payload, event_hash=event_hash))
            connection.execute(
                update(audit_chain_state)
                .where(audit_chain_state.c.chain_id == "global")
                .values(event_hash=event_hash)
            )
        return {**payload, "event_hash": event_hash}

    def audit_log(self, limit: int = 100) -> list[dict[str, object]]:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(audit_events).order_by(audit_events.c.sequence.desc()).limit(limit)
            ).mappings()
            return [dict(row) for row in rows]

    def verify_audit_chain(self) -> bool:
        with self.engine.connect() as connection:
            rows = list(
                connection.execute(
                    select(audit_events).order_by(audit_events.c.sequence)
                ).mappings()
            )
        previous_hash = None
        for row in rows:
            payload = {
                key: row[key]
                for key in (
                    "event_id",
                    "request_id",
                    "actor_role",
                    "actor_id",
                    "action",
                    "resource",
                    "outcome",
                    "status_code",
                    "occurred_at",
                    "previous_hash",
                )
            }
            expected = hashlib.sha256(_json(payload).encode("utf-8")).hexdigest()
            if row["previous_hash"] != previous_hash or row["event_hash"] != expected:
                return False
            previous_hash = row["event_hash"]
        with self.engine.connect() as connection:
            head = connection.execute(
                select(audit_chain_state.c.event_hash).where(
                    audit_chain_state.c.chain_id == "global"
                )
            ).scalar_one()
        return head == previous_hash

    def clear(self) -> None:
        with self.engine.begin() as connection:
            connection.execute(delete(audit_events))
            connection.execute(
                update(audit_chain_state)
                .where(audit_chain_state.c.chain_id == "global")
                .values(event_hash=None)
            )
            connection.execute(delete(knowledge_snapshots))
            connection.execute(delete(leaderboard_states))
            connection.execute(delete(leaderboard_results))
            connection.execute(delete(submission_decisions))
            connection.execute(delete(packets))
            connection.execute(delete(pods))
            connection.execute(delete(lane_charters))
            connection.execute(delete(iteration_settings))
            connection.execute(delete(iterations))

    def dispose(self) -> None:
        self.engine.dispose()


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _now() -> str:
    return datetime.now(UTC).isoformat()
