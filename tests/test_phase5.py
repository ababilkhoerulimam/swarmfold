import io
import json
import os
import sqlite3
import tarfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from swarmfold.backup import (
    apply_backup_retention,
    create_local_backup,
    restore_local_backup,
    verify_local_backup,
)
from swarmfold.coordinator_api import CoordinatorSettings, create_coordinator_app
from swarmfold.versioning import MigrationError, migrate_record

ADMIN_TOKEN = "phase-five-admin-token-123"
AUDITOR_TOKEN = "phase-five-auditor-token-123"
POD_TOKENS = {
    "pod-a": "phase-five-pod-a-token-123",
    "pod-b": "phase-five-pod-b-token-123",
    "pod-c": "phase-five-pod-c-token-123",
}


def test_access_audit_is_hash_chained_and_auditor_is_read_only(tmp_path: Path) -> None:
    settings = CoordinatorSettings(
        database_url=f"sqlite:///{(tmp_path / 'audit.db').as_posix()}",
        artifact_root=tmp_path / "artifacts",
        admin_token=ADMIN_TOKEN,
        pod_tokens=POD_TOKENS,
        auditor_token=AUDITOR_TOKEN,
    )
    app = create_coordinator_app(settings)
    with TestClient(app) as client:
        denied = client.get(
            "/v1/iterations/missing",
            headers=_auth("invalid-token-with-safe-length"),
        )
        assert denied.status_code == 401
        forbidden = client.get(
            "/v1/iterations/missing",
            headers=_auth(AUDITOR_TOKEN),
        )
        assert forbidden.status_code == 403
        audit = client.get("/v1/audit-events", headers=_auth(AUDITOR_TOKEN))
        assert audit.status_code == 200
        assert audit.json()["chain_valid"] is True
        events = audit.json()["events"]
        assert {event["actor_role"] for event in events} >= {"anonymous", "auditor"}
        assert all("token" not in json.dumps(event).casefold() for event in events)
    assert app.state.store.verify_audit_chain() is True
    app.state.store.dispose()


def test_backup_manifest_restore_drill_tamper_detection_and_retention(tmp_path: Path) -> None:
    database = tmp_path / "coordinator.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE evidence (value TEXT NOT NULL)")
        connection.execute("INSERT INTO evidence VALUES ('verified')")
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "packet.json").write_text('{"verified": true}\n', encoding="utf-8")
    backup_dir = tmp_path / "backups"
    backup = create_local_backup(database, artifacts, backup_dir / "current.tar.gz")

    verification = verify_local_backup(backup)
    assert verification["schema_version"] == "0.2"
    restored_database = tmp_path / "restored" / "coordinator.db"
    restored_artifacts = tmp_path / "restored" / "artifacts"
    restore_local_backup(backup, restored_database, restored_artifacts)
    with sqlite3.connect(restored_database) as connection:
        assert connection.execute("SELECT value FROM evidence").fetchone() == ("verified",)

    tampered = backup_dir / "tampered.tar.gz"
    _tamper_backup(backup, tampered)
    with pytest.raises(ValueError, match="integrity verification"):
        verify_local_backup(tampered)

    old_one = backup_dir / "old-one.tar.gz"
    old_two = backup_dir / "old-two.tar.gz"
    old_one.write_bytes(backup.read_bytes())
    old_two.write_bytes(backup.read_bytes())
    old_timestamp = (datetime.now(UTC) - timedelta(days=60)).timestamp()
    os.utime(old_one, (old_timestamp, old_timestamp))
    os.utime(old_two, (old_timestamp, old_timestamp))
    removed = apply_backup_retention(backup_dir, keep_last=1, max_age_days=30)
    assert {path.name for path in removed} == {"old-one.tar.gz", "old-two.tar.gz"}
    assert backup.is_file()


def test_version_registry_migrates_legacy_records_without_mutating_input() -> None:
    legacy_prompt = {
        "schema_version": "0.1",
        "prompt_id": "baseline-hypothesis",
        "prompt_version": "planner-v1",
        "prompt": "Generate a falsifiable hypothesis.",
    }
    migrated_prompt = migrate_record("prompt", legacy_prompt)
    assert legacy_prompt["schema_version"] == "0.1"
    assert migrated_prompt == {
        "schema_version": "0.2",
        "prompt_id": "baseline-hypothesis",
        "prompt_version": "planner-v1",
        "template": "Generate a falsifiable hypothesis.",
        "input_schema": "planner-context-v1",
        "output_schema": "planner-suggestion-v1",
    }

    legacy_manifest = {
        "schema_version": "0.1",
        "lane_id": "human-1",
    }
    migrated_manifest = migrate_record("packet_manifest", legacy_manifest)
    assert migrated_manifest["schema_version"] == "0.2"
    assert migrated_manifest["source_kind"] == "human"
    assert migrated_manifest["resource_usage"] == {}
    with pytest.raises(MigrationError, match="No migration path"):
        migrate_record("prompt", {"schema_version": "9.9"})


def test_authentication_policy_rejects_reused_role_tokens(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be distinct"):
        CoordinatorSettings(
            database_url=f"sqlite:///{(tmp_path / 'duplicate-token.db').as_posix()}",
            artifact_root=tmp_path / "artifacts",
            admin_token=ADMIN_TOKEN,
            pod_tokens={**POD_TOKENS, "pod-a": ADMIN_TOKEN},
            auditor_token=AUDITOR_TOKEN,
        )


def _tamper_backup(source: Path, destination: Path) -> None:
    with tarfile.open(source, "r:gz") as archive:
        members = archive.getmembers()
        values = {
            member.name: archive.extractfile(member).read() for member in members if member.isfile()
        }
    values["artifacts/packet.json"] = b'{"verified": false}\n'
    with tarfile.open(destination, "w:gz") as archive:
        for name, value in values.items():
            info = tarfile.TarInfo(name)
            info.size = len(value)
            archive.addfile(info, io.BytesIO(value))


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}
