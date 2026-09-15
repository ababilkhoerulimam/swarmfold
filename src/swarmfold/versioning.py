from collections.abc import Callable
from copy import deepcopy

CURRENT_VERSIONS = {
    "packet_manifest": "0.2",
    "packet_completion": "0.2",
    "knowledge_snapshot": "0.2",
    "prompt": "0.2",
}

Migration = Callable[[dict[str, object]], dict[str, object]]


class MigrationError(ValueError):
    pass


def migrate_record(
    record_type: str,
    value: dict[str, object],
    target_version: str | None = None,
) -> dict[str, object]:
    if record_type not in CURRENT_VERSIONS:
        raise MigrationError(f"Unknown record type: {record_type}")
    target = target_version or CURRENT_VERSIONS[record_type]
    migrated = deepcopy(value)
    version = migrated.get("schema_version")
    if not isinstance(version, str):
        raise MigrationError("Record schema_version is missing")
    visited = set()
    while version != target:
        key = (record_type, version)
        if key in visited or key not in _MIGRATIONS:
            raise MigrationError(
                f"No migration path for {record_type} from {version} to {target}"
            )
        visited.add(key)
        migrated = _MIGRATIONS[key](migrated)
        next_version = migrated.get("schema_version")
        if not isinstance(next_version, str) or next_version == version:
            raise MigrationError(f"Migration for {record_type} {version} did not advance")
        version = next_version
    return migrated


def _packet_manifest_01_to_02(value: dict[str, object]) -> dict[str, object]:
    value["schema_version"] = "0.2"
    source_kind = "human" if str(value.get("lane_id", "")).startswith("human-") else "ai"
    value.setdefault("source_kind", source_kind)
    value.setdefault("resource_usage", {})
    value.setdefault("evidence_schema_version", "evidence-v1")
    return value


def _completion_01_to_02(value: dict[str, object]) -> dict[str, object]:
    value["schema_version"] = "0.2"
    return value


def _snapshot_01_to_02(value: dict[str, object]) -> dict[str, object]:
    value["schema_version"] = "0.2"
    value.setdefault(
        "schema_versions",
        {"packet": "0.1", "finding": "evidence-v1", "prompt": "planner-v1"},
    )
    return value


def _prompt_01_to_02(value: dict[str, object]) -> dict[str, object]:
    prompt = value.pop("prompt", None)
    if prompt is None:
        raise MigrationError("Legacy prompt record is missing prompt")
    value["schema_version"] = "0.2"
    value["template"] = prompt
    value.setdefault("input_schema", "planner-context-v1")
    value.setdefault("output_schema", "planner-suggestion-v1")
    return value


_MIGRATIONS: dict[tuple[str, str], Migration] = {
    ("packet_manifest", "0.1"): _packet_manifest_01_to_02,
    ("packet_completion", "0.1"): _completion_01_to_02,
    ("knowledge_snapshot", "0.1"): _snapshot_01_to_02,
    ("prompt", "0.1"): _prompt_01_to_02,
}
