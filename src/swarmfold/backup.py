import json
import shutil
import sqlite3
import tarfile
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

from swarmfold.artifacts import sha256_file, write_json


def create_local_backup(
    database_path: str | Path,
    artifact_root: str | Path,
    output_path: str | Path,
) -> Path:
    database = Path(database_path).resolve()
    artifacts = Path(artifact_root).resolve()
    output = Path(output_path).resolve()
    if not database.is_file():
        raise ValueError("Coordinator SQLite database does not exist")
    if not artifacts.is_dir():
        raise ValueError("Coordinator artifact root does not exist")
    if output.exists():
        raise ValueError("Backup output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        package = Path(temporary) / "backup"
        package.mkdir()
        snapshot = package / "coordinator.db"
        source = sqlite3.connect(database)
        destination = sqlite3.connect(snapshot)
        try:
            source.backup(destination)
        finally:
            destination.close()
            source.close()
        _verify_sqlite(snapshot)
        for path in artifacts.rglob("*"):
            if path.is_symlink():
                raise ValueError("Artifact backup cannot contain symbolic links")
        packaged_artifacts = package / "artifacts"
        shutil.copytree(artifacts, packaged_artifacts)
        files = []
        for path in sorted(package.rglob("*")):
            if path.is_symlink():
                raise ValueError("Artifact backup cannot contain symbolic links")
            if path.is_file():
                files.append(
                    {
                        "name": path.relative_to(package).as_posix(),
                        "sha256": sha256_file(path),
                        "size_bytes": path.stat().st_size,
                    }
                )
        write_json(
            package / "backup-manifest.json",
            {
                "schema_version": "0.2",
                "created_at": datetime.now(UTC).isoformat(),
                "files": files,
            },
        )
        temporary_output = output.with_name(f".{output.name}.staging")
        try:
            with tarfile.open(temporary_output, "w:gz") as archive:
                for path in sorted(package.rglob("*")):
                    archive.add(path, arcname=path.relative_to(package))
            temporary_output.replace(output)
        finally:
            temporary_output.unlink(missing_ok=True)
    return output


def restore_local_backup(
    backup_path: str | Path,
    database_path: str | Path,
    artifact_root: str | Path,
) -> None:
    backup = Path(backup_path).resolve()
    database = Path(database_path).resolve()
    artifacts = Path(artifact_root).resolve()
    if database.exists() or artifacts.exists():
        raise ValueError("Restore targets must not already exist")
    with tempfile.TemporaryDirectory() as temporary:
        staging = Path(temporary).resolve()
        _extract_and_verify(backup, staging)
        restored_database = staging / "coordinator.db"
        restored_artifacts = staging / "artifacts"
        if not restored_database.is_file() or not restored_artifacts.is_dir():
            raise ValueError("Backup is missing coordinator state")
        database.parent.mkdir(parents=True, exist_ok=True)
        artifacts.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(restored_database, database)
        shutil.copytree(restored_artifacts, artifacts)


def verify_local_backup(backup_path: str | Path) -> dict[str, object]:
    backup = Path(backup_path).resolve()
    with tempfile.TemporaryDirectory() as temporary:
        staging = Path(temporary).resolve()
        manifest = _extract_and_verify(backup, staging)
    return {
        "backup_path": str(backup),
        "schema_version": manifest["schema_version"],
        "created_at": manifest["created_at"],
        "file_count": len(manifest["files"]),
        "sha256": sha256_file(backup),
    }


def apply_backup_retention(
    backup_directory: str | Path,
    *,
    keep_last: int = 7,
    max_age_days: int = 30,
    now: datetime | None = None,
) -> list[Path]:
    if keep_last < 1 or max_age_days < 1:
        raise ValueError("Backup retention values must be positive")
    directory = Path(backup_directory).resolve()
    if not directory.is_dir():
        raise ValueError("Backup directory does not exist")
    reference = now or datetime.now(UTC)
    cutoff = reference - timedelta(days=max_age_days)
    backups = sorted(
        directory.glob("*.tar.gz"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    removed = []
    for index, backup in enumerate(backups):
        modified = datetime.fromtimestamp(backup.stat().st_mtime, tz=UTC)
        if index >= keep_last and modified < cutoff:
            backup.unlink()
            removed.append(backup)
    return removed


def _extract_and_verify(backup: Path, staging: Path) -> dict[str, object]:
    if not backup.is_file():
        raise ValueError("Backup archive does not exist")
    with tarfile.open(backup, "r:gz") as archive:
        for member in archive.getmembers():
            target = (staging / member.name).resolve()
            if staging != target and staging not in target.parents:
                raise ValueError("Backup contains an unsafe path")
            if member.issym() or member.islnk():
                raise ValueError("Backup contains a link")
        archive.extractall(staging, filter="data")
    manifest_path = staging / "backup-manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Backup manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "0.2" or not isinstance(manifest.get("files"), list):
        raise ValueError("Backup manifest schema is unsupported")
    for record in manifest["files"]:
        name = record.get("name", "")
        path = (staging / name).resolve()
        if staging not in path.parents or not path.is_file():
            raise ValueError(f"Backup file is missing or unsafe: {name}")
        if path.stat().st_size != record.get("size_bytes") or sha256_file(path) != record.get(
            "sha256"
        ):
            raise ValueError(f"Backup file failed integrity verification: {name}")
    _verify_sqlite(staging / "coordinator.db")
    if not (staging / "artifacts").is_dir():
        raise ValueError("Backup is missing coordinator artifacts")
    return manifest


def _verify_sqlite(path: Path) -> None:
    if not path.is_file():
        raise ValueError("Backup is missing the coordinator database")
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        result = connection.execute("PRAGMA integrity_check").fetchone()
    finally:
        connection.close()
    if result != ("ok",):
        raise ValueError("Coordinator database failed integrity verification")
