from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

# Sensitive token keys that should not contain hardcoded secret values
_SECRET_KEY_PATTERNS = [
    re.compile(r'"(?:api[_-]?key|secret|token|password)":\s*"([^"]+)"', re.IGNORECASE),
    re.compile(r'(?:api[_-]?key|secret|token|password)\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE),
    re.compile(r"\b(sk-[A-Za-z0-9_-]{16,})\b"),
]
# Acceptable non-secret sentinel or placeholder values
_BENIGN_TOKEN_VALUES = {
    "",
    "null",
    "none",
    "mock",
    "mock-key",
    "placeholder",
    "local-iteration",
    "ai-a",
    "ai-b",
    "ai-c",
    "ai-d",
    "ai-e",
}

# Required critical artifacts in a valid portfolio package
_REQUIRED_PACKAGE_FILES = [
    "manifest.json",
    "submission.csv",
    "end-to-end.ipynb",
    "experiment-ledger.jsonl",
    "findings.jsonl",
    "experiment-knowledge.md",
    "ensemble-assessment.json",
]


def _sha256_bytes(data: bytes) -> str:
    hasher = hashlib.sha256()
    hasher.update(data)
    return hasher.hexdigest()


def verify_portfolio_package(package_path: str | Path) -> dict[str, Any]:
    """Verify integrity, file presence, relative path safety, and secret absence in a package.

    Args:
        package_path: Path to full-package.zip or an uncompressed package directory.

    Returns:
        Structured dictionary reporting validation results.
    """
    path = Path(package_path)
    if not path.exists():
        raise FileNotFoundError(f"Package path does not exist: {path}")

    errors: list[str] = []
    missing_files: list[str] = []
    checksum_mismatches: list[dict[str, str]] = []
    secret_violations: list[str] = []
    portability_violations: list[str] = []

    # Read package files and byte contents
    file_bytes: dict[str, bytes] = {}
    duplicate_entries: list[str] = []
    if path.is_file() and path.suffix.casefold() == ".zip":
        with zipfile.ZipFile(path, "r") as archive:
            for item in archive.infolist():
                if not item.is_dir():
                    norm_name = item.filename.replace("\\", "/")
                    if norm_name in file_bytes:
                        duplicate_entries.append(norm_name)
                    file_bytes[norm_name] = archive.read(item.filename)
    elif path.is_dir():
        for file in path.rglob("*"):
            if file.is_file():
                rel = file.relative_to(path).as_posix()
                file_bytes[rel] = file.read_bytes()
    else:
        raise ValueError(f"Package must be a .zip file or directory: {path}")

    # 1. Manifest presence and schema validation
    if "manifest.json" not in file_bytes:
        errors.append("Missing required manifest.json in package")
        return {
            "valid": False,
            "file_count": len(file_bytes),
            "errors": errors,
            "missing_files": ["manifest.json"],
            "checksum_mismatches": [],
            "secret_violations": [],
            "portability_violations": [],
        }

    try:
        manifest = json.loads(file_bytes["manifest.json"].decode("utf-8"))
    except Exception as exc:
        errors.append(f"Invalid manifest.json: {exc}")
        return {
            "valid": False,
            "file_count": len(file_bytes),
            "errors": errors,
            "missing_files": [],
            "checksum_mismatches": [],
            "secret_violations": [],
            "portability_violations": [],
        }

    # 2. Check required files
    for req in _REQUIRED_PACKAGE_FILES:
        if req not in file_bytes:
            missing_files.append(req)
            errors.append(f"Required artifact '{req}' is missing from package")

    # 3. Check checksums and manifest file entries
    manifest_files = {entry["path"]: entry for entry in manifest.get("files", [])}
    for duplicate in duplicate_entries:
        errors.append(f"Duplicate ZIP entry: {duplicate}")
    for file_rel, content in file_bytes.items():
        if file_rel == "manifest.json":
            continue
        entry = manifest_files.get(file_rel)
        if entry is None:
            errors.append(f"File is missing from manifest: {file_rel}")
            continue
        actual_sha = _sha256_bytes(content)
        expected_sha = entry.get("sha256")
        if actual_sha != expected_sha:
            mismatch = {
                "file": file_rel,
                "expected": expected_sha,
                "actual": actual_sha,
            }
            checksum_mismatches.append(mismatch)
            errors.append(f"Checksum mismatch for {file_rel}")
    for manifest_path in manifest_files:
        if manifest_path not in file_bytes:
            errors.append(f"Manifest references missing file: {manifest_path}")

    # 4. Portability check: ensure no absolute paths or Windows drive letters
    for file_rel in file_bytes:
        parts = PurePosixPath(file_rel).parts
        if (
            file_rel.startswith("/")
            or file_rel.startswith("\\")
            or (len(file_rel) > 1 and file_rel[1] == ":")
            or ".." in parts
        ):
            portability_violations.append(file_rel)
            errors.append(f"Non-portable absolute path in package: {file_rel}")

    textual_suffixes = (
        ".json",
        ".jsonl",
        ".md",
        ".csv",
        ".txt",
        ".yaml",
        ".yml",
        ".toml",
        ".ipynb",
        ".env",
    )
    abs_pattern = re.compile(
        r'([A-Za-z]:[\\/][^"\n\r]+|/(?:Users|home|tmp|var)/[^"\n\r]+)'
    )
    for name, content in file_bytes.items():
        if not name.casefold().endswith(textual_suffixes):
            continue
        text = content.decode("utf-8", errors="ignore")
        for match in abs_pattern.finditer(text):
            path_found = match.group(1)
            portability_violations.append(path_found)
            errors.append(f"Non-portable absolute path found in {name}: {path_found}")

    # 5. Security check: scan for leaked credentials
    for name, content in file_bytes.items():
        if name.casefold().endswith(textual_suffixes):
            text = content.decode("utf-8", errors="ignore")
            for pattern in _SECRET_KEY_PATTERNS:
                for match in pattern.finditer(text):
                    val = match.group(1).strip()
                    if val.casefold() not in _BENIGN_TOKEN_VALUES and len(val) >= 16:
                        violation = f"Possible secret found in {name}: '{match.group(0)[:40]}...'"
                        secret_violations.append(violation)
                        errors.append(violation)

    is_valid = len(errors) == 0
    return {
        "valid": is_valid,
        "file_count": len(file_bytes),
        "missing_files": missing_files,
        "checksum_mismatches": checksum_mismatches,
        "secret_violations": secret_violations,
        "portability_violations": portability_violations,
        "errors": errors,
    }
