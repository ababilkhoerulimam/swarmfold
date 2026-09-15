import csv
import json
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Literal, Protocol

import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field

from swarmfold.config import PlannerSettings


class FileUnderstanding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename: str
    role: str = Field(min_length=2)
    reasoning: str = Field(min_length=3)


class IssueResolution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    issue: str = Field(min_length=3)
    response: str = Field(min_length=3)
    status: Literal["resolved"] = "resolved"


class CompetitionProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=10)
    task_type: str = Field(min_length=2)
    metric: str | None = None
    training_file: str | None = None
    inference_files: list[str] = Field(default_factory=list)
    submission_template: str | None = None
    brief_file: str | None = None
    target_column: str | None = None
    prediction_columns: list[str] = Field(default_factory=list)
    id_columns: list[str] = Field(default_factory=list)
    validation_strategy: str = Field(min_length=3)
    files: list[FileUnderstanding]
    execution_ready: bool
    blockers: list[str] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)
    resolutions: list[IssueResolution] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class CompetitionProfiler(Protocol):
    def profile(self, files: list[Path]) -> CompetitionProfile: ...


class LLMCompetitionProfiler:
    def __init__(
        self,
        model_name: str | None = None,
        model_factory: Callable[[str], object] | None = None,
    ) -> None:
        settings = PlannerSettings.from_values()
        self.model_name = model_name or settings.model
        self.model_factory = model_factory

    def profile(self, files: list[Path]) -> CompetitionProfile:
        if not self.model_name:
            raise RuntimeError("Competition profiling requires SWARMFOLD_MODEL")
        inventory = [_inspect_file(path) for path in files]
        prompt = _profile_prompt(inventory)
        if self.model_factory is None:
            from swarmfold.planner import _init_chat_model, _validate_provider_credentials

            _validate_provider_credentials(self.model_name)
            factory = _init_chat_model
        else:
            factory = self.model_factory
        try:
            model = factory(self.model_name)
            structured = model.with_structured_output(CompetitionProfile)  # type: ignore[attr-defined]
            response = structured.invoke(prompt)  # type: ignore[attr-defined]
        except Exception as exc:
            raise RuntimeError(
                f"Competition profiler failed with {type(exc).__name__}; check the model and bundle"
            ) from exc
        return CompetitionProfile.model_validate(response)


def _inspect_file(path: Path) -> dict[str, object]:
    try:
        return _inspect_file_unchecked(path)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
        return {
            "filename": path.name,
            "format": path.suffix.casefold().lstrip(".") or "unknown",
            "size_bytes": path.stat().st_size,
            "inspection_error": f"{type(exc).__name__}: {exc}",
        }


def _inspect_file_unchecked(path: Path) -> dict[str, object]:
    suffix = path.suffix.casefold()
    base: dict[str, object] = {
        "filename": path.name,
        "format": suffix.lstrip(".") or "unknown",
        "size_bytes": path.stat().st_size,
    }
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            rows = []
            row_count = 0
            for row in reader:
                row_count += 1
                if len(rows) < 5:
                    rows.append(dict(row))
        base.update(columns=list(reader.fieldnames or []), rows=row_count, sample=rows)
    elif suffix == ".parquet":
        table = pq.read_table(path)
        base.update(
            columns=table.column_names,
            rows=table.num_rows,
            schema=str(table.schema),
            sample=table.slice(0, 5).to_pylist(),
        )
    elif suffix in {".md", ".txt"}:
        base["content"] = path.read_text(encoding="utf-8")[:30_000]
    elif suffix == ".json":
        base["sample"] = json.loads(path.read_text(encoding="utf-8")[:200_000])
    elif suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            base["sample"] = [
                json.loads(line) for _, line in zip(range(5), handle, strict=False)
            ]
    elif suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            base["members"] = [
                {"name": item.filename, "size_bytes": item.file_size}
                for item in archive.infolist()[:200]
                if not item.is_dir()
            ]
    else:
        base["inspection_note"] = "Binary format recorded but not structurally inspected"
    return base


def _profile_prompt(inventory: list[dict[str, object]]) -> str:
    return (
        "You are the intake analyst for a data-science competition system. Infer the role of "
        "each uploaded file from its name, schema, samples, and any brief or README. The files "
        "may use arbitrary names and may represent tabular, text, image, time-series, ranking, "
        "or another task. Never force a train/test assumption. Distinguish facts from inference. "
        "Set execution_ready=false whenever an essential role, target, metric, submission shape, "
        "or rule is ambiguous, and put one concrete question per ambiguity in questions. File "
        "fields must exactly match uploaded filenames or be null. Set brief_file to the one "
        "Markdown file that should govern the run; ask a question if it cannot be distinguished. "
        "Do not claim an experiment ran. "
        "The current built-in execution adapter supports binary classification with mixed numeric "
        "and categorical CSV features, string or numeric class labels, accuracy or ROC-AUC, one "
        "labeled training file, one inference file, and one submission template; "
        "other scenarios may be understood but must list the missing adapter as a blocker.\n\n"
        f"Uploaded bundle inventory:\n{json.dumps(inventory, ensure_ascii=False, default=str)}"
    )
