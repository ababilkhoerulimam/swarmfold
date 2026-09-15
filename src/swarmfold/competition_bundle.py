import csv
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CompetitionBundleSpec:
    target_column: str
    feature_columns: list[str]
    submission_columns: list[str]
    prediction_column: str
    id_columns: list[str]
    train_rows: int
    test_rows: int
    project_context: str


def inspect_competition_bundle(
    train_path: str | Path,
    test_path: str | Path,
    sample_submission_path: str | Path,
    project_path: str | Path,
    *,
    target_column: str | None = None,
) -> CompetitionBundleSpec:
    train_headers, train_rows = _read_csv(Path(train_path), "training data")
    test_headers, test_rows = _read_csv(Path(test_path), "inference data")
    submission_headers, submission_rows = _read_csv(
        Path(sample_submission_path), "submission template"
    )
    project_context = Path(project_path).read_text(encoding="utf-8").strip()
    if not project_context:
        raise ValueError("project.md must contain the competition brief")
    if len(project_context) > 100_000:
        raise ValueError("project.md exceeds the 100 KB limit")
    if len(train_rows) < 10:
        raise ValueError("train.csv must contain at least 10 rows")
    if not test_rows:
        raise ValueError("test.csv must contain at least one row")
    if len(submission_rows) != len(test_rows):
        raise ValueError("sample_submission.csv row count must match test.csv")

    target_candidates = [name for name in train_headers if name not in test_headers]
    declared_target = target_column or _declared_target(project_context)
    if declared_target:
        if declared_target not in train_headers or declared_target in test_headers:
            raise ValueError(
                f"project.md declares target '{declared_target}', but it is not train-only"
            )
        target_column = declared_target
    elif len(target_candidates) == 1:
        target_column = target_candidates[0]
    else:
        joined = ", ".join(target_candidates) or "none"
        raise ValueError(
            "Cannot infer one target column from train/test schemas "
            f"(train-only columns: {joined}). Add `target_column: NAME` to project.md"
        )

    feature_columns = [name for name in train_headers if name != target_column]
    if feature_columns != test_headers:
        raise ValueError(
            "test.csv columns must exactly match train.csv columns excluding the target"
        )
    id_columns = [name for name in submission_headers if name in test_headers]
    prediction_columns = [name for name in submission_headers if name not in id_columns]
    if len(prediction_columns) != 1:
        raise ValueError(
            "This release requires exactly one prediction column in sample_submission.csv"
        )
    prediction_column = prediction_columns[0]
    if not id_columns:
        raise ValueError(
            "sample_submission.csv must share at least one ID column with test.csv"
        )

    targets = [row[target_column] for row in train_rows]
    if any(not value for value in targets) or len(set(targets)) < 2:
        raise ValueError("The inferred target must contain at least two classes")
    for row_number, (test_row, submission_row) in enumerate(
        zip(test_rows, submission_rows, strict=True), start=2
    ):
        for column in id_columns:
            if test_row[column] != submission_row[column]:
                raise ValueError(
                    f"sample_submission.csv ID '{column}' differs from test.csv at row {row_number}"
                )

    return CompetitionBundleSpec(
        target_column=target_column,
        feature_columns=feature_columns,
        submission_columns=submission_headers,
        prediction_column=prediction_column,
        id_columns=id_columns,
        train_rows=len(train_rows),
        test_rows=len(test_rows),
        project_context=project_context,
    )


def load_test_features(path: str | Path, feature_columns: list[str]):
    import pandas as pd

    frame = pd.read_csv(Path(path))
    if list(frame.columns) != feature_columns:
        raise ValueError("test.csv feature schema changed after bundle validation")
    return frame


def _read_csv(path: Path, label: str) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise ValueError(f"{label} does not exist")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        headers = list(reader.fieldnames or [])
        if not headers:
            raise ValueError(f"{label} has no header")
        if len(headers) != len(set(headers)):
            raise ValueError(f"{label} contains duplicate column names")
        return headers, list(reader)


def _declared_target(project_context: str) -> str | None:
    match = re.search(
        r"(?im)^\s*(?:target|target_column)\s*:\s*[`\"']?([^`\"'\s]+)",
        project_context,
    )
    return match.group(1).strip() if match else None
