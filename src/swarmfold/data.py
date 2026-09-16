from pathlib import Path

import pandas as pd


def infer_task_type_dataset(dataset_path: str | Path, target_column: str) -> str:
    frame = _read_training_dataset(dataset_path, target_column)
    target = frame[target_column]
    unique_count = int(target.nunique(dropna=True))
    if unique_count == 2:
        return "binary_classification"
    if pd.api.types.is_numeric_dtype(target):
        return "regression"
    return "multiclass_classification"


infer_task_type_csv = infer_task_type_dataset


def load_training_dataset(
    dataset_path: str | Path,
    target_column: str,
    task_type: str,
) -> tuple[pd.DataFrame, pd.Series]:
    frame = _read_training_dataset(dataset_path, target_column)
    target = frame[target_column]
    if task_type in {"regression", "time_series_forecasting"}:
        numeric_target = pd.to_numeric(target, errors="coerce")
        if numeric_target.isna().any():
            raise ValueError(f"{task_type} target must be numeric")
        return frame.drop(columns=[target_column]), numeric_target.astype(float)
    if task_type == "multilabel_classification":
        numeric_target = pd.to_numeric(target, errors="coerce")
        if not numeric_target.isna().any():
            return frame.drop(columns=[target_column]), numeric_target.astype(int)
        return frame.drop(columns=[target_column]), target.astype(str)
    return frame.drop(columns=[target_column]), target.astype(str)


load_training_data = load_training_dataset
load_training_csv = load_training_dataset


def inspect_training_dataset(
    dataset_path: str | Path,
    target_column: str,
    task_type: str,
) -> tuple[int, list[str], list[str]]:
    frame = _read_training_dataset(dataset_path, target_column)
    target = frame[target_column]
    if task_type in {"regression", "time_series_forecasting"}:
        numeric_target = pd.to_numeric(target, errors="coerce")
        if numeric_target.isna().any():
            raise ValueError(f"{task_type} target must be numeric")
        target_values: list[str] = []
    else:
        target_values = sorted(target.astype(str).unique().tolist())
    return (
        len(frame),
        [name for name in frame.columns if name != target_column],
        target_values,
    )


inspect_training_data = inspect_training_dataset
inspect_training_csv = inspect_training_dataset


def load_classification_csv(
    dataset_path: str | Path, target_column: str
) -> tuple[pd.DataFrame, pd.Series]:
    return load_training_dataset(dataset_path, target_column, "binary_classification")


def inspect_classification_csv(
    dataset_path: str | Path, target_column: str
) -> tuple[int, list[str], list[str]]:
    return inspect_training_dataset(dataset_path, target_column, "binary_classification")


def _read_classification_csv(dataset_path: str | Path, target_column: str) -> pd.DataFrame:
    return _read_training_dataset(dataset_path, target_column)


def _read_training_dataset(dataset_path: str | Path, target_column: str) -> pd.DataFrame:
    path = Path(dataset_path)
    if not path.is_file():
        raise ValueError(f"Dataset does not exist: {path}")

    suffix = path.suffix.casefold()
    if suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
    elif suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        frame = pd.read_csv(path, delimiter=delimiter)
    else:
        frame = pd.read_csv(path)

    if target_column not in frame.columns:
        raise ValueError(f"Target column '{target_column}' is missing")
    if len(frame.columns) < 2:
        raise ValueError("Dataset must contain at least one feature column")
    if len(frame) < 10:
        raise ValueError("Dataset must contain at least 10 rows")
    if frame[target_column].isna().any():
        raise ValueError("Target must not contain missing values")
    if frame[target_column].nunique() < 2:
        raise ValueError("Target must contain at least two distinct values")
    return frame


_read_training_csv = _read_training_dataset
