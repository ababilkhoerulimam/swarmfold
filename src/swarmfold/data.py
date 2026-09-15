from pathlib import Path

import pandas as pd


def load_classification_csv(
    dataset_path: str | Path, target_column: str
) -> tuple[pd.DataFrame, pd.Series]:
    frame = _read_classification_csv(dataset_path, target_column)
    return frame.drop(columns=[target_column]), frame[target_column].astype(str)


def inspect_classification_csv(
    dataset_path: str | Path, target_column: str
) -> tuple[int, list[str], list[str]]:
    frame = _read_classification_csv(dataset_path, target_column)
    return (
        len(frame),
        [name for name in frame.columns if name != target_column],
        sorted(frame[target_column].astype(str).unique().tolist()),
    )


def _read_classification_csv(dataset_path: str | Path, target_column: str) -> pd.DataFrame:
    path = Path(dataset_path)
    if not path.is_file():
        raise ValueError(f"Dataset does not exist: {path}")

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
        raise ValueError("Target must contain at least two classes")
    return frame
