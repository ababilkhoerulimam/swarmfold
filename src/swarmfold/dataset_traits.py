from __future__ import annotations

import re

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

_TIME_NAME_PATTERNS = {
    "date",
    "datetime",
    "timestamp",
    "time",
    "month",
    "week",
    "year",
    "period",
    "day",
    "epoch",
}

_GROUP_NAME_PATTERNS = (
    "id",
    "user",
    "client",
    "customer",
    "patient",
    "subject",
    "device",
    "store",
    "entity",
    "group",
    "account",
    "hospital",
    "site",
    "batch",
    "cluster",
)


class DatasetTraits(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    row_count: int = Field(ge=0)
    feature_count: int = Field(ge=0)
    task_type: str
    time_columns: list[str] = Field(default_factory=list)
    primary_time_column: str | None = None
    group_columns: list[str] = Field(default_factory=list)
    primary_group_column: str | None = None
    has_repeated_entities: bool = False
    class_imbalance: bool = False
    rare_classes: list[str] = Field(default_factory=list)
    is_temporal: bool = False
    train_test_overlap: bool = False
    duplicate_feature_rows: int = 0
    conflicting_duplicate_groups: int = 0
    class_distribution: dict[str, int] = Field(default_factory=dict)
    summary_traits: list[str] = Field(default_factory=list)


def extract_dataset_traits(
    frame: pd.DataFrame,
    target_column: str,
    task_type: str = "binary_classification",
    id_columns: list[str] | None = None,
    test_frame: pd.DataFrame | None = None,
) -> DatasetTraits:
    row_count = len(frame)
    feature_cols = [c for c in frame.columns if c != target_column]
    feature_count = len(feature_cols)

    features = frame[feature_cols] if feature_cols else pd.DataFrame(index=frame.index)
    target = frame[target_column] if target_column in frame.columns else None

    # 1. Detect time and ordering columns
    time_cols: list[str] = []
    for col in feature_cols:
        norm_name = re.sub(r"[\s_-]+", " ", col.lower().strip())
        tokens = set(norm_name.split())
        is_named_time = bool(tokens.intersection(_TIME_NAME_PATTERNS))

        s = features[col]
        is_dt_dtype = pd.api.types.is_datetime64_any_dtype(s)
        if is_dt_dtype:
            time_cols.append(col)
            continue

        if is_named_time and pd.api.types.is_numeric_dtype(s):
            non_null = s.dropna()
            if len(non_null) > 5 and (non_null.is_monotonic_increasing or non_null.nunique() > 10):
                time_cols.append(col)
                continue

        if is_named_time and s.dtype == object:
            sample = s.dropna().head(100)
            if len(sample) > 0:
                parsed = pd.to_datetime(sample, errors="coerce")
                if parsed.notna().mean() >= 0.8:
                    time_cols.append(col)

    primary_time = time_cols[0] if time_cols else None
    is_temporal = bool(primary_time is not None)

    # 2. Detect entity/group identifier columns
    group_cols: list[str] = []
    ignored_ids = set(id_columns or [])
    for col in feature_cols:
        if col in time_cols or col in ignored_ids:
            continue
        norm_col = col.lower().strip()
        is_named_group = any(
            norm_col == p
            or norm_col.endswith(f"_{p}")
            or norm_col.endswith(f"{p}_id")
            or norm_col.startswith(f"{p}_")
            for p in _GROUP_NAME_PATTERNS
        )
        s = features[col]
        n_unique = s.nunique(dropna=True)
        if n_unique > 1 and n_unique < row_count:
            counts = s.value_counts()
            has_repeats = bool((counts > 1).any())
            if has_repeats and is_named_group and n_unique >= 3:
                group_cols.append(col)

    primary_group = group_cols[0] if group_cols else None
    has_repeated_entities = bool(primary_group is not None)

    # 3. Class imbalance and rare classes
    class_imbalance = False
    rare_classes: list[str] = []
    class_distribution: dict[str, int] = {}
    if target is not None and task_type in ("binary_classification", "multiclass_classification"):
        counts = target.astype(str).value_counts().to_dict()
        class_distribution = {str(k): int(v) for k, v in counts.items()}
        if class_distribution:
            min_c = min(class_distribution.values())
            max_c = max(class_distribution.values())
            total = sum(class_distribution.values())
            if total > 0 and (min_c / max_c < 0.25 or min_c / total < 0.15):
                class_imbalance = True
            for cls_name, cnt in class_distribution.items():
                if cnt < 10:
                    rare_classes.append(cls_name)

    # 4. Duplicate feature rows and conflicting targets
    duplicate_rows = 0
    conflicting_groups = 0
    if feature_count > 0:
        duplicate_rows = int(features.duplicated(keep=False).sum())
        if duplicate_rows > 0 and target is not None:
            joined = features.copy()
            joined["__target__"] = target.astype(str)
            conflicting_groups = int(
                (joined.groupby(feature_cols, dropna=False)["__target__"].nunique() > 1).sum()
            )

    # 5. Train-test overlap
    train_test_overlap = False
    if (
        test_frame is not None
        and feature_count > 0
        and primary_group
        and primary_group in test_frame.columns
    ):
        train_groups = set(features[primary_group].dropna().unique())
        test_groups = set(test_frame[primary_group].dropna().unique())
        if train_groups.intersection(test_groups):
            train_test_overlap = True

    # 6. Build summary trait tags
    summary_traits: list[str] = ["tabular"]
    if is_temporal:
        summary_traits.extend(["temporal_ordering", "time_series"])
    if has_repeated_entities:
        summary_traits.extend(["grouped_data", "entity_identifiers", "repeated_observations"])
    if class_imbalance:
        summary_traits.append("class_imbalance")
    if rare_classes:
        summary_traits.append("rare_classes")
    if duplicate_rows > 0:
        summary_traits.append("duplicate_rows")

    return DatasetTraits(
        row_count=row_count,
        feature_count=feature_count,
        task_type=task_type,
        time_columns=time_cols,
        primary_time_column=primary_time,
        group_columns=group_cols,
        primary_group_column=primary_group,
        has_repeated_entities=has_repeated_entities,
        class_imbalance=class_imbalance,
        rare_classes=rare_classes,
        is_temporal=is_temporal,
        train_test_overlap=train_test_overlap,
        duplicate_feature_rows=duplicate_rows,
        conflicting_duplicate_groups=conflicting_groups,
        class_distribution=class_distribution,
        summary_traits=summary_traits,
    )
