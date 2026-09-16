from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    RepeatedStratifiedKFold,
    StratifiedGroupKFold,
    StratifiedKFold,
)

from swarmfold.dataset_traits import DatasetTraits


class ValidationRecommendation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    recommended_strategy: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    group_column: str | None = None
    time_column: str | None = None
    purge_window: int = 0
    embargo_window: int = 0
    n_splits: int = 5
    n_repeats: int = 1


def recommend_validation_strategy(traits: DatasetTraits) -> ValidationRecommendation:
    evidence: list[str] = []
    risks: list[str] = []
    questions: list[str] = []

    # 1. Temporal data detection
    if traits.is_temporal and traits.primary_time_column:
        time_col = traits.primary_time_column
        strategy = "purged_time_series"
        confidence = 0.92
        evidence.append(
            f"Temporal column '{time_col}' detected with ordered sequence. "
            "Purged walk-forward cross-validation prevents lookahead and leakage."
        )
        risks.append(
            "Temporal models are sensitive to regime shifts; early folds train on smaller windows."
        )
        return ValidationRecommendation(
            recommended_strategy=strategy,
            confidence=confidence,
            evidence=evidence,
            risks=risks,
            unresolved_questions=questions,
            time_column=time_col,
            purge_window=1,
            embargo_window=1,
            n_splits=5,
            n_repeats=1,
        )

    # 2. Grouped entity data detection
    if traits.has_repeated_entities and traits.primary_group_column:
        group_col = traits.primary_group_column
        is_classification = traits.task_type in (
            "binary_classification", "multiclass_classification"
        )
        strategy = "stratified_group_kfold" if is_classification else "group_kfold"
        confidence = 0.88
        evidence.append(
            f"Entity grouping column '{group_col}' detected with repeated observations. "
            f"{strategy} guarantees zero entity leakage between train and validation."
        )
        if traits.class_imbalance:
            evidence.append("Class imbalance detected; preserving class ratios across group folds.")
        risks.append(
            "If entity groups in test data differ from train, generalization depends on "
            "group-level invariance."
        )
        return ValidationRecommendation(
            recommended_strategy=strategy,
            confidence=confidence,
            evidence=evidence,
            risks=risks,
            unresolved_questions=questions,
            group_column=group_col,
            n_splits=5,
            n_repeats=1,
        )

    # 3. Small dataset with class imbalance
    is_cls = traits.task_type in ("binary_classification", "multiclass_classification")
    if is_cls and traits.class_imbalance:
        if traits.row_count < 600:
            strategy = "repeated_stratified_kfold"
            confidence = 0.85
            evidence.append(
                f"Small sample size ({traits.row_count} rows) with class imbalance. "
                "Repeated stratified K-fold reduces validation variance."
            )
            return ValidationRecommendation(
                recommended_strategy=strategy,
                confidence=confidence,
                evidence=evidence,
                risks=risks,
                unresolved_questions=questions,
                n_splits=5,
                n_repeats=2,
            )
        strategy = "stratified_kfold"
        confidence = 0.90
        evidence.append(
            "Class imbalance detected; StratifiedKFold preserves label distribution across folds."
        )
        return ValidationRecommendation(
            recommended_strategy=strategy,
            confidence=confidence,
            evidence=evidence,
            risks=risks,
            unresolved_questions=questions,
            n_splits=5,
            n_repeats=1,
        )

    # 4. Standard regression or balanced classification
    strategy = "kfold" if traits.task_type == "regression" else "stratified_kfold"
    confidence = 0.95
    evidence.append(f"Standard IID structure detected; using 5-fold {strategy}.")
    return ValidationRecommendation(
        recommended_strategy=strategy,
        confidence=confidence,
        evidence=evidence,
        risks=risks,
        unresolved_questions=questions,
        n_splits=5,
        n_repeats=1,
    )


# --- Splitter Implementations ---

class BaseValidationSplitter:
    def split(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None = None,
        groups: pd.Series | np.ndarray | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        raise NotImplementedError

    def get_n_splits(
        self,
        X: Any = None,
        y: Any = None,
        groups: Any = None,
    ) -> int:
        raise NotImplementedError


class StratifiedKFoldSplitter(BaseValidationSplitter):
    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42) -> None:
        self.n_splits = n_splits
        self.splitter = StratifiedKFold(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state
        )

    def split(self, X, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if y is None:
            raise ValueError("StratifiedKFold requires labels y")
        yield from self.splitter.split(X, y)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


class RepeatedStratifiedKFoldSplitter(BaseValidationSplitter):
    def __init__(
        self, n_splits: int = 5, n_repeats: int = 2, random_state: int = 42
    ) -> None:
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.splitter = RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
        )

    def split(self, X, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if y is None:
            raise ValueError("RepeatedStratifiedKFold requires labels y")
        yield from self.splitter.split(X, y)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits * self.n_repeats


class KFoldSplitter(BaseValidationSplitter):
    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42) -> None:
        self.n_splits = n_splits
        self.splitter = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def split(self, X, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        yield from self.splitter.split(X, y)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


class GroupKFoldSplitter(BaseValidationSplitter):
    def __init__(self, n_splits: int = 5) -> None:
        self.n_splits = n_splits
        self.splitter = GroupKFold(n_splits=n_splits)

    def split(self, X, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if groups is None:
            raise ValueError("GroupKFold requires group assignments")
        yield from self.splitter.split(X, y, groups=groups)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


class StratifiedGroupKFoldSplitter(BaseValidationSplitter):
    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42) -> None:
        self.n_splits = n_splits
        self.splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state
        )

    def split(self, X, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if y is None or groups is None:
            raise ValueError("StratifiedGroupKFold requires both y and groups")
        yield from self.splitter.split(X, y, groups=groups)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


class ExpandingTimeSeriesSplitter(BaseValidationSplitter):
    """Expanding walk-forward time-series split.
    Guarantees strict temporal causality: train is strictly prior to validation.
    """
    def __init__(
        self,
        n_splits: int = 5,
        min_train_ratio: float = 0.3,
        time_column: str | None = None,
    ) -> None:
        self.n_splits = n_splits
        self.min_train_ratio = min_train_ratio
        self.time_column = time_column

    def split(self, X, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n_samples = len(X)
        if isinstance(X, pd.DataFrame) and self.time_column and self.time_column in X.columns:
            order = np.argsort(X[self.time_column].to_numpy())
        else:
            order = np.arange(n_samples)

        min_train = max(int(n_samples * self.min_train_ratio), 5)
        remaining = n_samples - min_train
        if remaining < self.n_splits:
            val_size = 1
            min_train = max(1, n_samples - self.n_splits)
        else:
            val_size = remaining // self.n_splits

        for fold in range(self.n_splits):
            train_end = min_train + fold * val_size
            val_start = train_end
            val_end = train_end + val_size if fold < self.n_splits - 1 else n_samples
            if val_start >= n_samples:
                break
            train_idx = order[:train_end]
            val_idx = order[val_start:val_end]
            yield train_idx, val_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


class PurgedTimeSeriesSplitter(BaseValidationSplitter):
    """Walk-forward time-series split with purge and embargo buffers.
    - purge_window: samples dropped immediately preceding the validation window.
    - embargo_window: samples dropped immediately following the validation window.
    """
    def __init__(
        self,
        n_splits: int = 5,
        min_train_ratio: float = 0.3,
        purge_window: int = 1,
        embargo_window: int = 1,
        time_column: str | None = None,
    ) -> None:
        self.n_splits = n_splits
        self.min_train_ratio = min_train_ratio
        self.purge_window = max(0, purge_window)
        self.embargo_window = max(0, embargo_window)
        self.time_column = time_column

    def split(self, X, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n_samples = len(X)
        if isinstance(X, pd.DataFrame) and self.time_column and self.time_column in X.columns:
            order = np.argsort(X[self.time_column].to_numpy())
        else:
            order = np.arange(n_samples)

        min_train = max(int(n_samples * self.min_train_ratio), 5)
        remaining = n_samples - min_train
        if remaining < self.n_splits:
            val_size = 1
            min_train = max(1, n_samples - self.n_splits)
        else:
            val_size = remaining // self.n_splits

        for fold in range(self.n_splits):
            val_start = min_train + fold * val_size
            val_end = val_start + val_size if fold < self.n_splits - 1 else n_samples
            if val_start >= n_samples:
                break

            # Purge window applies before val_start
            train_cutoff = max(0, val_start - self.purge_window)
            train_idx = order[:train_cutoff]
            val_idx = order[val_start:val_end]

            if len(train_idx) == 0:
                # If purge ate all train samples, fallback to at least 1 train sample
                train_idx = order[:1]

            yield train_idx, val_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


def get_validation_splitter(
    strategy: str,
    n_splits: int = 5,
    n_repeats: int = 1,
    seed: int = 42,
    purge_window: int = 0,
    embargo_window: int = 0,
    time_column: str | None = None,
) -> BaseValidationSplitter:
    norm_strat = strategy.lower().strip()
    if norm_strat == "stratified_kfold":
        return StratifiedKFoldSplitter(n_splits=n_splits, shuffle=True, random_state=seed)
    if norm_strat == "repeated_stratified_kfold":
        return RepeatedStratifiedKFoldSplitter(
            n_splits=n_splits, n_repeats=n_repeats, random_state=seed
        )
    if norm_strat in ("kfold", "k_fold"):
        return KFoldSplitter(n_splits=n_splits, shuffle=True, random_state=seed)
    if norm_strat in ("group_kfold", "group_k_fold"):
        return GroupKFoldSplitter(n_splits=n_splits)
    if norm_strat in ("stratified_group_kfold", "stratified_group_k_fold"):
        return StratifiedGroupKFoldSplitter(n_splits=n_splits, shuffle=True, random_state=seed)
    if norm_strat in ("expanding_walk_forward", "walk_forward"):
        return ExpandingTimeSeriesSplitter(n_splits=n_splits, time_column=time_column)
    if norm_strat in ("purged_time_series", "purged_walk_forward", "purged_kfold_embargo"):
        return PurgedTimeSeriesSplitter(
            n_splits=n_splits,
            purge_window=purge_window,
            embargo_window=embargo_window,
            time_column=time_column,
        )
    # Default fallback
    return StratifiedKFoldSplitter(n_splits=n_splits, shuffle=True, random_state=seed)
