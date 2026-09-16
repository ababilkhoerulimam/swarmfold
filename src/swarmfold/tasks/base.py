from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from swarmfold.candidates import CandidateSpec


@runtime_checkable
class TaskAdapter(Protocol):
    """Protocol governing task-specific schemas, baselines, validation,
    and submission formatting."""

    task_type: str
    default_metric: str
    supported_metrics: frozenset[str]
    recommended_validation_strategy: str
    supported_model_families: frozenset[str]

    def validate_target(self, target: pd.Series | pd.DataFrame | np.ndarray) -> None:
        """Validate target values, shapes, missingness, and range for the task."""
        ...

    def validate_target_classes(self, target_classes: list[str]) -> None:
        """Validate discrete class labels when applicable."""
        ...

    def prediction_shape(
        self, n_samples: int, n_classes: int | None = None
    ) -> tuple[int, ...]:
        """Expected output shape for OOF and test predictions."""
        ...

    def format_submission(
        self,
        id_frame: pd.DataFrame,
        predictions: np.ndarray,
        prediction_columns: list[str],
    ) -> pd.DataFrame:
        """Format model predictions into a submission DataFrame aligned with ID columns."""
        ...

    def sanity_baseline(self) -> CandidateSpec:
        """Provide at least one trivial or fast sanity baseline."""
        ...

    def strong_baseline(self) -> CandidateSpec:
        """Provide at least one competitive strong baseline (e.g. GBDT)."""
        ...

    def compatible_ensemble_strategies(self) -> list[str]:
        """List ensemble strategies compatible with this task's predictions."""
        ...
