from __future__ import annotations

import numpy as np
import pandas as pd

from swarmfold.candidates import CandidateSpec


class MultilabelClassificationAdapter:
    task_type: str = "multilabel_classification"
    default_metric: str = "multilabel_f1"
    supported_metrics: frozenset[str] = frozenset({"multilabel_f1"})
    recommended_validation_strategy: str = "kfold"
    supported_model_families: frozenset[str] = frozenset(
        {
            "logistic_regression",
            "random_forest",
            "extra_trees",
            "hist_gradient_boosting",
            "lightgbm",
        }
    )

    def validate_target(self, target: pd.Series | pd.DataFrame | np.ndarray) -> None:
        if isinstance(target, pd.DataFrame):
            if len(target) < 10:
                raise ValueError("Target must contain at least 10 samples")
            if target.isna().any().any():
                raise ValueError("Multilabel targets must not contain missing values")
            for col in target.columns:
                unique = set(target[col].dropna().unique())
                if not unique.issubset({0, 1, "0", "1"}):
                    raise ValueError(
                        f"Multilabel target column '{col}' must contain binary indicators (0 or 1)"
                    )
        else:
            arr = np.asarray(target)
            if len(arr) < 10:
                raise ValueError("Target must contain at least 10 samples")
            unique = set(np.unique(arr))
            if not unique.issubset({0, 1}):
                raise ValueError("Multilabel target must contain binary indicators (0 or 1)")

    def validate_target_classes(self, target_classes: list[str]) -> None:
        pass

    def prediction_shape(
        self, n_samples: int, n_classes: int | None = None
    ) -> tuple[int, ...]:
        if n_classes is not None and n_classes > 1:
            return (n_samples, n_classes)
        return (n_samples,)

    def format_submission(
        self,
        id_frame: pd.DataFrame,
        predictions: np.ndarray,
        prediction_columns: list[str],
    ) -> pd.DataFrame:
        out = id_frame.copy()
        if predictions.ndim == 2 and len(prediction_columns) == predictions.shape[1]:
            for idx, col in enumerate(prediction_columns):
                out[col] = predictions[:, idx]
        else:
            col = prediction_columns[0] if prediction_columns else "prediction"
            out[col] = predictions.ravel()
        return out

    def sanity_baseline(self) -> CandidateSpec:
        return CandidateSpec(
            hypothesis="Independent multi-output linear classifier baseline",
            model_family="logistic_regression",
            feature_strategy="scaled_numeric",
            parameters={"multi_output": True, "c": 1.0},
        )

    def strong_baseline(self) -> CandidateSpec:
        return CandidateSpec(
            hypothesis="Multi-output histogram gradient boosting trees baseline",
            model_family="hist_gradient_boosting",
            feature_strategy="raw_numeric",
            parameters={"multi_output": True, "max_iter": 100, "max_depth": 5},
        )

    def compatible_ensemble_strategies(self) -> list[str]:
        return ["weighted_blend", "rank_average"]
