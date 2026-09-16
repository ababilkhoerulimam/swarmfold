from __future__ import annotations

import numpy as np
import pandas as pd

from swarmfold.candidates import CandidateSpec


class RegressionAdapter:
    task_type: str = "regression"
    default_metric: str = "rmse"
    supported_metrics: frozenset[str] = frozenset({"rmse", "mae", "r2"})
    recommended_validation_strategy: str = "kfold"
    supported_model_families: frozenset[str] = frozenset(
        {
            "logistic_regression",
            "random_forest",
            "extra_trees",
            "hist_gradient_boosting",
            "lightgbm",
            "xgboost",
            "catboost",
            "soft_voting",
            "stacking",
        }
    )

    def validate_target(self, target: pd.Series | pd.DataFrame | np.ndarray) -> None:
        vals = (
            target.to_numpy()
            if isinstance(target, (pd.Series, pd.DataFrame))
            else np.asarray(target)
        )
        if len(vals) < 10:
            raise ValueError("Target must contain at least 10 samples")
        numeric = pd.to_numeric(pd.Series(vals.ravel()), errors="coerce")
        if numeric.isna().any():
            raise ValueError("Regression target must be numeric with zero missing values")

    def validate_target_classes(self, target_classes: list[str]) -> None:
        pass

    def prediction_shape(
        self, n_samples: int, n_classes: int | None = None
    ) -> tuple[int, ...]:
        return (n_samples,)

    def format_submission(
        self,
        id_frame: pd.DataFrame,
        predictions: np.ndarray,
        prediction_columns: list[str],
    ) -> pd.DataFrame:
        out = id_frame.copy()
        col = prediction_columns[0] if prediction_columns else "prediction"
        out[col] = predictions.ravel()
        return out

    def sanity_baseline(self) -> CandidateSpec:
        return CandidateSpec(
            hypothesis="Stable L2 regularized Ridge linear regression baseline",
            model_family="logistic_regression",
            feature_strategy="scaled_numeric",
            parameters={"alpha": 1.0},
        )

    def strong_baseline(self) -> CandidateSpec:
        return CandidateSpec(
            hypothesis="Nonlinear histogram gradient boosting regressor baseline",
            model_family="hist_gradient_boosting",
            feature_strategy="raw_numeric",
            parameters={"max_iter": 100, "max_depth": 5, "learning_rate": 0.1},
        )

    def compatible_ensemble_strategies(self) -> list[str]:
        return ["weighted_blend", "rank_average", "stacking"]
