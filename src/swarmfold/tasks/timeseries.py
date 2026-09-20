from __future__ import annotations

import numpy as np
import pandas as pd

from swarmfold.candidates import CandidateSpec


class TimeSeriesForecastingAdapter:
    task_type: str = "time_series_forecasting"
    default_metric: str = "rmse"
    supported_metrics: frozenset[str] = frozenset({"rmse", "mae", "mape", "r2"})
    recommended_validation_strategy: str = "purged_time_series"
    supported_model_families: frozenset[str] = frozenset(
        {
            "logistic_regression",
            "random_forest",
            "extra_trees",
            "hist_gradient_boosting",
            "lightgbm",
            "xgboost",
            "catboost",
        }
    )

    def validate_target(self, target: pd.Series | pd.DataFrame | np.ndarray) -> None:
        vals = (
            target.to_numpy()
            if isinstance(target, (pd.Series, pd.DataFrame))
            else np.asarray(target)
        )
        if len(vals) < 10:
            raise ValueError("Time series forecasting requires at least 10 observations")
        numeric = pd.to_numeric(pd.Series(vals.ravel()), errors="coerce")
        if numeric.isna().any():
            raise ValueError("Time series target must be numeric with zero missing values")

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
            hypothesis="Autoregressive regularized linear baseline",
            model_family="logistic_regression",
            feature_strategy="scaled_numeric",
            parameters={"alpha": 1.0},
        )

    def strong_baseline(self) -> CandidateSpec:
        return CandidateSpec(
            hypothesis="Temporal gradient boosting with lag interactions",
            model_family="hist_gradient_boosting",
            feature_strategy="raw_numeric",
            parameters={"max_iter": 100, "max_depth": 5, "learning_rate": 0.05},
        )

    def compatible_ensemble_strategies(self) -> list[str]:
        return ["weighted_blend", "rank_average"]
