from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold


class StackingEnsemble:
    """Cross-fitted level-2 model over aligned base-model predictions."""

    def __init__(
        self,
        task_type: str = "binary_classification",
        seed: int = 42,
        cross_fit_splits: int = 5,
    ) -> None:
        self.task_type = task_type
        self.seed = seed
        self.cross_fit_splits = cross_fit_splits
        if task_type in {"regression", "time_series_forecasting"}:
            self.meta_model: Ridge | LogisticRegression = Ridge(alpha=1.0)
        else:
            self.meta_model = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
        self.feature_count_: int | None = None
        self.pass_through_count_: int = 0
        self.meta_fold_ids_: np.ndarray | None = None

    @staticmethod
    def _prediction_features(predictions: Sequence[np.ndarray]) -> np.ndarray:
        if not predictions:
            raise ValueError("At least one base prediction array is required")
        blocks = []
        row_count: int | None = None
        for prediction in predictions:
            array = np.asarray(prediction, dtype=float)
            if array.ndim == 1:
                array = array.reshape(-1, 1)
            if array.ndim != 2:
                raise ValueError("Base predictions must be one- or two-dimensional")
            if row_count is None:
                row_count = len(array)
            elif len(array) != row_count:
                raise ValueError("Base prediction row counts do not match")
            blocks.append(array)
        return np.hstack(blocks)

    def _build_meta_features(
        self,
        predictions: Sequence[np.ndarray],
        pass_through_features: np.ndarray | None = None,
    ) -> np.ndarray:
        prediction_features = self._prediction_features(predictions)
        if pass_through_features is None:
            return prediction_features
        pass_through = np.asarray(pass_through_features, dtype=float)
        if pass_through.ndim == 1:
            pass_through = pass_through.reshape(-1, 1)
        if pass_through.ndim != 2 or len(pass_through) != len(prediction_features):
            raise ValueError("Pass-through features do not align with prediction rows")
        return np.hstack([prediction_features, pass_through])

    def _split(self, targets: np.ndarray):
        row_count = len(targets)
        if row_count < 4:
            raise ValueError("Cross-fitted stacking requires at least four rows")
        if self.task_type in {"regression", "time_series_forecasting"}:
            splits = min(self.cross_fit_splits, row_count)
            return KFold(n_splits=splits, shuffle=True, random_state=self.seed).split(targets)

        _, counts = np.unique(targets, return_counts=True)
        splits = min(self.cross_fit_splits, int(counts.min()))
        if splits < 2:
            raise ValueError("Cross-fitted classification requires two rows per class")
        splitter = StratifiedKFold(n_splits=splits, shuffle=True, random_state=self.seed)
        return splitter.split(np.zeros(row_count), targets)

    def _predict_for_metric(self, model, features: np.ndarray) -> np.ndarray:
        if self.task_type == "binary_classification":
            probabilities = model.predict_proba(features)
            return probabilities[:, 1]
        return np.asarray(model.predict(features))

    def cross_fit(
        self,
        oof_predictions: Sequence[np.ndarray],
        targets: np.ndarray,
        pass_through_features: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return held-out meta predictions, then fit the deployable final meta-model."""
        meta_features = self._build_meta_features(oof_predictions, pass_through_features)
        target_array = np.asarray(targets)
        meta_oof = np.empty(len(target_array), dtype=float)
        fold_ids = np.full(len(target_array), -1, dtype=int)

        for fold_index, (train_indices, validation_indices) in enumerate(
            self._split(target_array)
        ):
            fold_model = clone(self.meta_model)
            fold_model.fit(meta_features[train_indices], target_array[train_indices])
            fold_predictions = self._predict_for_metric(
                fold_model, meta_features[validation_indices]
            )
            meta_oof[validation_indices] = np.asarray(fold_predictions, dtype=float)
            fold_ids[validation_indices] = fold_index

        if np.any(fold_ids < 0):
            raise ValueError("Meta cross-fitting did not predict every row")

        self.meta_model.fit(meta_features, target_array)
        self.feature_count_ = self._prediction_features(oof_predictions).shape[1]
        self.pass_through_count_ = meta_features.shape[1] - self.feature_count_
        self.meta_fold_ids_ = fold_ids
        return meta_oof

    def fit(
        self,
        oof_predictions: Sequence[np.ndarray],
        targets: np.ndarray,
        pass_through_features: np.ndarray | None = None,
    ) -> StackingEnsemble:
        meta_features = self._build_meta_features(oof_predictions, pass_through_features)
        self.meta_model.fit(meta_features, np.asarray(targets))
        self.feature_count_ = self._prediction_features(oof_predictions).shape[1]
        self.pass_through_count_ = meta_features.shape[1] - self.feature_count_
        return self

    def predict(
        self,
        predictions: Sequence[np.ndarray],
        pass_through_features: np.ndarray | None = None,
    ) -> np.ndarray:
        meta_features = self._build_meta_features(predictions, pass_through_features)
        self._validate_feature_count(meta_features)
        return np.asarray(self.meta_model.predict(meta_features))

    def predict_proba(
        self,
        predictions: Sequence[np.ndarray],
        pass_through_features: np.ndarray | None = None,
    ) -> np.ndarray:
        meta_features = self._build_meta_features(predictions, pass_through_features)
        self._validate_feature_count(meta_features)
        if hasattr(self.meta_model, "predict_proba"):
            return np.asarray(self.meta_model.predict_proba(meta_features), dtype=float)
        values = np.asarray(self.meta_model.predict(meta_features), dtype=float)
        return np.column_stack([1.0 - values, values])

    def _validate_feature_count(self, features: np.ndarray) -> None:
        if self.feature_count_ is None:
            raise ValueError("StackingEnsemble must be fitted before prediction")
        expected = self.feature_count_ + self.pass_through_count_
        if features.shape[1] != expected:
            raise ValueError(
                f"Meta feature count mismatch: expected {expected}, got {features.shape[1]}"
            )

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: Path) -> StackingEnsemble:
        loaded = joblib.load(path)
        if not isinstance(loaded, cls):
            raise TypeError("Serialized artifact is not a StackingEnsemble")
        return loaded

    def get_meta_params(self) -> dict[str, Any]:
        classes = getattr(self.meta_model, "classes_", np.array([]))
        return {
            "task_type": self.task_type,
            "seed": self.seed,
            "cross_fit_splits": self.cross_fit_splits,
            "feature_count": self.feature_count_,
            "pass_through_count": self.pass_through_count_,
            "classes": classes.tolist(),
            "coef": getattr(self.meta_model, "coef_", np.array([])).tolist(),
            "intercept": getattr(self.meta_model, "intercept_", np.array([])).tolist(),
        }
