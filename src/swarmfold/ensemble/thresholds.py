from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


@dataclass(frozen=True)
class ThresholdOptimizationResult:
    metric_name: str
    threshold: float
    baseline_threshold: float
    baseline_score: float
    optimized_score: float
    candidate_count: int
    improved: bool

    def as_dict(self) -> dict[str, str | float | int | bool]:
        return asdict(self)


_SUPPORTED_METRICS = {"accuracy", "balanced_accuracy", "f1", "f1_macro"}


def optimize_classification_threshold(
    targets: np.ndarray,
    probabilities: np.ndarray,
    *,
    metric_name: str,
    positive_class: str | int | float | None = None,
    candidate_count: int = 199,
) -> ThresholdOptimizationResult:
    """Optimize a binary decision threshold from cross-fitted probabilities."""
    if metric_name not in _SUPPORTED_METRICS:
        raise ValueError(f"Threshold optimization is not supported for metric '{metric_name}'")
    target_array = np.asarray(targets)
    probability_array = np.asarray(probabilities, dtype=float)
    if probability_array.ndim != 1 or len(probability_array) != len(target_array):
        raise ValueError("Threshold probabilities must be a one-dimensional aligned array")
    classes = np.unique(target_array)
    if len(classes) != 2:
        raise ValueError("Threshold optimization requires a binary target")

    positive = positive_class if positive_class is not None else classes[-1]
    negative = next(value for value in classes if value != positive)
    if positive not in classes:
        raise ValueError("Configured positive class is absent from the target")

    def score(threshold: float) -> float:
        labels = np.where(probability_array >= threshold, positive, negative)
        if metric_name == "accuracy":
            return float(accuracy_score(target_array, labels))
        if metric_name == "balanced_accuracy":
            return float(balanced_accuracy_score(target_array, labels))
        if metric_name == "f1_macro":
            return float(f1_score(target_array, labels, average="macro", zero_division=0))
        return float(
            f1_score(
                target_array,
                labels,
                pos_label=positive,
                average="binary",
                zero_division=0,
            )
        )

    thresholds = np.linspace(0.005, 0.995, candidate_count)
    baseline_score = score(0.5)
    scored = [(float(threshold), score(float(threshold))) for threshold in thresholds]
    best_threshold, best_score = max(scored, key=lambda item: (item[1], -abs(item[0] - 0.5)))
    improved = best_score > baseline_score + 1e-12
    if not improved:
        best_threshold = 0.5
        best_score = baseline_score
    return ThresholdOptimizationResult(
        metric_name=metric_name,
        threshold=best_threshold,
        baseline_threshold=0.5,
        baseline_score=baseline_score,
        optimized_score=best_score,
        candidate_count=candidate_count,
        improved=improved,
    )
