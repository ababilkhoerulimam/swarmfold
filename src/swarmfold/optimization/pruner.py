from collections.abc import Sequence

import numpy as np


class TrialPruner:
    """Evaluates early stopping / fold pruning for training candidates."""

    def __init__(
        self,
        min_folds_before_pruning: int = 2,
        relative_threshold: float = 0.15,
        higher_is_better: bool = True,
    ) -> None:
        self.min_folds_before_pruning = min_folds_before_pruning
        self.relative_threshold = relative_threshold
        self.higher_is_better = higher_is_better

    def should_prune(
        self,
        completed_fold_scores: Sequence[float],
        best_benchmark_score: float | None = None,
    ) -> bool:
        if len(completed_fold_scores) < self.min_folds_before_pruning:
            return False
        if best_benchmark_score is None:
            return False

        current_mean = float(np.mean(completed_fold_scores))
        if self.higher_is_better:
            # If current mean is drastically worse than benchmark
            margin = abs(best_benchmark_score) * self.relative_threshold
            return current_mean < (best_benchmark_score - margin)
        else:
            margin = abs(best_benchmark_score) * self.relative_threshold
            return current_mean > (best_benchmark_score + margin)
