from collections import Counter
from collections.abc import Sequence

import numpy as np

from swarmfold.nodes import _compute_metric


class CaruanaEnsembleSelector:
    """Caruana Stepwise Greedy Ensemble Selection (Caruana et al., 2004).

    Greedily selects model predictions with replacement to optimize the OOF score.
    """

    def __init__(
        self,
        iterations: int = 50,
        metric_name: str = "accuracy",
        higher_is_better: bool = True,
        positive_class: str | None = None,
    ) -> None:
        self.iterations = iterations
        self.metric_name = metric_name
        self.higher_is_better = higher_is_better
        self.positive_class = positive_class
        self.selected_indices: list[int] = []
        self.weights_: dict[int, float] = {}
        self.best_score_: float | None = None

    def fit(
        self,
        predictions: Sequence[np.ndarray],
        targets: np.ndarray,
    ) -> "CaruanaEnsembleSelector":
        n_models = len(predictions)
        if n_models == 0:
            raise ValueError("At least one model prediction array is required.")

        targets = np.asarray(targets)
        # Find best starting single model
        best_single_idx = 0
        best_single_score = -float("inf") if self.higher_is_better else float("inf")

        for idx, pred in enumerate(predictions):
            score = self._score(targets, pred)
            if self._is_better(score, best_single_score):
                best_single_score = score
                best_single_idx = idx

        self.selected_indices = [best_single_idx]
        current_sum = np.array(predictions[best_single_idx], dtype=float)
        current_score = best_single_score

        for step in range(1, self.iterations):
            best_cand_idx = -1
            best_cand_score = current_score

            for idx, cand_pred in enumerate(predictions):
                tentative_sum = current_sum + cand_pred
                tentative_avg = tentative_sum / (step + 1)
                tentative_score = self._score(targets, tentative_avg)

                if self._is_better(tentative_score, best_cand_score):
                    best_cand_score = tentative_score
                    best_cand_idx = idx

            if best_cand_idx != -1 and self._is_better(best_cand_score, current_score):
                self.selected_indices.append(best_cand_idx)
                current_sum += predictions[best_cand_idx]
                current_score = best_cand_score
            else:
                break

        self.best_score_ = current_score
        counts = Counter(self.selected_indices)
        total_selected = len(self.selected_indices)
        self.weights_ = {idx: count / total_selected for idx, count in counts.items()}
        return self

    def _is_better(self, new_score: float, current_score: float) -> bool:
        if self.higher_is_better:
            return new_score > current_score + 1e-9
        return new_score < current_score - 1e-9

    def _score(self, y_true: np.ndarray, y_pred_or_prob: np.ndarray) -> float:
        if self.metric_name in ("roc_auc", "log_loss"):
            return _compute_metric(
                self.metric_name,
                y_true=y_true,
                y_pred=None,
                y_prob=y_pred_or_prob,
                positive_class=self.positive_class,
            )
        if self.metric_name in ("rmse", "mae", "r2"):
            return _compute_metric(
                self.metric_name,
                y_true=y_true,
                y_pred=y_pred_or_prob,
            )

        if (
            y_pred_or_prob.ndim == 1
            and np.issubdtype(y_pred_or_prob.dtype, np.floating)
            and self.metric_name in ("accuracy", "f1_macro")
        ):
            if self.positive_class is not None:
                discrete_pred = np.where(y_pred_or_prob >= 0.5, self.positive_class, 0)
            else:
                discrete_pred = np.where(y_pred_or_prob >= 0.5, 1, 0)
        else:
            discrete_pred = y_pred_or_prob

        return _compute_metric(
            self.metric_name,
            y_true=y_true,
            y_pred=discrete_pred,
            positive_class=self.positive_class,
        )

    def predict(self, test_predictions: Sequence[np.ndarray]) -> np.ndarray:
        ensemble_pred = np.zeros_like(test_predictions[0], dtype=float)
        for idx, weight in self.weights_.items():
            ensemble_pred += weight * test_predictions[idx]
        return ensemble_pred
