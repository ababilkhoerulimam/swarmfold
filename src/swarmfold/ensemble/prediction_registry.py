from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class MemberPrediction:
    member_id: str
    row_ids: list[str]
    predictions: np.ndarray
    targets: np.ndarray
    model_family: str
    metric_score: float
    scores: np.ndarray | None = None
    test_predictions: np.ndarray | None = None
    experiment_id: str | None = None
    split: str = "oof"
    fold: int | None = None
    prediction_hash: str = ""

    def __post_init__(self) -> None:
        if not self.prediction_hash:
            self.prediction_hash = hashlib.sha256(
                np.ascontiguousarray(self.predictions).tobytes()
            ).hexdigest()


class PredictionRegistry:
    """Shared registry for out-of-fold and test predictions across ensemble members."""

    def __init__(self) -> None:
        self.members: dict[str, MemberPrediction] = {}

    def add_member(
        self,
        member_id: str,
        row_ids: list[str | int],
        predictions: Any,
        targets: Any,
        model_family: str = "unknown",
        metric_score: float = 0.0,
        scores: Any | None = None,
        test_predictions: Any | None = None,
        experiment_id: str | None = None,
        split: str = "oof",
        fold: int | None = None,
        prediction_hash: str | None = None,
    ) -> None:
        str_row_ids = [str(r) for r in row_ids]
        preds_arr = np.asarray(predictions)
        targets_arr = np.asarray(targets)
        scores_arr = np.asarray(scores, dtype=float) if scores is not None else None
        test_preds_arr = np.asarray(test_predictions) if test_predictions is not None else None

        self.members[member_id] = MemberPrediction(
            member_id=member_id,
            row_ids=str_row_ids,
            predictions=preds_arr,
            targets=targets_arr,
            model_family=model_family,
            metric_score=float(metric_score),
            scores=scores_arr,
            test_predictions=test_preds_arr,
            experiment_id=experiment_id or member_id,
            split=split,
            fold=fold,
            prediction_hash=prediction_hash or "",
        )

    def validate_alignment(self) -> None:
        if not self.members:
            raise ValueError("No members registered in PredictionRegistry")

        first_id = next(iter(self.members))
        ref = self.members[first_id]
        ref_rows = ref.row_ids
        ref_targets = ref.targets

        for member_id, member in self.members.items():
            if len(member.row_ids) != len(ref_rows):
                raise ValueError(
                    f"Row count mismatch for {member_id}: "
                    f"expected {len(ref_rows)}, got {len(member.row_ids)}"
                )
            if member.row_ids != ref_rows:
                raise ValueError(f"Row IDs mismatch between {first_id} and {member_id}")
            if not np.array_equal(member.targets, ref_targets):
                raise ValueError(f"Target values mismatch between {first_id} and {member_id}")

    def correlation_matrix(self, metric_name: str = "accuracy") -> pd.DataFrame:
        self.validate_alignment()
        arrays = {}
        for member_id, member in self.members.items():
            if metric_name in {"roc_auc", "log_loss"} and member.scores is not None:
                arrays[member_id] = member.scores
            else:
                arrays[member_id] = pd.to_numeric(
                    pd.Series(member.predictions), errors="coerce"
                ).fillna(0.0)

        df = pd.DataFrame(arrays)
        return df.corr(method="pearson").fillna(0.0)

    def filter_diverse_members(
        self,
        metric_name: str = "accuracy",
        higher_is_better: bool = True,
        max_correlation: float = 0.999,
    ) -> list[str]:
        if len(self.members) <= 1:
            return list(self.members.keys())

        corr = self.correlation_matrix(metric_name)
        # Rank members by single score
        ranked_ids = sorted(
            self.members.keys(),
            key=lambda mid: self.members[mid].metric_score,
            reverse=higher_is_better,
        )

        selected: list[str] = []
        for cand_id in ranked_ids:
            collinear = False
            for sel_id in selected:
                if abs(corr.loc[cand_id, sel_id]) > max_correlation:
                    collinear = True
                    break
            if not collinear:
                selected.append(cand_id)

        return selected if selected else [ranked_ids[0]]

    def get_aligned_arrays(
        self,
        metric_name: str,
        member_ids: list[str] | None = None,
    ) -> tuple[list[np.ndarray], np.ndarray, list[str]]:
        self.validate_alignment()
        active_ids = member_ids or list(self.members.keys())
        preds_list: list[np.ndarray] = []

        for mid in active_ids:
            member = self.members[mid]
            if metric_name in {"roc_auc", "log_loss"} and member.scores is not None:
                preds_list.append(member.scores)
            else:
                preds_list.append(member.predictions)

        first_id = active_ids[0]
        targets = self.members[first_id].targets
        return preds_list, targets, active_ids
