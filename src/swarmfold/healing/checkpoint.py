from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


class CheckpointManager:
    """Manages fold-level experiment checkpoints to enable safe crash resumption."""

    def __init__(self, checkpoint_dir: Path) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _meta_path(self, fold_idx: int) -> Path:
        return self.checkpoint_dir / f"fold_{fold_idx}_meta.json"

    def _pred_path(self, fold_idx: int) -> Path:
        return self.checkpoint_dir / f"fold_{fold_idx}_predictions.parquet"

    def _test_pred_path(self, fold_idx: int) -> Path:
        return self.checkpoint_dir / f"fold_{fold_idx}_test_predictions.parquet"

    def _model_path(self, fold_idx: int) -> Path:
        return self.checkpoint_dir / f"fold_{fold_idx}_model.joblib"

    def has_fold(self, fold_idx: int) -> bool:
        """Return True if fold checkpoint metadata and predictions exist."""
        return self._meta_path(fold_idx).is_file() and self._pred_path(fold_idx).is_file()

    def completed_folds(self) -> list[int]:
        """Return list of completed fold indices sorted in ascending order."""
        indices = []
        for path in self.checkpoint_dir.glob("fold_*_meta.json"):
            stem = path.stem
            parts = stem.split("_")
            if len(parts) >= 2 and parts[1].isdigit():
                idx = int(parts[1])
                if self.has_fold(idx):
                    indices.append(idx)
        return sorted(indices)

    def save_fold(
        self,
        fold_idx: int,
        val_indices: list[int] | np.ndarray,
        predictions: np.ndarray,
        targets: np.ndarray,
        metric_score: float,
        scores: np.ndarray | None = None,
        test_predictions: np.ndarray | None = None,
        model: Any | None = None,
    ) -> None:
        """Persist intermediate predictions and metrics for a completed fold."""
        str_val_indices = [str(i) for i in val_indices]
        str_preds = [str(p) for p in predictions]
        str_targets = [str(t) for t in targets]

        pred_dict: dict[str, list[str]] = {
            "val_index": str_val_indices,
            "prediction": str_preds,
            "target": str_targets,
        }
        if scores is not None:
            pred_dict["score"] = [str(s) for s in scores]

        pq.write_table(pa.table(pred_dict), self._pred_path(fold_idx))

        if test_predictions is not None:
            test_dict = {"prediction": [str(p) for p in test_predictions]}
            pq.write_table(pa.table(test_dict), self._test_pred_path(fold_idx))

        if model is not None:
            joblib.dump(model, self._model_path(fold_idx))

        meta = {
            "fold_idx": fold_idx,
            "metric_score": float(metric_score),
            "num_val_rows": len(val_indices),
            "has_scores": scores is not None,
            "has_test": test_predictions is not None,
            "has_model": model is not None,
        }
        self._meta_path(fold_idx).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def load_fold(self, fold_idx: int) -> dict[str, Any]:
        """Load persisted predictions, targets, and scores for a fold."""
        if not self.has_fold(fold_idx):
            raise FileNotFoundError(f"No checkpoint found for fold {fold_idx}")

        meta = json.loads(self._meta_path(fold_idx).read_text(encoding="utf-8"))
        table = pq.read_table(self._pred_path(fold_idx)).to_pydict()

        val_indices = [int(x) for x in table["val_index"]]
        predictions = np.array(table["prediction"])
        targets = np.array(table["target"])
        scores = np.array([float(x) for x in table["score"]]) if "score" in table else None

        test_predictions = None
        if self._test_pred_path(fold_idx).is_file():
            test_table = pq.read_table(self._test_pred_path(fold_idx)).to_pydict()
            test_predictions = np.array(test_table["prediction"])

        model = None
        if self._model_path(fold_idx).is_file():
            model = joblib.load(self._model_path(fold_idx))

        return {
            "fold_idx": fold_idx,
            "metric_score": meta["metric_score"],
            "val_indices": val_indices,
            "predictions": predictions,
            "targets": targets,
            "scores": scores,
            "test_predictions": test_predictions,
            "model": model,
        }

    def clear(self) -> None:
        """Remove all checkpoint files."""
        for path in self.checkpoint_dir.glob("fold_*"):
            if path.is_file():
                path.unlink()
