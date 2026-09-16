from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, Field
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from swarmfold.ensemble.caruana import CaruanaEnsembleSelector
from swarmfold.ensemble.prediction_registry import PredictionRegistry
from swarmfold.ensemble.stacker import StackingEnsemble
from swarmfold.ensemble.thresholds import optimize_classification_threshold
from swarmfold.metrics import get_metric_spec


class EnsembleDecision(BaseModel):
    """Structured decision output from the unified ensemble engine."""

    status: Literal["ADOPTED", "REJECTED", "SKIPPED", "INELIGIBLE"] = "SKIPPED"
    adopted: bool = False
    improved_over_best_single: bool = False
    method: str = "none"
    metric_name: str = "accuracy"
    champion_lane: str | None = None
    champion_score: float = 0.0
    ensemble_score: float = 0.0
    ensemble_gain: float = 0.0
    weights: dict[str, float] = Field(default_factory=dict)
    selected_members: list[str] = Field(default_factory=list)
    rule: str = ""
    reason: str = ""
    reconstruction_spec: dict[str, Any] = Field(default_factory=dict)
    diagnostics: list[dict[str, str]] = Field(default_factory=list)


@dataclass
class _EnsembleCandidate:
    method: str
    rule: str
    weights: dict[str, float]
    oof_predictions: np.ndarray
    score: float
    test_predictions: np.ndarray | None = None
    extra: dict[str, Any] | None = None
    adoption_eligible: bool = True


def _positive_label(values: list[str] | np.ndarray) -> str:
    unique_vals = sorted(set(str(v) for v in values))
    for val in unique_vals:
        if val.casefold() in {"yes", "true", "1", "positive"}:
            return val
    return unique_vals[-1]


def _score_predictions(
    metric_name: str,
    targets: np.ndarray,
    predictions: np.ndarray,
) -> float:
    """Compute standard metric score for evaluation."""
    if metric_name == "roc_auc":
        if targets.dtype.kind in {"U", "O", "S"}:
            pos = _positive_label(targets)
            y_true = np.asarray(targets == pos, dtype=int)
        else:
            y_true = np.asarray(targets, dtype=int)
        return float(roc_auc_score(y_true, predictions))

    if metric_name == "log_loss":
        if targets.dtype.kind in {"U", "O", "S"}:
            pos = _positive_label(targets)
            y_true = np.asarray(targets == pos, dtype=int)
        else:
            y_true = np.asarray(targets, dtype=int)
        clipped = np.clip(predictions, 1e-15, 1.0 - 1e-15)
        return float(log_loss(y_true, clipped))

    if metric_name in {"rmse", "mae", "r2"}:
        y_true_num = np.asarray(targets, dtype=float)
        y_pred_num = np.asarray(predictions, dtype=float)
        if metric_name == "rmse":
            return float(mean_squared_error(y_true_num, y_pred_num) ** 0.5)
        if metric_name == "mae":
            return float(mean_absolute_error(y_true_num, y_pred_num))
        if metric_name == "r2":
            return float(r2_score(y_true_num, y_pred_num))

    # Discrete classification metrics
    if predictions.ndim == 1 and np.issubdtype(predictions.dtype, np.floating):
        if targets.dtype.kind in {"U", "O", "S"}:
            pos = _positive_label(targets)
            neg = [v for v in sorted(set(str(x) for x in targets)) if v != pos][0]
            discrete = np.where(predictions >= 0.5, pos, neg)
        else:
            discrete = np.where(predictions >= 0.5, 1, 0)
    else:
        discrete = predictions

    if metric_name == "accuracy":
        return float(accuracy_score(targets, discrete))
    if metric_name == "f1_macro":
        return float(f1_score(targets, discrete, average="macro"))
    if metric_name == "multilabel_f1":
        discrete_ml = np.where(predictions >= 0.5, 1, 0)
        return float(f1_score(targets, discrete_ml, average="macro", zero_division=0))

    raise ValueError(f"Unsupported metric '{metric_name}' in ensemble scorer")


class UnifiedEnsembleEngine:
    """Unified metric-aware ensemble engine for intra-lane and cross-lane selection."""

    def __init__(
        self,
        metric_name: str = "accuracy",
        task_type: str = "binary_classification",
        min_improvement: float = 1e-6,
        seed: int = 42,
        methods: list[str] | None = None,
        artifact_directory: Path | None = None,
        cross_fit_splits: int = 5,
    ) -> None:
        self.metric_name = metric_name
        self.task_type = task_type
        self.min_improvement = min_improvement
        self.seed = seed
        self.methods = methods
        self.artifact_directory = artifact_directory
        self.cross_fit_splits = cross_fit_splits
        self.metric_spec = get_metric_spec(metric_name)
        self.diagnostics: list[dict[str, str]] = []

    def evaluate_registry(
        self,
        registry: PredictionRegistry,
        member_ids: list[str] | None = None,
    ) -> tuple[EnsembleDecision, np.ndarray | None]:
        registry.validate_alignment()
        active_ids = member_ids or list(registry.members.keys())

        if len(active_ids) < 2:
            single_id = active_ids[0] if active_ids else None
            champ_score = (
                registry.members[single_id].metric_score if single_id else 0.0
            )
            decision = EnsembleDecision(
                status="SKIPPED",
                adopted=False,
                method="none",
                metric_name=self.metric_name,
                champion_lane=single_id,
                champion_score=champ_score,
                ensemble_score=champ_score,
                reason="Fewer than two candidate members for ensembling",
            )
            return decision, None

        preds_list, targets, member_names = registry.get_aligned_arrays(
            self.metric_name, active_ids
        )
        member_scores = {mid: registry.members[mid].metric_score for mid in member_names}

        test_preds = [
            registry.members[mid].test_predictions
            for mid in member_names
            if registry.members[mid].test_predictions is not None
        ]
        has_test = len(test_preds) == len(member_names)

        return self.evaluate_arrays(
            member_names=member_names,
            oof_predictions=preds_list,
            targets=targets,
            member_scores=member_scores,
            test_predictions=test_preds if has_test else None,
        )

    def evaluate_arrays(
        self,
        member_names: list[str],
        oof_predictions: list[np.ndarray],
        targets: np.ndarray,
        member_scores: dict[str, float],
        test_predictions: list[np.ndarray] | None = None,
        pass_through_oof: np.ndarray | None = None,
        pass_through_test: np.ndarray | None = None,
    ) -> tuple[EnsembleDecision, np.ndarray | None]:
        n_members = len(member_names)
        if n_members < 2:
            single_id = member_names[0] if member_names else None
            score = member_scores.get(single_id, 0.0) if single_id else 0.0
            return (
                EnsembleDecision(
                    status="SKIPPED",
                    adopted=False,
                    method="none",
                    metric_name=self.metric_name,
                    champion_lane=single_id,
                    champion_score=score,
                    ensemble_score=score,
                    reason="Fewer than two candidate members for ensembling",
                ),
                None,
            )

        ranked_names = sorted(
            member_names,
            key=lambda mid: member_scores[mid],
            reverse=self.metric_spec.higher_is_better,
        )
        champion_name = ranked_names[0]
        champion_score = member_scores[champion_name]

        candidates: list[_EnsembleCandidate] = []
        methods_to_run = set(
            self.methods
            if self.methods is not None
            else [
                "equal_average",
                "pair_blend",
                "rank_average",
                "constrained_weighted_blend",
                "caruana",
                "stacking",
                "majority_vote",
            ]
        )

        # 1. Equal-weight average
        if "equal_average" in methods_to_run:
            cand = self._eval_equal_average(
                member_names, oof_predictions, targets, test_predictions
            )
            if cand is not None:
                candidates.append(cand)

        # 2. Pair blends (champion/challenger grid)
        if "pair_blend" in methods_to_run:
            pair_cands = self._eval_pair_blends(
                member_names,
                ranked_names,
                oof_predictions,
                targets,
                test_predictions,
            )
            candidates.extend(pair_cands)

        # 3. Rank average
        if "rank_average" in methods_to_run and self.metric_name == "roc_auc":
            cand = self._eval_rank_average(
                member_names, oof_predictions, targets, test_predictions
            )
            if cand is not None:
                candidates.append(cand)

        # 4. Constrained weighted blend (SLSQP optimization)
        if "constrained_weighted_blend" in methods_to_run:
            cand = self._eval_constrained_blend(
                member_names, oof_predictions, targets, test_predictions
            )
            if cand is not None:
                candidates.append(cand)

        # 5. Caruana greedy forward selection
        if "caruana" in methods_to_run:
            cand = self._eval_caruana(
                member_names, oof_predictions, targets, test_predictions
            )
            if cand is not None:
                candidates.append(cand)

        # 6. Stacking meta-learner
        if "stacking" in methods_to_run:
            cand = self._eval_stacking(
                member_names,
                oof_predictions,
                targets,
                test_predictions,
                pass_through_oof,
                pass_through_test,
            )
            if cand is not None:
                candidates.append(cand)

        # 7. Majority vote
        if "majority_vote" in methods_to_run and self.metric_name in {"accuracy", "f1_macro"}:
            cand = self._eval_majority_vote(
                member_names, oof_predictions, targets, test_predictions
            )
            if cand is not None:
                candidates.append(cand)

        if not candidates:
            return (
                EnsembleDecision(
                    status="INELIGIBLE",
                    adopted=False,
                    method="none",
                    metric_name=self.metric_name,
                    champion_lane=champion_name,
                    champion_score=champion_score,
                    ensemble_score=champion_score,
                    reason="No valid ensemble candidates could be formed",
                ),
                None,
            )

        eligible_candidates = [candidate for candidate in candidates if candidate.adoption_eligible]
        if not eligible_candidates:
            best_diagnostic = sorted(
                candidates,
                key=lambda candidate: candidate.score,
                reverse=self.metric_spec.higher_is_better,
            )[0]
            return (
                EnsembleDecision(
                    status="INELIGIBLE",
                    adopted=False,
                    method=best_diagnostic.method,
                    metric_name=self.metric_name,
                    champion_lane=champion_name,
                    champion_score=champion_score,
                    ensemble_score=best_diagnostic.score,
                    reason="Ensemble candidates lack required reconstruction evidence",
                    diagnostics=self.diagnostics,
                ),
                None,
            )

        eligible_candidates.sort(
            key=lambda c: c.score,
            reverse=self.metric_spec.higher_is_better,
        )
        best = eligible_candidates[0]

        improved = self.metric_spec.is_better(
            best.score, champion_score, tolerance=self.min_improvement
        )
        gain = self.metric_spec.improvement(best.score, champion_score)

        status = "ADOPTED" if improved else "REJECTED"
        reason = (
            f"Ensemble ({best.rule}) improved upon best single model "
            f"({champion_score:.6f} -> {best.score:.6f})"
            if improved
            else f"Ensemble ({best.rule}) did not beat best single model "
            f"({champion_score:.6f} vs {best.score:.6f})"
        )

        selected_members = [mid for mid, w in best.weights.items() if w > 1e-4]
        if not selected_members:
            selected_members = list(best.weights.keys())

        reconstruction_spec = {
            "method": best.method,
            "rule": best.rule,
            "weights": best.weights,
            "selected_members": selected_members,
            "metric_name": self.metric_name,
            "task_type": self.task_type,
            "extra": best.extra or {},
        }

        decision = EnsembleDecision(
            status=status,
            adopted=improved,
            improved_over_best_single=improved,
            method=best.method,
            metric_name=self.metric_name,
            champion_lane=champion_name,
            champion_score=champion_score,
            ensemble_score=best.score,
            ensemble_gain=gain,
            weights=best.weights,
            selected_members=selected_members,
            rule=best.rule,
            reason=reason,
            reconstruction_spec=reconstruction_spec,
            diagnostics=self.diagnostics,
        )

        return decision, best.test_predictions

    def _eval_equal_average(
        self,
        member_names: list[str],
        oof_predictions: list[np.ndarray],
        targets: np.ndarray,
        test_predictions: list[np.ndarray] | None,
    ) -> _EnsembleCandidate | None:
        n = len(member_names)
        try:
            weights = {mid: 1.0 / n for mid in member_names}
            blend_oof = np.mean([np.asarray(p, dtype=float) for p in oof_predictions], axis=0)
            score = _score_predictions(self.metric_name, targets, blend_oof)
            blend_test = (
                np.mean([np.asarray(tp, dtype=float) for tp in test_predictions], axis=0)
                if test_predictions is not None
                else None
            )
            return _EnsembleCandidate(
                method="equal_average",
                rule="equal-weight average",
                weights=weights,
                oof_predictions=blend_oof,
                score=score,
                test_predictions=blend_test,
            )
        except Exception:
            return None

    def _eval_pair_blends(
        self,
        member_names: list[str],
        ranked_names: list[str],
        oof_predictions: list[np.ndarray],
        targets: np.ndarray,
        test_predictions: list[np.ndarray] | None,
    ) -> list[_EnsembleCandidate]:
        results: list[_EnsembleCandidate] = []
        name_to_idx = {name: idx for idx, name in enumerate(member_names)}
        champ_name = ranked_names[0]
        champ_idx = name_to_idx[champ_name]

        for challenger_name in ranked_names[1:]:
            challenger_idx = name_to_idx[challenger_name]
            for champ_w in (0.5, 0.65, 0.8, 0.9, 0.95):
                challenger_w = 1.0 - champ_w
                weights = {mid: 0.0 for mid in member_names}
                weights[champ_name] = champ_w
                weights[challenger_name] = challenger_w

                try:
                    p_champ = np.asarray(oof_predictions[champ_idx], dtype=float)
                    p_chall = np.asarray(oof_predictions[challenger_idx], dtype=float)
                    blend_oof = champ_w * p_champ + challenger_w * p_chall
                    score = _score_predictions(self.metric_name, targets, blend_oof)

                    blend_test = None
                    if test_predictions is not None:
                        t_champ = np.asarray(test_predictions[champ_idx], dtype=float)
                        t_chall = np.asarray(test_predictions[challenger_idx], dtype=float)
                        blend_test = champ_w * t_champ + challenger_w * t_chall

                    rule = (
                        f"OOF-selected pair blend "
                        f"({champ_w:.2f}/{challenger_w:.2f})"
                    )
                    results.append(
                        _EnsembleCandidate(
                            method="pair_blend",
                            rule=rule,
                            weights=weights,
                            oof_predictions=blend_oof,
                            score=score,
                            test_predictions=blend_test,
                        )
                    )
                except Exception:
                    continue
        return results

    def _eval_rank_average(
        self,
        member_names: list[str],
        oof_predictions: list[np.ndarray],
        targets: np.ndarray,
        test_predictions: list[np.ndarray] | None,
    ) -> _EnsembleCandidate | None:
        try:
            n = len(member_names)
            ranked_oof = []
            for p in oof_predictions:
                arr = np.asarray(p, dtype=float)
                ranked_oof.append(rankdata(arr) / len(arr))
            blend_oof = np.mean(ranked_oof, axis=0)
            score = _score_predictions(self.metric_name, targets, blend_oof)

            blend_test = None
            if test_predictions is not None:
                ranked_test = []
                for tp in test_predictions:
                    arr = np.asarray(tp, dtype=float)
                    ranked_test.append(rankdata(arr) / len(arr))
                blend_test = np.mean(ranked_test, axis=0)

            return _EnsembleCandidate(
                method="rank_average",
                rule="rank-average ensemble",
                weights={mid: 1.0 / n for mid in member_names},
                oof_predictions=blend_oof,
                score=score,
                test_predictions=blend_test,
            )
        except Exception:
            return None

    def _eval_constrained_blend(
        self,
        member_names: list[str],
        oof_predictions: list[np.ndarray],
        targets: np.ndarray,
        test_predictions: list[np.ndarray] | None,
    ) -> _EnsembleCandidate | None:
        n = len(member_names)
        try:
            p_matrix = np.column_stack([np.asarray(p, dtype=float) for p in oof_predictions])
            y_target = (
                (targets == _positive_label(targets)).astype(float)
                if self.metric_name in {"roc_auc", "log_loss"}
                else np.asarray(targets, dtype=float)
            )

            def objective(w: np.ndarray) -> float:
                pred = p_matrix @ w
                if self.metric_name == "log_loss":
                    clipped = np.clip(pred, 1e-15, 1.0 - 1e-15)
                    return float(log_loss(y_target, clipped))
                return float(np.mean((pred - y_target) ** 2))

            constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
            bounds = [(0.0, 1.0) for _ in range(n)]
            init_w = np.ones(n) / n

            res = minimize(
                objective,
                init_w,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 200},
            )

            if not res.success:
                return None

            opt_w = np.clip(res.x, 0.0, 1.0)
            opt_w = opt_w / np.sum(opt_w)
            weights = {mid: float(w) for mid, w in zip(member_names, opt_w, strict=True)}

            blend_oof = p_matrix @ opt_w
            score = _score_predictions(self.metric_name, targets, blend_oof)

            blend_test = None
            if test_predictions is not None:
                t_matrix = np.column_stack(
                    [np.asarray(tp, dtype=float) for tp in test_predictions]
                )
                blend_test = t_matrix @ opt_w

            return _EnsembleCandidate(
                method="constrained_weighted_blend",
                rule="SLSQP constrained weighted blend",
                weights=weights,
                oof_predictions=blend_oof,
                score=score,
                test_predictions=blend_test,
            )
        except Exception:
            return None

    def _eval_caruana(
        self,
        member_names: list[str],
        oof_predictions: list[np.ndarray],
        targets: np.ndarray,
        test_predictions: list[np.ndarray] | None,
    ) -> _EnsembleCandidate | None:
        try:
            selector = CaruanaEnsembleSelector(
                iterations=min(40, len(targets)),
                metric_name=self.metric_name,
                higher_is_better=self.metric_spec.higher_is_better,
            )
            selector.fit(oof_predictions, targets)

            weights = {
                member_names[i]: selector.weights_.get(i, 0.0)
                for i in range(len(member_names))
            }
            blend_oof = selector.predict(oof_predictions)
            score = selector.best_score_ or _score_predictions(
                self.metric_name, targets, blend_oof
            )

            blend_test = (
                selector.predict(test_predictions)
                if test_predictions is not None
                else None
            )

            return _EnsembleCandidate(
                method="caruana",
                rule="Caruana greedy stepwise forward selection",
                weights=weights,
                oof_predictions=blend_oof,
                score=score,
                test_predictions=blend_test,
                extra={"selected_steps": selector.selected_indices},
            )
        except Exception:
            return None

    def _eval_stacking(
        self,
        member_names: list[str],
        oof_predictions: list[np.ndarray],
        targets: np.ndarray,
        test_predictions: list[np.ndarray] | None,
        pass_through_oof: np.ndarray | None = None,
        pass_through_test: np.ndarray | None = None,
    ) -> _EnsembleCandidate | None:
        if self.task_type not in {
            "binary_classification",
            "regression",
            "time_series_forecasting",
        }:
            self.diagnostics.append(
                {
                    "method": "stacking",
                    "status": "INELIGIBLE",
                    "reason": f"Cross-fitted stacking is not implemented for {self.task_type}",
                }
            )
            return None
        try:
            stacker = StackingEnsemble(
                task_type=self.task_type,
                seed=self.seed,
                cross_fit_splits=self.cross_fit_splits,
            )
            meta_oof = stacker.cross_fit(
                oof_predictions,
                targets,
                pass_through_features=pass_through_oof,
            )
            threshold_result = None
            positive_class = None
            negative_class = None
            if self.task_type == "binary_classification":
                classes = list(stacker.meta_model.classes_)
                negative_class, positive_class = classes[0], classes[1]
                if hasattr(negative_class, "item"):
                    negative_class = negative_class.item()
                if hasattr(positive_class, "item"):
                    positive_class = positive_class.item()
                if self.metric_name in {"accuracy", "f1_macro"}:
                    threshold_result = optimize_classification_threshold(
                        targets,
                        meta_oof,
                        metric_name=self.metric_name,
                        positive_class=positive_class,
                    )
                    blend_oof = np.where(
                        meta_oof >= threshold_result.threshold,
                        positive_class,
                        negative_class,
                    )
                else:
                    blend_oof = meta_oof
            else:
                blend_oof = meta_oof

            score = _score_predictions(self.metric_name, targets, blend_oof)

            blend_test = None
            raw_test = None
            if test_predictions is not None:
                if self.task_type == "binary_classification":
                    t_probs = stacker.predict_proba(
                        test_predictions,
                        pass_through_features=pass_through_test,
                    )
                    raw_test = (
                        t_probs[:, 1]
                        if t_probs.ndim == 2 and t_probs.shape[1] > 1
                        else t_probs
                    )
                    if threshold_result is not None:
                        blend_test = np.where(
                            raw_test >= threshold_result.threshold,
                            positive_class,
                            negative_class,
                        )
                    else:
                        blend_test = raw_test
                else:
                    raw_test = stacker.predict(
                        test_predictions,
                        pass_through_features=pass_through_test,
                    )
                    blend_test = raw_test

            artifact_evidence: dict[str, Any] = {}
            adoption_eligible = False
            if self.artifact_directory is not None and test_predictions is not None:
                self.artifact_directory.mkdir(parents=True, exist_ok=True)
                model_path = stacker.save(self.artifact_directory / "meta-model.joblib")
                pq.write_table(
                    pa.table(
                        {
                            "row": [str(index) for index in range(len(meta_oof))],
                            "fold": stacker.meta_fold_ids_.tolist(),
                            "prediction": np.asarray(meta_oof, dtype=float).tolist(),
                        }
                    ),
                    self.artifact_directory / "meta-oof-predictions.parquet",
                )
                reloaded = StackingEnsemble.load(model_path)
                if self.task_type == "binary_classification":
                    reloaded_probabilities = reloaded.predict_proba(
                        test_predictions,
                        pass_through_features=pass_through_test,
                    )[:, 1]
                    expected = np.asarray(raw_test, dtype=float)
                    reconstructed = reloaded_probabilities
                else:
                    reconstructed = reloaded.predict(
                        test_predictions,
                        pass_through_features=pass_through_test,
                    )
                    expected = np.asarray(raw_test)
                parity = bool(np.allclose(reconstructed, expected, rtol=1e-10, atol=1e-12))
                reconstruction_record = {
                    "passed": parity,
                    "row_count": len(expected),
                    "relative_tolerance": 1e-10,
                    "absolute_tolerance": 1e-12,
                }
                (self.artifact_directory / "ensemble-reconstruction-test.json").write_text(
                    json.dumps(reconstruction_record, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                if threshold_result is not None:
                    (self.artifact_directory / "threshold.json").write_text(
                        json.dumps(threshold_result.as_dict(), indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                if pass_through_test is not None:
                    pass_through_array = np.asarray(pass_through_test, dtype=float)
                    if pass_through_array.ndim == 1:
                        pass_through_array = pass_through_array.reshape(-1, 1)
                    pq.write_table(
                        pa.table(
                            {
                                f"feature_{index}": pass_through_array[:, index].tolist()
                                for index in range(pass_through_array.shape[1])
                            }
                        ),
                        self.artifact_directory / "pass-through-test.parquet",
                    )
                    artifact_evidence["pass_through_test_path"] = (
                        "ensemble/pass-through-test.parquet"
                    )
                adoption_eligible = parity
                artifact_evidence = {
                    "meta_model_path": "ensemble/meta-model.joblib",
                    "meta_oof_path": "ensemble/meta-oof-predictions.parquet",
                    "reconstruction_test_path": (
                        "ensemble/ensemble-reconstruction-test.json"
                    ),
                    "reconstruction_passed": parity,
                }

            weights = {mid: 1.0 / len(member_names) for mid in member_names}
            extra = {
                **stacker.get_meta_params(),
                **artifact_evidence,
                "member_order": member_names,
                "positive_class": positive_class,
                "negative_class": negative_class,
                "threshold": threshold_result.as_dict() if threshold_result else None,
            }
            return _EnsembleCandidate(
                method="stacking",
                rule="cross-fitted stacking meta-learner",
                weights=weights,
                oof_predictions=blend_oof,
                score=score,
                test_predictions=blend_test,
                extra=extra,
                adoption_eligible=adoption_eligible,
            )
        except Exception as exc:
            self.diagnostics.append(
                {
                    "method": "stacking",
                    "status": "FAILED",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            return None

    def _eval_majority_vote(
        self,
        member_names: list[str],
        oof_predictions: list[np.ndarray],
        targets: np.ndarray,
        test_predictions: list[np.ndarray] | None,
    ) -> _EnsembleCandidate | None:
        try:
            from collections import Counter

            def vote_columns(cols: list[np.ndarray]) -> np.ndarray:
                n_rows = len(cols[0])
                return np.array(
                    [
                        Counter(
                            str(cols[c_idx][r_idx]) for c_idx in range(len(cols))
                        ).most_common(1)[0][0]
                        for r_idx in range(n_rows)
                    ]
                )

            blend_oof = vote_columns(oof_predictions)
            score = _score_predictions(self.metric_name, targets, blend_oof)

            blend_test = (
                vote_columns(test_predictions)
                if test_predictions is not None
                else None
            )

            return _EnsembleCandidate(
                method="majority_vote",
                rule="majority vote",
                weights={mid: 1.0 / len(member_names) for mid in member_names},
                oof_predictions=blend_oof,
                score=score,
                test_predictions=blend_test,
            )
        except Exception:
            return None


def reconstruct_ensemble_submission(
    ensemble_spec_path: Path,
    member_artifacts_dir: Path,
    output_path: Path,
    template_submission_path: Path | None = None,
    contracts_path: Path | None = None,
) -> Path:
    """Deterministically regenerate submission CSV from member artifacts and ensemble spec."""
    spec = json.loads(ensemble_spec_path.read_text(encoding="utf-8"))
    reconstruction_spec = spec.get("reconstruction_spec") or spec
    weights: dict[str, float] = reconstruction_spec.get("weights", spec.get("weights", {}))
    method: str = reconstruction_spec.get("method") or spec.get("method", "")
    method = method or spec.get("rule", "")
    champion_lane: str | None = spec.get("champion_lane")

    # If ensemble was not adopted, fallback to champion submission
    if not spec.get("adopted", False):
        champ_sub = _find_member_file(
            member_artifacts_dir, champion_lane or "", "submission.csv"
        )
        if champ_sub and champ_sub.is_file():
            output_path.write_bytes(champ_sub.read_bytes())
            return output_path
        raise ValueError("Ensemble not adopted and champion submission not found")

    # Load member test prediction columns
    selected_members = reconstruction_spec.get("selected_members") or []
    active_members = [str(member) for member in selected_members]
    if not active_members:
        active_members = [mid for mid, w in weights.items() if w > 1e-4]
    if not active_members:
        active_members = list(weights.keys())

    member_test_preds: dict[str, list[str]] = {}
    for mid in active_members:
        t_path = _find_member_file(
            member_artifacts_dir, mid, "test_predictions.parquet"
        )
        if not t_path or not t_path.is_file():
            raise FileNotFoundError(f"Missing test_predictions.parquet for member '{mid}'")
        table = pq.read_table(t_path).to_pydict()
        member_test_preds[mid] = [str(v) for v in table["prediction"]]

    first_len = len(next(iter(member_test_preds.values())))
    for mid, col in member_test_preds.items():
        if len(col) != first_len:
            raise ValueError(f"Member test length mismatch for {mid}")

    # Compute reconstructed predictions
    if "stacking" in method.casefold():
        extra = reconstruction_spec.get("extra") or {}
        model_reference = extra.get("meta_model_path")
        if not isinstance(model_reference, str):
            raise ValueError("Stacking reconstruction is missing meta_model_path")
        model_path = Path(model_reference)
        if not model_path.is_absolute():
            model_path = ensemble_spec_path.parent / model_path
        stacker = StackingEnsemble.load(model_path)
        member_order = [str(value) for value in extra.get("member_order", active_members)]
        if set(member_order) != set(active_members):
            raise ValueError("Stacking member order does not match selected members")
        ordered_predictions = [
            np.asarray(member_test_preds[member], dtype=float) for member in member_order
        ]
        pass_through_features = None
        pass_through_reference = extra.get("pass_through_test_path")
        if isinstance(pass_through_reference, str):
            pass_through_path = Path(pass_through_reference)
            if not pass_through_path.is_absolute():
                pass_through_path = ensemble_spec_path.parent / pass_through_path
            pass_through_table = pq.read_table(pass_through_path).to_pydict()
            pass_through_features = np.column_stack(
                [
                    np.asarray(pass_through_table[column], dtype=float)
                    for column in sorted(
                        pass_through_table,
                        key=lambda value: int(value.removeprefix("feature_")),
                    )
                ]
            )
        if stacker.task_type == "binary_classification":
            probabilities = stacker.predict_proba(
                ordered_predictions,
                pass_through_features=pass_through_features,
            )[:, 1]
            threshold_record = extra.get("threshold")
            if threshold_record:
                threshold = float(threshold_record["threshold"])
                positive = extra.get("positive_class")
                negative = extra.get("negative_class")
                reconstructed = np.where(
                    probabilities >= threshold, positive, negative
                ).tolist()
            else:
                reconstructed = probabilities.tolist()
        else:
            reconstructed = stacker.predict(
                ordered_predictions,
                pass_through_features=pass_through_features,
            ).tolist()
    elif "majority" in method.casefold():
        from collections import Counter

        reconstructed = [
            Counter(member_test_preds[mid][idx] for mid in active_members).most_common(1)[0][0]
            for idx in range(first_len)
        ]
    elif "rank" in method.casefold():
        ranked_cols = [
            rankdata(np.array(member_test_preds[mid], dtype=float)) / first_len
            for mid in active_members
        ]
        reconstructed = [
            float(np.mean([ranked_cols[c][idx] for c in range(len(active_members))]))
            for idx in range(first_len)
        ]
    else:
        norm_weights = {
            mid: weights.get(mid, 0.0) / sum(weights.get(m, 0.0) for m in active_members)
            for mid in active_members
        }
        reconstructed = [
            sum(
                float(member_test_preds[mid][idx]) * norm_weights[mid]
                for mid in active_members
            )
            for idx in range(first_len)
        ]

    # Write submission CSV
    template = template_submission_path or _find_member_file(
        member_artifacts_dir, champion_lane or active_members[0], "submission.csv"
    )
    if not template or not template.is_file():
        raise FileNotFoundError("No template submission.csv found to construct submission")

    contracts_file = contracts_path or _find_member_file(
        member_artifacts_dir, champion_lane or active_members[0], "contracts.json"
    )
    if contracts_file and contracts_file.is_file():
        contracts_data = json.loads(contracts_file.read_text(encoding="utf-8"))
        pred_col = contracts_data.get("competition", {}).get(
            "prediction_column", "prediction"
        )
    else:
        pred_col = "prediction"

    with template.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if len(rows) != len(reconstructed):
        raise ValueError("Reconstructed predictions count does not match template rows")

    for row, val in zip(rows, reconstructed, strict=True):
        row[pred_col] = str(val)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def _find_member_file(root: Path, member_id: str, filename: str) -> Path | None:
    candidates = [
        root / member_id / filename,
        root / "ensemble-members" / member_id / filename,
        root / member_id / "packet" / filename,
        root / "experiments" / member_id / "packet" / filename,
        root / "source-packets" / member_id / filename,
        root / filename,
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    return None
