from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from swarmfold.model_registry import model_capability_registry
from swarmfold.optimization.controller import HPOController


@dataclass(frozen=True)
class CandidateSpec:
    hypothesis: str
    model_family: str
    feature_strategy: str
    parameters: dict[str, Any]
    hpo_study_id: str | None = None
    hpo_trial_number: int | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "hypothesis": self.hypothesis,
            "model_family": self.model_family,
            "feature_strategy": self.feature_strategy,
            "parameters": dict(self.parameters),
            "hpo_study_id": self.hpo_study_id,
            "hpo_trial_number": self.hpo_trial_number,
        }


class CandidateFactory:
    """Build executable, task-compatible candidates for one modeling lane."""

    def __init__(
        self,
        registry: list[dict[str, object]] | None = None,
        hpo_controller: HPOController | None = None,
    ) -> None:
        self.registry = registry or model_capability_registry()
        self.hpo_controller = hpo_controller
        self._capabilities = {str(entry["model_family"]): entry for entry in self.registry}

    def create(
        self,
        lane_id: str,
        limit: int,
        *,
        task_type: str = "binary_classification",
    ) -> list[dict[str, object]]:
        if limit < 1:
            return []
        builders = {
            "ai-a": self._conservative,
            "ai-b": self._boosting,
            "ai-c": self._feature_ablation,
            "ai-d": self._alternative_geometry,
            "ai-e": self._robustness,
        }
        builder = builders.get(lane_id, self._conservative)
        candidates: list[dict[str, object]] = []
        index = 0
        max_attempts = max(limit * 8, 40)
        while len(candidates) < limit and index < max_attempts:
            candidate = builder(index, task_type)
            if self._eligible(candidate.model_family, lane_id, task_type):
                candidates.append(candidate.as_dict())
            index += 1
        return candidates

    def eligibility_report(self, lane_id: str, task_type: str) -> list[dict[str, object]]:
        report = []
        for family, entry in self._capabilities.items():
            report.append(
                {
                    "model_family": family,
                    "declared": bool(entry.get("declared")),
                    "installed": bool(entry.get("installed")),
                    "executable": bool(entry.get("executable")),
                    "applicable": task_type in entry.get("task_types", ()),
                    "scheduled": lane_id in entry.get("recommended_lanes", ()),
                    "eligible": self._eligible(family, lane_id, task_type),
                }
            )
        return report

    def _eligible(self, family: str, lane_id: str, task_type: str) -> bool:
        entry = self._capabilities.get(family)
        if entry is None:
            return False
        return bool(
            entry.get("declared")
            and entry.get("installed")
            and entry.get("executable")
            and task_type in entry.get("task_types", ())
            and lane_id in entry.get("recommended_lanes", ())
        )

    def _conservative(self, index: int, task_type: str) -> CandidateSpec:
        templates = (
            ("stable linear baseline", "logistic_regression", "scaled_numeric", {"c": 1.0}),
            (
                "bounded nonlinear baseline",
                "random_forest",
                "raw_numeric",
                {"n_estimators": 100, "max_depth": 5},
            ),
            (
                "regularized boosting baseline",
                "hist_gradient_boosting",
                "raw_numeric",
                {"max_iter": 100, "max_depth": 3},
            ),
            ("conservative family blend", "soft_voting", "scaled_numeric", {}),
            ("stacked conservative families", "stacking", "raw_numeric", {}),
        )
        if index < len(templates):
            mechanism, family, strategy, parameters = templates[index]
        elif index % 2:
            mechanism, family, strategy = (
                "conservative tree stability",
                "random_forest",
                "raw_numeric",
            )
            parameters = {
                "n_estimators": 125 + 25 * (index % 6),
                "max_depth": 3 + index % 6,
                "min_samples_leaf": 1 + index % 4,
            }
        else:
            mechanism, family, strategy = (
                "conservative linear stability",
                "logistic_regression",
                "scaled_numeric",
            )
            parameters = {"c": (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)[index % 6]}
        return self._candidate(mechanism, index, family, strategy, parameters, task_type)

    def _boosting(self, index: int, task_type: str) -> CandidateSpec:
        families = ("hist_gradient_boosting", "lightgbm", "xgboost", "catboost", "stacking")
        family = families[index % len(families)]
        mechanism = "bounded boosting search"
        if family == "hist_gradient_boosting":
            parameters = {
                "max_iter": 75 + 25 * (index % 8),
                "max_depth": 2 + index % 7,
                "learning_rate": (0.03, 0.05, 0.08, 0.1)[index % 4],
                "l2_regularization": float(index % 6),
            }
        elif family == "stacking":
            mechanism = "boosting-led stacked ensemble"
            parameters = {"meta_c": 1.0, "cv": 3}
        else:
            parameters = {
                "n_estimators": 150 + 50 * (index % 6),
                "max_depth": 3 + index % 5,
                "learning_rate": (0.02, 0.04, 0.07, 0.1)[index % 4],
                "subsample": (0.75, 0.9, 1.0)[index % 3],
                "colsample_bytree": (0.7, 0.85, 1.0)[index % 3],
            }
        candidate = self._candidate(
            mechanism, index, family, "raw_numeric", parameters, task_type
        )
        if self.hpo_controller is not None and family != "stacking":
            parameters = self.hpo_controller.suggest(family, index)
            candidate = replace(
                candidate,
                hypothesis=(
                    f"adaptive boosting trial {index + 1} uses {family} for {task_type}; "
                    f"parameters {parameters} are sampled from the persistent HPO study."
                ),
                parameters=parameters,
                hpo_study_id=self.hpo_controller.config.study_id,
                hpo_trial_number=index,
            )
        return candidate

    def _feature_ablation(self, index: int, task_type: str) -> CandidateSpec:
        templates = (
            ("selected_50", "fold-safe feature selection 50%", "hist_gradient_boosting"),
            ("selected_75", "fold-safe feature selection 75%", "logistic_regression"),
            ("target_encoded", "smoothed out-of-fold target encoding", "hist_gradient_boosting"),
            (
                "power_transformed",
                "variance-stabilizing power transformation",
                "logistic_regression",
            ),
            ("selected_75", "feature-selected stacked ensemble", "stacking"),
            (
                "interaction_pairs",
                "pairwise interaction polynomial expansion",
                "hist_gradient_boosting",
            ),
            (
                "frequency_encoded",
                "frequency-encoded categorical representations",
                "hist_gradient_boosting",
            ),
        )
        strategy, mechanism, family = templates[index % len(templates)]
        if family == "stacking":
            parameters = {"meta_c": 0.5, "cv": 3, "passthrough": True}
        elif family == "logistic_regression":
            parameters = {"c": (0.05, 0.2, 1.0, 5.0, 20.0)[index % 5]}
        else:
            parameters = {
                "max_iter": 75 + 25 * (index % 6),
                "max_depth": 2 + index % 5,
                "learning_rate": (0.04, 0.07, 0.1)[index % 3],
            }
        return self._candidate(
            mechanism, index, family, strategy, parameters, task_type
        )

    def _alternative_geometry(self, index: int, task_type: str) -> CandidateSpec:
        slot = index % 5
        if slot in {1, 4}:
            family = "extra_trees"
            parameters = {
                "n_estimators": 100 + 50 * (index % 6),
                "max_depth": 4 + index % 10,
                "min_samples_leaf": 1 + index % 4,
                "max_features": ("sqrt", "log2", 0.75)[index % 3],
            }
        elif slot == 0:
            family = "sgd_log_loss"
            parameters = {
                "alpha": (0.00001, 0.0001, 0.001, 0.01)[index % 4],
                "penalty": ("l2", "l1", "elasticnet")[index % 3],
            }
        elif slot == 2:
            family = "bagging"
            parameters = {"n_estimators": 100, "max_samples": 0.8, "max_features": 0.9}
        else:
            family = "stacking"
            parameters = {"passthrough": True, "cv": 3}
        return self._candidate(
            "alternative prediction geometry",
            index,
            family,
            "scaled_numeric",
            parameters,
            task_type,
        )

    def _robustness(self, index: int, task_type: str) -> CandidateSpec:
        families = (
            "logistic_regression",
            "random_forest",
            "hist_gradient_boosting",
            "soft_voting",
            "stacking",
        )
        family = families[index % len(families)]
        if family == "logistic_regression":
            parameters = {"c": (0.05, 0.2, 1.0, 5.0)[index % 4]}
            if task_type != "regression":
                parameters["class_weight"] = "balanced"
        elif family == "random_forest":
            parameters = {
                "n_estimators": 150 + 25 * (index % 6),
                "max_depth": 4 + index % 7,
                "min_samples_leaf": 2 + index % 4,
            }
            if task_type != "regression":
                parameters["class_weight"] = "balanced"
        elif family == "hist_gradient_boosting":
            parameters = {
                "max_iter": 100 + 25 * (index % 6),
                "max_depth": 2 + index % 5,
                "l2_regularization": 1.0 + index % 6,
            }
        elif family == "soft_voting":
            parameters = {"weights": [1 + index % 3, 1, 1 + (index + 1) % 3]}
        else:
            parameters = {"passthrough": bool(index % 2), "cv": 3}
        return self._candidate(
            "robustness and ensemble stress test",
            index,
            family,
            "scaled_numeric",
            parameters,
            task_type,
        )

    @staticmethod
    def _candidate(
        mechanism: str,
        index: int,
        family: str,
        strategy: str,
        parameters: dict[str, Any],
        task_type: str,
    ) -> CandidateSpec:
        return CandidateSpec(
            hypothesis=(
                f"{mechanism} candidate {index + 1} uses {family} with {strategy} for "
                f"{task_type}; parameters {parameters} should improve the frozen validation metric."
            ),
            model_family=family,
            feature_strategy=strategy,
            parameters=parameters,
        )
