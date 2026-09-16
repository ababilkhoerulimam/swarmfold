from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class TechniqueCard(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    card_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]+$")
    title: str = Field(min_length=2)
    category: str
    model_family: str | None = None
    canonical_role: str = Field(default="model_engineer", min_length=2)
    secondary_roles: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    content_hash: str = Field(default="")
    task_types: list[str] = Field(
        default_factory=lambda: [
            "binary_classification",
            "multiclass_classification",
            "regression",
        ]
    )
    prerequisites: list[str] = Field(default_factory=list)
    leakage_risk: Literal["NONE", "LOW", "MEDIUM", "HIGH"] = "LOW"
    fold_boundary_rule: str
    recommended_parameters: dict[str, Any] = Field(default_factory=dict)
    strengths: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    rationale: str

    # 20-Field Extended Specifications
    definition: str = ""
    when_to_use: str = ""
    when_not_to_use: str = ""
    assumptions: str = ""
    preprocessing: str = ""
    dataset_scale: str = ""
    metric_compatibility: str = ""
    key_hyperparameters: str = ""
    initial_parameter_ranges: str = ""
    compute_profile: str = ""
    failure_modes: str = ""
    comparison_baselines: str = ""
    ensemble_compatibility: str = ""
    exact_package: str = ""
    example_hypothesis: str = ""
    acceptance_criteria: str = ""
    rejection_criteria: str = ""
    source: str = ""
    evidence_status: str = "V"


DEFAULT_PREREQUISITES: dict[str, list[str]] = {
    "catboost_categorical_boosting": ["tabular", "categorical_features", "high_cardinality"],
    "kfold_out_of_fold_target_encoding": [
        "tabular", "high_cardinality_categorical", "categorical_features",
    ],
    "purged_walk_forward_temporal_split": ["temporal_ordering", "time_series"],
    "lightgbm_tabular_boosting": ["tabular", "numeric_or_categorical"],
    "xgboost_regularized_boosting": ["tabular"],
    "random_forest_robust_nonlinear": ["tabular"],
    "logistic_regression_regularized": ["scaled_numeric"],
    "caruana_greedy_ensemble_selection": ["diverse_oof_predictions"],
}

# Default parameter profiles for key models & algorithms
RECOMMENDED_PARAM_DEFAULTS: dict[str, dict[str, Any]] = {
    "lightgbm_tabular_boosting": {
        "n_estimators": 100,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
    "xgboost_regularized_boosting": {
        "n_estimators": 100,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
    },
    "catboost_categorical_boosting": {
        "iterations": 150,
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 3.0,
    },
    "random_forest_robust_nonlinear": {
        "n_estimators": 100,
        "max_depth": 8,
        "min_samples_leaf": 2,
    },
    "logistic_regression_regularized": {
        "c": 1.0,
        "max_iter": 1000,
    },
    "caruana_greedy_ensemble_selection": {
        "iterations": 50,
        "replacement": True,
        "fraction": 0.8,
    },
    "kfold_out_of_fold_target_encoding": {
        "smoothing": 10.0,
        "inner_splits": 5,
    },
    "purged_walk_forward_temporal_split": {
        "n_splits": 5,
        "purge_window_days": 7,
    },
}

DEFAULT_STRENGTHS: dict[str, list[str]] = {
    "lightgbm_tabular_boosting": [
        "Fast histogram training",
        "Native missing value handling",
        "State-of-the-art tabular accuracy",
    ],
    "xgboost_regularized_boosting": [
        "L1/L2 leaf regularization",
        "Proven generalization on dense tabular problems",
    ],
    "catboost_categorical_boosting": [
        "Native target statistics without leakage",
        "Minimal hyperparameter tuning required",
    ],
    "caruana_greedy_ensemble_selection": [
        "Monotonically improves validation metric",
        "Prunes redundant models",
    ],
    "kfold_out_of_fold_target_encoding": [
        "Replaces high-dimensional one-hot vectors with compact numerical representations",
    ],
}


def load_technique_corpus() -> list[TechniqueCard]:
    compiled_path = Path(__file__).resolve().parent / "data" / "compiled_technique_corpus.json"
    if not compiled_path.exists():
        try:
            from swarmfold.knowledge_parser import compile_corpus_to_json
            compile_corpus_to_json(compiled_path)
        except Exception:
            pass

    if compiled_path.exists():
        try:
            with open(compiled_path, encoding="utf-8") as f:
                raw_cards = json.load(f)

            cards: list[TechniqueCard] = []
            for item in raw_cards:
                cid = item.get("card_id", "")
                # Enhance default params & prerequisites if available
                if cid in DEFAULT_PREREQUISITES:
                    existing_prereqs = item.get("prerequisites", [])
                    merged_prereqs = existing_prereqs + DEFAULT_PREREQUISITES[cid]
                    item["prerequisites"] = list(dict.fromkeys(merged_prereqs))
                if cid in RECOMMENDED_PARAM_DEFAULTS and not item.get("recommended_parameters"):
                    item["recommended_parameters"] = RECOMMENDED_PARAM_DEFAULTS[cid]
                if cid in DEFAULT_STRENGTHS and not item.get("strengths"):
                    item["strengths"] = DEFAULT_STRENGTHS[cid]
                cards.append(TechniqueCard(**item))
            return cards
        except Exception as err:
            print(f"Warning: Failed to load compiled corpus from {compiled_path}: {err}")

    return [
        TechniqueCard(
            card_id="lightgbm_tabular_boosting",
            title="LightGBM Histogram-based Gradient Boosting",
            category="model",
            model_family="lightgbm",
            canonical_role="model_engineer",
            task_types=["binary_classification", "multiclass_classification", "regression"],
            prerequisites=["tabular"],
            leakage_risk="LOW",
            fold_boundary_rule=(
                "Fit exclusively on training fold; compute OOF predictions on validation fold."
            ),
            recommended_parameters=RECOMMENDED_PARAM_DEFAULTS["lightgbm_tabular_boosting"],
            strengths=DEFAULT_STRENGTHS["lightgbm_tabular_boosting"],
            rationale="Histogram gradient boosting.",
        )
    ]


TECHNIQUE_CORPUS: list[TechniqueCard] = load_technique_corpus()
