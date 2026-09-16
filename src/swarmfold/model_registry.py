from importlib.util import find_spec
from typing import Literal

CapabilityState = Literal["DECLARED", "INSTALLED", "EXECUTABLE"]

# Current executable model families with built-in runtime execution adapters
EXECUTABLE_MODEL_FAMILIES: frozenset[str] = frozenset(
    {
        "logistic_regression",
        "random_forest",
        "hist_gradient_boosting",
        "extra_trees",
        "sgd_log_loss",
        "bagging",
        "soft_voting",
        "stacking",
        "lightgbm",
        "xgboost",
        "catboost",
    }
)


def model_capability_registry() -> list[dict[str, object]]:
    entries = [
        {
            "model_family": "logistic_regression",
            "estimator": "sklearn.linear_model.LogisticRegression",
            "backend": "scikit-learn",
            "declared": True,
            "installed": True,
            "executable": "logistic_regression" in EXECUTABLE_MODEL_FAMILIES,
            "strengths": ["fast", "stable baseline", "probability output"],
            "limitations": ["limited nonlinear interactions", "one-hot expansion"],
            "native_categorical": False,
            "recommended_lanes": ["ai-a", "ai-c", "ai-e"],
        },
        {
            "model_family": "random_forest",
            "estimator": "sklearn.ensemble.RandomForestClassifier",
            "backend": "scikit-learn",
            "declared": True,
            "installed": True,
            "executable": "random_forest" in EXECUTABLE_MODEL_FAMILIES,
            "strengths": ["nonlinear interactions", "robust defaults"],
            "limitations": ["memory use", "weak extrapolation"],
            "native_categorical": False,
            "recommended_lanes": ["ai-a", "ai-b", "ai-e"],
        },
        {
            "model_family": "hist_gradient_boosting",
            "estimator": "sklearn.ensemble.HistGradientBoostingClassifier",
            "backend": "scikit-learn",
            "declared": True,
            "installed": True,
            "executable": "hist_gradient_boosting" in EXECUTABLE_MODEL_FAMILIES,
            "strengths": ["efficient boosting", "nonlinear fit", "regularization"],
            "limitations": ["categorical encoding required"],
            "native_categorical": False,
            "recommended_lanes": ["ai-a", "ai-b", "ai-c", "ai-e"],
        },
        {
            "model_family": "extra_trees",
            "estimator": "sklearn.ensemble.ExtraTreesClassifier",
            "backend": "scikit-learn",
            "declared": True,
            "installed": True,
            "executable": "extra_trees" in EXECUTABLE_MODEL_FAMILIES,
            "strengths": ["prediction diversity", "nonlinear interactions"],
            "limitations": ["larger artifacts", "memory use"],
            "native_categorical": False,
            "recommended_lanes": ["ai-d"],
        },
        {
            "model_family": "sgd_log_loss",
            "estimator": "sklearn.linear_model.SGDClassifier(loss='log_loss')",
            "backend": "scikit-learn",
            "declared": True,
            "installed": True,
            "executable": "sgd_log_loss" in EXECUTABLE_MODEL_FAMILIES,
            "strengths": ["large sparse data", "fast regularized linear search"],
            "limitations": ["scale sensitivity", "optimization variance"],
            "native_categorical": False,
            "recommended_lanes": ["ai-d", "ai-e"],
        },
        {
            "model_family": "bagging",
            "estimator": "sklearn.ensemble.BaggingClassifier(DecisionTreeClassifier)",
            "backend": "scikit-learn",
            "declared": True,
            "installed": True,
            "executable": "bagging" in EXECUTABLE_MODEL_FAMILIES,
            "strengths": ["variance reduction", "nonlinear diversity"],
            "limitations": ["higher training cost", "correlated members can limit gains"],
            "native_categorical": False,
            "recommended_lanes": ["ai-d"],
        },
        {
            "model_family": "soft_voting",
            "estimator": "sklearn.ensemble.VotingClassifier(voting='soft')",
            "backend": "scikit-learn",
            "declared": True,
            "installed": True,
            "executable": "soft_voting" in EXECUTABLE_MODEL_FAMILIES,
            "strengths": ["family averaging", "variance reduction"],
            "limitations": ["requires credible members", "higher training cost"],
            "native_categorical": False,
            "recommended_lanes": ["ai-a", "ai-e"],
        },
        {
            "model_family": "stacking",
            "estimator": "sklearn.ensemble.StackingClassifier(LogisticRegression meta-learner)",
            "backend": "scikit-learn",
            "declared": True,
            "installed": True,
            "executable": "stacking" in EXECUTABLE_MODEL_FAMILIES,
            "strengths": ["learned meta-combination", "family diversity"],
            "limitations": ["expensive nested fitting", "requires leakage-safe folds"],
            "native_categorical": False,
            "recommended_lanes": ["ai-a", "ai-b", "ai-c", "ai-d", "ai-e"],
        },
        {
            "model_family": "catboost",
            "estimator": "catboost.CatBoostClassifier",
            "backend": "catboost",
            "declared": True,
            "installed": find_spec("catboost") is not None,
            "executable": "catboost" in EXECUTABLE_MODEL_FAMILIES,
            "strengths": ["native categorical", "strong mixed tabular baseline"],
            "limitations": ["optional dependency", "larger artifacts"],
            "native_categorical": True,
            "recommended_lanes": ["ai-b"],
        },
        {
            "model_family": "lightgbm",
            "estimator": "lightgbm.LGBMClassifier",
            "backend": "lightgbm",
            "declared": True,
            "installed": find_spec("lightgbm") is not None,
            "executable": "lightgbm" in EXECUTABLE_MODEL_FAMILIES,
            "strengths": ["fast large-scale boosting", "memory efficiency"],
            "limitations": ["optional dependency", "leaf complexity can overfit"],
            "native_categorical": True,
            "recommended_lanes": ["ai-b"],
        },
        {
            "model_family": "xgboost",
            "estimator": "xgboost.XGBClassifier",
            "backend": "xgboost",
            "declared": True,
            "installed": find_spec("xgboost") is not None,
            "executable": "xgboost" in EXECUTABLE_MODEL_FAMILIES,
            "strengths": ["regularized boosting", "flexible objectives"],
            "limitations": ["optional dependency", "more tuning decisions"],
            "native_categorical": True,
            "recommended_lanes": ["ai-b", "ai-d"],
        },
    ]
    for item in entries:
        item["task_types"] = [
            "binary_classification",
            "multiclass_classification",
            "regression",
        ]
        if item["executable"] and item["installed"]:
            item["capability_state"] = "EXECUTABLE"
        elif item["installed"]:
            item["capability_state"] = "INSTALLED"
        else:
            item["capability_state"] = "DECLARED"
    return entries


def is_model_executable(model_family: str) -> bool:
    for entry in model_capability_registry():
        if entry["model_family"] == model_family:
            return bool(entry["executable"] and entry["installed"])
    return False


def is_model_installed(model_family: str) -> bool:
    for entry in model_capability_registry():
        if entry["model_family"] == model_family:
            return bool(entry["installed"])
    return False


def is_model_declared(model_family: str) -> bool:
    return any(entry["model_family"] == model_family for entry in model_capability_registry())


def executable_model_families() -> list[str]:
    return [
        str(entry["model_family"])
        for entry in model_capability_registry()
        if entry["executable"] and entry["installed"]
    ]


def validate_proposal_executability(model_family: str) -> None:
    if not is_model_executable(model_family):
        installed = is_model_installed(model_family)
        state_name = "INSTALLED" if installed else "DECLARED"
        raise ValueError(
            f"Model family '{model_family}' is {state_name} but not EXECUTABLE. "
            f"Only models with an active runtime adapter may be executed."
        )
