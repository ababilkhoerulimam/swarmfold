from importlib.util import find_spec
from typing import Any

from swarmfold.adapters.base import BaseModelAdapter


class XGBoostAdapter(BaseModelAdapter):
    def is_available(self) -> bool:
        return find_spec("xgboost") is not None

    def build_estimator(
        self,
        task_type: str = "binary_classification",
        hyperparameters: dict[str, Any] | None = None,
        random_state: int = 42,
    ) -> Any:
        if not self.is_available():
            raise RuntimeError("xgboost package is not installed.")

        import xgboost as xgb

        params = dict(hyperparameters or {})
        n_estimators = int(params.get("n_estimators", 100))
        learning_rate = float(params.get("learning_rate", 0.05))
        max_depth = int(params.get("max_depth", 6))
        subsample = float(params.get("subsample", 1.0))
        colsample_bytree = float(params.get("colsample_bytree", 1.0))
        reg_alpha = float(params.get("reg_alpha", 0.0))
        reg_lambda = float(params.get("reg_lambda", 1.0))

        common_kwargs = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "random_state": random_state,
            "n_jobs": 1,
            "verbosity": 0,
        }

        if task_type == "regression":
            return xgb.XGBRegressor(**common_kwargs)
        if task_type == "multiclass_classification":
            return xgb.XGBClassifier(**common_kwargs, eval_metric="mlogloss")
        return xgb.XGBClassifier(**common_kwargs, eval_metric="logloss")
