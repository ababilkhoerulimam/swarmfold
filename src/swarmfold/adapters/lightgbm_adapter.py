from importlib.util import find_spec
from typing import Any

from swarmfold.adapters.base import BaseModelAdapter


class LightGBMAdapter(BaseModelAdapter):
    def is_available(self) -> bool:
        return find_spec("lightgbm") is not None

    def build_estimator(
        self,
        task_type: str = "binary_classification",
        hyperparameters: dict[str, Any] | None = None,
        random_state: int = 42,
    ) -> Any:
        if not self.is_available():
            raise RuntimeError("lightgbm package is not installed.")

        import lightgbm as lgb

        params = dict(hyperparameters or {})
        n_estimators = int(params.get("n_estimators", 100))
        learning_rate = float(params.get("learning_rate", 0.05))
        max_depth = int(params.get("max_depth", -1))
        num_leaves = int(params.get("num_leaves", 31))
        subsample = float(params.get("subsample", 1.0))
        colsample_bytree = float(params.get("colsample_bytree", 1.0))
        reg_alpha = float(params.get("reg_alpha", 0.0))
        reg_lambda = float(params.get("reg_lambda", 0.0))

        common_kwargs = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "num_leaves": num_leaves,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "random_state": random_state,
            "n_jobs": 1,
            "verbose": -1,
        }

        if task_type == "regression":
            return lgb.LGBMRegressor(**common_kwargs)
        return lgb.LGBMClassifier(**common_kwargs)
