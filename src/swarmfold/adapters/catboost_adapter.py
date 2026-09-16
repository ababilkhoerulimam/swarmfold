from importlib.util import find_spec
from typing import Any

from swarmfold.adapters.base import BaseModelAdapter


class CatBoostAdapter(BaseModelAdapter):
    def is_available(self) -> bool:
        return find_spec("catboost") is not None

    def build_estimator(
        self,
        task_type: str = "binary_classification",
        hyperparameters: dict[str, Any] | None = None,
        random_state: int = 42,
    ) -> Any:
        if not self.is_available():
            raise RuntimeError("catboost package is not installed.")

        import catboost as cb

        params = dict(hyperparameters or {})
        iterations = int(params.get("iterations", params.get("n_estimators", 100)))
        learning_rate = float(params.get("learning_rate", 0.05))
        depth = int(params.get("depth", params.get("max_depth", 6)))
        l2_leaf_reg = float(params.get("l2_leaf_reg", params.get("reg_lambda", 3.0)))

        common_kwargs = {
            "iterations": iterations,
            "learning_rate": learning_rate,
            "depth": depth,
            "l2_leaf_reg": l2_leaf_reg,
            "random_seed": random_state,
            "thread_count": 1,
            "verbose": 0,
        }

        if task_type == "regression":
            return cb.CatBoostRegressor(**common_kwargs)
        return cb.CatBoostClassifier(**common_kwargs)
