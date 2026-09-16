from swarmfold.adapters.base import BaseModelAdapter
from swarmfold.adapters.catboost_adapter import CatBoostAdapter
from swarmfold.adapters.lightgbm_adapter import LightGBMAdapter
from swarmfold.adapters.xgboost_adapter import XGBoostAdapter

_ADAPTER_REGISTRY: dict[str, type[BaseModelAdapter]] = {
    "lightgbm": LightGBMAdapter,
    "xgboost": XGBoostAdapter,
    "catboost": CatBoostAdapter,
}


def get_adapter(model_family: str) -> BaseModelAdapter:
    adapter_cls = _ADAPTER_REGISTRY.get(model_family)
    if adapter_cls is None:
        raise ValueError(f"No adapter registered for model family '{model_family}'.")
    return adapter_cls()


def is_adapter_available(model_family: str) -> bool:
    if model_family not in _ADAPTER_REGISTRY:
        return False
    return get_adapter(model_family).is_available()


__all__ = [
    "BaseModelAdapter",
    "LightGBMAdapter",
    "XGBoostAdapter",
    "CatBoostAdapter",
    "get_adapter",
    "is_adapter_available",
]
