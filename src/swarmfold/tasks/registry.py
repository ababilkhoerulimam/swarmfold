from __future__ import annotations

from swarmfold.tasks.base import TaskAdapter
from swarmfold.tasks.binary import BinaryClassificationAdapter
from swarmfold.tasks.multiclass import MulticlassClassificationAdapter
from swarmfold.tasks.multilabel import MultilabelClassificationAdapter
from swarmfold.tasks.regression import RegressionAdapter
from swarmfold.tasks.timeseries import TimeSeriesForecastingAdapter


class TaskAdapterRegistry:
    """Central registry of task adapters."""

    def __init__(self) -> None:
        self._adapters: dict[str, TaskAdapter] = {}
        self._register_defaults()

    def register(self, adapter: TaskAdapter) -> None:
        self._adapters[adapter.task_type] = adapter

    def get(self, task_type: str) -> TaskAdapter:
        if task_type not in self._adapters:
            supported = ", ".join(sorted(self._adapters.keys()))
            raise ValueError(
                f"Task type '{task_type}' is not registered; supported task types: {supported}"
            )
        return self._adapters[task_type]

    def is_supported(self, task_type: str) -> bool:
        return task_type in self._adapters

    def list_adapters(self) -> list[TaskAdapter]:
        return [self._adapters[k] for k in sorted(self._adapters.keys())]

    def _register_defaults(self) -> None:
        self.register(BinaryClassificationAdapter())
        self.register(MulticlassClassificationAdapter())
        self.register(RegressionAdapter())
        self.register(MultilabelClassificationAdapter())
        self.register(TimeSeriesForecastingAdapter())


_GLOBAL_TASK_REGISTRY: TaskAdapterRegistry | None = None


def get_task_registry() -> TaskAdapterRegistry:
    global _GLOBAL_TASK_REGISTRY
    if _GLOBAL_TASK_REGISTRY is None:
        _GLOBAL_TASK_REGISTRY = TaskAdapterRegistry()
    return _GLOBAL_TASK_REGISTRY


def get_task_adapter(task_type: str) -> TaskAdapter:
    return get_task_registry().get(task_type)
