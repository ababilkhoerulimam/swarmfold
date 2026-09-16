from swarmfold.tasks.base import TaskAdapter
from swarmfold.tasks.binary import BinaryClassificationAdapter
from swarmfold.tasks.multiclass import MulticlassClassificationAdapter
from swarmfold.tasks.multilabel import MultilabelClassificationAdapter
from swarmfold.tasks.registry import (
    TaskAdapterRegistry,
    get_task_adapter,
    get_task_registry,
)
from swarmfold.tasks.regression import RegressionAdapter
from swarmfold.tasks.timeseries import TimeSeriesForecastingAdapter

__all__ = [
    "BinaryClassificationAdapter",
    "MulticlassClassificationAdapter",
    "MultilabelClassificationAdapter",
    "RegressionAdapter",
    "TaskAdapter",
    "TaskAdapterRegistry",
    "TimeSeriesForecastingAdapter",
    "get_task_adapter",
    "get_task_registry",
]
