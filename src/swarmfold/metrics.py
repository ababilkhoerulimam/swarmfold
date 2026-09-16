from dataclasses import dataclass
from typing import Literal

MetricDirection = Literal["maximize", "minimize"]
PredictionKind = Literal["label", "binary_probability", "continuous"]
TaskType = Literal[
    "binary_classification",
    "multiclass_classification",
    "regression",
    "multilabel_classification",
    "time_series_forecasting",
]


@dataclass(frozen=True)
class MetricSpec:
    name: str
    version: str
    direction: MetricDirection
    prediction_kind: PredictionKind
    task_types: frozenset[TaskType]
    minimum: float | None
    maximum: float | None

    @property
    def higher_is_better(self) -> bool:
        return self.direction == "maximize"

    def validate_task(self, task_type: str) -> None:
        if task_type not in self.task_types:
            supported = ", ".join(sorted(self.task_types))
            raise ValueError(
                f"Metric '{self.name}' does not support task '{task_type}'; "
                f"supported tasks: {supported}"
            )

    def is_better(self, candidate: float, incumbent: float, *, tolerance: float = 1e-12) -> bool:
        if self.higher_is_better:
            return candidate > incumbent + tolerance
        return candidate < incumbent - tolerance

    def improvement(self, candidate: float, incumbent: float) -> float:
        delta = candidate - incumbent
        return delta if self.higher_is_better else -delta

    def best(self, scores: list[float]) -> float:
        if not scores:
            raise ValueError("At least one score is required")
        return max(scores) if self.higher_is_better else min(scores)


_CLASSIFICATION = frozenset({"binary_classification", "multiclass_classification"})
_REGRESSION_LIKE = frozenset({"regression", "time_series_forecasting"})

METRIC_SPECS: dict[str, MetricSpec] = {
    "accuracy": MetricSpec(
        name="accuracy",
        version="accuracy-v1",
        direction="maximize",
        prediction_kind="label",
        task_types=_CLASSIFICATION,
        minimum=0.0,
        maximum=1.0,
    ),
    "roc_auc": MetricSpec(
        name="roc_auc",
        version="roc-auc-v1",
        direction="maximize",
        prediction_kind="binary_probability",
        task_types=frozenset({"binary_classification"}),
        minimum=0.0,
        maximum=1.0,
    ),
    "log_loss": MetricSpec(
        name="log_loss",
        version="log-loss-v1",
        direction="minimize",
        prediction_kind="binary_probability",
        task_types=frozenset({"binary_classification"}),
        minimum=0.0,
        maximum=None,
    ),
    "f1_macro": MetricSpec(
        name="f1_macro",
        version="f1-macro-v1",
        direction="maximize",
        prediction_kind="label",
        task_types=_CLASSIFICATION,
        minimum=0.0,
        maximum=1.0,
    ),
    "multilabel_f1": MetricSpec(
        name="multilabel_f1",
        version="multilabel-f1-v1",
        direction="maximize",
        prediction_kind="label",
        task_types=frozenset({"multilabel_classification"}),
        minimum=0.0,
        maximum=1.0,
    ),
    "rmse": MetricSpec(
        name="rmse",
        version="rmse-v1",
        direction="minimize",
        prediction_kind="continuous",
        task_types=_REGRESSION_LIKE,
        minimum=0.0,
        maximum=None,
    ),
    "mae": MetricSpec(
        name="mae",
        version="mae-v1",
        direction="minimize",
        prediction_kind="continuous",
        task_types=_REGRESSION_LIKE,
        minimum=0.0,
        maximum=None,
    ),
    "r2": MetricSpec(
        name="r2",
        version="r2-v1",
        direction="maximize",
        prediction_kind="continuous",
        task_types=_REGRESSION_LIKE,
        minimum=None,
        maximum=1.0,
    ),
}


def get_metric_spec(metric_name: str) -> MetricSpec:
    try:
        return METRIC_SPECS[metric_name]
    except KeyError as exc:
        supported = ", ".join(sorted(METRIC_SPECS))
        raise ValueError(
            f"Unsupported metric '{metric_name}'; supported metrics: {supported}"
        ) from exc


def default_metric_for_task(task_type: str) -> str:
    if task_type in {"regression", "time_series_forecasting"}:
        return "rmse"
    if task_type == "multilabel_classification":
        return "multilabel_f1"
    return "accuracy"
