from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class FeatureAblationReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    strategy_name: str
    baseline_strategy: str = "raw_numeric"
    input_feature_count: int = 0
    output_feature_count: int = 0
    feature_count_delta: int = 0
    memory_bytes_before: int = 0
    memory_bytes_after: int = 0
    memory_growth_ratio: float = 1.0
    train_time_seconds: float = 0.0
    baseline_metric_score: float | None = None
    candidate_metric_score: float | None = None
    metric_delta: float | None = None
    higher_is_better: bool = True
    is_worthwhile: bool = True
    rationale: str = ""


def compute_feature_ablation(
    strategy_name: str,
    baseline_strategy: str = "raw_numeric",
    *,
    input_feature_count: int,
    output_feature_count: int,
    memory_bytes_before: int,
    memory_bytes_after: int,
    train_time_seconds: float = 0.0,
    baseline_metric_score: float | None = None,
    candidate_metric_score: float | None = None,
    higher_is_better: bool = True,
) -> FeatureAblationReport:
    feature_delta = output_feature_count - input_feature_count
    growth_ratio = (
        float(memory_bytes_after) / max(float(memory_bytes_before), 1.0)
        if memory_bytes_before > 0
        else 1.0
    )

    metric_delta = None
    if baseline_metric_score is not None and candidate_metric_score is not None:
        metric_delta = candidate_metric_score - baseline_metric_score

    # Determine if worthwhile
    is_worthwhile = True
    reasons = []

    if metric_delta is not None:
        improved = (metric_delta > 0) if higher_is_better else (metric_delta < 0)
        if not improved:
            is_worthwhile = False
            reasons.append(f"Validation metric degraded by {abs(metric_delta):.4f}")
        else:
            reasons.append(f"Validation metric improved by {abs(metric_delta):.4f}")
    else:
        reasons.append("Metric comparison pending")

    if growth_ratio > 10.0:
        is_worthwhile = False
        reasons.append(f"Excessive memory growth ratio of {growth_ratio:.1f}x")

    return FeatureAblationReport(
        strategy_name=strategy_name,
        baseline_strategy=baseline_strategy,
        input_feature_count=input_feature_count,
        output_feature_count=output_feature_count,
        feature_count_delta=feature_delta,
        memory_bytes_before=memory_bytes_before,
        memory_bytes_after=memory_bytes_after,
        memory_growth_ratio=round(growth_ratio, 3),
        train_time_seconds=round(train_time_seconds, 3),
        baseline_metric_score=baseline_metric_score,
        candidate_metric_score=candidate_metric_score,
        metric_delta=metric_delta,
        higher_is_better=higher_is_better,
        is_worthwhile=is_worthwhile,
        rationale="; ".join(reasons),
    )
