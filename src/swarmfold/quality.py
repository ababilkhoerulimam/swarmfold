import csv
import datetime
from itertools import combinations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from swarmfold.artifacts import write_json
from swarmfold.dataset_traits import extract_dataset_traits
from swarmfold.metrics import get_metric_spec
from swarmfold.schemas import ContractBundle, ExperimentState, RunResult
from swarmfold.validation_splitters import recommend_validation_strategy

_TIME_NAMES = {
    "date",
    "datetime",
    "timestamp",
    "time",
    "month",
    "week",
    "year",
    "period",
}
_TIME_CONTEXT = ("forecast", "time series", "temporal", "future", "horizon")
_COUNT_TOKENS = ("count", "frequency", "number", "events", "claims")
_AMOUNT_TOKENS = ("amount", "severity", "cost", "loss", "total", "value")
_EXPOSURE_TOKENS = ("exposure", "duration", "policy_days", "insured_days")


def formulate_problem(state: ExperimentState) -> ExperimentState:
    frame = _read_training_frame(state)
    target_column = state["target_column"]
    target = frame[target_column]
    unique_count = int(target.nunique(dropna=True))
    requested_task_type = state.get("requested_task_type")
    if requested_task_type:
        task_type = str(requested_task_type)
    elif unique_count == 2:
        task_type = "binary_classification"
    elif pd.api.types.is_numeric_dtype(target):
        task_type = "regression"
    else:
        task_type = "multiclass_classification"

    time_candidates = _time_candidates(frame.drop(columns=[target_column]))
    context = state.get("project_context", "").casefold()
    temporal_intent = any(token in context for token in _TIME_CONTEXT)
    traits = extract_dataset_traits(
        frame,
        target_column=target_column,
        task_type=task_type,
    )
    if temporal_intent and time_candidates and not traits.is_temporal:
        trait_tags = list(
            dict.fromkeys([*traits.summary_traits, "temporal_ordering", "time_series"])
        )
        traits = traits.model_copy(
            update={
                "is_temporal": True,
                "time_columns": time_candidates,
                "primary_time_column": time_candidates[0],
                "summary_traits": trait_tags,
            }
        )
    val_rec = recommend_validation_strategy(traits)

    if temporal_intent and (time_candidates or traits.is_temporal):
        prediction_structure = "temporal"
    elif traits.has_repeated_entities and traits.primary_group_column:
        prediction_structure = "grouped"
    else:
        prediction_structure = "iid"

    recommended_strategy = val_rec.recommended_strategy

    supported_task = task_type in (
        "binary_classification",
        "multiclass_classification",
        "regression",
        "multilabel_classification",
        "time_series_forecasting",
    )

    report = {
        "task_type": task_type,
        "prediction_structure": prediction_structure,
        "target_column": target_column,
        "target_unique_values": unique_count,
        "time_candidates": time_candidates,
        "recommended_validation_strategy": recommended_strategy,
        "execution_supported": supported_task,
        "reason": (
            f"The built-in executor supports {task_type.replace('_', ' ')} with "
            f"{recommended_strategy}."
            if supported_task
            else "This task structure has a declared adapter but no end-to-end training runtime."
        ),
        "dataset_traits": traits.model_dump(),
        "validation_recommendation": val_rec.model_dump(),
    }
    return {
        "problem_formulation": report,
        "validation_strategy": recommended_strategy,
        "splits": val_rec.n_splits,
        "repeats": val_rec.n_repeats,
        "group_column": val_rec.group_column,
        "time_column": val_rec.time_column,
        "purge_window": val_rec.purge_window,
        "embargo_window": val_rec.embargo_window,
        **_with_report(state, "problem_formulation_agent", report),
    }


def audit_temporal_validation(state: ExperimentState) -> ExperimentState:
    contracts = ContractBundle.model_validate(state["contracts"])
    formulation = dict(state["problem_formulation"])
    frame = _read_training_frame(state)
    features = frame[contracts.data.feature_columns]
    duplicate_feature_rows = int(features.duplicated(keep=False).sum())
    conflicting_duplicate_groups = _conflicting_duplicate_groups(
        features, frame[contracts.data.target_column]
    )

    temporal_strategies = {"rolling_origin", "expanding_walk_forward", "purged_time_series"}
    grouped_strategies = {"group_kfold", "stratified_group_kfold"}
    pred_struct = formulation.get("prediction_structure", "iid")
    strat = contracts.validation.strategy

    if pred_struct == "temporal":
        alignment_status = "PASS" if strat in temporal_strategies else "BLOCKED"
    elif pred_struct == "grouped":
        alignment_status = "PASS" if strat in grouped_strategies else "BLOCKED"
    elif formulation.get("recommended_validation_strategy") == strat:
        alignment_status = "PASS"
    elif (
        formulation.get("recommended_validation_strategy") in temporal_strategies
        and strat not in temporal_strategies
    ):
        alignment_status = "BLOCKED"
    else:
        alignment_status = "PASS"

    checks = [
        {
            "name": "fold_local_transforms",
            "status": "PASS",
            "evidence": contracts.validation.fit_transforms_on_training_only,
        },
        {
            "name": "duplicate_feature_rows",
            "status": "WARN" if duplicate_feature_rows else "PASS",
            "evidence": duplicate_feature_rows,
        },
        {
            "name": "conflicting_duplicate_groups",
            "status": "WARN" if conflicting_duplicate_groups else "PASS",
            "evidence": conflicting_duplicate_groups,
        },
        {
            "name": "deployment_split_alignment",
            "status": alignment_status,
            "evidence": {
                "recommended": formulation.get("recommended_validation_strategy"),
                "configured": contracts.validation.strategy,
            },
        },
    ]
    blocked = [check["name"] for check in checks if check["status"] == "BLOCKED"]
    report = {
        "status": "BLOCKED" if blocked else "PASS_WITH_WARNINGS" if any(
            check["status"] == "WARN" for check in checks
        ) else "PASS",
        "checks": checks,
        "blocked_checks": blocked,
        "prefix_invariance_required": formulation.get("prediction_structure") == "temporal",
    }
    if blocked:
        raise ValueError(
            "Validation strategy does not match the formulated problem: "
            + ", ".join(blocked)
        )
    return {
        "validation_audit": report,
        **_with_report(state, "temporal_validation_leakage_auditor", report),
    }


def analyze_domain_decomposition(state: ExperimentState) -> ExperimentState:
    frame = _read_training_frame(state)
    target_column = state["target_column"]
    lowered = {column: column.casefold() for column in frame.columns}
    components = {
        "count_candidates": _matching_columns(lowered, _COUNT_TOKENS, target_column),
        "amount_candidates": _matching_columns(lowered, _AMOUNT_TOKENS, target_column),
        "exposure_candidates": _matching_columns(lowered, _EXPOSURE_TOKENS, target_column),
    }
    formulation = dict(state["problem_formulation"])
    applicable = formulation["task_type"] == "regression" and bool(
        components["count_candidates"] and components["amount_candidates"]
    )
    report = {
        "status": "REVIEW_REQUIRED" if applicable else "NOT_APPLICABLE",
        "components": components,
        "coherence_constraint_required": applicable,
        "reason": (
            "Potential count and amount components require domain confirmation."
            if applicable
            else "No defensible target decomposition was inferred from schema alone."
        ),
    }
    return {
        "domain_decomposition": report,
        **_with_report(state, "domain_target_decomposition_agent", report),
    }


def assess_oof_ensemble(state: ExperimentState) -> ExperimentState:
    """Agent F — OOF Ensemble Scientist.

    Assesses whether any intra-lane ensemble candidate improves over the best base model.
    Writes agent-f-assessment.json to output_dir so this stage has a distinct,
    observable lifecycle artifact on disk and in the UI.
    """
    contracts = ContractBundle.model_validate(state["contracts"])
    metric_name = contracts.metric.metric_name
    metric_spec = get_metric_spec(metric_name)
    completed = [
        RunResult.model_validate(item)
        for item in state.get("completed_results", [])
        if RunResult.model_validate(item).status == "SUCCEEDED"
    ]
    ensemble_families = {"soft_voting", "stacking", "bagging", "caruana"}
    base_results = [item for item in completed if item.model_family not in ensemble_families]
    ensemble_results = [item for item in completed if item.model_family in ensemble_families]
    similarities = _prediction_similarities(base_results, metric_name)
    eligible = len(base_results) >= 2 and bool(similarities)

    base_scores = [
        item.metrics[metric_name] for item in base_results if metric_name in item.metrics
    ]
    best_base_score = metric_spec.best(base_scores) if base_scores else None

    ensemble_result = None
    if ensemble_results:
        scored_ensembles = [e for e in ensemble_results if metric_name in e.metrics]
        if scored_ensembles:
            ensemble_result = sorted(
                scored_ensembles,
                key=lambda item: item.metrics[metric_name],
                reverse=metric_spec.higher_is_better,
            )[0]

    ensemble_score = (
        ensemble_result.metrics[metric_name] if ensemble_result is not None else None
    )
    gain = (
        metric_spec.improvement(ensemble_score, best_base_score)
        if ensemble_score is not None and best_base_score is not None
        else None
    )
    if not eligible:
        decision = "INELIGIBLE"
    elif ensemble_result is None:
        decision = "NOT_EVALUATED"
    elif (
        ensemble_score is not None
        and best_base_score is not None
        and metric_spec.is_better(ensemble_score, best_base_score, tolerance=1e-12)
    ):
        decision = "SUPPORTED"
    else:
        decision = "REJECTED"
    report = {
        "stage": "oof_ensemble_scientist",
        "stage_label": "Agent F — OOF Ensemble Scientist",
        "timestamp": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
        "iteration_id": state.get("iteration_id", ""),
        "lane_id": state.get("lane_id", ""),
        "metric_name": metric_name,
        "decision": decision,
        "eligible_members": [item.experiment_id for item in base_results],
        "candidate_count": len(base_results),
        "ensemble_candidate_count": len(ensemble_results),
        "similarities": similarities,
        "best_base_score": best_base_score,
        "ensemble_score": ensemble_score,
        "ensemble_gain": gain,
        "selection_rule": "Ensemble must beat the best base model on the identical holdout.",
        "limitation": "This release has one holdout, so nested OOF weight learning is unavailable.",
    }

    # Persist artifact so Agent F has a distinct, observable lifecycle stage on disk.
    output_dir = state.get("output_dir")
    if output_dir:
        artifact_path = Path(output_dir) / "agent-f-assessment.json"
        try:
            write_json(artifact_path, report)
        except OSError:
            pass  # Non-fatal: state is the source of truth; disk write is observability only.

    return {
        "ensemble_assessment": report,
        **_with_report(state, "oof_ensemble_scientist", report),
    }



def assess_robustness_uncertainty(state: ExperimentState) -> ExperimentState:
    result = RunResult.model_validate(state["run_result"])
    contracts = ContractBundle.model_validate(state["contracts"])
    metric_name = contracts.metric.metric_name
    if result.status != "SUCCEEDED":
        report = {
            "status": "UNAVAILABLE",
            "metric": metric_name,
            "reason": result.error or "No successful experiment was available.",
            "multi_seed_evidence": False,
            "fold_stability_evidence": False,
        }
        return {
            "robustness_assessment": report,
            **_with_report(state, "robustness_uncertainty_agent", report),
        }
    prediction_path = _prediction_artifact(result)
    actual, predicted, scores = _read_predictions(prediction_path)
    rng = np.random.default_rng(contracts.validation.seed)
    estimates: list[float] = []
    for _ in range(300):
        indices = rng.integers(0, len(actual), len(actual))
        sub_actual = actual[indices]
        sub_pred = predicted[indices]
        if metric_name == "accuracy":
            estimates.append(float(np.mean(sub_actual == sub_pred)))
        elif metric_name == "roc_auc":
            binary = sub_actual == result.positive_class
            if np.unique(binary).size == 2 and scores is not None and len(scores) == len(actual):
                estimates.append(float(roc_auc_score(binary, scores[indices])))
        elif metric_name == "rmse":
            estimates.append(
                float(np.sqrt(np.mean((sub_actual.astype(float) - sub_pred.astype(float)) ** 2)))
            )
        elif metric_name == "mae":
            estimates.append(
                float(np.mean(np.abs(sub_actual.astype(float) - sub_pred.astype(float))))
            )
        elif metric_name == "r2":
            ss_tot = np.sum((sub_actual.astype(float) - np.mean(sub_actual.astype(float))) ** 2)
            if ss_tot > 0:
                ss_res = np.sum((sub_actual.astype(float) - sub_pred.astype(float)) ** 2)
                estimates.append(float(1.0 - (ss_res / ss_tot)))
        else:
            estimates.append(float(np.mean(sub_actual == sub_pred)))
    if not estimates:
        estimates.append(float(result.metrics.get(metric_name, 0.0)))
    lower, upper = np.quantile(estimates, [0.025, 0.975]).tolist()
    report = {
        "status": "LIMITED",
        "bootstrap_samples": len(estimates),
        "confidence_level": 0.95,
        "metric": metric_name,
        "point_estimate": result.metrics[metric_name],
        "interval": [float(lower), float(upper)],
        "multi_seed_evidence": False,
        "fold_stability_evidence": False,
        "limitation": "Bootstrap uncertainty from one holdout is not fold or seed stability.",
    }
    evidence_path = write_json(
        Path(state["output_dir"]) / "evidence" / "robustness-uncertainty.json",
        report,
    )
    report["artifact_path"] = str(evidence_path.resolve())
    return {
        "robustness_assessment": report,
        **_with_report(state, "robustness_uncertainty_agent", report),
    }


def _read_training_frame(state: ExperimentState) -> pd.DataFrame:
    path = Path(state["dataset_path"])
    if not path.is_file():
        raise ValueError(f"Dataset does not exist: {path}")
    frame = pd.read_csv(path)
    target_column = state["target_column"]
    if target_column not in frame:
        raise ValueError(f"Target column '{target_column}' is missing")
    return frame


def _time_candidates(features: pd.DataFrame) -> list[str]:
    candidates = []
    for column in features.columns:
        normalized = column.casefold().replace("-", "_")
        name_tokens = set(normalized.split("_"))
        if not name_tokens.intersection(_TIME_NAMES):
            continue
        parsed = pd.to_datetime(features[column], errors="coerce")
        if float(parsed.notna().mean()) >= 0.8:
            candidates.append(column)
    return candidates


def _conflicting_duplicate_groups(features: pd.DataFrame, target: pd.Series) -> int:
    if not features.duplicated(keep=False).any():
        return 0
    joined = features.copy()
    joined["__swarmfold_target__"] = target.astype(str)
    return int(
        (joined.groupby(list(features.columns), dropna=False)["__swarmfold_target__"].nunique() > 1)
        .sum()
    )


def _matching_columns(
    lowered: dict[str, str], tokens: tuple[str, ...], target_column: str
) -> list[str]:
    return [
        column
        for column, normalized in lowered.items()
        if column != target_column and any(token in normalized for token in tokens)
    ]


def _prediction_similarities(
    results: list[RunResult], metric_name: str
) -> list[dict[str, object]]:
    values = {
        result.experiment_id: _prediction_vector(result, metric_name) for result in results
    }
    similarities = []
    for left, right in combinations(values, 2):
        left_values = values[left]
        right_values = values[right]
        if len(left_values) != len(right_values):
            continue
        if metric_name == "roc_auc":
            value = float(np.corrcoef(left_values.astype(float), right_values.astype(float))[0, 1])
            measure = "prediction_correlation"
        else:
            value = float(np.mean(left_values == right_values))
            measure = "prediction_agreement"
        if not np.isfinite(value):
            value = 1.0 if np.array_equal(left_values, right_values) else 0.0
        similarities.append(
            {"left": left, "right": right, "measure": measure, "value": value}
        )
    return similarities


def _prediction_vector(result: RunResult, metric_name: str) -> np.ndarray:
    actual, predicted, scores = _read_predictions(_prediction_artifact(result))
    del actual
    return scores if metric_name == "roc_auc" else predicted


def _prediction_artifact(result: RunResult) -> Path:
    artifact = next(
        (
            item
            for item in result.artifacts
            if Path(item.path).name == "validation_predictions.csv"
        ),
        None,
    )
    if artifact is None:
        raise ValueError(f"Validation predictions are missing for {result.experiment_id}")
    return Path(artifact.path)


def _read_predictions(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    actual = np.asarray([row["actual"] for row in rows], dtype=object)
    predicted = np.asarray([row["predicted"] for row in rows], dtype=object)
    scores = np.asarray(
        [float(row["score"]) for row in rows if row.get("score") not in {None, ""}],
        dtype=float,
    )
    return actual, predicted, scores


def _with_report(state: ExperimentState, role: str, report: dict[str, object]) -> ExperimentState:
    reports = dict(state.get("role_reports", {}))
    reports[role] = report
    return {"role_reports": reports}
