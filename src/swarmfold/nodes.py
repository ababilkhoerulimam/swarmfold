import csv
import math
import time
from pathlib import Path
from uuid import uuid4

import joblib
import mlflow
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.ensemble import (
    BaggingClassifier,
    BaggingRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge, SGDClassifier, SGDRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.metrics import (
    log_loss as sklearn_log_loss,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from swarmfold.artifacts import sha256_file, write_json
from swarmfold.competition_bundle import load_test_features
from swarmfold.contracts import contract_bundle_sha256
from swarmfold.data import load_training_csv
from swarmfold.features.executor import FeaturePipelineExecutor
from swarmfold.healing import (
    CheckpointManager,
    ResourcePolicy,
    SelfHealingDebugger,
    classify_failure,
)
from swarmfold.metrics import get_metric_spec
from swarmfold.model_registry import validate_proposal_executability
from swarmfold.optimization import HPOConfig, HPOController, TrialPruner
from swarmfold.planner import ProposalPlanner
from swarmfold.resources import LOCAL_TRAINING_POOL, ProcessResourceMeter
from swarmfold.schemas import (
    ArtifactRecord,
    ContractBundle,
    ExperimentProposal,
    ExperimentState,
    RunResult,
    VerificationResult,
)
from swarmfold.validation_splitters import get_validation_splitter


class TrialPrunedError(RuntimeError):
    def __init__(
        self,
        fold_scores: list[float],
        benchmark: float,
        reason: str | None = None,
    ) -> None:
        self.fold_scores = fold_scores
        self.benchmark = benchmark
        message = reason or (
            f"mean fold score was not competitive with benchmark {benchmark:.6g}"
        )
        super().__init__(message)


def _check_trial_pruning(
    proposal: ExperimentProposal,
    fold_scores: list[float],
    fold_index: int,
    hpo_controller: HPOController | None,
    benchmark_score: float | None,
    higher_is_better: bool,
) -> None:
    if hpo_controller is not None and proposal.hpo_trial_number is not None:
        decision = hpo_controller.report_fold(
            proposal.hpo_trial_number,
            fold_index,
            fold_scores[-1],
        )
        if decision.should_prune and decision.reference_median is not None:
            raise TrialPrunedError(
                fold_scores,
                decision.reference_median,
                decision.reason,
            )
        return

    pruner = TrialPruner(higher_is_better=higher_is_better)
    if pruner.should_prune(fold_scores, benchmark_score):
        raise TrialPrunedError(fold_scores, float(benchmark_score))


def create_plan_node(planner: ProposalPlanner):
    def plan_experiment(state: ExperimentState) -> ExperimentState:
        proposal = planner.propose(state)
        return {"proposal": proposal.model_dump(), "status": "PROPOSED"}

    return plan_experiment


def _execute_single_attempt(
    state: ExperimentState,
    proposal: ExperimentProposal,
    contracts: ContractBundle,
    run_id: str,
    run_directory: Path,
    output_root: Path,
    resource_meter: ProcessResourceMeter,
    resource_policy: ResourcePolicy,
    checkpoint_manager: CheckpointManager,
    benchmark_score: float | None,
    hpo_controller: HPOController | None,
    started_at: float,
) -> RunResult:
    resumed_from_checkpoint = False
    with LOCAL_TRAINING_POOL.slot(), resource_meter:
        resource_policy.check_cancellation()
        dataset_sha256 = sha256_file(proposal.dataset_path)
        task_type = contracts.competition.task_type
        features, target = load_training_csv(
            proposal.dataset_path, proposal.target_column, task_type
        )
        id_cols = getattr(contracts.competition, "id_columns", ["row"])
        id_col = next((col for col in id_cols if col in features.columns), None)
        if id_col is not None:
            features.index = features[id_col].astype(str)
            features = features.drop(columns=[id_col])

        strategy = getattr(contracts.validation, "strategy", "stratified_holdout")
        positive_class = (
            _positive_class(target) if task_type == "binary_classification" else None
        )
        metric_name = contracts.metric.metric_name
        metric_spec = get_metric_spec(metric_name)
        fold_scores: list[float] = []

        if strategy not in ("holdout", "stratified_holdout"):
            n_splits = getattr(contracts.validation, "splits", 5)
            n_repeats = getattr(contracts.validation, "repeats", 1)
            group_col = getattr(contracts.validation, "group_column", None)
            time_col = getattr(contracts.validation, "time_column", None)
            purge_win = getattr(contracts.validation, "purge_window", 0)
            embargo_win = getattr(contracts.validation, "embargo_window", 0)

            splitter = get_validation_splitter(
                strategy,
                n_splits=n_splits,
                n_repeats=n_repeats,
                seed=proposal.seed,
                purge_window=purge_win,
                embargo_window=embargo_win,
                time_column=time_col,
            )

            groups_input = (
                features[group_col]
                if group_col and group_col in features.columns
                else None
            )
            y_input = target if task_type != "regression" else None

            splits = list(splitter.split(features, y=y_input, groups=groups_input))

            # Group isolation verification
            if groups_input is not None and strategy in (
                "group_kfold",
                "stratified_group_kfold",
            ):
                for fold_idx, (t_idx, v_idx) in enumerate(splits):
                    t_groups = set(groups_input.iloc[t_idx])
                    v_groups = set(groups_input.iloc[v_idx])
                    overlap = t_groups.intersection(v_groups)
                    if overlap:
                        raise ValueError(
                            f"Entity group overlap in fold {fold_idx}: {overlap}"
                        )

            # Temporal causality verification
            if strategy in ("expanding_walk_forward", "purged_time_series"):
                for fold_idx, (t_idx, v_idx) in enumerate(splits):
                    if (
                        len(t_idx) > 0
                        and len(v_idx) > 0
                        and time_col
                        and time_col in features.columns
                    ):
                        t_max = features[time_col].iloc[t_idx].max()
                        v_min = features[time_col].iloc[v_idx].min()
                        if t_max >= v_min:
                            raise ValueError(
                                f"Temporal lookahead in fold {fold_idx}: {t_max} >= {v_min}"
                            )

            is_single_partition = strategy in (
                "stratified_kfold",
                "kfold",
                "group_kfold",
                "stratified_group_kfold",
            )

            models = []
            test_preds_folds = []

            if is_single_partition:
                oof_predictions = [None] * len(features)
                oof_scores = (
                    [None] * len(features)
                    if metric_name in {"roc_auc", "log_loss"}
                    else None
                )
                oof_fold_ids = [0] * len(features)
                oof_row_ids = [str(features.index[i]) for i in range(len(features))]

                for fold_idx, (train_idx, val_idx) in enumerate(splits):
                    resource_policy.check_cancellation()
                    resource_policy.check_total_budget(time.perf_counter() - started_at)

                    if checkpoint_manager.has_fold(fold_idx):
                        resumed_from_checkpoint = True
                        saved = checkpoint_manager.load_fold(fold_idx)
                        fold_preds = saved["predictions"]
                        for i, idx in enumerate(val_idx):
                            oof_predictions[idx] = fold_preds[i]
                            oof_fold_ids[idx] = fold_idx
                        if oof_scores is not None and saved.get("scores") is not None:
                            for i, idx in enumerate(val_idx):
                                oof_scores[idx] = float(saved["scores"][i])
                        if saved.get("model") is not None:
                            models.append(saved["model"])
                        if (
                            contracts.data.test_dataset_path
                            and saved.get("test_predictions") is not None
                        ):
                            test_preds_folds.append(saved["test_predictions"])
                        fold_scores.append(float(saved["metric_score"]))
                    else:
                        fold_start = time.perf_counter()
                        x_train_fold = features.iloc[train_idx]
                        y_train_fold = target.iloc[train_idx]
                        x_val_fold = features.iloc[val_idx]
                        fold_model = _build_model(proposal, x_train_fold, task_type=task_type)
                        fold_model.fit(x_train_fold, y_train_fold)
                        models.append(fold_model)

                        fold_elapsed = time.perf_counter() - fold_start
                        resource_policy.check_fold_budget(fold_idx, fold_elapsed)
                        resource_policy.check_total_budget(time.perf_counter() - started_at)
                        if hasattr(resource_meter, "_peak_rss"):
                            resource_policy.check_memory(resource_meter._peak_rss / (1024 * 1024))

                        fold_preds = fold_model.predict(x_val_fold)
                        for i, idx in enumerate(val_idx):
                            oof_predictions[idx] = fold_preds[i]
                            oof_fold_ids[idx] = fold_idx

                        fold_probability_scores = None
                        if metric_name in {"roc_auc", "log_loss"}:
                            fold_probability_scores = _positive_class_scores(
                                fold_model, x_val_fold, positive_class
                            )
                            for i, idx in enumerate(val_idx):
                                oof_scores[idx] = fold_probability_scores[i]

                        fold_metric = _compute_metric(
                            metric_name=metric_name,
                            y_true=target.iloc[val_idx],
                            y_pred=fold_preds,
                            y_prob=fold_probability_scores,
                            positive_class=positive_class,
                        )
                        fold_scores.append(fold_metric)

                        test_pred_fold = None
                        if contracts.data.test_dataset_path:
                            test_features = load_test_features(
                                contracts.data.test_dataset_path, contracts.data.feature_columns
                            )
                            if metric_name in {"roc_auc", "log_loss"}:
                                test_pred_fold = _positive_class_scores(
                                    fold_model, test_features, positive_class
                                )
                            else:
                                test_pred_fold = fold_model.predict(test_features)
                            test_preds_folds.append(test_pred_fold)

                        checkpoint_manager.save_fold(
                            fold_idx=fold_idx,
                            val_indices=val_idx,
                            predictions=fold_preds,
                            targets=np.array(target.iloc[val_idx]),
                            metric_score=fold_metric,
                            scores=fold_probability_scores,
                            test_predictions=test_pred_fold,
                            model=fold_model,
                        )

                    _check_trial_pruning(
                        proposal,
                        fold_scores,
                        fold_idx,
                        hpo_controller,
                        benchmark_score,
                        metric_spec.higher_is_better,
                    )

                y_validation = target
                predictions = oof_predictions
                validation_scores = oof_scores
                validation_row_ids = oof_row_ids
                validation_fold_ids = oof_fold_ids
                val_rows_count = len(features)
            else:
                # Multi-window or repeated splits
                val_row_ids: list[str] = []
                val_fold_ids: list[int] = []
                val_actual: list[object] = []
                val_preds_list: list[object] = []
                val_scores_list: list[float] | None = (
                    [] if metric_name in {"roc_auc", "log_loss"} else None
                )

                for fold_idx, (train_idx, val_idx) in enumerate(splits):
                    resource_policy.check_cancellation()
                    resource_policy.check_total_budget(time.perf_counter() - started_at)

                    if checkpoint_manager.has_fold(fold_idx):
                        resumed_from_checkpoint = True
                        saved = checkpoint_manager.load_fold(fold_idx)
                        fold_preds = saved["predictions"]
                        fold_probability_scores = saved.get("scores")
                        for i, idx in enumerate(val_idx):
                            val_row_ids.append(str(features.index[idx]))
                            val_fold_ids.append(fold_idx)
                            val_actual.append(target.iloc[idx])
                            val_preds_list.append(fold_preds[i])
                            if val_scores_list is not None and fold_probability_scores is not None:
                                val_scores_list.append(fold_probability_scores[i])
                        if saved.get("model") is not None:
                            models.append(saved["model"])
                        if (
                            contracts.data.test_dataset_path
                            and saved.get("test_predictions") is not None
                        ):
                            test_preds_folds.append(saved["test_predictions"])
                        fold_scores.append(float(saved["metric_score"]))
                    else:
                        fold_start = time.perf_counter()
                        x_train_fold = features.iloc[train_idx]
                        y_train_fold = target.iloc[train_idx]
                        x_val_fold = features.iloc[val_idx]
                        fold_model = _build_model(proposal, x_train_fold, task_type=task_type)
                        fold_model.fit(x_train_fold, y_train_fold)
                        models.append(fold_model)

                        fold_elapsed = time.perf_counter() - fold_start
                        resource_policy.check_fold_budget(fold_idx, fold_elapsed)
                        resource_policy.check_total_budget(time.perf_counter() - started_at)
                        if hasattr(resource_meter, "_peak_rss"):
                            resource_policy.check_memory(resource_meter._peak_rss / (1024 * 1024))

                        fold_preds = fold_model.predict(x_val_fold)
                        fold_probability_scores = None
                        if metric_name in {"roc_auc", "log_loss"}:
                            fold_probability_scores = _positive_class_scores(
                                fold_model, x_val_fold, positive_class
                            )

                        for i, idx in enumerate(val_idx):
                            val_row_ids.append(str(features.index[idx]))
                            val_fold_ids.append(fold_idx)
                            val_actual.append(target.iloc[idx])
                            val_preds_list.append(fold_preds[i])
                            if val_scores_list is not None and fold_probability_scores is not None:
                                val_scores_list.append(fold_probability_scores[i])

                        fold_metric = _compute_metric(
                            metric_name=metric_name,
                            y_true=target.iloc[val_idx],
                            y_pred=fold_preds,
                            y_prob=fold_probability_scores,
                            positive_class=positive_class,
                        )
                        fold_scores.append(fold_metric)

                        test_pred_fold = None
                        if contracts.data.test_dataset_path:
                            test_features = load_test_features(
                                contracts.data.test_dataset_path, contracts.data.feature_columns
                            )
                            if metric_name in {"roc_auc", "log_loss"}:
                                test_pred_fold = _positive_class_scores(
                                    fold_model, test_features, positive_class
                                )
                            else:
                                test_pred_fold = fold_model.predict(test_features)
                            test_preds_folds.append(test_pred_fold)

                        checkpoint_manager.save_fold(
                            fold_idx=fold_idx,
                            val_indices=val_idx,
                            predictions=fold_preds,
                            targets=np.array(target.iloc[val_idx]),
                            metric_score=fold_metric,
                            scores=fold_probability_scores,
                            test_predictions=test_pred_fold,
                            model=fold_model,
                        )

                    _check_trial_pruning(
                        proposal,
                        fold_scores,
                        fold_idx,
                        hpo_controller,
                        benchmark_score,
                        metric_spec.higher_is_better,
                    )

                y_validation = pd.Series(val_actual)
                predictions = val_preds_list
                validation_scores = val_scores_list
                validation_row_ids = val_row_ids
                validation_fold_ids = val_fold_ids
                val_rows_count = len(val_row_ids)

            model = models[0] if models else None
            train_rows_count = len(splits[0][0])
            x_train = features.iloc[splits[0][0]]
            x_validation = features

            metric_value = _compute_metric(
                metric_name=metric_name,
                y_true=y_validation,
                y_pred=predictions,
                y_prob=validation_scores,
                positive_class=positive_class,
            )
            metrics = {metric_name: metric_value}

            test_predictions = None
            if contracts.data.test_dataset_path and test_preds_folds:
                if metric_name in {"roc_auc", "log_loss"} or task_type == "regression":
                    test_predictions = np.mean(test_preds_folds, axis=0).tolist()
                else:
                    from collections import Counter

                    test_predictions = [
                        Counter(preds).most_common(1)[0][0]
                        for preds in zip(*test_preds_folds, strict=True)
                    ]
        else:
            # holdout
            resource_policy.check_cancellation()
            resource_policy.check_total_budget(time.perf_counter() - started_at)

            x_train, x_validation, y_train, y_validation = train_test_split(
                features,
                target,
                test_size=proposal.test_size,
                random_state=proposal.seed,
                stratify=target if task_type != "regression" else None,
            )
            validation_row_ids = [str(idx) for idx in x_validation.index]
            validation_fold_ids = [0] * len(x_validation)

            if checkpoint_manager.has_fold(0):
                resumed_from_checkpoint = True
                saved = checkpoint_manager.load_fold(0)
                model = saved.get("model")
                predictions = saved["predictions"]
                validation_scores = saved.get("scores")
                metric_value = float(saved["metric_score"])
                metrics = {metric_name: metric_value}
                test_predictions = saved.get("test_predictions")
                train_rows_count = len(x_train)
                val_rows_count = len(x_validation)
            else:
                fold_start = time.perf_counter()
                model = _build_model(proposal, x_train, task_type=task_type)
                model.fit(x_train, y_train)

                fold_elapsed = time.perf_counter() - fold_start
                resource_policy.check_fold_budget(0, fold_elapsed)
                resource_policy.check_total_budget(time.perf_counter() - started_at)
                if hasattr(resource_meter, "_peak_rss"):
                    resource_policy.check_memory(resource_meter._peak_rss / (1024 * 1024))

                predictions = model.predict(x_validation)
                validation_scores = None
                if metric_name in ("roc_auc", "log_loss"):
                    validation_scores = _positive_class_scores(model, x_validation, positive_class)
                metric_value = _compute_metric(
                    metric_name=metric_name,
                    y_true=y_validation,
                    y_pred=predictions,
                    y_prob=validation_scores,
                    positive_class=positive_class,
                )
                metrics = {metric_name: metric_value}
                test_predictions = None
                if contracts.data.test_dataset_path:
                    test_features = load_test_features(
                        contracts.data.test_dataset_path, contracts.data.feature_columns
                    )
                    model.fit(features, target)
                    test_predictions = (
                        _positive_class_scores(model, test_features, positive_class)
                        if metric_name in {"roc_auc", "log_loss"}
                        else model.predict(test_features)
                    )
                train_rows_count = len(x_train)
                val_rows_count = len(x_validation)

                checkpoint_manager.save_fold(
                    fold_idx=0,
                    val_indices=list(range(len(x_validation))),
                    predictions=predictions,
                    targets=np.array(y_validation),
                    metric_score=metric_value,
                    scores=validation_scores,
                    test_predictions=test_predictions,
                    model=model,
                )

    resource_usage = resource_meter.result()

    proposal_path = write_json(run_directory / "proposal.json", proposal.model_dump())
    model_path = run_directory / "model.joblib"
    target_model = models[0] if "models" in locals() and models else model
    if model is None and target_model is not None:
        model = target_model
    joblib.dump(model, model_path)
    predictions_path = _write_predictions(
        run_directory / "validation_predictions.csv",
        y_validation,
        predictions,
        validation_scores,
        row_ids=validation_row_ids,
    )
    _write_oof_parquet(
        run_directory / "oof_predictions.parquet",
        validation_row_ids,
        validation_fold_ids,
        y_validation,
        predictions,
        validation_scores,
    )
    feature_pipeline_path = run_directory / "feature_pipeline.joblib"
    preprocessor_obj = getattr(target_model, "named_steps", {}).get("preprocessor")
    if preprocessor_obj is not None:
        joblib.dump(preprocessor_obj, feature_pipeline_path)
        metrics["feature_count_in"] = float(
            getattr(preprocessor_obj, "input_feature_count", len(features.columns))
        )
        metrics["feature_count_out"] = float(
            getattr(preprocessor_obj, "output_feature_count", len(features.columns))
        )
    artifact_paths = [proposal_path, model_path, predictions_path]
    if test_predictions is not None:
        artifact_paths.append(
            _write_external_predictions(
                run_directory / "test_predictions.csv", test_predictions
            )
        )
    artifacts = [_artifact_record(path) for path in artifact_paths]

    tracking_uri = (
        state.get("tracking_uri") or f"sqlite:///{(output_root / 'mlflow.db').as_posix()}"
    )
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name("swarmfold-demo")
    experiment_id = (
        experiment.experiment_id
        if experiment is not None
        else mlflow.create_experiment(
            "swarmfold-demo",
            artifact_location=(output_root / "mlartifacts").resolve().as_uri(),
        )
    )
    with mlflow.start_run(
        experiment_id=experiment_id, run_name=proposal.experiment_id
    ) as tracking_run:
        mlflow.log_params(
            {
                "experiment_id": proposal.experiment_id,
                "model_family": proposal.model_family,
                "seed": proposal.seed,
                "test_size": proposal.test_size,
                "dataset_sha256": dataset_sha256,
                "feature_strategy": proposal.feature_strategy,
                "prompt_version": proposal.prompt_version,
                **{f"model_{key}": value for key, value in proposal.parameters.items()},
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(str(run_directory), artifact_path="run")
        tracking_run_id = tracking_run.info.run_id
        artifact_uri = tracking_run.info.artifact_uri

    return RunResult(
        experiment_id=proposal.experiment_id,
        run_id=run_id,
        status="SUCCEEDED",
        model_family=proposal.model_family,
        metrics=metrics,
        train_rows=train_rows_count,
        validation_rows=val_rows_count,
        test_rows=(len(test_predictions) if test_predictions is not None else 0),
        positive_class=positive_class,
        duration_seconds=time.perf_counter() - started_at,
        tracking_run_id=tracking_run_id,
        artifact_uri=artifact_uri,
        dataset_sha256=dataset_sha256,
        contract_sha256=contracts.contract_sha256,
        data_version=proposal.data_version,
        split_id=proposal.split_id,
        metric_version=proposal.metric_version,
        artifacts=artifacts,
        resource_usage=resource_usage,
        fold_scores=fold_scores,
        resumed_from_checkpoint=resumed_from_checkpoint,
    )


def run_experiment_candidate(state: ExperimentState) -> ExperimentState:
    proposal = ExperimentProposal.model_validate(state["proposal"])
    contracts = ContractBundle.model_validate(state["contracts"])
    _validate_proposal_against_contracts(proposal, contracts)
    cached_result = _load_cached_result(state, proposal)
    if cached_result is not None:
        return _record_completed_result(state, cached_result)

    root_experiment_id = proposal.experiment_id
    output_root = Path(state["output_dir"])
    checkpoint_dir = output_root / "checkpoints" / root_experiment_id
    checkpoint_manager = CheckpointManager(checkpoint_dir)

    budget = state.get("budget") or {}
    resource_policy = ResourcePolicy(
        max_training_seconds_per_fold=(
            state.get("max_training_seconds_per_fold")
            or budget.get("max_training_seconds_per_fold")
        ),
        max_total_seconds=(
            state.get("max_total_seconds")
            or budget.get("max_total_seconds")
        ),
        max_memory_mb=(
            state.get("max_memory_mb")
            or budget.get("max_memory_mb")
        ),
        cancellation_token=state.get("cancellation_token"),
    )

    debugger = SelfHealingDebugger(max_retries=2)
    metric_name = contracts.metric.metric_name
    metric_spec = get_metric_spec(metric_name)
    prior_scores = [
        float(item.get("metrics", {}).get(metric_name))
        for item in state.get("completed_results", [])
        if item.get("status") == "SUCCEEDED" and metric_name in item.get("metrics", {})
    ]
    benchmark_score = metric_spec.best(prior_scores) if prior_scores else None
    hpo_controller = None
    if proposal.hpo_study_id is not None and proposal.hpo_trial_number is not None:
        hpo_controller = HPOController(
            output_root / "hpo.sqlite",
            HPOConfig(
                study_id=proposal.hpo_study_id,
                direction="maximize" if metric_spec.higher_is_better else "minimize",
                seed=contracts.validation.seed,
            ),
        )

    while True:
        run_id = uuid4().hex
        run_directory = output_root / "runs" / run_id
        run_directory.mkdir(parents=True, exist_ok=False)

        started_at = time.perf_counter()
        resource_meter = ProcessResourceMeter()

        try:
            result = _execute_single_attempt(
                state=state,
                proposal=proposal,
                contracts=contracts,
                run_id=run_id,
                run_directory=run_directory,
                output_root=output_root,
                resource_meter=resource_meter,
                resource_policy=resource_policy,
                checkpoint_manager=checkpoint_manager,
                benchmark_score=benchmark_score,
                hpo_controller=hpo_controller,
                started_at=started_at,
            )
            result.original_experiment_id = root_experiment_id
            result.repair_history = list(debugger.repair_history)
            if hpo_controller is not None and proposal.hpo_trial_number is not None:
                hpo_controller.complete(
                    proposal.hpo_trial_number,
                    result.metrics[metric_name],
                )
            break
        except TrialPrunedError as exc:
            try:
                resource_usage = resource_meter.result()
            except Exception:
                resource_usage = {}
            result = RunResult(
                experiment_id=proposal.experiment_id,
                run_id=run_id,
                status="PRUNED",
                model_family=proposal.model_family,
                error=f"TrialPrunedError: {exc}",
                duration_seconds=time.perf_counter() - started_at,
                contract_sha256=contracts.contract_sha256,
                data_version=proposal.data_version,
                split_id=proposal.split_id,
                metric_version=proposal.metric_version,
                resource_usage=resource_usage,
                fold_scores=exc.fold_scores,
                pruning_reason=str(exc),
                original_experiment_id=root_experiment_id,
                repair_history=list(debugger.repair_history),
            )
            if hpo_controller is not None and proposal.hpo_trial_number is not None:
                hpo_controller.prune(
                    proposal.hpo_trial_number,
                    float(np.mean(exc.fold_scores)),
                )
            break
        except Exception as exc:
            try:
                resource_usage = resource_meter.result()
            except Exception:
                resource_usage = {}
            failure_record = classify_failure(exc)

            # Provider/network failures get bounded exponential backoff retries,
            # separate from the training parameter repair budget.
            if debugger.can_retry_provider(root_experiment_id, failure_record):
                backoff = debugger.get_provider_backoff_seconds(root_experiment_id)
                time.sleep(backoff)
                continue

            if debugger.can_repair(root_experiment_id, failure_record):
                repaired_proposal, action = debugger.attempt_repair(proposal, failure_record)
                failure_record.repair_action = action
                proposal = repaired_proposal
                checkpoint_manager.clear()
                continue

            result = RunResult(
                experiment_id=proposal.experiment_id,
                run_id=run_id,
                status="FAILED",
                model_family=proposal.model_family,
                error=f"{type(exc).__name__}: {exc}",
                duration_seconds=time.perf_counter() - started_at,
                contract_sha256=contracts.contract_sha256,
                data_version=proposal.data_version,
                split_id=proposal.split_id,
                metric_version=proposal.metric_version,
                resource_usage=resource_usage,
                failure_record=failure_record.model_dump(),
                repair_history=list(debugger.repair_history),
                original_experiment_id=root_experiment_id,
            )
            break


    cache_path = _result_cache_path(state, proposal)
    write_json(cache_path, result.model_dump())
    output_state = _record_completed_result(state, result)
    output_state["proposal"] = proposal.model_dump()
    return output_state


run_baseline = run_experiment_candidate


def verify_evidence(state: ExperimentState) -> ExperimentState:
    result = RunResult.model_validate(state["run_result"])
    contracts = ContractBundle.model_validate(state["contracts"])
    reasons: list[str] = []

    if result.status != "SUCCEEDED":
        reasons.append(result.error or "Experiment did not succeed")
    metric_name = contracts.metric.metric_name
    metric_value = result.metrics.get(metric_name)
    metric_spec = get_metric_spec(metric_name)
    if (
        metric_value is None
        or not math.isfinite(metric_value)
        or (metric_spec.minimum is not None and metric_value < metric_spec.minimum)
        or (metric_spec.maximum is not None and metric_value > metric_spec.maximum)
    ):
        reasons.append(f"{metric_name} metric is missing or invalid")
    if result.train_rows == 0 or result.validation_rows == 0:
        reasons.append("Train and validation row counts must be positive")

    expected_provenance = {
        "contract_sha256": contracts.contract_sha256,
        "data_version": contracts.data.data_version,
        "split_id": contracts.validation.split_id,
        "metric_version": contracts.metric.metric_version,
    }
    for field, expected in expected_provenance.items():
        if getattr(result, field) != expected:
            reasons.append(f"Run provenance does not match frozen {field}")

    contracts_source = Path(state["output_dir"]) / "contracts" / "contracts.json"
    if not contracts_source.is_file():
        reasons.append("Frozen contract source is missing")
    else:
        source_contracts = ContractBundle.model_validate_json(
            contracts_source.read_text(encoding="utf-8")
        )
        if contracts.contract_sha256 != contract_bundle_sha256(source_contracts):
            reasons.append("Frozen contract hash mismatch")

    dataset_path = Path(state["dataset_path"])
    if not dataset_path.is_file():
        reasons.append("Dataset cannot be resolved during verification")
    elif result.dataset_sha256 and sha256_file(dataset_path) != result.dataset_sha256:
        reasons.append("Dataset hash changed after execution")
    for field in ("test_dataset", "sample_submission", "project"):
        path_value = getattr(contracts.data, f"{field}_path")
        expected_hash = getattr(contracts.data, f"{field}_sha256")
        if path_value and expected_hash and sha256_file(path_value) != expected_hash:
            reasons.append(f"{field.replace('_', ' ').title()} changed after contract freeze")

    for artifact in result.artifacts:
        path = Path(artifact.path)
        if not path.is_file():
            reasons.append(f"Artifact is missing: {path.name}")
        elif sha256_file(path) != artifact.sha256:
            reasons.append(f"Artifact hash mismatch: {path.name}")

    prediction_artifact = next(
        (
            artifact
            for artifact in result.artifacts
            if Path(artifact.path).name == "validation_predictions.csv"
        ),
        None,
    )
    if prediction_artifact is None:
        reasons.append("Validation predictions artifact is missing")
    elif Path(prediction_artifact.path).is_file() and metric_value is not None:
        try:
            recalculated_metric = _metric_from_predictions(
                Path(prediction_artifact.path), metric_name, result.positive_class
            )
        except (KeyError, ValueError) as exc:
            reasons.append(f"Validation predictions cannot be verified: {exc}")
        else:
            if not math.isclose(recalculated_metric, metric_value, rel_tol=0.0, abs_tol=1e-12):
                reasons.append(f"Recorded {metric_name} does not match saved predictions")
    if contracts.data.test_dataset_path:
        external_artifact = next(
            (
                artifact
                for artifact in result.artifacts
                if Path(artifact.path).name == "test_predictions.csv"
            ),
            None,
        )
        if external_artifact is None:
            reasons.append("Test predictions artifact is missing")
        elif result.test_rows != contracts.data.test_row_count:
            reasons.append("Test prediction row count does not match frozen test data")

    verification = VerificationResult(
        experiment_id=result.experiment_id,
        accepted=not reasons,
        reasons=reasons,
    )
    return {
        "verification": verification.model_dump(),
        "status": "ACCEPTED" if verification.accepted else "REJECTED",
    }


def _write_predictions(
    path: Path, actual, predicted, scores=None, row_ids=None
) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        headers = ["row", "row_id", "actual", "predicted"]
        if scores is not None:
            headers.append("score")
        writer.writerow(headers)
        score_values = scores if scores is not None else [None] * len(actual)
        row_values = (
            [str(r) for r in row_ids] if row_ids is not None else list(range(len(actual)))
        )
        for row_id, actual_value, predicted_value, score in zip(
            row_values, actual, predicted, score_values, strict=True
        ):
            row = [row_id, row_id, actual_value, predicted_value]
            if scores is not None:
                row.append(score)
            writer.writerow(row)
    return path


def _write_oof_parquet(
    path: Path,
    row_ids: list[int | str],
    fold_ids: list[int],
    actual,
    predicted,
    scores=None,
) -> Path:
    actual_list = list(actual)
    pred_list = list(predicted)
    data: dict[str, list] = {
        "row": [str(r) for r in row_ids],
        "row_id": [str(r) for r in row_ids],
        "fold": [int(f) for f in fold_ids],
        "target": [str(a) for a in actual_list],
        "prediction": [
            str(scores[i]) if scores is not None and scores[i] is not None else str(pred_list[i])
            for i in range(len(actual_list))
        ],
    }
    if scores is not None and any(s is not None for s in scores):
        data["score"] = [
            float(scores[i]) if scores[i] is not None else 0.0 for i in range(len(actual_list))
        ]
    table = pa.Table.from_pydict(data)
    pq.write_table(table, path)
    return path


def _write_external_predictions(path: Path, predicted) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row", "predicted"])
        for row_number, predicted_value in enumerate(predicted):
            writer.writerow([row_number, predicted_value])
    return path


def _artifact_record(path: Path) -> ArtifactRecord:
    resolved = path.resolve()
    return ArtifactRecord(
        path=str(resolved),
        sha256=sha256_file(resolved),
        size_bytes=resolved.stat().st_size,
    )


def _metric_from_predictions(path: Path, metric_name: str, positive_class: str | None) -> float:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Validation predictions are empty")
    if metric_name == "accuracy":
        correct = sum(row["actual"] == row["predicted"] for row in rows)
        return correct / len(rows)
    if metric_name == "roc_auc":
        if positive_class is None:
            raise ValueError("Positive class is missing for ROC-AUC")
        return float(
            roc_auc_score(
                [row["actual"] == positive_class for row in rows],
                [float(row["score"]) for row in rows],
            )
        )
    if metric_name == "log_loss":
        from sklearn.metrics import log_loss

        y_true = [row["actual"] for row in rows]
        y_score = [
            float(row["score"]) if row.get("score") else float(row["predicted"]) for row in rows
        ]
        return float(log_loss(y_true, y_score))
    if metric_name == "f1_macro":
        from sklearn.metrics import f1_score

        return float(
            f1_score(
                [row["actual"] for row in rows],
                [row["predicted"] for row in rows],
                average="macro",
            )
        )
    if metric_name == "multilabel_f1":
        from sklearn.metrics import f1_score

        y_t = np.asarray([row["actual"] for row in rows])
        y_p = np.asarray([row["predicted"] for row in rows])
        return float(f1_score(y_t, y_p, average="macro", zero_division=0))
    if metric_name == "rmse":
        from sklearn.metrics import root_mean_squared_error

        return float(
            root_mean_squared_error(
                [float(row["actual"]) for row in rows],
                [float(row["predicted"]) for row in rows],
            )
        )
    if metric_name == "mae":
        from sklearn.metrics import mean_absolute_error

        return float(
            mean_absolute_error(
                [float(row["actual"]) for row in rows],
                [float(row["predicted"]) for row in rows],
            )
        )
    if metric_name == "mape":
        from sklearn.metrics import mean_absolute_percentage_error

        return float(
            mean_absolute_percentage_error(
                [float(row["actual"]) for row in rows],
                [float(row["predicted"]) for row in rows],
            )
        )
    if metric_name == "r2":
        from sklearn.metrics import r2_score

        return float(
            r2_score(
                [float(row["actual"]) for row in rows],
                [float(row["predicted"]) for row in rows],
            )
        )
    raise ValueError(f"Unsupported verification metric: {metric_name}")


def _validate_proposal_against_contracts(
    proposal: ExperimentProposal, contracts: ContractBundle
) -> None:
    validate_proposal_executability(proposal.model_family)
    expected = {
        "dataset_path": contracts.data.dataset_path,
        "target_column": contracts.data.target_column,
        "data_version": contracts.data.data_version,
        "split_id": contracts.validation.split_id,
        "metric_version": contracts.metric.metric_version,
        "seed": contracts.validation.seed,
        "test_size": contracts.validation.test_size,
    }
    actual = {
        "dataset_path": str(Path(proposal.dataset_path).resolve()),
        "target_column": proposal.target_column,
        "data_version": proposal.data_version,
        "split_id": proposal.split_id,
        "metric_version": proposal.metric_version,
        "seed": proposal.seed,
        "test_size": proposal.test_size,
    }
    mismatches = [name for name, value in expected.items() if actual[name] != value]
    if mismatches:
        raise ValueError(f"Proposal violates frozen contracts: {', '.join(mismatches)}")


def _build_model(
    proposal: ExperimentProposal,
    features,
    task_type: str = "binary_classification",
):
    use_one_hot = proposal.model_family in {"logistic_regression", "sgd_log_loss"}
    preprocessor = FeaturePipelineExecutor(
        strategy=proposal.feature_strategy,
        task_type=task_type,
        use_one_hot=use_one_hot,
    )
    selector = None

    estimator = None
    if proposal.model_family == "logistic_regression":
        if task_type == "regression":
            c_val = proposal.parameters.get("c", proposal.parameters.get("C", 1.0))
            estimator = Ridge(
                alpha=1.0 / max(float(c_val), 1e-12),
                random_state=proposal.seed,
            )
        else:
            lr_kwargs = {
                "C": float(proposal.parameters.get("c", proposal.parameters.get("C", 1.0))),
                "class_weight": proposal.parameters.get("class_weight"),
                "max_iter": 1_000,
                "random_state": proposal.seed,
            }
            if "solver" in proposal.parameters:
                lr_kwargs["solver"] = proposal.parameters["solver"]
            if "penalty" in proposal.parameters:
                lr_kwargs["penalty"] = proposal.parameters["penalty"]
            if "l1_ratio" in proposal.parameters:
                lr_kwargs["l1_ratio"] = float(proposal.parameters["l1_ratio"])
            estimator = LogisticRegression(**lr_kwargs)
    elif proposal.model_family == "sgd_log_loss":
        if task_type == "regression":
            estimator = SGDRegressor(
                loss="squared_error",
                alpha=float(proposal.parameters.get("alpha", 0.0001)),
                penalty=str(proposal.parameters.get("penalty", "l2")),
                max_iter=2_000,
                random_state=proposal.seed,
            )
        else:
            estimator = SGDClassifier(
                loss="log_loss",
                alpha=float(proposal.parameters.get("alpha", 0.0001)),
                penalty=str(proposal.parameters.get("penalty", "l2")),
                class_weight=proposal.parameters.get("class_weight"),
                max_iter=2_000,
                random_state=proposal.seed,
            )
    elif proposal.model_family == "random_forest":
        common = {
            "n_estimators": int(proposal.parameters.get("n_estimators", 100)),
            "max_depth": int(proposal.parameters.get("max_depth", 5)),
            "min_samples_leaf": int(proposal.parameters.get("min_samples_leaf", 1)),
            "random_state": proposal.seed,
            "n_jobs": int(proposal.parameters.get("n_jobs", 1)),
        }
        if "criterion" in proposal.parameters:
            common["criterion"] = proposal.parameters["criterion"]
        estimator = (
            RandomForestRegressor(**common)
            if task_type == "regression"
            else RandomForestClassifier(
                **common,
                class_weight=proposal.parameters.get("class_weight"),
            )
        )
    elif proposal.model_family == "extra_trees":
        common = {
            "n_estimators": int(proposal.parameters.get("n_estimators", 200)),
            "max_depth": int(proposal.parameters.get("max_depth", 8)),
            "min_samples_leaf": int(proposal.parameters.get("min_samples_leaf", 1)),
            "max_features": proposal.parameters.get("max_features", "sqrt"),
            "random_state": proposal.seed,
            "n_jobs": int(proposal.parameters.get("n_jobs", 1)),
        }
        if "criterion" in proposal.parameters:
            common["criterion"] = proposal.parameters["criterion"]
        estimator = (
            ExtraTreesRegressor(**common)
            if task_type == "regression"
            else ExtraTreesClassifier(
                **common,
                class_weight=proposal.parameters.get("class_weight"),
            )
        )
    elif proposal.model_family == "hist_gradient_boosting":
        estimator_type = (
            HistGradientBoostingRegressor
            if task_type == "regression"
            else HistGradientBoostingClassifier
        )
        hgb_kwargs = {
            "max_iter": int(proposal.parameters.get("max_iter", 100)),
            "max_depth": int(proposal.parameters.get("max_depth", 3)),
            "learning_rate": float(proposal.parameters.get("learning_rate", 0.1)),
            "l2_regularization": float(proposal.parameters.get("l2_regularization", 0.0)),
            "random_state": proposal.seed,
        }
        if "max_leaf_nodes" in proposal.parameters:
            hgb_kwargs["max_leaf_nodes"] = int(proposal.parameters["max_leaf_nodes"])
        estimator = estimator_type(**hgb_kwargs)
    elif proposal.model_family == "bagging":
        tree_type = DecisionTreeRegressor if task_type == "regression" else DecisionTreeClassifier
        bagging_type = BaggingRegressor if task_type == "regression" else BaggingClassifier
        estimator = bagging_type(
            estimator=tree_type(
                max_depth=int(proposal.parameters.get("max_depth", 8)),
                min_samples_leaf=int(proposal.parameters.get("min_samples_leaf", 2)),
                random_state=proposal.seed,
            ),
            n_estimators=int(proposal.parameters.get("n_estimators", 100)),
            max_samples=float(proposal.parameters.get("max_samples", 0.8)),
            max_features=float(proposal.parameters.get("max_features", 1.0)),
            bootstrap=bool(proposal.parameters.get("bootstrap", True)),
            random_state=proposal.seed,
            n_jobs=1,
        )
    elif proposal.model_family == "soft_voting":
        if task_type == "regression":
            estimator = VotingRegressor(
                estimators=[
                    ("ridge", make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
                    (
                        "forest",
                        RandomForestRegressor(
                            n_estimators=100,
                            max_depth=5,
                            random_state=proposal.seed,
                            n_jobs=1,
                        ),
                    ),
                    (
                        "histogram",
                        HistGradientBoostingRegressor(
                            max_iter=100,
                            max_depth=3,
                            random_state=proposal.seed,
                        ),
                    ),
                ],
                weights=proposal.parameters.get("weights"),
            )
        else:
            estimator = VotingClassifier(
                estimators=[
                    (
                        "logistic",
                        make_pipeline(
                            StandardScaler(),
                            LogisticRegression(max_iter=1_000, random_state=proposal.seed),
                        ),
                    ),
                    (
                        "forest",
                        RandomForestClassifier(
                            n_estimators=100,
                            max_depth=5,
                            random_state=proposal.seed,
                            n_jobs=1,
                        ),
                    ),
                    (
                        "histogram",
                        HistGradientBoostingClassifier(
                            max_iter=100,
                            max_depth=3,
                            random_state=proposal.seed,
                        ),
                    ),
                ],
                voting="soft",
                weights=proposal.parameters.get("weights"),
            )
    elif proposal.model_family == "stacking":
        if task_type == "regression":
            estimator = StackingRegressor(
                estimators=[
                    ("ridge", make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
                    (
                        "extra_trees",
                        ExtraTreesRegressor(
                            n_estimators=150,
                            max_depth=8,
                            min_samples_leaf=2,
                            random_state=proposal.seed,
                            n_jobs=1,
                        ),
                    ),
                    (
                        "histogram",
                        HistGradientBoostingRegressor(
                            max_iter=125,
                            max_depth=4,
                            random_state=proposal.seed,
                        ),
                    ),
                ],
                final_estimator=Ridge(
                    alpha=1.0 / max(float(proposal.parameters.get("meta_c", 1.0)), 1e-12)
                ),
                cv=int(proposal.parameters.get("cv", 3)),
                passthrough=bool(proposal.parameters.get("passthrough", False)),
                n_jobs=1,
            )
        else:
            estimator = StackingClassifier(
                estimators=[
                    (
                        "logistic",
                        make_pipeline(
                            StandardScaler(),
                            LogisticRegression(max_iter=1_000, random_state=proposal.seed),
                        ),
                    ),
                    (
                        "extra_trees",
                        ExtraTreesClassifier(
                            n_estimators=150,
                            max_depth=8,
                            min_samples_leaf=2,
                            random_state=proposal.seed,
                            n_jobs=1,
                        ),
                    ),
                    (
                        "histogram",
                        HistGradientBoostingClassifier(
                            max_iter=125,
                            max_depth=4,
                            random_state=proposal.seed,
                        ),
                    ),
                ],
                final_estimator=LogisticRegression(
                    C=float(proposal.parameters.get("meta_c", 1.0)),
                    max_iter=1_000,
                    random_state=proposal.seed,
                ),
                cv=int(proposal.parameters.get("cv", 3)),
                passthrough=bool(proposal.parameters.get("passthrough", False)),
                stack_method="predict_proba",
                n_jobs=1,
            )
    elif proposal.model_family in ("lightgbm", "xgboost", "catboost"):
        from swarmfold.adapters import get_adapter

        adapter = get_adapter(proposal.model_family)
        estimator = adapter.build_estimator(
            task_type=task_type,
            hyperparameters=proposal.parameters,
            random_state=proposal.seed,
        )
    if estimator is None:
        raise ValueError(f"Unsupported model family: {proposal.model_family}")
    steps = [("preprocessor", preprocessor)]
    if selector is not None:
        steps.append(("feature_selection", selector))
    steps.append(("model", estimator))
    return Pipeline(steps)


def _positive_class(target) -> str:
    classes = sorted(str(value) for value in np.unique(target))
    for value in classes:
        if value.casefold() in {"yes", "true", "1", "positive"}:
            return value
    return classes[-1]


def _positive_class_scores(model, features, positive_class: str | None):
    if positive_class is None:
        raise ValueError("Binary probability metrics require a positive class")
    classes = [str(value) for value in model.classes_]
    if positive_class not in classes:
        raise ValueError("Positive class is absent from the fitted model")
    return model.predict_proba(features)[:, classes.index(positive_class)]


def _compute_metric(
    metric_name: str,
    y_true,
    y_pred,
    y_prob=None,
    positive_class: str | None = None,
) -> float:
    if metric_name == "roc_auc":
        if y_prob is None:
            raise ValueError("roc_auc requires predicted probabilities/scores")
        if positive_class is not None:
            return float(roc_auc_score(y_true == positive_class, y_prob))
        return float(roc_auc_score(y_true, y_prob))
    elif metric_name == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    elif metric_name == "f1_macro":
        return float(f1_score(y_true, y_pred, average="macro"))
    elif metric_name == "log_loss":
        if y_prob is None:
            raise ValueError("log_loss requires predicted probabilities")
        return float(sklearn_log_loss(y_true, y_prob))
    elif metric_name == "rmse":
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
    elif metric_name == "mae":
        return float(mean_absolute_error(y_true, y_pred))
    elif metric_name == "mape":
        from sklearn.metrics import mean_absolute_percentage_error
        return float(mean_absolute_percentage_error(y_true, y_pred))
    elif metric_name == "r2":
        return float(r2_score(y_true, y_pred))
    elif metric_name == "multilabel_f1":
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    else:
        raise ValueError(f"Unsupported metric: {metric_name}")


def _result_cache_path(state: ExperimentState, proposal: ExperimentProposal) -> Path:
    return Path(state["output_dir"]) / "results" / f"{proposal.experiment_id}.json"


def _load_cached_result(state: ExperimentState, proposal: ExperimentProposal) -> RunResult | None:
    path = _result_cache_path(state, proposal)
    if not path.is_file():
        return None
    result = RunResult.model_validate_json(path.read_text(encoding="utf-8"))
    if result.experiment_id != proposal.experiment_id:
        raise ValueError("Cached result does not match the current experiment")
    for artifact in result.artifacts:
        artifact_path = Path(artifact.path)
        if not artifact_path.is_file() or sha256_file(artifact_path) != artifact.sha256:
            raise ValueError(f"Cached run artifact failed verification: {artifact_path.name}")
    return result


def _record_completed_result(state: ExperimentState, result: RunResult) -> ExperimentState:
    completed = [RunResult.model_validate(item) for item in state.get("completed_results", [])]
    if not any(item.experiment_id == result.experiment_id for item in completed):
        completed.append(result)
    resource_usage = dict(state.get("resource_usage", {}))
    resource_usage["observed_peak_training"] = max(
        int(resource_usage.get("observed_peak_training", 0)),
        LOCAL_TRAINING_POOL.peak_active,
    )
    reports = dict(state.get("role_reports", {}))
    reports["experiment_runner"] = {
        "completed": len(completed),
        "latest_experiment_id": result.experiment_id,
        "latest_status": result.status,
    }
    return {
        "run_result": result.model_dump(),
        "completed_results": [item.model_dump() for item in completed],
        "resource_usage": resource_usage,
        "role_reports": reports,
        "status": result.status,
    }
