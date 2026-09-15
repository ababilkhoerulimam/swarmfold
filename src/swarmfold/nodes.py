import csv
import math
import time
from pathlib import Path
from uuid import uuid4

import joblib
import mlflow
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from swarmfold.artifacts import sha256_file, write_json
from swarmfold.competition_bundle import load_test_features
from swarmfold.contracts import contract_bundle_sha256
from swarmfold.data import load_classification_csv
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


def create_plan_node(planner: ProposalPlanner):
    def plan_experiment(state: ExperimentState) -> ExperimentState:
        proposal = planner.propose(state)
        return {"proposal": proposal.model_dump(), "status": "PROPOSED"}

    return plan_experiment


def run_experiment_candidate(state: ExperimentState) -> ExperimentState:
    proposal = ExperimentProposal.model_validate(state["proposal"])
    contracts = ContractBundle.model_validate(state["contracts"])
    _validate_proposal_against_contracts(proposal, contracts)
    cached_result = _load_cached_result(state, proposal)
    if cached_result is not None:
        return _record_completed_result(state, cached_result)

    run_id = uuid4().hex
    output_root = Path(state["output_dir"])
    run_directory = output_root / "runs" / run_id
    run_directory.mkdir(parents=True, exist_ok=False)

    started_at = time.monotonic()
    resource_usage: dict[str, float] = {}
    resource_meter = ProcessResourceMeter()
    try:
        with LOCAL_TRAINING_POOL.slot(), resource_meter:
            dataset_sha256 = sha256_file(proposal.dataset_path)
            features, target = load_classification_csv(
                proposal.dataset_path, proposal.target_column
            )
            x_train, x_validation, y_train, y_validation = train_test_split(
                features,
                target,
                test_size=proposal.test_size,
                random_state=proposal.seed,
                stratify=target,
            )
            model = _build_model(proposal, x_train)
            model.fit(x_train, y_train)
            predictions = model.predict(x_validation)
            metric_name = contracts.metric.metric_name
            positive_class = _positive_class(target)
            validation_scores = None
            if metric_name == "roc_auc":
                validation_scores = _positive_class_scores(model, x_validation, positive_class)
                metric_value = float(
                    roc_auc_score(y_validation == positive_class, validation_scores)
                )
            else:
                metric_value = float(accuracy_score(y_validation, predictions))
            metrics = {metric_name: metric_value}
            test_predictions = None
            if contracts.data.test_dataset_path:
                test_features = load_test_features(
                    contracts.data.test_dataset_path, contracts.data.feature_columns
                )
                model.fit(features, target)
                test_predictions = (
                    _positive_class_scores(model, test_features, positive_class)
                    if metric_name == "roc_auc"
                    else model.predict(test_features)
                )
        resource_usage = resource_meter.result()

        proposal_path = write_json(run_directory / "proposal.json", proposal.model_dump())
        model_path = run_directory / "model.joblib"
        joblib.dump(model, model_path)
        predictions_path = _write_predictions(
            run_directory / "validation_predictions.csv",
            y_validation,
            predictions,
            validation_scores,
        )
        artifact_paths = [proposal_path, model_path, predictions_path]
        if test_predictions is not None:
            artifact_paths.append(
                _write_external_predictions(
                    run_directory / "test_predictions.csv", test_predictions
                )
            )
        artifacts = [_artifact_record(path) for path in artifact_paths]

        tracking_uri = state["tracking_uri"]
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

        result = RunResult(
            experiment_id=proposal.experiment_id,
            run_id=run_id,
            status="SUCCEEDED",
            model_family=proposal.model_family,
            metrics=metrics,
            train_rows=len(x_train),
            validation_rows=len(x_validation),
            test_rows=(len(test_predictions) if test_predictions is not None else 0),
            positive_class=positive_class,
            duration_seconds=time.monotonic() - started_at,
            tracking_run_id=tracking_run_id,
            artifact_uri=artifact_uri,
            dataset_sha256=dataset_sha256,
            contract_sha256=contracts.contract_sha256,
            data_version=proposal.data_version,
            split_id=proposal.split_id,
            metric_version=proposal.metric_version,
            artifacts=artifacts,
            resource_usage=resource_usage,
        )
    except Exception as exc:
        resource_usage = resource_meter.result()
        result = RunResult(
            experiment_id=proposal.experiment_id,
            run_id=run_id,
            status="FAILED",
            model_family=proposal.model_family,
            error=f"{type(exc).__name__}: {exc}",
            duration_seconds=time.monotonic() - started_at,
            contract_sha256=contracts.contract_sha256,
            data_version=proposal.data_version,
            split_id=proposal.split_id,
            metric_version=proposal.metric_version,
            resource_usage=resource_usage,
        )
    cache_path = _result_cache_path(state, proposal)
    write_json(cache_path, result.model_dump())
    return _record_completed_result(state, result)


run_baseline = run_experiment_candidate


def verify_evidence(state: ExperimentState) -> ExperimentState:
    result = RunResult.model_validate(state["run_result"])
    contracts = ContractBundle.model_validate(state["contracts"])
    reasons: list[str] = []

    if result.status != "SUCCEEDED":
        reasons.append(result.error or "Experiment did not succeed")
    metric_name = contracts.metric.metric_name
    metric_value = result.metrics.get(metric_name)
    if metric_value is None or not math.isfinite(metric_value) or not 0.0 <= metric_value <= 1.0:
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
            if not math.isclose(
                recalculated_metric, metric_value, rel_tol=0.0, abs_tol=1e-12
            ):
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


def _write_predictions(path: Path, actual, predicted, scores=None) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        headers = ["row", "actual", "predicted"]
        if scores is not None:
            headers.append("score")
        writer.writerow(headers)
        score_values = scores if scores is not None else [None] * len(actual)
        for row_number, (actual_value, predicted_value, score) in enumerate(
            zip(actual, predicted, score_values, strict=True)
        ):
            row = [row_number, actual_value, predicted_value]
            if scores is not None:
                row.append(score)
            writer.writerow(row)
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


def _metric_from_predictions(
    path: Path, metric_name: str, positive_class: str | None
) -> float:
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
    raise ValueError(f"Unsupported verification metric: {metric_name}")


def _validate_proposal_against_contracts(
    proposal: ExperimentProposal, contracts: ContractBundle
) -> None:
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


def _build_model(proposal: ExperimentProposal, features):
    numeric_columns = features.select_dtypes(include="number").columns.tolist()
    categorical_columns = [name for name in features.columns if name not in numeric_columns]
    use_one_hot = proposal.model_family == "logistic_regression"
    transformers = []
    if numeric_columns:
        numeric_steps = [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True))
        ]
        if use_one_hot:
            numeric_steps.append(("scaler", StandardScaler()))
        transformers.append(("numeric", Pipeline(numeric_steps), numeric_columns))
    if categorical_columns:
        encoder = (
            OneHotEncoder(handle_unknown="ignore", min_frequency=2)
            if use_one_hot
            else OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                encoded_missing_value=-2,
            )
        )
        transformers.append(
            (
                "categorical",
                make_pipeline(
                    SimpleImputer(strategy="constant", fill_value="__missing__"),
                    encoder,
                ),
                categorical_columns,
            )
        )
    preprocessor = ColumnTransformer(transformers)
    if proposal.model_family == "logistic_regression":
        return make_pipeline(
            preprocessor,
            LogisticRegression(
                C=float(proposal.parameters.get("c", 1.0)),
                max_iter=1_000,
                random_state=proposal.seed,
            ),
        )
    if proposal.model_family == "random_forest":
        return make_pipeline(
            preprocessor,
            RandomForestClassifier(
                n_estimators=int(proposal.parameters.get("n_estimators", 100)),
                max_depth=int(proposal.parameters.get("max_depth", 5)),
                random_state=proposal.seed,
                n_jobs=1,
            ),
        )
    if proposal.model_family == "hist_gradient_boosting":
        return make_pipeline(
            preprocessor,
            HistGradientBoostingClassifier(
                max_iter=int(proposal.parameters.get("max_iter", 100)),
                max_depth=int(proposal.parameters.get("max_depth", 3)),
                random_state=proposal.seed,
            ),
        )
    if proposal.model_family == "soft_voting":
        return make_pipeline(
            preprocessor,
            VotingClassifier(
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
            ),
        )
    raise ValueError(f"Unsupported model family: {proposal.model_family}")


def _positive_class(target) -> str:
    classes = sorted(str(value) for value in np.unique(target))
    for value in classes:
        if value.casefold() in {"yes", "true", "1", "positive"}:
            return value
    return classes[-1]


def _positive_class_scores(model, features, positive_class: str):
    classes = [str(value) for value in model.classes_]
    if positive_class not in classes:
        raise ValueError("Positive class is absent from the fitted model")
    return model.predict_proba(features)[:, classes.index(positive_class)]


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
