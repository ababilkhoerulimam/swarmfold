from pathlib import Path
from pprint import pformat

from swarmfold.artifacts import write_json
from swarmfold.schemas import ContractBundle, ExperimentProposal, RunResult


def write_end_to_end_notebook(
    path: str | Path,
    *,
    proposal: ExperimentProposal,
    result: RunResult,
    contracts: ContractBundle,
    literature_review: dict[str, object] | None = None,
    quality_evidence: dict[str, object] | None = None,
) -> Path:
    if not (
        contracts.data.test_dataset_path
        and contracts.data.sample_submission_path
        and contracts.data.project_path
    ):
        raise ValueError("End-to-end notebook requires a confirmed competition bundle")
    train_name = Path(contracts.data.dataset_path).name
    test_name = Path(contracts.data.test_dataset_path).name
    submission_name = Path(contracts.data.sample_submission_path).name
    project_name = Path(contracts.data.project_path).name
    metric_name = contracts.metric.metric_name
    metric_value = result.metrics[metric_name]
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
            "swarmfold": {
                "run_id": result.run_id,
                "experiment_id": result.experiment_id,
                "contract_sha256": contracts.contract_sha256,
                "evidence_status": "recorded_notebook_unexecuted",
            },
        },
        "cells": [
            _markdown_cell(
                "# End-to-end competition notebook\n\n"
                "Generated from a verified SwarmFold run using the confirmed competition "
                "profile. The cells are intentionally unexecuted: the recorded metric below "
                "comes from `run-result.json`, not from a claimed notebook execution.\n\n"
                f"- Experiment: `{result.experiment_id}`\n"
                f"- Recorded held-out {metric_name}: `{metric_value}`\n"
                f"- Split: `{proposal.split_id}`\n"
                f"- Model family: `{proposal.model_family}`\n"
                "- Delivery status: conditional until this notebook is rerun against the "
                "original uploaded bundle."
            ),
            _literature_cell(literature_review),
            _quality_evidence_cell(quality_evidence),
            _code_cell(
                1,
                "Configure evidence-linked inputs",
                f"""from pathlib import Path
import platform

import joblib
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

BUNDLE_DIR = Path.cwd()
TRAIN_PATH = BUNDLE_DIR / {train_name!r}
TEST_PATH = BUNDLE_DIR / {test_name!r}
SUBMISSION_TEMPLATE_PATH = BUNDLE_DIR / {submission_name!r}
PROJECT_BRIEF_PATH = BUNDLE_DIR / {project_name!r}
TARGET_COLUMN = {proposal.target_column!r}
FEATURE_COLUMNS = {pformat(contracts.data.feature_columns)}
ID_COLUMNS = {pformat(contracts.competition.id_columns)}
PREDICTION_COLUMN = {contracts.competition.prediction_column!r}
METRIC_NAME = {metric_name!r}
POSITIVE_CLASS = {result.positive_class!r}
SEED = {proposal.seed}
TEST_SIZE = {proposal.test_size!r}

print({{
    "python": platform.python_version(),
    "pandas": pd.__version__,
    "sklearn": sklearn.__version__,
}})""",
            ),
            _code_cell(
                2,
                "Load the confirmed bundle",
                """project_brief = PROJECT_BRIEF_PATH.read_text(encoding="utf-8")
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
submission_template = pd.read_csv(SUBMISSION_TEMPLATE_PATH)

print({
    "train_shape": train.shape,
    "test_shape": test.shape,
    "submission_shape": submission_template.shape,
    "brief_characters": len(project_brief),
})""",
            ),
            _code_cell(
                3,
                "Audit schema, grain, missingness, and leakage risks",
                """assert TARGET_COLUMN in train.columns
assert TARGET_COLUMN not in test.columns
assert list(train.drop(columns=[TARGET_COLUMN]).columns) == list(test.columns)
assert len(test) == len(submission_template)
assert list(submission_template.columns) == ID_COLUMNS + [PREDICTION_COLUMN]

quality = pd.DataFrame({
    "dtype": train.dtypes.astype(str),
    "missing_count": train.isna().sum(),
    "missing_rate": train.isna().mean(),
    "unique_count": train.nunique(dropna=False),
})
print({
    "duplicate_training_rows": int(train.duplicated().sum()),
    "target_distribution": train[TARGET_COLUMN].value_counts(dropna=False).to_dict(),
})
display(quality)""",
            ),
            _code_cell(
                4,
                "Create a leakage-safe held-out split and model",
                f"""X = train[FEATURE_COLUMNS]
y = train[TARGET_COLUMN]
X_train, X_validation, y_train, y_validation = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=SEED,
    stratify=y,
)

{_model_source(proposal)}
print({{"train_rows": len(X_train), "validation_rows": len(X_validation)}})""",
            ),
            _code_cell(
                5,
                "Fit and evaluate only on held-out observations",
                f"""model.fit(X_train, y_train)
validation_predictions = model.predict(X_validation)
if METRIC_NAME == "roc_auc":
    class_index = [str(value) for value in model.classes_].index(POSITIVE_CLASS)
    validation_scores = model.predict_proba(X_validation)[:, class_index]
    validation_metric = roc_auc_score(y_validation == POSITIVE_CLASS, validation_scores)
else:
    validation_metric = accuracy_score(y_validation, validation_predictions)

print({{
    "metric": METRIC_NAME,
    "held_out_value": float(validation_metric),
    "recorded_swarmfold_value": {float(metric_value)!r},
    "absolute_difference": abs(float(validation_metric) - {float(metric_value)!r}),
}})""",
            ),
            _code_cell(
                6,
                "Refit on all labeled rows and predict inference rows",
                """model.fit(X, y)
if METRIC_NAME == "roc_auc":
    class_index = [str(value) for value in model.classes_].index(POSITIVE_CLASS)
    test_predictions = model.predict_proba(test[FEATURE_COLUMNS])[:, class_index]
else:
    test_predictions = model.predict(test[FEATURE_COLUMNS])
assert len(test_predictions) == len(test)
print({{
    "prediction_rows": len(test_predictions),
    "missing_predictions": int(pd.isna(test_predictions).sum()),
}})""",
            ),
            _code_cell(
                7,
                "Build and validate the submission artifact",
                """submission = submission_template.copy()
submission[PREDICTION_COLUMN] = test_predictions

assert list(submission.columns) == list(submission_template.columns)
assert len(submission) == len(test)
assert submission[PREDICTION_COLUMN].notna().all()
for column in ID_COLUMNS:
    assert submission[column].equals(submission_template[column])

OUTPUT_PATH = BUNDLE_DIR / "submission.csv"
submission.to_csv(OUTPUT_PATH, index=False)
print({{
    "output": str(OUTPUT_PATH),
    "rows": len(submission),
    "columns": list(submission.columns),
}})""",
            ),
            _code_cell(
                8,
                "Persist the fitted pipeline and reproduction metadata",
                f"""MODEL_PATH = BUNDLE_DIR / "model.joblib"
joblib.dump(model, MODEL_PATH)
reproduction = {{
    "experiment_id": {result.experiment_id!r},
    "source_run_id": {result.run_id!r},
    "split_id": {proposal.split_id!r},
    "seed": SEED,
    "model_family": {proposal.model_family!r},
    "feature_columns": FEATURE_COLUMNS,
    "target_column": TARGET_COLUMN,
    "submission_path": str(OUTPUT_PATH),
    "model_path": str(MODEL_PATH),
}}
print(reproduction)""",
            ),
        ],
    }
    return write_json(path, notebook)


def _literature_cell(review: dict[str, object] | None) -> dict[str, object]:
    if not review:
        return _markdown_cell(
            "## Literature research\n\nNo literature review was attached to this run."
        )
    lines = [
        "## Literature research",
        "",
        "These sources informed hypotheses only. Local validation remains the evidence for "
        "dataset-specific performance.",
        "",
    ]
    for source in review.get("sources", []):
        if not isinstance(source, dict):
            continue
        title = str(source.get("title", "Untitled work"))
        url = str(source.get("url", ""))
        year = source.get("year")
        suffix = f" ({year})" if year else ""
        lines.append(f"- [{title}]({url}){suffix}")
    if len(lines) == 4:
        lines.append("No citable source metadata was retrieved.")
    recommendations = review.get("handling_recommendations", [])
    if isinstance(recommendations, list) and recommendations:
        lines.extend(["", "### Handling recommendations", ""])
        lines.extend(f"- {item}" for item in recommendations)
    return _markdown_cell("\n".join(lines))


def _quality_evidence_cell(evidence: dict[str, object] | None) -> dict[str, object]:
    if not evidence:
        return _markdown_cell(
            "## Modeling quality gates\n\nNo modeling quality evidence was attached."
        )
    formulation = evidence.get("problem_formulation", {})
    validation = evidence.get("validation_audit", {})
    ensemble = evidence.get("ensemble_assessment", {})
    robustness = evidence.get("robustness_assessment", {})
    return _markdown_cell(
        "## Modeling quality gates\n\n"
        "The complete machine-readable record is stored in "
        "`modeling-quality-evidence.json`.\n\n"
        f"- Formulated task: `{formulation.get('task_type', 'unknown')}`\n"
        f"- Prediction structure: `{formulation.get('prediction_structure', 'unknown')}`\n"
        f"- Validation audit: `{validation.get('status', 'unknown')}`\n"
        f"- Ensemble decision: `{ensemble.get('decision', 'unknown')}`\n"
        f"- Robustness evidence: `{robustness.get('status', 'unknown')}`\n\n"
        "An ensemble is optional and is promoted only when its recorded score improves "
        "over the best eligible base model under the same validation contract."
    )


def _markdown_cell(source: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def _code_cell(number: int, title: str, source: str) -> dict[str, object]:
    value = f"# Cell {number} - {title}\n{source.strip()}\n"
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": value.splitlines(keepends=True),
    }


def _model_source(proposal: ExperimentProposal) -> str:
    parameters = proposal.parameters
    preamble = (
        "from sklearn.compose import ColumnTransformer\n"
        "from sklearn.impute import SimpleImputer\n"
        "from sklearn.pipeline import make_pipeline\n"
        "from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler\n\n"
        "numeric_columns = X_train.select_dtypes(include='number').columns.tolist()\n"
        "categorical_columns = [name for name in FEATURE_COLUMNS if name not in numeric_columns]\n"
    )
    if proposal.model_family == "logistic_regression":
        return preamble + (
            "from sklearn.linear_model import LogisticRegression\n"
            "transformers = []\n"
            "if numeric_columns:\n"
            "    transformers.append(('numeric', make_pipeline("
            "SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler()), "
            "numeric_columns))\n"
            "if categorical_columns:\n"
            "    transformers.append(('categorical', make_pipeline("
            "SimpleImputer(strategy='constant', fill_value='__missing__'), "
            "OneHotEncoder(handle_unknown='ignore', min_frequency=2)), "
            "categorical_columns))\n"
            "feature_pipeline = ColumnTransformer(transformers)\n\n"
            "model = make_pipeline(\n"
            "    feature_pipeline,\n"
            f"    LogisticRegression(C={float(parameters.get('c', 1.0))!r}, "
            "max_iter=1000, random_state=SEED),\n"
            ")"
        )
    if proposal.model_family == "random_forest":
        return preamble + (
            "from sklearn.ensemble import RandomForestClassifier\n\n"
            + _ordinal_preprocessor_source()
            + "model = make_pipeline(feature_pipeline, RandomForestClassifier(\n"
            f"    n_estimators={int(parameters.get('n_estimators', 100))},\n"
            f"    max_depth={int(parameters.get('max_depth', 5))},\n"
            "    random_state=SEED,\n"
            "    n_jobs=1,\n"
            "))"
        )
    if proposal.model_family == "hist_gradient_boosting":
        return preamble + (
            "from sklearn.ensemble import HistGradientBoostingClassifier\n\n"
            + _ordinal_preprocessor_source()
            + "model = make_pipeline(feature_pipeline, HistGradientBoostingClassifier(\n"
            f"    max_iter={int(parameters.get('max_iter', 100))},\n"
            f"    max_depth={int(parameters.get('max_depth', 3))},\n"
            "    random_state=SEED,\n"
            "))"
        )
    return preamble + (
        "from sklearn.ensemble import HistGradientBoostingClassifier, "
        "RandomForestClassifier, VotingClassifier\n"
        "from sklearn.linear_model import LogisticRegression\n"
        + _ordinal_preprocessor_source()
        + "model = make_pipeline(feature_pipeline, VotingClassifier(estimators=[\n"
        "    ('logistic', make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, "
        "random_state=SEED))),\n"
        "    ('forest', RandomForestClassifier(n_estimators=100, max_depth=5, "
        "random_state=SEED, n_jobs=1)),\n"
        "    ('histogram', HistGradientBoostingClassifier(max_iter=100, max_depth=3, "
        "random_state=SEED)),\n"
        "], voting='soft'))"
    )


def _ordinal_preprocessor_source() -> str:
    return (
        "transformers = []\n"
        "if numeric_columns:\n"
        "    transformers.append(('numeric', SimpleImputer("
        "strategy='median', keep_empty_features=True), numeric_columns))\n"
        "if categorical_columns:\n"
        "    transformers.append(('categorical', make_pipeline("
        "SimpleImputer(strategy='constant', fill_value='__missing__'), "
        "OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, "
        "encoded_missing_value=-2)), categorical_columns))\n"
        "feature_pipeline = ColumnTransformer(transformers)\n\n"
    )
