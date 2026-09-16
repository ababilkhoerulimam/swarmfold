from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from swarmfold.contracts import freeze_contracts
from swarmfold.metrics import get_metric_spec
from swarmfold.model_registry import (
    executable_model_families,
    is_model_executable,
    model_capability_registry,
    validate_proposal_executability,
)
from swarmfold.nodes import _compute_metric, run_experiment_candidate
from swarmfold.quality import formulate_problem
from swarmfold.schemas import (
    ContractBundle,
    ExperimentProposal,
)

FIXTURE = Path(__file__).parent / "fixtures" / "demo_classification.csv"


def test_capability_tri_state_registry() -> None:
    registry = model_capability_registry()
    family_map = {entry["model_family"]: entry for entry in registry}

    assert "logistic_regression" in family_map
    assert family_map["logistic_regression"]["capability_state"] == "EXECUTABLE"
    assert is_model_executable("logistic_regression") is True

    assert "catboost" in family_map
    assert family_map["catboost"]["capability_state"] in {"DECLARED", "INSTALLED"}
    assert is_model_executable("catboost") is False

    exec_families = executable_model_families()
    assert "logistic_regression" in exec_families
    assert "random_forest" in exec_families
    assert "hist_gradient_boosting" in exec_families
    assert "catboost" not in exec_families


def test_proposal_executability_validation() -> None:
    # Valid executable proposal raises no error
    validate_proposal_executability("logistic_regression")
    validate_proposal_executability("random_forest")

    # Declared but unexecutable proposal raises ValueError with state details
    with pytest.raises(ValueError, match="not EXECUTABLE"):
        validate_proposal_executability("catboost")


def test_stratified_kfold_cv_generates_complete_oof_parquet(tmp_path: Path) -> None:
    contracts_state = freeze_contracts(
        {
            "dataset_path": str(FIXTURE.resolve()),
            "target_column": "target",
            "seed": 42,
            "test_size": 0.25,
            "output_dir": str(tmp_path / "contracts"),
        }
    )
    contracts = ContractBundle.model_validate(contracts_state["contracts"])

    # Update contracts to use stratified_kfold with 3 splits
    contracts_dict = contracts.model_dump()
    contracts_dict["validation"]["strategy"] = "stratified_kfold"
    contracts_dict["validation"]["splits"] = 3
    modified_contracts = ContractBundle.model_validate(contracts_dict)

    proposal = ExperimentProposal(
        experiment_id="test-exp-001",
        hypothesis="Test random forest with 3-fold CV",
        hypothesis_id="0123456789abcdef",
        model_family="random_forest",
        parameters={"n_estimators": 10},
        dataset_path=str(FIXTURE.resolve()),
        target_column="target",
        data_version=contracts.data.data_version,
        split_id=contracts.validation.split_id,
        metric_version=contracts.metric.metric_version,
        test_size=0.25,
        seed=42,
    )

    state = {
        **contracts_state,
        "contracts": modified_contracts.model_dump(),
        "proposal": proposal.model_dump(),
        "output_dir": str(tmp_path / "output"),
    }

    result = run_experiment_candidate(state)

    assert result["run_result"]["status"] == "SUCCEEDED"
    assert "accuracy" in result["run_result"]["metrics"]
    assert 0.5 <= result["run_result"]["metrics"]["accuracy"] <= 1.0

    run_id = result["run_result"]["run_id"]
    run_dir = tmp_path / "output" / "runs" / run_id
    oof_path = run_dir / "oof_predictions.parquet"
    assert oof_path.is_file()

    oof_df = pd.read_parquet(oof_path)
    assert set(oof_df.columns) >= {"row", "fold", "target", "prediction"}

    # Total rows in full dataset
    raw_df = pd.read_csv(FIXTURE)
    assert len(oof_df) == len(raw_df)
    assert set(oof_df["fold"].unique()) == {0, 1, 2}


def test_metric_calculations_expanded() -> None:
    # Classification metrics
    targets = np.array([0, 1, 1, 0, 1, 0])
    preds = np.array([0, 1, 1, 0, 0, 0])
    probs = np.array([0.1, 0.9, 0.8, 0.2, 0.4, 0.3])

    acc = _compute_metric("accuracy", targets, preds)
    assert 0.0 <= acc <= 1.0

    roc = _compute_metric("roc_auc", targets, preds, probs)
    assert 0.0 <= roc <= 1.0

    f1 = _compute_metric("f1_macro", targets, preds)
    assert 0.0 <= f1 <= 1.0

    loss = _compute_metric("log_loss", targets, preds, probs)
    assert loss >= 0.0

    # Regression metrics
    reg_targets = np.array([10.0, 20.0, 30.0, 40.0])
    reg_preds = np.array([12.0, 19.0, 28.0, 42.0])

    rmse = _compute_metric("rmse", reg_targets, reg_preds)
    assert rmse > 0.0

    mae = _compute_metric("mae", reg_targets, reg_preds)
    assert mae > 0.0

    r2 = _compute_metric("r2", reg_targets, reg_preds)
    assert r2 <= 1.0


def test_metric_specs_define_direction_and_task_compatibility() -> None:
    assert get_metric_spec("roc_auc").higher_is_better is True
    assert get_metric_spec("rmse").higher_is_better is False
    assert get_metric_spec("rmse").is_better(1.0, 2.0) is True
    assert get_metric_spec("accuracy").is_better(0.9, 0.8) is True

    with pytest.raises(ValueError, match="does not support task"):
        get_metric_spec("rmse").validate_task("binary_classification")


@pytest.mark.parametrize(
    ("task_type", "metric_name", "target_values"),
    [
        ("multiclass_classification", "accuracy", ["cat", "dog", "bird"] * 8),
        ("regression", "rmse", [float(index * 3 + 1) for index in range(24)]),
    ],
)
def test_multiclass_and_regression_candidates_execute(
    tmp_path: Path,
    task_type: str,
    metric_name: str,
    target_values: list[str] | list[float],
) -> None:
    dataset = tmp_path / f"{task_type}.csv"
    pd.DataFrame(
        {
            "numeric": np.arange(len(target_values), dtype=float),
            "category": ["left", "right"] * (len(target_values) // 2),
            "target": target_values,
        }
    ).to_csv(dataset, index=False)
    state = {
        "dataset_path": str(dataset),
        "target_column": "target",
        "seed": 42,
        "test_size": 0.25,
        "output_dir": str(tmp_path / f"contracts-{task_type}"),
        "metric_name": metric_name,
        "problem_formulation": {"task_type": task_type},
    }
    state.update(freeze_contracts(state))
    contracts = ContractBundle.model_validate(state["contracts"])
    proposal = ExperimentProposal(
        experiment_id=f"{task_type}-candidate",
        hypothesis=f"A mixed feature {task_type} baseline should execute successfully.",
        hypothesis_id="fedcba9876543210",
        model_family="logistic_regression",
        dataset_path=str(dataset.resolve()),
        target_column="target",
        data_version=contracts.data.data_version,
        split_id=contracts.validation.split_id,
        metric_version=contracts.metric.metric_version,
        test_size=contracts.validation.test_size,
        seed=contracts.validation.seed,
    )
    state.update(
        proposal=proposal.model_dump(),
        output_dir=str(tmp_path / f"run-{task_type}"),
    )

    result = run_experiment_candidate(state)["run_result"]

    assert result["status"] == "SUCCEEDED"
    assert metric_name in result["metrics"]
    assert result["positive_class"] is None


def test_expanded_problem_formulation(tmp_path: Path) -> None:
    # Multiclass dataset
    multiclass_csv = tmp_path / "multiclass.csv"
    multiclass_csv.write_text(
        "feat1,feat2,label\n"
        "1.0,2.0,cat\n"
        "2.0,1.5,dog\n"
        "3.0,3.5,bird\n"
        "1.2,2.2,cat\n"
        "2.1,1.7,dog\n"
        "3.2,3.1,bird\n",
        encoding="utf-8",
    )
    state = {
        "dataset_path": str(multiclass_csv),
        "target_column": "label",
        "output_dir": str(tmp_path / "output"),
        "role_reports": {},
    }
    result = formulate_problem(state)
    assert result["problem_formulation"]["task_type"] == "multiclass_classification"
    assert result["problem_formulation"]["execution_supported"] is True

    # Regression dataset
    reg_csv = tmp_path / "regression.csv"
    reg_csv.write_text(
        "feat1,feat2,price\n"
        + "".join(f"{i * 0.5},{i * 1.2},{i * 10.0 + 5.5}\n" for i in range(20)),
        encoding="utf-8",
    )
    reg_state = {
        "dataset_path": str(reg_csv),
        "target_column": "price",
        "output_dir": str(tmp_path / "output_reg"),
        "role_reports": {},
    }
    reg_result = formulate_problem(reg_state)
    assert reg_result["problem_formulation"]["task_type"] == "regression"
    assert reg_result["problem_formulation"]["execution_supported"] is True
