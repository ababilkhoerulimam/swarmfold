from pathlib import Path

import pytest

from swarmfold.adapters import (
    CatBoostAdapter,
    LightGBMAdapter,
    XGBoostAdapter,
    get_adapter,
    is_adapter_available,
)
from swarmfold.contracts import freeze_contracts
from swarmfold.nodes import run_experiment_candidate
from swarmfold.schemas import ContractBundle, ExperimentProposal

FIXTURE = Path(__file__).parent / "fixtures" / "demo_classification.csv"


def test_adapter_registry_lookup() -> None:
    lgb_adapter = get_adapter("lightgbm")
    assert isinstance(lgb_adapter, LightGBMAdapter)

    xgb_adapter = get_adapter("xgboost")
    assert isinstance(xgb_adapter, XGBoostAdapter)

    cb_adapter = get_adapter("catboost")
    assert isinstance(cb_adapter, CatBoostAdapter)

    with pytest.raises(ValueError, match="No adapter registered"):
        get_adapter("non_existent_family")


def test_adapter_build_estimators_if_installed() -> None:
    for family in ("lightgbm", "xgboost", "catboost"):
        adapter = get_adapter(family)
        if adapter.is_available():
            clf = adapter.build_estimator(
                task_type="binary_classification",
                hyperparameters={"n_estimators": 10},
                random_state=42,
            )
            assert hasattr(clf, "fit")
            assert hasattr(clf, "predict")
            assert hasattr(clf, "predict_proba")

            reg = adapter.build_estimator(
                task_type="regression",
                hyperparameters={"n_estimators": 10},
                random_state=42,
            )
            assert hasattr(reg, "fit")
            assert hasattr(reg, "predict")


def test_gradient_boosting_cv_experiment(tmp_path: Path) -> None:
    # Test end-to-end execution of a boosting model if installed
    installed_boosting = [
        family for family in ("lightgbm", "xgboost", "catboost") if is_adapter_available(family)
    ]
    if not installed_boosting:
        pytest.skip("No boosting backend is currently installed in this environment")

    test_family = installed_boosting[0]

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
    contracts_dict = contracts.model_dump()
    contracts_dict["validation"]["strategy"] = "stratified_kfold"
    contracts_dict["validation"]["splits"] = 3
    modified_contracts = ContractBundle.model_validate(contracts_dict)

    proposal = ExperimentProposal(
        experiment_id=f"test-{test_family}-001",
        hypothesis=f"Test {test_family} with 3-fold CV",
        hypothesis_id="abcdef0123456789",
        model_family=test_family,  # type: ignore[arg-type]
        parameters={"n_estimators": 15},
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
    oof_path = tmp_path / "output" / "runs" / run_id / "oof_predictions.parquet"
    assert oof_path.is_file()
