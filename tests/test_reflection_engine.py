from pathlib import Path

from swarmfold.contracts import freeze_contracts
from swarmfold.graph import run_experiment
from swarmfold.reflection import (
    IterationReflector,
    ReflectionMemoryStore,
)
from swarmfold.roles import inspect_experiment_memory

FIXTURE = Path(__file__).parent / "fixtures" / "demo_classification.csv"


def test_iteration_reflector_generates_insights() -> None:
    reflector = IterationReflector()

    completed_results = [
        {
            "experiment_id": "exp-logreg",
            "model_family": "logistic_regression",
            "status": "SUCCEEDED",
            "metrics": {"roc_auc": 0.76},
        },
        {
            "experiment_id": "exp-lgbm",
            "model_family": "lightgbm",
            "status": "SUCCEEDED",
            "metrics": {"roc_auc": 0.88},
        },
        {
            "experiment_id": "exp-bad",
            "model_family": "random_forest",
            "status": "FAILED",
            "error": "Memory limit exceeded",
        },
    ]

    ensemble_assessment = {
        "decision": "SUPPORTED",
        "ensemble_gain": 0.025,
    }

    report = reflector.generate_reflection(
        iteration_id="iter-01",
        completed_results=completed_results,
        metric_name="roc_auc",
        higher_is_better=True,
        ensemble_assessment=ensemble_assessment,
    )

    assert report.iteration_id == "iter-01"
    assert report.champion_experiment_id == "exp-lgbm"
    assert report.champion_score == 0.88
    assert report.best_family == "lightgbm"
    assert len(report.insights) >= 3
    assert len(report.failed_hypotheses) == 1
    assert "exp-bad" in report.failed_hypotheses[0]


def test_reflection_memory_store_save_load_and_transfer(tmp_path: Path) -> None:
    reflector = IterationReflector()
    store = ReflectionMemoryStore()

    completed = [
        {
            "experiment_id": "exp-rf",
            "model_family": "random_forest",
            "status": "SUCCEEDED",
            "metrics": {"accuracy": 0.82},
        }
    ]
    report = reflector.generate_reflection(
        iteration_id="iter-02",
        completed_results=completed,
        metric_name="accuracy",
    )

    saved_json = store.save_reflection(tmp_path, report)
    assert saved_json.is_file()
    assert (tmp_path / "memory" / "iter-02-reflection.md").is_file()

    loaded_report = store.load_reflection(saved_json)
    assert loaded_report.iteration_id == report.iteration_id
    assert loaded_report.champion_score == report.champion_score

    # Convert to prior knowledge package
    source_profile = {"target_column": "target", "metric": "accuracy"}
    prior_pkg = store.to_prior_knowledge_package(loaded_report, source_profile=source_profile)

    # Test compatibility with inspect_experiment_memory
    contracts_state = freeze_contracts(
        {
            "dataset_path": str(FIXTURE.resolve()),
            "target_column": "target",
            "seed": 42,
            "test_size": 0.25,
            "output_dir": str(tmp_path / "contracts"),
        }
    )
    state = {
        **contracts_state,
        "prior_knowledge": [prior_pkg],
    }

    result_state = inspect_experiment_memory(state)
    classified_prior = result_state["prior_knowledge"]
    assert len(classified_prior) == 1
    assert classified_prior[0]["transfer_status"] == "TRANSFERABLE_HYPOTHESIS"


def test_transferable_reflection_reorders_next_iteration_portfolio(tmp_path: Path) -> None:
    prior = {
        "source_iteration_id": "previous",
        "source_profile": {"target_column": "target", "metric": "accuracy"},
        "champion_model_family": "hist_gradient_boosting",
        "champion_score": 0.9,
        "insights": [],
        "failed_hypotheses": [],
    }

    result = run_experiment(
        FIXTURE,
        output_dir=tmp_path,
        max_experiments=1,
        prior_knowledge=[prior],
    )

    assert result["proposal"]["model_family"] == "hist_gradient_boosting"
    planner_report = result["role_reports"]["portfolio_planner"]
    assert planner_report["memory_changed_priority"] is True
