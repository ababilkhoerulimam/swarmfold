from pathlib import Path

from swarmfold.candidates import CandidateFactory
from swarmfold.contracts import freeze_contracts
from swarmfold.nodes import run_experiment_candidate
from swarmfold.optimization import AdaptiveScheduler, TrialPruner
from swarmfold.schemas import ContractBundle, ExperimentProposal

FIXTURE = Path(__file__).parent / "fixtures" / "demo_classification.csv"


def test_trial_pruner_higher_is_better() -> None:
    pruner = TrialPruner(min_folds_before_pruning=2, relative_threshold=0.15, higher_is_better=True)

    # 1 fold completed -> do not prune yet
    assert pruner.should_prune([0.4], best_benchmark_score=0.85) is False

    # 2 folds completed with severe degradation (0.4 vs 0.85) -> prune
    assert pruner.should_prune([0.4, 0.42], best_benchmark_score=0.85) is True

    # 2 folds completed with competitive score (0.83 vs 0.85) -> keep
    assert pruner.should_prune([0.83, 0.84], best_benchmark_score=0.85) is False


def test_trial_pruner_lower_is_better() -> None:
    pruner = TrialPruner(
        min_folds_before_pruning=2, relative_threshold=0.20, higher_is_better=False
    )

    # Low error vs benchmark -> keep
    assert pruner.should_prune([1.05, 1.02], best_benchmark_score=1.0) is False

    # High error vs benchmark -> prune
    assert pruner.should_prune([1.5, 1.6], best_benchmark_score=1.0) is True


def test_adaptive_scheduler_prioritization() -> None:
    scheduler = AdaptiveScheduler(exploration_weight=0.1)

    queue = [
        {"model_family": "logistic_regression", "parameters": {"c": 0.1}},
        {"model_family": "hist_gradient_boosting", "parameters": {"max_iter": 100}},
        {"model_family": "random_forest", "parameters": {"n_estimators": 50}},
    ]

    completed_results = [
        {"model_family": "logistic_regression", "metrics": {"accuracy": 0.65}},
        {"model_family": "hist_gradient_boosting", "metrics": {"accuracy": 0.85}},
    ]

    prioritized = scheduler.prioritize_queue(queue, completed_results, metric_name="accuracy")

    families_ordered = [cand["model_family"] for cand in prioritized]
    assert families_ordered.index("hist_gradient_boosting") < families_ordered.index(
        "logistic_regression"
    )


def test_candidate_factory_schedules_only_eligible_installed_models() -> None:
    registry = [
        {
            "model_family": "hist_gradient_boosting",
            "declared": True,
            "installed": True,
            "executable": True,
            "task_types": ["binary_classification"],
            "recommended_lanes": ["ai-b"],
        },
        {
            "model_family": "lightgbm",
            "declared": True,
            "installed": False,
            "executable": True,
            "task_types": ["binary_classification"],
            "recommended_lanes": ["ai-b"],
        },
        {
            "model_family": "stacking",
            "declared": True,
            "installed": True,
            "executable": True,
            "task_types": ["binary_classification"],
            "recommended_lanes": ["ai-b"],
        },
    ]

    candidates = CandidateFactory(registry).create("ai-b", 4)

    assert {item["model_family"] for item in candidates} == {
        "hist_gradient_boosting",
        "stacking",
    }
    assert all(item["model_family"] != "lightgbm" for item in candidates)


def test_kfold_runtime_records_pruned_trial(
    tmp_path: Path, monkeypatch
) -> None:
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
    contract_data = contracts.model_dump()
    contract_data["validation"]["strategy"] = "stratified_kfold"
    contract_data["validation"]["splits"] = 3
    contracts = ContractBundle.model_validate(contract_data)
    proposal = ExperimentProposal(
        experiment_id="pruned-candidate",
        hypothesis="A deliberately weak candidate should be pruned after fold evidence.",
        hypothesis_id="1234567890abcdef",
        model_family="logistic_regression",
        parameters={"c": 0.01},
        dataset_path=str(FIXTURE.resolve()),
        target_column="target",
        data_version=contracts.data.data_version,
        split_id=contracts.validation.split_id,
        metric_version=contracts.metric.metric_version,
        test_size=contracts.validation.test_size,
        seed=contracts.validation.seed,
    )
    monkeypatch.setattr(
        TrialPruner,
        "should_prune",
        lambda self, scores, benchmark: len(scores) >= 2 and benchmark is not None,
    )
    state = {
        **contracts_state,
        "contracts": contracts.model_dump(),
        "proposal": proposal.model_dump(),
        "output_dir": str(tmp_path / "run"),
        "completed_results": [
            {
                "experiment_id": "benchmark",
                "run_id": "benchmark-run",
                "status": "SUCCEEDED",
                "model_family": "random_forest",
                "metrics": {"accuracy": 0.95},
            }
        ],
    }

    result = run_experiment_candidate(state)["run_result"]

    assert result["status"] == "PRUNED"
    assert len(result["fold_scores"]) == 2
    assert result["pruning_reason"]
