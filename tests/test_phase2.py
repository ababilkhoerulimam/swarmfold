import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from swarmfold.graph import run_experiment
from swarmfold.model_registry import is_model_executable
from swarmfold.resources import TrainingResourcePool
from swarmfold.roles import LOGICAL_ROLES, _lane_candidates
from swarmfold.schemas import BudgetPolicy, ExperimentProposal

FIXTURE = Path(__file__).parent / "fixtures" / "demo_classification.csv"


def test_local_pod_covers_roles_models_and_one_packet(tmp_path: Path) -> None:
    result = run_experiment(FIXTURE, output_dir=tmp_path)

    assert result["status"] == "ACCEPTED"
    assert set(LOGICAL_ROLES).issubset(result["role_reports"])
    assert len(LOGICAL_ROLES) == 19
    assert len(result["completed_results"]) == 4
    assert len(result["duplicate_hypotheses"]) == 1
    assert result["ensemble_gate"] == "OPEN"
    assert {
        "logistic_regression",
        "random_forest",
        "hist_gradient_boosting",
        "soft_voting",
    }.issubset(result["model_coverage"])
    assert result["resource_usage"]["observed_peak_training"] == 1
    assert result["problem_formulation"]["task_type"] == "binary_classification"
    assert result["validation_audit"]["status"] in {"PASS", "PASS_WITH_WARNINGS"}
    assert result["ensemble_assessment"]["decision"] in {"SUPPORTED", "REJECTED"}
    assert len(result["robustness_assessment"]["interval"]) == 2

    packets = list((tmp_path / "packets").glob("*/candidate-packet/COMPLETED.json"))
    assert len(packets) == 1
    packet_path = packets[0].parent
    assert len((packet_path / "experiment-ledger.jsonl").read_text().splitlines()) == 4
    assert len((packet_path / "findings.jsonl").read_text().splitlines()) >= 4
    assert (packet_path / "iteration-reflection.json").is_file()
    assert (packet_path / "iteration-reflection.md").is_file()
    assert (packet_path / "modeling-quality-evidence.json").is_file()


def test_experiment_budget_limits_queue_execution(tmp_path: Path) -> None:
    result = run_experiment(FIXTURE, output_dir=tmp_path, max_experiments=2)

    assert len(result["experiment_queue"]) == 2
    assert len(result["completed_results"]) == 2
    assert result["stop_reason"] == "experiment budget exhausted"
    assert result["ensemble_gate"] == "OPEN"


def test_boosting_lane_executes_available_lightgbm_adapter(tmp_path: Path) -> None:
    if not is_model_executable("lightgbm"):
        pytest.skip("LightGBM is not installed in this environment")

    result = run_experiment(
        FIXTURE,
        output_dir=tmp_path,
        lane_id="ai-b",
        max_experiments=2,
    )

    assert "lightgbm" in result["model_coverage"]
    eligibility = result["role_reports"]["model_scout"]["eligibility"]
    lightgbm = next(item for item in eligibility if item["model_family"] == "lightgbm")
    assert lightgbm == {
        "model_family": "lightgbm",
        "declared": True,
        "installed": True,
        "executable": True,
        "applicable": True,
        "scheduled": True,
        "eligible": True,
    }


@pytest.mark.parametrize(
    ("lane_id", "expected_families"),
    [
        ("ai-d", {"bagging", "stacking"}),
        ("ai-e", {"soft_voting", "stacking"}),
    ],
)
def test_ensemble_lanes_execute_full_portfolio_and_select_best_candidate(
    tmp_path: Path, lane_id: str, expected_families: set[str]
) -> None:
    result = run_experiment(
        FIXTURE,
        output_dir=tmp_path / lane_id,
        lane_id=lane_id,
        max_experiments=5,
    )

    assert len(result["completed_results"]) == 5
    assert expected_families.issubset(result["model_coverage"])
    metric_name = result["contracts"]["metric"]["metric_name"]
    successful_scores = [
        item["metrics"][metric_name]
        for item in result["completed_results"]
        if item["status"] == "SUCCEEDED"
    ]
    assert result["run_result"]["metrics"][metric_name] == pytest.approx(
        max(successful_scores)
    )
    assert result["stop_reason"] == "experiment budget exhausted"


def test_checkpoint_resume_does_not_repeat_completed_training(tmp_path: Path) -> None:
    thread_id = "resume-phase2-test"
    interrupted = run_experiment(
        FIXTURE,
        output_dir=tmp_path,
        thread_id=thread_id,
        interrupt_before="result_critic",
    )

    assert interrupted["status"] == "INTERRUPTED"
    assert interrupted["next_nodes"] == ["result_critic"]
    assert len(interrupted["completed_results"]) == 1
    first_run_id = interrupted["completed_results"][0]["run_id"]

    resumed = run_experiment(
        FIXTURE,
        output_dir=tmp_path,
        thread_id=thread_id,
        resume=True,
    )

    assert resumed["status"] == "ACCEPTED"
    assert len(resumed["completed_results"]) == 4
    assert resumed["completed_results"][0]["run_id"] == first_run_id
    assert len(list((tmp_path / "runs").iterdir())) == 4
    cached = json.loads((tmp_path / "results" / "p2-1-logistic_regression-s42.json").read_text())
    assert cached["run_id"] == first_run_id


def test_training_resource_pool_enforces_concurrency_limit() -> None:
    pool = TrainingResourcePool(max_concurrent=1)

    def train() -> None:
        with pool.slot():
            time.sleep(0.03)

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(lambda _: train(), range(2)))

    assert pool.peak_active == 1


def test_lane_candidates_preserve_distinct_modeling_charters_and_ensembles() -> None:
    candidates = {
        lane_id: _lane_candidates(lane_id, 5)
        for lane_id in ("ai-a", "ai-b", "ai-c", "ai-d", "ai-e")
    }

    assert candidates["ai-a"][0]["model_family"] == "logistic_regression"
    assert candidates["ai-b"][0]["model_family"] == "hist_gradient_boosting"
    assert candidates["ai-d"][0]["model_family"] == "sgd_log_loss"
    assert {item["model_family"] for item in candidates["ai-d"]} >= {
        "bagging",
        "stacking",
    }
    for lane_candidates in candidates.values():
        assert any(
            item["model_family"] in {"bagging", "soft_voting", "stacking"}
            for item in lane_candidates
        )


def test_training_budget_has_no_default_time_limit_and_accepts_ensemble_weights() -> None:
    assert BudgetPolicy(max_experiments=5).max_training_seconds is None
    fields = {
        "experiment_id": "ensemble-1",
        "hypothesis": "Weighted diverse members improve ranking stability.",
        "hypothesis_id": "a" * 16,
        "model_family": "soft_voting",
        "parameters": {"weights": [2, 1, 1]},
        "dataset_path": "train.csv",
        "target_column": "target",
        "data_version": "sha256:" + "b" * 64,
        "split_id": "holdout-" + "c" * 12,
        "metric_version": "roc-auc-v1",
        "test_size": 0.25,
        "seed": 42,
    }

    assert ExperimentProposal(**fields).parameters["weights"] == [2, 1, 1]
