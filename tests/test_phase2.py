import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from swarmfold.graph import run_experiment
from swarmfold.resources import TrainingResourcePool
from swarmfold.roles import LOGICAL_ROLES

FIXTURE = Path(__file__).parent / "fixtures" / "demo_classification.csv"


def test_local_pod_covers_roles_models_and_one_packet(tmp_path: Path) -> None:
    result = run_experiment(FIXTURE, output_dir=tmp_path)

    assert result["status"] == "ACCEPTED"
    assert set(LOGICAL_ROLES).issubset(result["role_reports"])
    assert len(LOGICAL_ROLES) == 18
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
    assert len((packet_path / "findings.jsonl").read_text().splitlines()) == 4
    assert (packet_path / "modeling-quality-evidence.json").is_file()


def test_experiment_budget_limits_queue_execution(tmp_path: Path) -> None:
    result = run_experiment(FIXTURE, output_dir=tmp_path, max_experiments=2)

    assert len(result["experiment_queue"]) == 2
    assert len(result["completed_results"]) == 2
    assert result["stop_reason"] == "experiment budget exhausted"
    assert result["ensemble_gate"] == "PENDING"


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
