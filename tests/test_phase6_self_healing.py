import csv
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from swarmfold.artifacts import sha256_file
from swarmfold.contracts import contract_bundle_sha256
from swarmfold.healing import (
    CancellationToken,
    CheckpointManager,
    ExecutionCancelledError,
    FailureKind,
    FailureRecord,
    ResourceExceededError,
    ResourcePolicy,
    SelfHealingDebugger,
    classify_failure,
)
from swarmfold.nodes import run_experiment_candidate
from swarmfold.schemas import (
    CompetitionContract,
    ContractBundle,
    DataContract,
    ExperimentProposal,
    MetricContract,
    ValidationContract,
)


def test_failure_taxonomy_classification() -> None:
    # 1. Out of memory
    oom_exc = MemoryError("Unable to allocate 4.00 GiB for an array")
    oom_record = classify_failure(oom_exc)
    assert oom_record.kind == FailureKind.OUT_OF_MEMORY
    assert oom_record.recoverable is True

    # 2. Dependency unavailable
    dep_exc = ModuleNotFoundError("No module named 'catboost'")
    dep_record = classify_failure(dep_exc)
    assert dep_record.kind == FailureKind.DEPENDENCY_UNAVAILABLE
    assert dep_record.recoverable is False

    # 3. Timeout or stalled
    time_exc = TimeoutError("Execution timed out after 30s")
    time_record = classify_failure(time_exc)
    assert time_record.kind == FailureKind.TIMEOUT_OR_STALLED
    assert time_record.recoverable is False

    res_exc = ResourceExceededError("total_duration_seconds", 10.0, 15.0)
    res_record = classify_failure(res_exc)
    assert res_record.kind == FailureKind.TIMEOUT_OR_STALLED

    # 4. Numerical instability
    num_exc = FloatingPointError("overflow encountered in exp")
    num_record = classify_failure(num_exc)
    assert num_record.kind == FailureKind.NUMERICAL_INSTABILITY
    assert num_record.recoverable is True

    # 5. Invalid parameter
    param_exc = ValueError("Solver lbfgs supports only 'l2' or None penalties, got l1 penalty.")
    param_record = classify_failure(param_exc)
    assert param_record.kind == FailureKind.INVALID_PARAMETER
    assert param_record.recoverable is True

    # 6. Incompatible data shape or dtype
    shape_exc = ValueError("Incompatible shapes: expected 2D array, got 1D")
    shape_record = classify_failure(shape_exc)
    assert shape_record.kind == FailureKind.INCOMPATIBLE_DATA_SHAPE_OR_DTYPE
    assert shape_record.recoverable is True

    # 7. Provider or network failure
    net_exc = ConnectionError("Failed to connect to remote host: connection refused")
    net_record = classify_failure(net_exc)
    assert net_record.kind == FailureKind.PROVIDER_OR_NETWORK_FAILURE
    assert net_record.recoverable is True
    assert net_record.details["retry_domain"] == "provider_operation"
    assert SelfHealingDebugger().can_repair("provider-call", net_record) is False
    # can_retry_provider should allow retries (up to max)
    debugger_prov = SelfHealingDebugger(max_provider_retries=3)
    assert debugger_prov.can_retry_provider("provider-call", net_record) is True
    # Training repair budget is unaffected by provider retry
    assert debugger_prov.can_repair("provider-call", net_record) is False


    # 8. Artifact verification
    art_exc = ValueError("Artifact verification failed: hash mismatch")
    art_record = classify_failure(art_exc)
    assert art_record.kind == FailureKind.ARTIFACT_VERIFICATION_FAILURE
    assert art_record.recoverable is False

    # 9. Unknown
    unk_exc = RuntimeError("Something completely unhandled occurred")
    unk_record = classify_failure(unk_exc)
    assert unk_record.kind == FailureKind.UNKNOWN
    assert unk_record.recoverable is False


def test_debugger_oom_reduction() -> None:
    debugger = SelfHealingDebugger(max_retries=2)
    proposal = ExperimentProposal(
        experiment_id="exp-oom",
        hypothesis="Testing OOM recovery with large trees",
        hypothesis_id="0123456789abcdef",
        model_family="random_forest",
        feature_strategy="scaled_numeric",
        parameters={
            "max_depth": 16,
            "num_leaves": 128,
            "n_estimators": 200,
            "n_jobs": 8,
        },
        dataset_path="data.csv",
        target_column="target",
        split_id="split-01",
        data_version="sha256:" + "a" * 64,
        metric_version="accuracy-v1",
        seed=42,
    )
    failure = FailureRecord(
        kind=FailureKind.OUT_OF_MEMORY,
        exception_type="MemoryError",
        message="Unable to allocate memory",
        recoverable=True,
    )
    assert debugger.can_repair(proposal.experiment_id, failure) is True
    repaired, action = debugger.attempt_repair(proposal, failure)
    assert repaired.parameters["max_depth"] == 8
    assert repaired.parameters["num_leaves"] == 64
    assert repaired.parameters["n_estimators"] == 100
    assert repaired.parameters["n_jobs"] == 1
    assert "reduced max_depth" in action
    assert "forced n_jobs=1" in action
    assert repaired.experiment_id == "exp-oom-repair-1"


def test_debugger_invalid_param_sanitization() -> None:
    debugger = SelfHealingDebugger(max_retries=2)
    # 1. lbfgs with l1 penalty
    proposal = ExperimentProposal(
        experiment_id="exp-param-1",
        hypothesis="Testing invalid solver penalty combination",
        hypothesis_id="0123456789abcdef",
        model_family="logistic_regression",
        feature_strategy="scaled_numeric",
        parameters={"solver": "lbfgs", "penalty": "l1"},
        dataset_path="data.csv",
        target_column="target",
        split_id="split-01",
        data_version="sha256:" + "a" * 64,
        metric_version="accuracy-v1",
        seed=42,
    )
    failure = classify_failure(ValueError("Solver lbfgs supports only 'l2' or None penalties"))
    repaired, action = debugger.attempt_repair(proposal, failure)
    assert repaired.parameters["penalty"] == "l2"
    assert "switched incompatible penalty='l1' to penalty='l2'" in action

    # 2. Non-positive alpha and learning_rate
    proposal2 = ExperimentProposal(
        experiment_id="exp-param-2",
        hypothesis="Testing non-positive learning rate and regularizer",
        hypothesis_id="0123456789abcdef",
        model_family="sgd_log_loss",
        feature_strategy="scaled_numeric",
        parameters={"alpha": -0.5, "learning_rate": -0.01},
        dataset_path="data.csv",
        target_column="target",
        split_id="split-01",
        data_version="sha256:" + "a" * 64,
        metric_version="accuracy-v1",
        seed=42,
    )
    failure2 = classify_failure(ValueError("Invalid parameter alpha must be > 0"))
    repaired2, action2 = debugger.attempt_repair(proposal2, failure2)
    assert repaired2.parameters["alpha"] == 1.0
    assert repaired2.parameters["learning_rate"] == 0.1
    assert "clamped non-positive alpha" in action2


def test_debugger_numerical_instability() -> None:
    debugger = SelfHealingDebugger(max_retries=2)
    proposal = ExperimentProposal(
        experiment_id="exp-num",
        hypothesis="Testing numerical instability mitigation",
        hypothesis_id="0123456789abcdef",
        model_family="logistic_regression",
        feature_strategy="scaled_numeric",
        parameters={"C": 10.0, "learning_rate": 0.1},
        dataset_path="data.csv",
        target_column="target",
        split_id="split-01",
        data_version="sha256:" + "a" * 64,
        metric_version="accuracy-v1",
        seed=42,
    )
    failure = classify_failure(FloatingPointError("loss is nan in iteration"))
    repaired, action = debugger.attempt_repair(proposal, failure)
    assert repaired.parameters["C"] == pytest.approx(1.0)
    assert repaired.parameters["learning_rate"] == pytest.approx(0.05)
    assert "halved learning_rate" in action


def test_debugger_feature_fallback() -> None:
    debugger = SelfHealingDebugger(max_retries=2)
    proposal = ExperimentProposal(
        experiment_id="exp-shape",
        hypothesis="Testing feature fallback for shape incompatibility",
        hypothesis_id="0123456789abcdef",
        model_family="logistic_regression",
        feature_strategy="interaction_pairs",
        parameters={},
        dataset_path="data.csv",
        target_column="target",
        split_id="split-01",
        data_version="sha256:" + "a" * 64,
        metric_version="accuracy-v1",
        seed=42,
    )
    failure = classify_failure(ValueError("Dense matrix expected for sparse features"))
    repaired, action = debugger.attempt_repair(proposal, failure)
    assert repaired.feature_strategy == "robust_scaled"
    assert "fallback feature strategy" in action


def test_debugger_bounded_retries() -> None:
    debugger = SelfHealingDebugger(max_retries=2)
    proposal = ExperimentProposal(
        experiment_id="exp-bound",
        hypothesis="Testing retry boundary limit",
        hypothesis_id="0123456789abcdef",
        model_family="logistic_regression",
        feature_strategy="scaled_numeric",
        parameters={"solver": "lbfgs", "penalty": "l1"},
        dataset_path="data.csv",
        target_column="target",
        split_id="split-01",
        data_version="sha256:" + "a" * 64,
        metric_version="accuracy-v1",
        seed=42,
    )
    failure = classify_failure(ValueError("Solver lbfgs supports only l2"))

    # Attempt 1
    assert debugger.can_repair(proposal.experiment_id, failure) is True
    rep1, _ = debugger.attempt_repair(proposal, failure)
    assert rep1.experiment_id == "exp-bound-repair-1"

    # Attempt 2
    assert debugger.can_repair(proposal.experiment_id, failure) is True
    rep2, _ = debugger.attempt_repair(rep1, failure)
    assert rep2.experiment_id == "exp-bound-repair-2"

    # Attempt 3 - must be rejected by bound!
    assert debugger.can_repair(proposal.experiment_id, failure) is False
    assert len(debugger.repair_history) == 2


def test_checkpoint_manager_save_load_clear(tmp_path: Path) -> None:
    mgr = CheckpointManager(tmp_path / "checkpoints" / "exp-01")
    assert mgr.completed_folds() == []
    assert mgr.has_fold(0) is False

    # Save fold 0
    val_indices = np.array([10, 11, 12])
    predictions = np.array([1, 0, 1])
    targets = np.array([1, 1, 0])
    test_preds = np.array([1, 1])
    mock_model = LogisticRegression().fit([[0], [1]], [0, 1])

    mgr.save_fold(
        fold_idx=0,
        val_indices=val_indices,
        predictions=predictions,
        targets=targets,
        metric_score=0.6667,
        scores=np.array([0.9, 0.2, 0.8]),
        test_predictions=test_preds,
        model=mock_model,
    )

    assert mgr.has_fold(0) is True
    assert mgr.has_fold(1) is False
    assert mgr.completed_folds() == [0]

    # Load fold 0
    loaded = mgr.load_fold(0)
    assert loaded["fold_idx"] == 0
    assert loaded["metric_score"] == pytest.approx(0.6667)
    np.testing.assert_array_equal(loaded["val_indices"], val_indices)
    np.testing.assert_array_equal(loaded["predictions"], predictions.astype(str))
    np.testing.assert_array_equal(loaded["targets"], targets.astype(str))
    assert loaded["scores"] is not None
    np.testing.assert_allclose(loaded["scores"], [0.9, 0.2, 0.8])
    assert loaded["model"] is not None

    # Clear
    mgr.clear()
    assert mgr.has_fold(0) is False
    assert mgr.completed_folds() == []


def test_resource_policy_and_cancellation() -> None:
    policy = ResourcePolicy(
        max_training_seconds_per_fold=1.0,
        max_total_seconds=5.0,
        max_memory_mb=1024.0,
    )

    # Within limits
    policy.check_fold_budget(0, 0.5)
    policy.check_total_budget(2.0)
    policy.check_memory(512.0)
    policy.check_cancellation()

    # Exceed fold budget
    with pytest.raises(ResourceExceededError, match="fold_0_duration_seconds"):
        policy.check_fold_budget(0, 1.5)

    # Exceed total budget
    with pytest.raises(ResourceExceededError, match="total_duration_seconds"):
        policy.check_total_budget(6.0)

    # Exceed memory budget
    with pytest.raises(ResourceExceededError, match="peak_memory_mb"):
        policy.check_memory(2048.0)

    # Cancellation token
    token = CancellationToken()
    policy_with_token = ResourcePolicy(cancellation_token=token)
    policy_with_token.check_cancellation()
    assert token.is_cancelled is False

    token.cancel("User requested stop")
    assert token.is_cancelled is True
    with pytest.raises(ExecutionCancelledError, match="User requested stop"):
        policy_with_token.check_cancellation()


def _create_sample_csv(path: Path, n_rows: int = 40) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row", "f1", "f2", "target"])
        for i in range(n_rows):
            writer.writerow([i, float(i * 0.1), float(i * -0.2), i % 2])


def _create_contracts(dataset_path: Path, strategy: str = "stratified_holdout") -> ContractBundle:
    data_hash = sha256_file(dataset_path)
    bundle = ContractBundle(
        competition=CompetitionContract(
            competition_id="comp-test",
            task_type="binary_classification",
            target_column="target",
            id_columns=["row"],
        ),
        data=DataContract(
            data_version=f"sha256:{data_hash}",
            dataset_path=str(dataset_path),
            dataset_sha256=data_hash,
            row_count=40,
            feature_columns=["f1", "f2"],
            target_column="target",
            target_classes=["0", "1"],
        ),
        metric=MetricContract(
            metric_name="accuracy",
            metric_version="accuracy-v1",
            higher_is_better=True,
            minimum=0.0,
            maximum=1.0,
        ),
        validation=ValidationContract(
            split_id="split-01",
            strategy=strategy,
            seed=42,
            splits=3,
        ),
    )
    return bundle.model_copy(update={"contract_sha256": contract_bundle_sha256(bundle)})


def test_node_self_healing_recovery(tmp_path: Path) -> None:
    data_csv = tmp_path / "data.csv"
    _create_sample_csv(data_csv, 40)
    contracts = _create_contracts(data_csv, strategy="stratified_holdout")
    contracts_dir = tmp_path / "contracts"
    contracts_dir.mkdir(parents=True, exist_ok=True)
    (contracts_dir / "contracts.json").write_text(contracts.model_dump_json(), encoding="utf-8")

    # Proposal with invalid parameter solver=lbfgs, penalty=l1
    proposal = ExperimentProposal(
        experiment_id="exp-invalid-solver",
        hypothesis="Testing automated recovery of invalid solver penalty",
        hypothesis_id="0123456789abcdef",
        model_family="logistic_regression",
        feature_strategy="scaled_numeric",
        parameters={"solver": "lbfgs", "penalty": "l1"},
        dataset_path=str(data_csv),
        target_column="target",
        split_id="split-01",
        data_version=contracts.data.data_version,
        metric_version=contracts.metric.metric_version,
        seed=42,
    )

    state = {
        "proposal": proposal.model_dump(),
        "contracts": contracts.model_dump(),
        "output_dir": str(tmp_path),
        "completed_results": [],
    }

    out_state = run_experiment_candidate(state)
    result = out_state["run_result"]
    assert result["status"] == "SUCCEEDED"
    assert result["original_experiment_id"] == "exp-invalid-solver"
    assert len(result["repair_history"]) == 1
    assert result["repair_history"][0]["failure_kind"] == "invalid_parameter"
    assert "penalty='l2'" in result["repair_history"][0]["repair_action"]


def test_node_checkpoint_resumption(tmp_path: Path) -> None:
    data_csv = tmp_path / "data.csv"
    _create_sample_csv(data_csv, 60)
    contracts = _create_contracts(data_csv, strategy="stratified_kfold")
    contracts_dir = tmp_path / "contracts"
    contracts_dir.mkdir(parents=True, exist_ok=True)
    (contracts_dir / "contracts.json").write_text(contracts.model_dump_json(), encoding="utf-8")

    proposal = ExperimentProposal(
        experiment_id="exp-checkpoint-resume",
        hypothesis="Testing fold checkpoint resumption",
        hypothesis_id="0123456789abcdef",
        model_family="logistic_regression",
        feature_strategy="scaled_numeric",
        parameters={},
        dataset_path=str(data_csv),
        target_column="target",
        split_id="split-01",
        data_version=contracts.data.data_version,
        metric_version=contracts.metric.metric_version,
        seed=42,
    )

    # Pre-populate fold 0 checkpoint
    ckpt_dir = tmp_path / "checkpoints" / proposal.experiment_id
    ckpt_mgr = CheckpointManager(ckpt_dir)
    mock_model = LogisticRegression().fit([[0, 0], [1, 1]], [0, 1])
    ckpt_mgr.save_fold(
        fold_idx=0,
        val_indices=list(range(20)),
        predictions=np.zeros(20, dtype=int),
        targets=np.zeros(20, dtype=int),
        metric_score=1.0,
        scores=np.zeros(20),
        model=mock_model,
    )

    state = {
        "proposal": proposal.model_dump(),
        "contracts": contracts.model_dump(),
        "output_dir": str(tmp_path),
        "completed_results": [],
    }

    out_state = run_experiment_candidate(state)
    result = out_state["run_result"]
    assert result["status"] == "SUCCEEDED"
    assert result["resumed_from_checkpoint"] is True
    # Folds 0, 1, 2 should all be in checkpoint manager now
    assert set(ckpt_mgr.completed_folds()) == {0, 1, 2}


def test_node_resource_budget_exceeded(tmp_path: Path) -> None:
    data_csv = tmp_path / "data.csv"
    _create_sample_csv(data_csv, 40)
    contracts = _create_contracts(data_csv, strategy="stratified_holdout")

    proposal = ExperimentProposal(
        experiment_id="exp-timeout",
        hypothesis="Testing timeout budget enforcement",
        hypothesis_id="0123456789abcdef",
        model_family="logistic_regression",
        feature_strategy="scaled_numeric",
        parameters={},
        dataset_path=str(data_csv),
        target_column="target",
        split_id="split-01",
        data_version=contracts.data.data_version,
        metric_version=contracts.metric.metric_version,
        seed=42,
    )

    state = {
        "proposal": proposal.model_dump(),
        "contracts": contracts.model_dump(),
        "output_dir": str(tmp_path),
        "completed_results": [],
        "max_training_seconds_per_fold": 0.0000001,  # Guaranteed to trigger after fold fit
    }

    out_state = run_experiment_candidate(state)
    result = out_state["run_result"]
    assert result["status"] == "FAILED"
    assert "Resource limit exceeded for fold_0_duration_seconds" in result["error"]
    assert result["failure_record"]["kind"] == "timeout_or_stalled"


def test_provider_retry_budget() -> None:
    """SelfHealingDebugger should enforce provider retry budget and produce correct backoff."""
    failure = FailureRecord(
        kind=FailureKind.PROVIDER_OR_NETWORK_FAILURE,
        exception_type="ConnectionError",
        message="network unreachable",
        recoverable=True,
        details={"retry_domain": "provider_operation"},
    )
    oom_failure = FailureRecord(
        kind=FailureKind.OUT_OF_MEMORY,
        exception_type="MemoryError",
        message="oom",
        recoverable=True,
    )

    debugger = SelfHealingDebugger(max_retries=2, max_provider_retries=3)
    pid = "exp-net-001"

    # First 3 calls: eligible
    assert debugger.can_retry_provider(pid, failure) is True
    b0 = debugger.get_provider_backoff_seconds(pid)  # increments to 1
    assert debugger.can_retry_provider(pid, failure) is True
    b1 = debugger.get_provider_backoff_seconds(pid)  # increments to 2
    assert debugger.can_retry_provider(pid, failure) is True
    b2 = debugger.get_provider_backoff_seconds(pid)  # increments to 3
    # Budget exhausted
    assert debugger.can_retry_provider(pid, failure) is False

    # Backoff should be 1s, 2s, 4s (bounded exponential)
    assert b0 == pytest.approx(1.0)
    assert b1 == pytest.approx(2.0)
    assert b2 == pytest.approx(4.0)

    # Training repair budget must be independent of provider retry count
    assert debugger.can_repair(pid, oom_failure) is True

    # Non-provider failures must not match can_retry_provider
    assert debugger.can_retry_provider(pid, oom_failure) is False


def test_provider_backoff_cap() -> None:
    """Backoff must cap at 30 s regardless of retry count."""
    debugger = SelfHealingDebugger(max_provider_retries=10)
    pid = "exp-net-002"
    for _ in range(10):
        debugger.get_provider_backoff_seconds(pid)
    # After 10 increments (2^10 = 1024), result must be capped at 30
    debugger2 = SelfHealingDebugger(max_provider_retries=20)
    pid2 = "exp-net-003"
    # Manually set high count to verify cap
    debugger2.provider_retry_counts[pid2] = 10
    assert debugger2.get_provider_backoff_seconds(pid2) == pytest.approx(30.0)

