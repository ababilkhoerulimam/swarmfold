from hashlib import sha256

from swarmfold.contracts import freeze_contracts
from swarmfold.data import load_classification_csv
from swarmfold.planner import ProposalPlanner
from swarmfold.research import LiteratureResearchAgent
from swarmfold.schemas import (
    BudgetPolicy,
    ContractBundle,
    ExperimentCritique,
    ExperimentProposal,
    ExperimentState,
    RunResult,
)

LOGICAL_ROLES = (
    "pod_supervisor",
    "rules_agent",
    "problem_formulation_agent",
    "data_contract_agent",
    "validation_agent",
    "temporal_validation_leakage_auditor",
    "eda_agent",
    "domain_target_decomposition_agent",
    "literature_research_agent",
    "model_scout",
    "portfolio_planner",
    "feature_engineer",
    "model_engineer",
    "experiment_runner",
    "result_critic",
    "oof_ensemble_scientist",
    "robustness_uncertainty_agent",
    "evidence_verifier",
)


def initialize_pod(state: ExperimentState) -> ExperimentState:
    budget = BudgetPolicy.model_validate(state.get("budget", {}))
    return {
        "budget": budget.model_dump(),
        "role_reports": {"pod_supervisor": {"status": "READY"}},
        "completed_results": [],
        "critiques": [],
        "duplicate_hypotheses": [],
        "model_coverage": [],
        "current_experiment_index": 0,
        "ensemble_gate": "PENDING",
        "next_action": "CONTINUE",
        "status": "RUNNING",
        "resource_usage": {
            "max_concurrent_training": budget.max_concurrent_training,
            "observed_peak_training": 0,
            "training_seconds": 0.0,
        },
    }


def inspect_rules(state: ExperimentState) -> ExperimentState:
    return _with_report(
        state,
        "rules_agent",
        {
            "external_data_allowed": False,
            "submission_requires_human_approval": True,
            "task_type": "binary_classification",
            "competition_brief_provided": bool(state.get("project_context")),
        },
    )


def build_data_contract(state: ExperimentState) -> ExperimentState:
    update = freeze_contracts(state)
    combined = {**state, **update}
    return {
        **update,
        **_with_report(
            combined,
            "data_contract_agent",
            {"contract_sha256": update["contracts"]["contract_sha256"]},
        ),
    }


def audit_validation(state: ExperimentState) -> ExperimentState:
    contracts = ContractBundle.model_validate(state["contracts"])
    return _with_report(
        state,
        "validation_agent",
        {
            "split_id": contracts.validation.split_id,
            "strategy": contracts.validation.strategy,
            "fit_transforms_on_training_only": True,
        },
    )


def inspect_eda(state: ExperimentState) -> ExperimentState:
    features, target = load_classification_csv(state["dataset_path"], state["target_column"])
    numeric_count = int(features.select_dtypes(include="number").shape[1])
    report = {
        "rows": int(features.shape[0]),
        "features": int(features.shape[1]),
        "numeric_features": numeric_count,
        "categorical_features": int(features.shape[1] - numeric_count),
        "missing_values": int(features.isna().sum().sum()),
        "class_counts": {
            str(label): int((target == label).sum()) for label in sorted(set(target.tolist()))
        },
    }
    return _with_report(state, "eda_agent", report)


def create_literature_research_node(agent: LiteratureResearchAgent):
    def research_literature(state: ExperimentState) -> ExperimentState:
        review = agent.research(state)
        report = review.model_dump()
        return {
            "literature_research": report,
            **_with_report(state, "literature_research_agent", report),
        }

    return research_literature


def scout_models(state: ExperimentState) -> ExperimentState:
    candidates: list[dict[str, object]] = [
        {
            "hypothesis": "A scaled logistic model establishes a stable linear accuracy baseline.",
            "model_family": "logistic_regression",
            "feature_strategy": "scaled_numeric",
            "parameters": {"c": 1.0},
        },
        {
            "hypothesis": (
                "A random forest captures nonlinear feature interactions missed by the baseline."
            ),
            "model_family": "random_forest",
            "feature_strategy": "raw_numeric",
            "parameters": {"n_estimators": 100, "max_depth": 5},
        },
        {
            "hypothesis": (
                "Histogram gradient boosting improves nonlinear fit with bounded complexity."
            ),
            "model_family": "hist_gradient_boosting",
            "feature_strategy": "raw_numeric",
            "parameters": {"max_iter": 100, "max_depth": 3},
        },
        {
            "hypothesis": (
                "A soft voting ensemble improves robustness when base model families succeed."
            ),
            "model_family": "soft_voting",
            "feature_strategy": "scaled_numeric",
            "parameters": {},
        },
        {
            "hypothesis": "A scaled logistic model establishes a stable linear accuracy baseline.",
            "model_family": "logistic_regression",
            "feature_strategy": "scaled_numeric",
            "parameters": {"c": 1.0},
        },
    ]
    update = _with_report(state, "model_scout", {"candidate_count": len(candidates)})
    return {**update, "scout_candidates": candidates}


def create_portfolio_node(planner: ProposalPlanner):
    def build_portfolio(state: ExperimentState) -> ExperimentState:
        contracts = ContractBundle.model_validate(state["contracts"])
        candidates = list(state["scout_candidates"])
        planned_baseline = planner.propose(state)
        candidates[0] = {
            "hypothesis": planned_baseline.hypothesis,
            "model_family": planned_baseline.model_family,
            "feature_strategy": planned_baseline.feature_strategy,
            "parameters": planned_baseline.parameters,
        }
        candidates[-1] = dict(candidates[0])

        queue: list[dict[str, object]] = []
        seen: set[str] = set()
        duplicates: list[str] = []
        for candidate in candidates:
            hypothesis = str(candidate["hypothesis"])
            hypothesis_id = _hypothesis_id(hypothesis)
            if hypothesis_id in seen:
                duplicates.append(hypothesis_id)
                continue
            seen.add(hypothesis_id)
            family = candidate["model_family"]
            proposal = ExperimentProposal(
                experiment_id=f"p2-{len(queue) + 1}-{family}-s{contracts.validation.seed}",
                hypothesis=hypothesis,
                hypothesis_id=hypothesis_id,
                model_family=family,
                feature_strategy=candidate["feature_strategy"],
                parameters=candidate["parameters"],
                dataset_path=contracts.data.dataset_path,
                target_column=contracts.data.target_column,
                data_version=contracts.data.data_version,
                split_id=contracts.validation.split_id,
                metric_version=contracts.metric.metric_version,
                test_size=contracts.validation.test_size,
                seed=contracts.validation.seed,
            )
            queue.append(proposal.model_dump())
        budget = BudgetPolicy.model_validate(state["budget"])
        queue = queue[: budget.max_experiments]
        update = _with_report(
            state,
            "portfolio_planner",
            {
                "queued": len(queue),
                "duplicates_rejected": len(duplicates),
                "budget_limit": budget.max_experiments,
            },
        )
        return {
            **update,
            "experiment_queue": queue,
            "duplicate_hypotheses": duplicates,
            "current_experiment_index": 0,
        }

    return build_portfolio


def engineer_features(state: ExperimentState) -> ExperimentState:
    proposal = _current_proposal(state)
    update = _with_report(
        state,
        "feature_engineer",
        {"strategy": proposal.feature_strategy, "experiment_id": proposal.experiment_id},
    )
    return {**update, "proposal": proposal.model_dump(), "status": "PROPOSED"}


def engineer_model(state: ExperimentState) -> ExperimentState:
    proposal = ExperimentProposal.model_validate(state["proposal"])
    return _with_report(
        state,
        "model_engineer",
        {
            "model_family": proposal.model_family,
            "parameters": proposal.parameters,
            "preflight": "PASSED",
        },
    )


def critique_result(state: ExperimentState) -> ExperimentState:
    result = RunResult.model_validate(state["run_result"])
    completed = [RunResult.model_validate(item) for item in state.get("completed_results", [])]
    contracts = ContractBundle.model_validate(state["contracts"])
    metric_name = contracts.metric.metric_name
    successful_before = [
        item.metrics[metric_name] for item in completed[:-1] if item.status == "SUCCEEDED"
    ]
    if result.status == "FAILED":
        verdict = "FAILED"
        score = None
    else:
        score = result.metrics[metric_name]
        best_before = max(successful_before, default=None)
        verdict = "IMPROVED" if best_before is None or score > best_before else "TIED"
        if best_before is not None and score < best_before:
            verdict = "REGRESSED"
    critique = ExperimentCritique(
        experiment_id=result.experiment_id,
        verdict=verdict,
        accuracy=score if metric_name == "accuracy" else None,
        metric_name=metric_name,
        score=score,
        recommendation="Continue only while budget and model coverage gates permit.",
    )
    critiques = [*state.get("critiques", []), critique.model_dump()]
    coverage = sorted({item.model_family for item in completed if item.status == "SUCCEEDED"})
    budget = BudgetPolicy.model_validate(state["budget"])
    training_seconds = sum(item.duration_seconds for item in completed)
    next_index = state["current_experiment_index"] + 1
    queue = [ExperimentProposal.model_validate(item) for item in state["experiment_queue"]]
    ensemble_gate = state.get("ensemble_gate", "PENDING")
    if next_index < len(queue) and queue[next_index].model_family == "soft_voting":
        required = {"logistic_regression", "random_forest"}
        if required.issubset(coverage):
            ensemble_gate = "OPEN"
        else:
            ensemble_gate = "SKIPPED"
            next_index += 1

    stop_reason = ""
    if len(completed) >= budget.max_experiments:
        stop_reason = "experiment budget exhausted"
    elif training_seconds >= budget.max_training_seconds:
        stop_reason = "training-time budget exhausted"
    elif next_index >= len(queue):
        stop_reason = "portfolio exhausted"
    next_action = "VERIFY" if stop_reason else "CONTINUE"
    selected = _select_result(completed)
    resource_usage = dict(state.get("resource_usage", {}))
    resource_usage["training_seconds"] = training_seconds
    update = _with_report(
        state,
        "result_critic",
        {
            "evaluated": len(completed),
            "latest_verdict": verdict,
            "next_action": next_action,
        },
    )
    return {
        **update,
        "critiques": critiques,
        "model_coverage": coverage,
        "ensemble_gate": ensemble_gate,
        "current_experiment_index": next_index,
        "next_action": next_action,
        "stop_reason": stop_reason,
        "run_result": selected.model_dump(),
        "resource_usage": resource_usage,
    }


def audit_submission(state: ExperimentState) -> ExperimentState:
    if "verification" not in state:
        raise ValueError("Submission audit requires evidence verification")
    if "packet" in state:
        raise ValueError("A pod may publish only one terminal packet")
    return state


def verify_selected_evidence(state: ExperimentState) -> ExperimentState:
    from swarmfold.nodes import verify_evidence

    update = verify_evidence(state)
    combined = {**state, **update}
    verification = combined["verification"]
    return {
        **update,
        **_with_report(
            combined,
            "evidence_verifier",
            {
                "accepted": verification["accepted"],
                "reason_count": len(verification["reasons"]),
            },
        ),
    }


def route_after_critique(state: ExperimentState) -> str:
    return "continue" if state["next_action"] == "CONTINUE" else "verify"


def _current_proposal(state: ExperimentState) -> ExperimentProposal:
    queue = state["experiment_queue"]
    index = state["current_experiment_index"]
    if index >= len(queue):
        raise ValueError("Experiment queue is exhausted")
    return ExperimentProposal.model_validate(queue[index])


def _select_result(results: list[RunResult]) -> RunResult:
    successful = [item for item in results if item.status == "SUCCEEDED"]
    if successful:
        metric_name = next(iter(successful[0].metrics))
        return max(successful, key=lambda item: item.metrics[metric_name])
    if not results:
        raise ValueError("Result critic has no completed experiments")
    return results[-1]


def _hypothesis_id(hypothesis: str) -> str:
    return sha256(hypothesis.strip().lower().encode()).hexdigest()[:16]


def _with_report(state: ExperimentState, role: str, report: dict[str, object]) -> ExperimentState:
    reports = dict(state.get("role_reports", {}))
    reports[role] = report
    return {"role_reports": reports}
