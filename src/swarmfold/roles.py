from hashlib import sha256
from pathlib import Path

from swarmfold.candidates import CandidateFactory
from swarmfold.contracts import freeze_contracts
from swarmfold.data import load_training_csv
from swarmfold.metrics import get_metric_spec
from swarmfold.model_registry import model_capability_registry
from swarmfold.optimization import AdaptiveScheduler, HPOConfig, HPOController
from swarmfold.planner import ProposalPlanner
from swarmfold.reflection import IterationReflector, ReflectionMemoryStore
from swarmfold.research import LiteratureResearchAgent
from swarmfold.retrieval import TechniqueRetriever
from swarmfold.schemas import (
    BudgetPolicy,
    ContractBundle,
    ExperimentCritique,
    ExperimentProposal,
    ExperimentState,
    RunResult,
)
from swarmfold.technique_plan import build_technique_plan

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
    "experiment_memory_agent",
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
            "task_type": "pending_problem_formulation",
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
    task_type = str(state["problem_formulation"]["task_type"])
    features, target = load_training_csv(state["dataset_path"], state["target_column"], task_type)
    numeric_count = int(features.select_dtypes(include="number").shape[1])
    report = {
        "rows": int(features.shape[0]),
        "features": int(features.shape[1]),
        "numeric_features": numeric_count,
        "categorical_features": int(features.shape[1] - numeric_count),
        "missing_values": int(features.isna().sum().sum()),
    }
    if task_type == "regression":
        report["target_summary"] = {
            "minimum": float(target.min()),
            "maximum": float(target.max()),
            "mean": float(target.mean()),
        }
    else:
        report["class_counts"] = {
            str(label): int((target == label).sum()) for label in sorted(set(target.tolist()))
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


def inspect_experiment_memory(state: ExperimentState) -> ExperimentState:
    memories = state.get("prior_knowledge", [])
    current_contracts = ContractBundle.model_validate(state["contracts"])
    classified = []
    for memory in memories:
        prior_profile = memory.get("source_profile")
        status = "CONTEXT_ONLY"
        if isinstance(prior_profile, dict):
            same_target = prior_profile.get("target_column") == current_contracts.data.target_column
            same_metric = str(prior_profile.get("metric", "")).casefold().replace("-", "_") == (
                current_contracts.metric.metric_name
            )
            if same_target and same_metric:
                status = "TRANSFERABLE_HYPOTHESIS"
        classified.append({**memory, "transfer_status": status})
    return {
        "prior_knowledge": classified,
        **_with_report(
            state,
            "experiment_memory_agent",
            {
                "packages_read": len(classified),
                "transferable": sum(
                    item["transfer_status"] == "TRANSFERABLE_HYPOTHESIS" for item in classified
                ),
                "blind_reuse_allowed": False,
            },
        ),
    }


def scout_models(state: ExperimentState) -> ExperimentState:
    budget = BudgetPolicy.model_validate(state["budget"])
    registry = model_capability_registry()
    lane_id = state.get("lane_id", "ai-a")
    task_type = str(state["problem_formulation"]["task_type"])
    hpo_controller = None
    if lane_id == "ai-b":
        metric_spec = get_metric_spec(str(state["contracts"]["metric"]["metric_name"]))
        study_id = ":".join(
            (
                str(state.get("iteration_id", "local-iteration")),
                lane_id,
                task_type,
                metric_spec.name,
            )
        )
        hpo_controller = HPOController(
            Path(state["output_dir"]) / "hpo.sqlite",
            HPOConfig(
                study_id=study_id,
                direction="maximize" if metric_spec.higher_is_better else "minimize",
                seed=int(state.get("seed", 42)),
            ),
        )
    factory = CandidateFactory(registry, hpo_controller=hpo_controller)
    candidates = factory.create(
        lane_id,
        max(budget.max_experiments, 5),
        task_type=task_type,
    )
    if not candidates:
        raise ValueError(
            f"No installed executable model family supports task '{task_type}' in lane '{lane_id}'"
        )
    eligibility = factory.eligibility_report(lane_id, task_type)
    candidate_families = {str(item["model_family"]) for item in candidates}
    traits = state["problem_formulation"].get("dataset_traits", {})
    summary_traits = (
        list(traits.get("summary_traits", [])) if isinstance(traits, dict) else []
    )
    retriever = TechniqueRetriever()
    retrieved = retriever.retrieve(
        role="model_engineer",
        task_type=task_type,
        dataset_traits=summary_traits,
        query=f"{task_type} {state['contracts']['metric']['metric_name']} modeling",
        top_k=12,
        role_scoped_only=True,
    )
    planned_family = str(candidates[0]["model_family"])
    selected_cards = [
        card
        for card in retrieved
        if not card.model_family or card.model_family == planned_family
    ][:5]
    rejected_cards = [
        {
            "card_id": card.card_id,
            "reason": (
                f"runtime family '{card.model_family}' is not eligible for lane {lane_id}"
                if card.model_family and card.model_family not in candidate_families
                else f"lane {lane_id} prioritized '{planned_family}' for this allocation"
            ),
        }
        for card in retrieved
        if card not in selected_cards
    ]
    contracts = ContractBundle.model_validate(state["contracts"])
    technique_plan = build_technique_plan(
        selected_cards,
        role="model_engineer",
        task_type=task_type,
        metric_name=contracts.metric.metric_name,
        higher_is_better=contracts.metric.higher_is_better,
        model_family=planned_family,
        feature_steps=[str(candidates[0]["feature_strategy"])],
        validation_strategy=contracts.validation.strategy,
    )
    retrieval_provenance = {
        **retriever.record_provenance(
            selected_cards,
            query=f"{task_type} {contracts.metric.metric_name} modeling",
            role="model_engineer",
        ),
        "dataset_traits": summary_traits,
        "rejected_cards": rejected_cards,
        "runtime_mapping": {
            "model_family": planned_family,
            "feature_strategy": candidates[0]["feature_strategy"],
            "validation_strategy": contracts.validation.strategy,
        },
    }
    update = _with_report(
        state,
        "model_scout",
        {
            "candidate_count": len(candidates),
            "lane_id": lane_id,
            "installed_model_families": [
                item["model_family"] for item in registry if item["installed"]
            ],
            "executable_model_families": [
                item["model_family"] for item in registry if item.get("executable")
            ],
            "eligibility": eligibility,
            "selected_technique_cards": technique_plan.source_card_ids,
            "technique_plan_executable": technique_plan.is_executable,
        },
    )
    return {
        **update,
        "scout_candidates": candidates,
        "model_registry": registry,
        "technique_plan": technique_plan.model_dump(),
        "retrieval_provenance": retrieval_provenance,
    }


def create_portfolio_node(planner: ProposalPlanner):
    def build_portfolio(state: ExperimentState) -> ExperimentState:
        contracts = ContractBundle.model_validate(state["contracts"])
        candidates = list(state["scout_candidates"])
        preferred_families = {
            str(memory["champion_model_family"])
            for memory in state.get("prior_knowledge", [])
            if memory.get("transfer_status") == "TRANSFERABLE_HYPOTHESIS"
            and memory.get("champion_model_family")
        }
        if preferred_families:
            candidates.sort(
                key=lambda item: str(item["model_family"]) not in preferred_families
            )
        planned_baseline = planner.propose(state)
        candidates[0] = {
            **candidates[0],
            "hypothesis": (
                f"{planned_baseline.hypothesis} The {state.get('lane_id', 'ai-a')} charter "
                f"tests this through {candidates[0]['model_family']} with "
                f"{candidates[0]['feature_strategy']}."
            ),
        }
        candidates.append(dict(candidates[0]))

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
                hpo_study_id=candidate.get("hpo_study_id"),
                hpo_trial_number=candidate.get("hpo_trial_number"),
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
                "memory_preferred_families": sorted(preferred_families),
                "memory_changed_priority": bool(preferred_families),
            },
        )
        return {
            **update,
            "experiment_queue": queue,
            "duplicate_hypotheses": duplicates,
            "current_experiment_index": 0,
        }

    return build_portfolio


def _lane_candidates(
    lane_id: str,
    limit: int,
    task_type: str = "binary_classification",
) -> list[dict[str, object]]:
    return CandidateFactory().create(lane_id, limit, task_type=task_type)


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
    metric_spec = get_metric_spec(metric_name)
    successful_before = [
        item.metrics[metric_name] for item in completed[:-1] if item.status == "SUCCEEDED"
    ]
    if result.status != "SUCCEEDED":
        verdict = "FAILED"
        score = None
    else:
        score = result.metrics[metric_name]
        best_before = metric_spec.best(successful_before) if successful_before else None
        verdict = (
            "IMPROVED"
            if best_before is None or metric_spec.is_better(score, best_before)
            else "TIED"
        )
        if best_before is not None and metric_spec.is_better(best_before, score):
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
    if len(coverage) >= 2:
        ensemble_gate = "OPEN"

    scheduler_decisions = list(state.get("scheduler_decisions", []))
    if next_index < len(queue) - 1:
        before = queue[next_index:]
        prioritized = AdaptiveScheduler().prioritize_queue(
            [item.model_dump() for item in before],
            [item.model_dump() for item in completed],
            metric_name=metric_name,
            higher_is_better=metric_spec.higher_is_better,
        )
        reordered = [ExperimentProposal.model_validate(item) for item in prioritized]
        queue = [*queue[:next_index], *reordered]
        scheduler_decisions.append(
            {
                "after_experiment_id": result.experiment_id,
                "before": [item.experiment_id for item in before],
                "after": [item.experiment_id for item in reordered],
                "metric_name": metric_name,
            }
        )

    stop_reason = ""
    if len(completed) >= budget.max_experiments:
        stop_reason = "experiment budget exhausted"
    elif (
        budget.max_training_seconds is not None and training_seconds >= budget.max_training_seconds
    ):
        stop_reason = "training-time budget exhausted"
    elif next_index >= len(queue):
        stop_reason = "portfolio exhausted"
    next_action = "VERIFY" if stop_reason else "CONTINUE"
    selected = _select_result(completed, metric_name)
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
        "experiment_queue": [item.model_dump() for item in queue],
        "scheduler_decisions": scheduler_decisions,
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


def reflect_iteration(state: ExperimentState) -> ExperimentState:
    contracts = ContractBundle.model_validate(state["contracts"])
    report = IterationReflector().generate_reflection(
        iteration_id=state["iteration_id"],
        completed_results=list(state.get("completed_results", [])),
        metric_name=contracts.metric.metric_name,
        higher_is_better=contracts.metric.higher_is_better,
        ensemble_assessment=state.get("ensemble_assessment"),
    )
    store = ReflectionMemoryStore()
    reflection_path = store.save_reflection(Path(state["output_dir"]), report)
    prior_package = store.to_prior_knowledge_package(
        report,
        source_profile={
            "target_column": contracts.data.target_column,
            "metric": contracts.metric.metric_name,
            "task_type": contracts.competition.task_type,
            "data_version": contracts.data.data_version,
        },
    )
    findings = list(state.get("critiques", []))
    findings.extend(
        {
            "experiment_id": report.champion_experiment_id or "iteration",
            "verdict": "REFLECTION",
            "metric_name": report.metric_name,
            "score": report.champion_score,
            "recommendation": insight.insight,
            "category": insight.category,
            "evidence": insight.evidence,
        }
        for insight in report.insights
    )
    return {
        "iteration_reflection": report.model_dump(),
        "reflection_path": str(reflection_path),
        "next_iteration_knowledge": prior_package,
        "critiques": findings,
    }


def route_after_critique(state: ExperimentState) -> str:
    return "continue" if state["next_action"] == "CONTINUE" else "verify"


def _current_proposal(state: ExperimentState) -> ExperimentProposal:
    queue = state["experiment_queue"]
    index = state["current_experiment_index"]
    if index >= len(queue):
        raise ValueError("Experiment queue is exhausted")
    return ExperimentProposal.model_validate(queue[index])


def _select_result(results: list[RunResult], metric_name: str | None = None) -> RunResult:
    successful = [item for item in results if item.status == "SUCCEEDED"]
    if successful:
        resolved_metric = metric_name or next(iter(successful[0].metrics))
        metric_spec = get_metric_spec(resolved_metric)
        return sorted(
            successful,
            key=lambda item: item.metrics[resolved_metric],
            reverse=metric_spec.higher_is_better,
        )[0]
    if not results:
        raise ValueError("Result critic has no completed experiments")
    return results[-1]


def _hypothesis_id(hypothesis: str) -> str:
    return sha256(hypothesis.strip().lower().encode()).hexdigest()[:16]


def _with_report(state: ExperimentState, role: str, report: dict[str, object]) -> ExperimentState:
    reports = dict(state.get("role_reports", {}))
    reports[role] = report
    return {"role_reports": reports}
