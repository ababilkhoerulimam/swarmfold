import os
import sqlite3
from pathlib import Path
from uuid import uuid4

os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from swarmfold.config import PlannerMode, PlannerSettings
from swarmfold.nodes import run_experiment_candidate
from swarmfold.packets import publish_terminal_packet
from swarmfold.planner import ProposalPlanner, create_planner
from swarmfold.quality import (
    analyze_domain_decomposition,
    assess_oof_ensemble,
    assess_robustness_uncertainty,
    audit_temporal_validation,
    formulate_problem,
)
from swarmfold.research import LiteratureResearchAgent, create_literature_research_agent
from swarmfold.roles import (
    audit_submission,
    audit_validation,
    build_data_contract,
    create_literature_research_node,
    create_portfolio_node,
    critique_result,
    engineer_features,
    engineer_model,
    initialize_pod,
    inspect_eda,
    inspect_rules,
    route_after_critique,
    scout_models,
    verify_selected_evidence,
)
from swarmfold.schemas import BudgetPolicy, ContractBundle, ExperimentState


def build_experiment_graph(
    planner: ProposalPlanner | None = None,
    *,
    checkpointer: BaseCheckpointSaver | None = None,
    research_agent: LiteratureResearchAgent | None = None,
):
    resolved_planner = planner or create_planner(PlannerSettings())
    resolved_research_agent = research_agent or create_literature_research_agent()
    builder = StateGraph(ExperimentState)
    builder.add_node("pod_supervisor", initialize_pod)
    builder.add_node("rules_agent", inspect_rules)
    builder.add_node("problem_formulation_agent", formulate_problem)
    builder.add_node("data_contract_agent", build_data_contract)
    builder.add_node("validation_agent", audit_validation)
    builder.add_node(
        "temporal_validation_leakage_auditor", audit_temporal_validation
    )
    builder.add_node("eda_agent", inspect_eda)
    builder.add_node("domain_target_decomposition_agent", analyze_domain_decomposition)
    builder.add_node(
        "literature_research_agent",
        create_literature_research_node(resolved_research_agent),
    )
    builder.add_node("model_scout", scout_models)
    builder.add_node("portfolio_planner", create_portfolio_node(resolved_planner))
    builder.add_node("feature_engineer", engineer_features)
    builder.add_node("model_engineer", engineer_model)
    builder.add_node("experiment_runner", run_experiment_candidate)
    builder.add_node("result_critic", critique_result)
    builder.add_node("oof_ensemble_scientist", assess_oof_ensemble)
    builder.add_node("robustness_uncertainty_agent", assess_robustness_uncertainty)
    builder.add_node("evidence_verifier", verify_selected_evidence)
    builder.add_node("submission_auditor", audit_submission)
    builder.add_node("publisher", publish_terminal_packet)

    builder.add_edge(START, "pod_supervisor")
    builder.add_edge("pod_supervisor", "rules_agent")
    builder.add_edge("rules_agent", "problem_formulation_agent")
    builder.add_edge("problem_formulation_agent", "data_contract_agent")
    builder.add_edge("data_contract_agent", "validation_agent")
    builder.add_edge("validation_agent", "temporal_validation_leakage_auditor")
    builder.add_edge("temporal_validation_leakage_auditor", "eda_agent")
    builder.add_edge("eda_agent", "domain_target_decomposition_agent")
    builder.add_edge("domain_target_decomposition_agent", "literature_research_agent")
    builder.add_edge("literature_research_agent", "model_scout")
    builder.add_edge("model_scout", "portfolio_planner")
    builder.add_edge("portfolio_planner", "feature_engineer")
    builder.add_edge("feature_engineer", "model_engineer")
    builder.add_edge("model_engineer", "experiment_runner")
    builder.add_edge("experiment_runner", "result_critic")
    builder.add_conditional_edges(
        "result_critic",
        route_after_critique,
        {"continue": "feature_engineer", "verify": "oof_ensemble_scientist"},
    )
    builder.add_edge("oof_ensemble_scientist", "robustness_uncertainty_agent")
    builder.add_edge("robustness_uncertainty_agent", "evidence_verifier")
    builder.add_edge("evidence_verifier", "submission_auditor")
    builder.add_edge("submission_auditor", "publisher")
    builder.add_edge("publisher", END)
    return builder.compile(checkpointer=checkpointer)


def run_experiment(
    dataset_path: str | Path,
    *,
    target_column: str = "target",
    metric_name: str = "accuracy",
    test_dataset_path: str | Path | None = None,
    sample_submission_path: str | Path | None = None,
    project_path: str | Path | None = None,
    seed: int = 42,
    test_size: float = 0.25,
    output_dir: str | Path = ".swarmfold",
    tracking_uri: str | None = None,
    planner_mode: PlannerMode | None = None,
    model: str | None = None,
    planner: ProposalPlanner | None = None,
    research_agent: LiteratureResearchAgent | None = None,
    max_experiments: int = 4,
    max_training_seconds: float = 120.0,
    thread_id: str | None = None,
    resume: bool = False,
    interrupt_before: str | None = None,
    iteration_id: str = "local-iteration",
    lane_id: str = "ai-a",
    contracts: ContractBundle | dict[str, object] | None = None,
) -> ExperimentState:
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    database_path = (output_path / "mlflow.db").as_posix()
    resolved_tracking_uri = tracking_uri or f"sqlite:///{database_path}"
    resolved_planner = planner or create_planner(
        PlannerSettings.from_values(mode=planner_mode, model=model)
    )
    resolved_thread_id = thread_id or uuid4().hex
    if resume and thread_id is None:
        raise ValueError("Resume requires an explicit thread_id")
    budget = BudgetPolicy(
        max_experiments=max_experiments,
        max_training_seconds=max_training_seconds,
    )
    initial_state: ExperimentState = {
        "dataset_path": str(Path(dataset_path).resolve()),
        "target_column": target_column,
        "metric_name": metric_name,
        "seed": seed,
        "test_size": test_size,
        "output_dir": str(output_path),
        "tracking_uri": resolved_tracking_uri,
        "thread_id": resolved_thread_id,
        "iteration_id": iteration_id,
        "lane_id": lane_id,
        "budget": budget.model_dump(),
        "status": "PENDING",
    }
    if test_dataset_path is not None:
        initial_state["test_dataset_path"] = str(Path(test_dataset_path).resolve())
    if sample_submission_path is not None:
        initial_state["sample_submission_path"] = str(
            Path(sample_submission_path).resolve()
        )
    if project_path is not None:
        resolved_project_path = Path(project_path).resolve()
        initial_state["project_path"] = str(resolved_project_path)
        initial_state["project_context"] = resolved_project_path.read_text(
            encoding="utf-8"
        ).strip()
    if contracts is not None:
        initial_state["contracts"] = ContractBundle.model_validate(contracts).model_dump()
    checkpoint_path = output_path / "checkpoints.sqlite"
    connection = sqlite3.connect(checkpoint_path, check_same_thread=False)
    checkpointer = SqliteSaver(connection)
    graph = build_experiment_graph(
        resolved_planner,
        checkpointer=checkpointer,
        research_agent=research_agent,
    )
    config = {"configurable": {"thread_id": resolved_thread_id}}
    try:
        existing = checkpointer.get_tuple(config)
        if resume and existing is None:
            raise ValueError(f"No checkpoint exists for thread_id '{resolved_thread_id}'")
        if not resume and thread_id is not None and existing is not None:
            raise ValueError(f"thread_id '{resolved_thread_id}' already exists; use resume=True")
        graph.invoke(
            None if resume else initial_state,
            config,
            interrupt_before=[interrupt_before] if interrupt_before else None,
        )
        snapshot = graph.get_state(config)
        result = dict(snapshot.values)
        if snapshot.next:
            result["status"] = "INTERRUPTED"
            result["next_nodes"] = list(snapshot.next)
        else:
            result["next_nodes"] = []
        return result
    finally:
        connection.close()
