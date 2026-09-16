import os
from collections.abc import Callable
from hashlib import sha256
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from swarmfold.config import PlannerSettings
from swarmfold.data import load_training_csv
from swarmfold.prompts import PLANNER_PROMPT, render_planner_prompt
from swarmfold.retrieval import TechniqueRetriever
from swarmfold.schemas import ContractBundle, ExperimentProposal, ExperimentState


class PlannerSuggestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hypothesis: str = Field(
        min_length=10,
        description="A testable mechanism explaining why the baseline could be useful.",
    )
    technique_card_id: str | None = Field(
        default=None,
        description="The card_id of the retrieved technique card backing this proposal.",
    )


class ProposalPlanner(Protocol):
    def propose(self, state: ExperimentState) -> ExperimentProposal: ...


class DeterministicPlanner:
    def propose(self, state: ExperimentState) -> ExperimentProposal:
        retriever = TechniqueRetriever()
        task_type = state.get("problem_formulation", {}).get("task_type", "binary_classification")
        cards = retriever.retrieve(role="model_scout", task_type=task_type, top_k=1)
        card_mention = f" [{cards[0].card_id}]" if cards else ""
        return _proposal_from_state(
            state,
            "A fold-safe mixed-feature baseline"
            f"{card_mention} can establish a reproducible score floor.",
        )


class LangChainPlanner:
    def __init__(
        self,
        model_name: str,
        model_factory: Callable[[str], object] | None = None,
    ) -> None:
        self.model_name = model_name
        self.model_factory = model_factory

    def propose(self, state: ExperimentState) -> ExperimentProposal:
        task_type = state.get("problem_formulation", {}).get("task_type", "binary_classification")
        features, target = load_training_csv(
            state["dataset_path"], state.get("target_column", "target"), task_type
        )
        if self.model_factory is None:
            _validate_provider_credentials(self.model_name)
        factory = self.model_factory or _init_chat_model

        retriever = TechniqueRetriever()
        cards = retriever.retrieve(
            role="model_scout",
            task_type=task_type,
            query=state.get("project_context", ""),
            top_k=3,
        )
        retrieved_techniques = retriever.format_for_prompt(cards)

        try:
            model = factory(self.model_name)
            structured_model = model.with_structured_output(PlannerSuggestion)  # type: ignore[attr-defined]
            response = structured_model.invoke(  # type: ignore[attr-defined]
                render_planner_prompt(
                    rows=features.shape[0],
                    features=features.shape[1],
                    classes=len(set(target.tolist())),
                    project_context=state.get("project_context", "Not provided."),
                    retrieved_techniques=retrieved_techniques,
                )
            )
        except Exception as exc:
            raise RuntimeError(
                f"Live planner failed with {type(exc).__name__}; check the provider package, "
                "model name, credentials, and network"
            ) from exc
        suggestion = PlannerSuggestion.model_validate(response)
        return _proposal_from_state(state, suggestion.hypothesis)


def create_planner(settings: PlannerSettings) -> ProposalPlanner:
    if settings.mode == "live":
        if settings.model is None:
            raise ValueError("Live planner model is missing")
        return LangChainPlanner(settings.model)
    return DeterministicPlanner()


def _proposal_from_state(state: ExperimentState, hypothesis: str) -> ExperimentProposal:
    seed = state.get("seed", 42)
    contracts = ContractBundle.model_validate(state["contracts"])
    return ExperimentProposal(
        experiment_id=f"baseline-logreg-s{seed}",
        hypothesis=hypothesis,
        hypothesis_id=sha256(hypothesis.strip().lower().encode()).hexdigest()[:16],
        model_family="logistic_regression",
        dataset_path=state["dataset_path"],
        target_column=state.get("target_column", "target"),
        data_version=contracts.data.data_version,
        split_id=contracts.validation.split_id,
        metric_version=contracts.metric.metric_version,
        test_size=state.get("test_size", 0.25),
        seed=seed,
        prompt_version=PLANNER_PROMPT.prompt_version,
    )


def _init_chat_model(model_name: str):
    from langchain.chat_models import init_chat_model

    try:
        return init_chat_model(model_name, temperature=0)
    except ImportError as exc:
        provider = model_name.partition(":")[0]
        raise RuntimeError(
            f"Model provider '{provider}' is not installed; install its SwarmFold extra"
        ) from exc


def _validate_provider_credentials(model_name: str) -> None:
    provider = model_name.partition(":")[0]
    key_names = {
        "openai": ("OPENAI_API_KEY",),
        "anthropic": ("ANTHROPIC_API_KEY",),
        "google_genai": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    }
    required = key_names.get(provider)
    if required and not any(os.getenv(name) for name in required):
        joined = " or ".join(required)
        raise RuntimeError(f"Live model '{provider}' requires {joined}")
