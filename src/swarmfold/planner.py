import os
from collections.abc import Callable
from hashlib import sha256
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from swarmfold.config import PlannerSettings
from swarmfold.data import load_classification_csv
from swarmfold.prompts import PLANNER_PROMPT, render_planner_prompt
from swarmfold.schemas import ContractBundle, ExperimentProposal, ExperimentState


class PlannerSuggestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hypothesis: str = Field(
        min_length=10,
        description="A testable mechanism explaining why the baseline could be useful.",
    )


class ProposalPlanner(Protocol):
    def propose(self, state: ExperimentState) -> ExperimentProposal: ...


class DeterministicPlanner:
    def propose(self, state: ExperimentState) -> ExperimentProposal:
        return _proposal_from_state(
            state,
            "A fold-safe mixed-feature baseline can establish a reproducible score floor.",
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
        features, target = load_classification_csv(
            state["dataset_path"], state.get("target_column", "target")
        )
        if self.model_factory is None:
            _validate_provider_credentials(self.model_name)
        factory = self.model_factory or _init_chat_model
        try:
            model = factory(self.model_name)
            structured_model = model.with_structured_output(PlannerSuggestion)  # type: ignore[attr-defined]
            response = structured_model.invoke(  # type: ignore[attr-defined]
                render_planner_prompt(
                    rows=features.shape[0],
                    features=features.shape[1],
                    classes=len(set(target.tolist())),
                    project_context=state.get("project_context", "Not provided."),
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
