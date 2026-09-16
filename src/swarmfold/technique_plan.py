from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from swarmfold.model_registry import is_model_executable
from swarmfold.technique_cards import TechniqueCard


class TechniquePlan(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    plan_id: str = Field(default_factory=lambda: f"plan-{uuid.uuid4().hex[:8]}")
    role: str = Field(min_length=2)
    task_type: str
    metric_name: str
    higher_is_better: bool = True
    source_card_ids: list[str] = Field(default_factory=list)
    model_family: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    feature_steps: list[str] = Field(default_factory=list)
    validation_strategy: str = "k_fold"
    fold_boundary_rule: str = ""
    leakage_rules: list[str] = Field(default_factory=list)
    ensemble_compatibility: str = ""
    compute_profile: str = ""
    acceptance_criteria: str = ""
    rejection_criteria: str = ""
    is_executable: bool = True
    execution_block_reason: str | None = None


def build_technique_plan(
    cards: list[TechniqueCard],
    *,
    role: str,
    task_type: str = "binary_classification",
    metric_name: str = "accuracy",
    higher_is_better: bool = True,
    model_family: str | None = None,
    parameters: dict[str, Any] | None = None,
    feature_steps: list[str] | None = None,
    validation_strategy: str = "k_fold",
) -> TechniquePlan:
    source_ids = [c.card_id for c in cards]

    # Aggregate fold boundary rules and leakage rules
    fold_rules = [c.fold_boundary_rule for c in cards if c.fold_boundary_rule]
    primary_fold_rule = (
        fold_rules[0] if fold_rules else "Fit strictly within train fold boundaries."
    )
    leakage_rules = [
        f"[{c.card_id}] {c.fold_boundary_rule}"
        for c in cards
        if c.leakage_risk in {"MEDIUM", "HIGH"}
    ]

    # Identify model family from cards if not explicitly provided
    resolved_family = model_family
    if not resolved_family:
        for c in cards:
            if c.model_family:
                resolved_family = c.model_family
                break

    # Aggregate params from recommended params
    combined_params = dict(parameters or {})
    for c in cards:
        if c.recommended_parameters:
            for k, v in c.recommended_parameters.items():
                if k not in combined_params:
                    combined_params[k] = v

    # Acceptance / rejection criteria from cards
    accept_crits = [c.acceptance_criteria for c in cards if c.acceptance_criteria]
    reject_crits = [c.rejection_criteria for c in cards if c.rejection_criteria]

    # Compute profile
    comp_profiles = [c.compute_profile for c in cards if c.compute_profile]

    ens_compat = "; ".join(c.ensemble_compatibility for c in cards if c.ensemble_compatibility)
    default_accept = "Improve validation metric over baseline."
    accept_str = "; ".join(accept_crits[:2]) if accept_crits else default_accept
    default_reject = "Degrade validation score or trigger leakage."
    reject_str = "; ".join(reject_crits[:2]) if reject_crits else default_reject

    plan = TechniquePlan(
        role=role,
        task_type=task_type,
        metric_name=metric_name,
        higher_is_better=higher_is_better,
        source_card_ids=source_ids,
        model_family=resolved_family,
        parameters=combined_params,
        feature_steps=feature_steps or [],
        validation_strategy=validation_strategy,
        fold_boundary_rule=primary_fold_rule,
        leakage_rules=leakage_rules,
        ensemble_compatibility=ens_compat,
        compute_profile=comp_profiles[0] if comp_profiles else "standard_cpu",
        acceptance_criteria=accept_str,
        rejection_criteria=reject_str,
    )
    return validate_technique_plan(plan)


def validate_technique_plan(
    plan: TechniquePlan,
    executable_families: set[str] | list[str] | None = None,
) -> TechniquePlan:
    if not plan.model_family:
        return plan

    if executable_families is not None:
        is_exec = plan.model_family in executable_families
    else:
        is_exec = is_model_executable(plan.model_family)

    if not is_exec:
        block_msg = (
            f"Model family '{plan.model_family}' is DECLARED in knowledge corpus "
            "but has no EXECUTABLE runtime adapter installed."
        )
        return plan.model_copy(
            update={
                "is_executable": False,
                "execution_block_reason": block_msg,
            }
        )
    return plan
