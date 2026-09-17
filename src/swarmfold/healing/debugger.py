from __future__ import annotations

from typing import Any

from swarmfold.healing.failure_taxonomy import FailureKind, FailureRecord
from swarmfold.schemas import ExperimentProposal


class SelfHealingDebugger:
    """Bounded, deterministic debugger for automated experiment recovery."""

    def __init__(self, max_retries: int = 2, max_provider_retries: int = 3) -> None:
        self.max_retries = max_retries
        self.max_provider_retries = max_provider_retries
        self.retry_counts: dict[str, int] = {}
        self.provider_retry_counts: dict[str, int] = {}
        self.repair_history: list[dict[str, Any]] = []

    def can_repair(self, proposal_id: str, failure: FailureRecord) -> bool:
        """Check whether a training failure can be repaired within the bounded retry budget."""
        if not failure.recoverable:
            return False
        if failure.kind == FailureKind.PROVIDER_OR_NETWORK_FAILURE:
            return False
        current_attempts = self.retry_counts.get(proposal_id, 0)
        return current_attempts < self.max_retries

    def can_retry_provider(self, proposal_id: str, failure: FailureRecord) -> bool:
        """Check whether a provider/network failure can be retried with exponential backoff.

        Provider retries are tracked separately from training parameter repairs so that
        a network hiccup does not consume the training repair budget.
        """
        if failure.kind != FailureKind.PROVIDER_OR_NETWORK_FAILURE:
            return False
        current_attempts = self.provider_retry_counts.get(proposal_id, 0)
        return current_attempts < self.max_provider_retries

    def get_provider_backoff_seconds(self, proposal_id: str) -> float:
        """Return the backoff duration for the next provider retry attempt.

        Uses bounded exponential backoff: 1s → 2s → 4s (capped at 30s).
        Increments the retry counter as a side-effect.
        """
        attempt = self.provider_retry_counts.get(proposal_id, 0)
        self.provider_retry_counts[proposal_id] = attempt + 1
        return min(2.0 ** attempt, 30.0)


    def attempt_repair(
        self,
        proposal: ExperimentProposal,
        failure: FailureRecord,
    ) -> tuple[ExperimentProposal, str]:
        """Apply a deterministic, machine-readable parameter repair action."""
        root_id = getattr(proposal, "original_experiment_id", None) or proposal.experiment_id
        current_attempt = self.retry_counts.get(root_id, 0) + 1
        self.retry_counts[root_id] = current_attempt

        params = dict(proposal.parameters)
        feature_strategy = proposal.feature_strategy
        repair_actions = []

        if failure.kind == FailureKind.OUT_OF_MEMORY:
            # 1. Reduce depth and leaves
            if "max_depth" in params:
                old_depth = int(params["max_depth"])
                params["max_depth"] = max(2, old_depth // 2)
                repair_actions.append(f"reduced max_depth {old_depth} -> {params['max_depth']}")
            else:
                params["max_depth"] = 4
                repair_actions.append("capped max_depth at 4")

            if "num_leaves" in params:
                old_leaves = int(params["num_leaves"])
                params["num_leaves"] = max(8, old_leaves // 2)
                repair_actions.append(f"reduced num_leaves {old_leaves} -> {params['num_leaves']}")

            # 2. Reduce estimators
            if "n_estimators" in params:
                old_est = int(params["n_estimators"])
                params["n_estimators"] = max(10, old_est // 2)
                repair_actions.append(f"reduced n_estimators {old_est} -> {params['n_estimators']}")

            # 3. Limit parallelism to 1
            params["n_jobs"] = 1
            repair_actions.append("forced n_jobs=1")

        elif failure.kind == FailureKind.INVALID_PARAMETER:
            msg_lower = failure.message.casefold()

            # LogisticRegression penalty / solver incompatibility
            if "solver" in params and "penalty" in params:
                if params["solver"] in ("lbfgs", "newton-cg", "sag") and params["penalty"] == "l1":
                    params["penalty"] = "l2"
                    repair_actions.append("switched incompatible penalty='l1' to penalty='l2'")
                elif params["penalty"] == "elasticnet" and "l1_ratio" not in params:
                    params["l1_ratio"] = 0.5
                    repair_actions.append("added missing l1_ratio=0.5 for elasticnet penalty")

            # Incompatible criterion
            if "criterion" in params and (
                "unknown criterion" in msg_lower or "criterion must be" in msg_lower
            ):
                params.pop("criterion", None)
                repair_actions.append("removed incompatible criterion")

            # Non-positive regularizers or rates
            for key in ("C", "alpha", "learning_rate"):
                if key in params:
                    try:
                        val = float(params[key])
                        if val <= 0:
                            params[key] = 1.0 if key != "learning_rate" else 0.1
                            repair_actions.append(f"clamped non-positive {key} to {params[key]}")
                    except (ValueError, TypeError):
                        params.pop(key, None)

            # Strip unrecognized params mentioned in error message
            if "unexpected keyword" in msg_lower or "unknown parameter" in msg_lower:
                for k in list(params.keys()):
                    if f"'{k}'" in msg_lower or f"'{k.casefold()}'" in msg_lower:
                        params.pop(k, None)
                        repair_actions.append(f"removed unrecognized parameter '{k}'")

            if not repair_actions:
                params = {}
                repair_actions.append("reset all parameters to model defaults")

        elif failure.kind == FailureKind.NUMERICAL_INSTABILITY:
            if "learning_rate" in params:
                params["learning_rate"] = max(0.001, float(params["learning_rate"]) * 0.5)
                repair_actions.append(f"halved learning_rate to {params['learning_rate']}")
            if "alpha" in params:
                params["alpha"] = max(1.0, float(params["alpha"]) * 10.0)
                repair_actions.append(f"increased regularization alpha to {params['alpha']}")
            if "C" in params:
                params["C"] = max(0.01, float(params["C"]) * 0.1)
                repair_actions.append(f"increased L2 penalty (reduced C to {params['C']})")
            if not repair_actions:
                params["alpha"] = 1.0
                repair_actions.append("added regularization alpha=1.0")

        elif failure.kind == FailureKind.INCOMPATIBLE_DATA_SHAPE_OR_DTYPE:
            if feature_strategy in ("interaction_pairs", "power_transformed", "date_expanded"):
                feature_strategy = "robust_scaled"
                repair_actions.append(
                    f"fallback feature strategy from {proposal.feature_strategy} to robust_scaled"
                )
            else:
                feature_strategy = "scaled_numeric"
                repair_actions.append("fallback feature strategy to scaled_numeric")

        action_summary = "; ".join(repair_actions) if repair_actions else "fallback adjustments"
        repaired_id = f"{root_id}-repair-{current_attempt}"

        record_entry = {
            "root_experiment_id": root_id,
            "original_experiment_id": proposal.experiment_id,
            "repaired_experiment_id": repaired_id,
            "attempt": current_attempt,
            "failure_kind": str(failure.kind),
            "original_parameters": proposal.parameters,
            "repaired_parameters": params,
            "repair_action": action_summary,
        }
        self.repair_history.append(record_entry)

        repaired_proposal = proposal.model_copy(
            update={
                "experiment_id": repaired_id,
                "original_experiment_id": root_id,
                "repair_attempt": current_attempt,
                "parameters": params,
                "feature_strategy": feature_strategy,
            }
        )
        return repaired_proposal, action_summary
