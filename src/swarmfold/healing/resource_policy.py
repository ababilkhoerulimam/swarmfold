from __future__ import annotations

from typing import Any


class ExecutionCancelledError(RuntimeError):
    """Raised when execution is cooperatively cancelled by user or coordinator."""

    def __init__(self, reason: str = "Execution cancelled") -> None:
        super().__init__(reason)
        self.reason = reason


class ResourceExceededError(RuntimeError):
    """Raised when execution exceeds allocated resource limits."""

    def __init__(self, resource_type: str, limit: float, observed: float) -> None:
        super().__init__(
            f"Resource limit exceeded for {resource_type}: "
            f"observed {observed:.2f} > limit {limit:.2f}"
        )
        self.resource_type = resource_type
        self.limit = limit
        self.observed = observed


class CancellationToken:
    """Thread-safe cooperative cancellation token."""

    def __init__(self) -> None:
        self._cancelled: bool = False
        self._reason: str = ""

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    def cancel(self, reason: str = "Execution cancelled") -> None:
        self._cancelled = True
        self._reason = reason

    def throw_if_cancelled(self) -> None:
        if self._cancelled:
            raise ExecutionCancelledError(self._reason or "Execution cancelled")


class ResourcePolicy:
    """Configurable resource limits and cooperative cancellation checks."""

    def __init__(
        self,
        max_training_seconds_per_fold: float | None = None,
        max_total_seconds: float | None = None,
        max_memory_mb: float | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> None:
        self.max_training_seconds_per_fold = max_training_seconds_per_fold
        self.max_total_seconds = max_total_seconds
        self.max_memory_mb = max_memory_mb
        self.cancellation_token = cancellation_token

    def check_fold_budget(self, fold_idx: int, elapsed_seconds: float) -> None:
        """Verify elapsed time for a single fold remains within budget."""
        if (
            self.max_training_seconds_per_fold is not None
            and elapsed_seconds > self.max_training_seconds_per_fold
        ):
            raise ResourceExceededError(
                f"fold_{fold_idx}_duration_seconds",
                self.max_training_seconds_per_fold,
                elapsed_seconds,
            )

    def check_total_budget(self, elapsed_seconds: float) -> None:
        """Verify overall training duration remains within budget."""
        if self.max_total_seconds is not None and elapsed_seconds > self.max_total_seconds:
            raise ResourceExceededError(
                "total_duration_seconds", self.max_total_seconds, elapsed_seconds
            )

    def check_memory(self, memory_mb: float) -> None:
        """Verify process memory consumption remains within budget."""
        if self.max_memory_mb is not None and memory_mb > self.max_memory_mb:
            raise ResourceExceededError("peak_memory_mb", self.max_memory_mb, memory_mb)

    def check_cancellation(self) -> None:
        """Check whether cooperative cancellation has been requested."""
        if self.cancellation_token is not None:
            self.cancellation_token.throw_if_cancelled()

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_training_seconds_per_fold": self.max_training_seconds_per_fold,
            "max_total_seconds": self.max_total_seconds,
            "max_memory_mb": self.max_memory_mb,
            "cancelled": self.cancellation_token.is_cancelled
            if self.cancellation_token
            else False,
        }
