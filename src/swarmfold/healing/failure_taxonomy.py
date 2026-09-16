from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class FailureKind(StrEnum):
    """Structured taxonomy of execution failures in SwarmFold."""

    OUT_OF_MEMORY = "out_of_memory"
    INVALID_PARAMETER = "invalid_parameter"
    NUMERICAL_INSTABILITY = "numerical_instability"
    INCOMPATIBLE_DATA_SHAPE_OR_DTYPE = "incompatible_data_shape_or_dtype"
    TIMEOUT_OR_STALLED = "timeout_or_stalled"
    DEPENDENCY_UNAVAILABLE = "dependency_unavailable"
    PROVIDER_OR_NETWORK_FAILURE = "provider_or_network_failure"
    ARTIFACT_VERIFICATION_FAILURE = "artifact_verification_failure"
    UNKNOWN = "unknown"


class FailureRecord(BaseModel):
    """Machine-readable record capturing an observed execution failure."""

    kind: FailureKind
    exception_type: str
    message: str
    recoverable: bool = False
    repair_action: str | None = None
    attempt_count: int = 1
    details: dict[str, Any] = Field(default_factory=dict)


def classify_failure(exc: BaseException) -> FailureRecord:
    """Classify a runtime exception into the structured FailureTaxonomy."""
    exc_type = type(exc).__name__
    msg = str(exc)
    msg_lower = msg.casefold()

    # 1. Out of Memory
    oom_tokens = (
        "out of memory",
        "cuda oom",
        "oom",
        "unable to allocate",
        "memoryerror",
        "peak_memory_mb",
        "memory limit",
        "memory exceeded",
    )
    if isinstance(exc, MemoryError) or any(kw in msg_lower for kw in oom_tokens):
        return FailureRecord(
            kind=FailureKind.OUT_OF_MEMORY,
            exception_type=exc_type,
            message=msg,
            recoverable=True,
            details={"memory_related": True},
        )

    # 2. Dependency Unavailable
    if isinstance(exc, (ImportError, ModuleNotFoundError)) or "no module named" in msg_lower:
        return FailureRecord(
            kind=FailureKind.DEPENDENCY_UNAVAILABLE,
            exception_type=exc_type,
            message=msg,
            recoverable=False,
            details={"module": getattr(exc, "name", None)},
        )

    # 3. Timeout or Stalled
    timeout_tokens = (
        "timed out",
        "timeout",
        "execution budget exceeded",
        "resource limit exceeded",
        "duration_seconds",
        "stalled",
        "cancelled",
        "cancellation",
    )
    if isinstance(exc, TimeoutError) or any(kw in msg_lower for kw in timeout_tokens):
        return FailureRecord(
            kind=FailureKind.TIMEOUT_OR_STALLED,
            exception_type=exc_type,
            message=msg,
            recoverable=False,
        )

    # 4. Numerical Instability
    if isinstance(exc, FloatingPointError) or any(
        kw in msg_lower
        for kw in (
            "divide by zero",
            "contains nan",
            "infinity",
            "singular matrix",
            "linalgerror",
            "loss is nan",
            "overflow encountered",
        )
    ):
        return FailureRecord(
            kind=FailureKind.NUMERICAL_INSTABILITY,
            exception_type=exc_type,
            message=msg,
            recoverable=True,
        )

    # 5. Invalid Parameter Combination
    if any(
        kw in msg_lower
        for kw in (
            "penalty",
            "solver",
            "hyperparameter",
            "unexpected keyword",
            "unknown parameter",
            "invalid parameter",
            "must be in",
            "criterion",
            "not supported by solver",
        )
    ):
        return FailureRecord(
            kind=FailureKind.INVALID_PARAMETER,
            exception_type=exc_type,
            message=msg,
            recoverable=True,
        )

    # 6. Incompatible Data Shape or Dtype
    if any(
        kw in msg_lower
        for kw in (
            "shape mismatch",
            "could not convert",
            "incompatible shape",
            "dimension mismatch",
            "sparse",
            "dense matrix",
            "expected 2d",
        )
    ):
        return FailureRecord(
            kind=FailureKind.INCOMPATIBLE_DATA_SHAPE_OR_DTYPE,
            exception_type=exc_type,
            message=msg,
            recoverable=True,
        )

    # 7. Provider or Network Failure
    if isinstance(exc, (ConnectionError, OSError)) and any(
        kw in msg_lower for kw in ("connection refused", "network", "remote host", "socket", "http")
    ):
        return FailureRecord(
            kind=FailureKind.PROVIDER_OR_NETWORK_FAILURE,
            exception_type=exc_type,
            message=msg,
            recoverable=True,
            details={"retry_domain": "provider_operation"},
        )

    # 8. Artifact Verification Failure
    if any(kw in msg_lower for kw in ("artifact", "verification failed", "missing required file")):
        return FailureRecord(
            kind=FailureKind.ARTIFACT_VERIFICATION_FAILURE,
            exception_type=exc_type,
            message=msg,
            recoverable=False,
        )

    return FailureRecord(
        kind=FailureKind.UNKNOWN,
        exception_type=exc_type,
        message=msg,
        recoverable=False,
    )
