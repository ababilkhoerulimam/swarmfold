import os
from dataclasses import dataclass
from typing import Literal, cast

PlannerMode = Literal["mock", "live"]


@dataclass(frozen=True)
class PlannerSettings:
    mode: PlannerMode = "mock"
    model: str | None = None

    @classmethod
    def from_values(
        cls,
        mode: PlannerMode | None = None,
        model: str | None = None,
    ) -> "PlannerSettings":
        resolved_mode = mode or os.getenv("SWARMFOLD_PLANNER", "mock")
        if resolved_mode not in {"mock", "live"}:
            raise ValueError("SWARMFOLD_PLANNER must be 'mock' or 'live'")
        resolved_model = model or os.getenv("SWARMFOLD_MODEL")
        if resolved_mode == "live" and not resolved_model:
            raise ValueError("Live planning requires --model or SWARMFOLD_MODEL")
        return cls(mode=cast(PlannerMode, resolved_mode), model=resolved_model)
