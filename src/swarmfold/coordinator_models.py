from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from swarmfold.schemas import ContractBundle


class LaneCharter(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    lane_id: str = Field(pattern=r"^(ai-[a-e]|human-[123])$")
    objective: str = Field(min_length=10)
    model_focus: str = Field(min_length=3)
    risk_level: Literal["low", "medium", "high"]
    expected_diversity: str = Field(min_length=10)


class IterationCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    iteration_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]+$")
    contracts: ContractBundle
    lane_charters: list[LaneCharter] = Field(min_length=3, max_length=8)
    submission_quota: int = Field(default=3, ge=0, le=8)
    knowledge_snapshot_id: str | None = Field(
        default=None,
        pattern=r"^v[0-9]{3,}$",
    )

    @model_validator(mode="after")
    def unique_lanes(self) -> "IterationCreate":
        lanes = [charter.lane_id for charter in self.lane_charters]
        if len(set(lanes)) != len(lanes):
            raise ValueError("lane charters must have unique lane_id values")
        legacy = {"ai-a", "ai-b", "ai-c"}
        expected = {"ai-a", "ai-b", "ai-c", "ai-d", "ai-e"}
        if set(lanes) not in (
            legacy,
            legacy | {"human-1", "human-2", "human-3"},
            expected,
            expected | {"human-1", "human-2", "human-3"},
        ):
            raise ValueError(
                "lane charters must define a supported three-lane or five-lane topology"
            )
        return self


class PodRegistration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pod_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]+$")
    iteration_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]+$")
    lane_id: str = Field(pattern=r"^ai-[a-e]$")
    capabilities: dict[str, str | int | float | bool] = Field(default_factory=dict)


class PodHeartbeat(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["IDLE", "RUNNING", "PACKET_READY", "FAILED"]


class PodRecord(BaseModel):
    pod_id: str
    iteration_id: str
    lane_id: str
    status: str
    last_seen_at: str


class PacketReceipt(BaseModel):
    iteration_id: str
    lane_id: str
    packet_type: Literal["candidate", "failure", "waiver"]
    packet_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    artifact_path: str
    idempotent_replay: bool = False


class BarrierStatus(BaseModel):
    iteration_id: str
    status: Literal["PENDING", "READY"]
    received_lanes: list[str]
    missing_lanes: list[str]


class SubmissionDecisionCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lane_id: str = Field(pattern=r"^(ai-[a-e]|human-[123])$")
    decision: Literal["SUBMITTED", "WITHHELD"]
    approver: str = Field(min_length=2)
    reason: str = Field(min_length=5)


class SubmissionGateStatus(BaseModel):
    iteration_id: str
    status: Literal["BLOCKED", "PENDING", "READY"]
    quota: int
    candidate_lanes: list[str]
    submitted_lanes: list[str]
    withheld_lanes: list[str]
    pending_lanes: list[str]


class LeaderboardResultCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lane_id: str = Field(pattern=r"^(ai-[a-e]|human-[123])$")
    score: float
    rank: int | None = Field(default=None, ge=1)
    recorded_by: str = Field(min_length=2)
    local_supported: bool = False
    notes: str = ""


class KnowledgeMergeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    snapshot_id: str = Field(pattern=r"^v[0-9]{3,}$")


class KnowledgeSnapshotReceipt(BaseModel):
    snapshot_id: str
    source_iteration_id: str
    manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    artifact_path: str
    idempotent_replay: bool = False
