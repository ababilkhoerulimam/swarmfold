from typing import Annotated, Any, Literal, TypeAlias, TypedDict

from pydantic import BaseModel, ConfigDict, Field, model_validator

from swarmfold.metrics import get_metric_spec

ModelFamily: TypeAlias = Literal[
    "logistic_regression",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
    "sgd_log_loss",
    "bagging",
    "soft_voting",
    "stacking",
    "lightgbm",
    "xgboost",
    "catboost",
]
FeatureStrategy: TypeAlias = Literal[
    "scaled_numeric",
    "raw_numeric",
    "selected_50",
    "selected_75",
    "target_encoded",
    "frequency_encoded",
    "power_transformed",
    "robust_scaled",
    "date_expanded",
    "interaction_pairs",
    "full_pipeline",
    "text_tfidf",
]
MetricName: TypeAlias = Literal[
    "accuracy",
    "roc_auc",
    "log_loss",
    "f1_macro",
    "rmse",
    "mae",
    "r2",
    "multilabel_f1",
]
MetricVersion: TypeAlias = Literal[
    "accuracy-v1",
    "roc-auc-v1",
    "log-loss-v1",
    "f1-macro-v1",
    "rmse-v1",
    "mae-v1",
    "r2-v1",
    "multilabel-f1-v1",
]
LaneId: TypeAlias = Literal[
    "ai-a",
    "ai-b",
    "ai-c",
    "ai-d",
    "ai-e",
    "human-1",
    "human-2",
    "human-3",
]


class CompetitionContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["0.1"] = "0.1"
    competition_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]+$")
    task_type: Literal[
        "binary_classification",
        "multiclass_classification",
        "regression",
        "multilabel_classification",
        "time_series_forecasting",
    ] = "binary_classification"
    target_column: str = Field(min_length=1)
    submission_columns: tuple[str, ...] = Field(default=("row", "prediction"), min_length=1)
    prediction_column: str = Field(default="prediction", min_length=1)
    id_columns: list[str] = Field(default_factory=lambda: ["row"])


class DataContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    data_version: str = Field(pattern=r"^sha256:[a-f0-9]{64}$")
    dataset_path: str
    dataset_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    row_count: int = Field(ge=10)
    feature_columns: list[str] = Field(min_length=1)
    target_column: str
    target_classes: list[str] = Field(default_factory=list)
    test_dataset_path: str | None = None
    test_dataset_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    sample_submission_path: str | None = None
    sample_submission_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    project_path: str | None = None
    project_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    test_row_count: int | None = Field(default=None, gt=0)


class MetricContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    metric_name: MetricName = "accuracy"
    metric_version: MetricVersion = "accuracy-v1"
    higher_is_better: bool = True
    minimum: float | None = 0.0
    maximum: float | None = 1.0

    @model_validator(mode="after")
    def version_matches_name(self) -> "MetricContract":
        spec = get_metric_spec(self.metric_name)
        if self.metric_version != spec.version:
            raise ValueError("metric_version does not match metric_name")
        if self.higher_is_better != spec.higher_is_better:
            raise ValueError("higher_is_better does not match metric_name")
        if self.minimum != spec.minimum or self.maximum != spec.maximum:
            raise ValueError("metric bounds do not match metric_name")
        return self


class ValidationContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    split_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]+$")
    strategy: Literal[
        "holdout",
        "stratified_holdout",
        "stratified_kfold",
        "repeated_stratified_kfold",
        "kfold",
        "group_kfold",
        "stratified_group_kfold",
        "expanding_walk_forward",
        "purged_time_series",
    ] = "stratified_holdout"
    seed: int = Field(ge=0)
    test_size: float = Field(default=0.25, gt=0.0, lt=1.0)
    splits: int = Field(default=5, ge=2)
    repeats: int = Field(default=1, ge=1)
    group_column: str | None = None
    time_column: str | None = None
    purge_window: int = Field(default=0, ge=0)
    embargo_window: int = Field(default=0, ge=0)
    fit_transforms_on_training_only: Literal[True] = True


class ContractBundle(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    competition: CompetitionContract
    data: DataContract
    metric: MetricContract
    validation: ValidationContract
    contract_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")


class ExperimentProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]+$")
    hypothesis: str = Field(min_length=10)
    hypothesis_id: str = Field(pattern=r"^[a-f0-9]{16}$")
    model_family: ModelFamily
    feature_strategy: FeatureStrategy = "scaled_numeric"
    parameters: dict[str, int | float | str | bool | list[int | float | str | bool]] = Field(
        default_factory=dict
    )
    dataset_path: str
    target_column: str
    data_version: str = Field(pattern=r"^sha256:[a-f0-9]{64}$")
    split_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]+$")
    metric_version: MetricVersion
    test_size: float = Field(default=0.25, gt=0.0, lt=1.0)
    seed: int = Field(ge=0)
    prompt_version: str = "planner-v1"
    original_experiment_id: str | None = None
    repair_attempt: int = 0
    hpo_study_id: str | None = None
    hpo_trial_number: int | None = Field(default=None, ge=0)


class RunResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    run_id: str
    status: Literal["SUCCEEDED", "FAILED", "PRUNED"]
    model_family: ModelFamily
    metrics: dict[str, float] = Field(default_factory=dict)
    train_rows: int = Field(default=0, ge=0)
    validation_rows: int = Field(default=0, ge=0)
    test_rows: int = Field(default=0, ge=0)
    positive_class: str | None = None
    duration_seconds: float = Field(default=0.0, ge=0.0)
    error: str | None = None
    tracking_run_id: str | None = None
    artifact_uri: str | None = None
    dataset_sha256: str | None = None
    contract_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    data_version: str | None = Field(default=None, pattern=r"^sha256:[a-f0-9]{64}$")
    split_id: str | None = Field(default=None, pattern=r"^[a-z0-9][a-z0-9_-]+$")
    metric_version: MetricVersion | None = None
    artifacts: list["ArtifactRecord"] = Field(default_factory=list)
    resource_usage: dict[str, float] = Field(default_factory=dict)
    fold_scores: list[float] = Field(default_factory=list)
    pruning_reason: str | None = None
    failure_record: dict[str, Any] | None = None
    repair_history: list[dict[str, Any]] = Field(default_factory=list)
    original_experiment_id: str | None = None
    resumed_from_checkpoint: bool = False


class VerificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    accepted: bool
    reasons: list[str] = Field(default_factory=list)


class BudgetPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_experiments: int = Field(default=4, ge=1, le=200)
    max_training_seconds: float | None = Field(default=None, gt=0.0)
    max_concurrent_training: int = Field(default=1, ge=1, le=5)


class ExperimentCritique(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    verdict: Literal["IMPROVED", "TIED", "REGRESSED", "FAILED"]
    accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    metric_name: MetricName = "accuracy"
    score: float | None = None
    recommendation: str


class ArtifactRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    size_bytes: int = Field(ge=0)


class PacketResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    packet_type: Literal["candidate", "failure", "waiver"]
    iteration_id: str
    lane_id: str
    packet_path: str
    manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class PacketFileRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    size_bytes: int = Field(ge=0)


class CandidateManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["0.1", "0.2"]
    evidence_schema_version: Literal["evidence-v1"] = "evidence-v1"
    packet_type: Literal["candidate"]
    iteration_id: str
    lane_id: str = Field(pattern=r"^(ai-[a-e]|human-[123])$")
    source_kind: Literal["ai", "human"] = "ai"
    experiment_id: str
    run_id: str
    tracking_run_id: str | None = None
    contract_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    data_version: str = Field(pattern=r"^sha256:[a-f0-9]{64}$")
    metric_version: MetricVersion
    split_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]+$")
    dataset_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    metrics: dict[str, float]
    submission_rows: int = Field(gt=0)
    artifacts: list[ArtifactRecord]
    resource_usage: dict[str, float] = Field(default_factory=dict)
    packet_files: list[PacketFileRecord]

    @model_validator(mode="after")
    def source_matches_lane(self) -> "CandidateManifest":
        if self.lane_id.startswith("human-") != (self.source_kind == "human"):
            raise ValueError("source_kind does not match lane_id")
        return self


class FailureManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["0.1", "0.2"]
    evidence_schema_version: Literal["evidence-v1"] = "evidence-v1"
    packet_type: Literal["failure"]
    iteration_id: str
    lane_id: str = Field(pattern=r"^(ai-[a-e]|human-[123])$")
    source_kind: Literal["ai", "human"] = "ai"
    experiment_id: str
    run_id: str
    contract_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    terminal_reason: str = Field(min_length=1)
    completed_run_ids: list[str]
    missing_deliverables: list[str] = Field(min_length=1)
    retry_useful: bool
    artifacts: list[ArtifactRecord]
    resource_usage: dict[str, float] = Field(default_factory=dict)
    packet_files: list[PacketFileRecord]

    @model_validator(mode="after")
    def source_matches_lane(self) -> "FailureManifest":
        if self.lane_id.startswith("human-") != (self.source_kind == "human"):
            raise ValueError("source_kind does not match lane_id")
        return self


class WaiverManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["0.1", "0.2"]
    evidence_schema_version: Literal["evidence-v1"] = "evidence-v1"
    packet_type: Literal["waiver"]
    iteration_id: str
    lane_id: str = Field(pattern=r"^human-[123]$")
    source_kind: Literal["human"] = "human"
    run_id: str
    contract_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    reason: str = Field(min_length=10)
    approver: str = Field(min_length=2)
    incomplete_work: list[str] = Field(default_factory=list)
    preserved_artifacts: bool = False
    packet_files: list[PacketFileRecord]


PacketManifest = Annotated[
    CandidateManifest | FailureManifest | WaiverManifest,
    Field(discriminator="packet_type"),
]


class CompletionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["0.1", "0.2"]
    packet_type: Literal["candidate", "failure", "waiver"]
    iteration_id: str
    lane_id: str = Field(pattern=r"^(ai-[a-e]|human-[123])$")
    run_id: str
    manifest_name: Literal[
        "artifact-manifest.json",
        "failure-manifest.json",
        "waiver-manifest.json",
    ]
    manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    published_at: str

    @model_validator(mode="after")
    def manifest_matches_packet_type(self) -> "CompletionRecord":
        expected = {
            "candidate": "artifact-manifest.json",
            "failure": "failure-manifest.json",
            "waiver": "waiver-manifest.json",
        }[self.packet_type]
        if self.manifest_name != expected:
            raise ValueError("manifest_name does not match packet_type")
        return self


class KnowledgeFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    finding_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]+$")
    iteration_id: str
    source_lane: str = Field(pattern=r"^(ai-[a-e]|human-[123])$")
    status: Literal[
        "HYPOTHESIS",
        "OBSERVED",
        "VALIDATED",
        "CROSS_LANE",
        "LEADERBOARD_ONLY",
        "REJECTED",
        "CONFLICT",
        "UNKNOWN",
    ]
    claim: str = Field(min_length=5)
    mechanism: str = Field(min_length=3)
    source_runs: list[str]
    data_version: str = Field(pattern=r"^sha256:[a-f0-9]{64}$")
    split_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]+$")
    metric_name: str = Field(min_length=1)
    local_delta: float | None = None
    leaderboard_delta: float | None = None
    limitations: list[str] = Field(default_factory=list)
    artifact_uris: list[str] = Field(default_factory=list)


class ExperimentState(TypedDict, total=False):
    thread_id: str
    iteration_id: str
    lane_id: str
    dataset_path: str
    test_dataset_path: str
    sample_submission_path: str
    project_path: str
    project_context: str
    target_column: str
    requested_task_type: str
    metric_name: MetricName
    seed: int
    test_size: float
    output_dir: str
    tracking_uri: str
    contracts: dict[str, object]
    contracts_path: str
    role_reports: dict[str, object]
    problem_formulation: dict[str, object]
    validation_audit: dict[str, object]
    domain_decomposition: dict[str, object]
    ensemble_assessment: dict[str, object]
    robustness_assessment: dict[str, object]
    iteration_reflection: dict[str, object]
    reflection_path: str
    next_iteration_knowledge: dict[str, object]
    literature_research: dict[str, object]
    budget: dict[str, object]
    scout_candidates: list[dict[str, object]]
    model_registry: list[dict[str, object]]
    technique_plan: dict[str, object]
    retrieval_provenance: dict[str, object]
    prior_knowledge: list[dict[str, object]]
    experiment_queue: list[dict[str, object]]
    scheduler_decisions: list[dict[str, object]]
    duplicate_hypotheses: list[str]
    current_experiment_index: int
    completed_results: list[dict[str, object]]
    critiques: list[dict[str, object]]
    model_coverage: list[str]
    ensemble_gate: Literal["PENDING", "OPEN", "SKIPPED"]
    stop_reason: str
    next_action: Literal["CONTINUE", "VERIFY"]
    resource_usage: dict[str, int | float]
    proposal: dict[str, object]
    run_result: dict[str, object]
    verification: dict[str, object]
    packet: dict[str, object]
    status: Literal[
        "PENDING",
        "RUNNING",
        "PROPOSED",
        "SUCCEEDED",
        "FAILED",
        "ACCEPTED",
        "REJECTED",
        "INTERRUPTED",
    ]
    next_nodes: list[str]
