from pydantic import BaseModel, ConfigDict, Field

from swarmfold.versioning import migrate_record


class PromptEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(pattern=r"^0\.[0-9]+$")
    prompt_id: str
    prompt_version: str
    template: str
    input_schema: str
    output_schema: str


PLANNER_PROMPT = PromptEnvelope(
    schema_version="0.2",
    prompt_id="baseline-hypothesis",
    prompt_version="planner-v1",
    template=(
        "Create one concise, falsifiable hypothesis for a scaled logistic-regression "
        "baseline on a numeric classification dataset with {rows} rows, {features} features, "
        "and {classes} target classes. Use the competition brief below when it is provided, "
        "but do not invent rules that are not written there. Do not claim that an experiment "
        "has already run.\n\nCompetition brief:\n{project_context}"
    ),
    input_schema="planner-context-v1",
    output_schema="planner-suggestion-v1",
)


def load_prompt_envelope(value: dict[str, object]) -> PromptEnvelope:
    return PromptEnvelope.model_validate(migrate_record("prompt", value))


def render_planner_prompt(
    *,
    rows: int,
    features: int,
    classes: int,
    project_context: str = "Not provided.",
    retrieved_techniques: str = "",
) -> str:
    base = PLANNER_PROMPT.template.format(
        rows=rows,
        features=features,
        classes=classes,
        project_context=project_context[:20_000] or "Not provided.",
    )
    if retrieved_techniques:
        base += f"\n\nRelevant Method Knowledge:\n{retrieved_techniques}"
    return base
