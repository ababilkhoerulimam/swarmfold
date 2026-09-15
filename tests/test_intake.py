from pathlib import Path

from swarmfold.intake import CompetitionProfile, LLMCompetitionProfiler


class _StructuredProfiler:
    def __init__(self, response: dict[str, object], prompts: list[str]) -> None:
        self.response = response
        self.prompts = prompts

    def invoke(self, prompt: str) -> dict[str, object]:
        self.prompts.append(prompt)
        return self.response


class _ChatModel:
    def __init__(self, response: dict[str, object], prompts: list[str]) -> None:
        self.response = response
        self.prompts = prompts

    def with_structured_output(self, schema):
        assert schema is CompetitionProfile
        return _StructuredProfiler(self.response, self.prompts)


def test_profiler_uses_content_not_prescribed_filenames(tmp_path: Path) -> None:
    observations = tmp_path / "alpha_observations.tsv"
    observations.write_text("case\ttext\tlabel\n1\thello\tpositive\n", encoding="utf-8")
    brief = tmp_path / "rules-any-name.md"
    brief.write_text("Rank sentiment predictions using macro F1.", encoding="utf-8")
    prompts: list[str] = []
    response = {
        "summary": "Text classification scored with macro F1 on labeled examples.",
        "task_type": "text_classification",
        "metric": "macro_f1",
        "training_file": "alpha_observations.tsv",
        "inference_files": [],
        "submission_template": None,
        "target_column": "label",
        "prediction_columns": [],
        "id_columns": ["case"],
        "validation_strategy": "stratified cross-validation",
        "files": [
            {
                "filename": "alpha_observations.tsv",
                "role": "labeled_examples",
                "reasoning": "Contains text and label columns",
            },
            {
                "filename": "rules-any-name.md",
                "role": "competition_brief",
                "reasoning": "Describes metric and task",
            },
        ],
        "execution_ready": False,
        "blockers": ["Text classification execution adapter is not installed"],
        "questions": ["Which file contains inference rows?"],
        "confidence": 0.82,
    }
    profiler = LLMCompetitionProfiler(
        "fake:model", model_factory=lambda _: _ChatModel(response, prompts)
    )

    profile = profiler.profile([observations, brief])

    assert profile.task_type == "text_classification"
    assert profile.execution_ready is False
    assert "macro F1" in prompts[0]
    assert "Never force a train/test assumption" in prompts[0]
