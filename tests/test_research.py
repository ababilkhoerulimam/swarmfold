from pathlib import Path

from swarmfold.graph import run_experiment
from swarmfold.research import LiteratureResearchAgent, LiteratureSource

FIXTURE = Path(__file__).parent / "fixtures" / "demo_classification.csv"


class FakeLiteratureTool:
    name = "fake-journal-index"

    def search(self, query: str, *, limit: int) -> list[LiteratureSource]:
        assert "machine learning" in query
        assert limit == 2
        return [
            LiteratureSource(
                source_id="10.1000/example",
                provider="crossref",
                title="Leakage-safe preprocessing for predictive modeling",
                year=2024,
                venue="Journal of Reproducible Models",
                doi="https://doi.org/10.1000/example",
                url="https://doi.org/10.1000/example",
                cited_by_count=12,
            )
        ]


def test_literature_agent_runs_before_modeling_and_publishes_evidence(tmp_path: Path) -> None:
    agent = LiteratureResearchAgent([FakeLiteratureTool()], results_per_tool=2)

    result = run_experiment(
        FIXTURE,
        output_dir=tmp_path,
        max_experiments=1,
        research_agent=agent,
    )

    review = result["literature_research"]
    assert review["status"] == "COMPLETED"
    assert review["sources"][0]["doi"] == "https://doi.org/10.1000/example"
    assert result["role_reports"]["literature_research_agent"]["status"] == "COMPLETED"
    packet = Path(result["packet"]["packet_path"])
    assert (packet / "literature-review.json").is_file()


def test_literature_sources_are_in_generated_notebook(tmp_path: Path) -> None:
    agent = LiteratureResearchAgent([FakeLiteratureTool()], results_per_tool=2)
    bundle = Path(__file__).parent / "fixtures"

    result = run_experiment(
        bundle / "bundle_learn.csv",
        target_column="outcome",
        test_dataset_path=bundle / "bundle_score.csv",
        sample_submission_path=bundle / "bundle_template.csv",
        project_path=bundle / "bundle_brief.md",
        output_dir=tmp_path,
        max_experiments=1,
        research_agent=agent,
    )

    packet = Path(result["packet"]["packet_path"])
    notebook = (packet / "end-to-end.ipynb").read_text(encoding="utf-8")
    assert "Leakage-safe preprocessing for predictive modeling" in notebook
    assert "https://doi.org/10.1000/example" in notebook
