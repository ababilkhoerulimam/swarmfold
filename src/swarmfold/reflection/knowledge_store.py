from pathlib import Path
from typing import Any

from swarmfold.reflection.reflector import ReflectionReport


class ReflectionMemoryStore:
    """Persists and retrieves reflection memory packages across iterations."""

    def save_reflection(self, output_dir: Path, report: ReflectionReport) -> Path:
        memory_dir = Path(output_dir) / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)

        json_path = memory_dir / f"{report.iteration_id}-reflection.json"
        json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

        md_path = memory_dir / f"{report.iteration_id}-reflection.md"
        md_lines = [
            f"# Iteration Reflection: {report.iteration_id}",
            f"- **Metric:** {report.metric_name}",
            f"- **Champion:** {report.champion_experiment_id} "
            f"({report.best_family}) with score {report.champion_score}",
            "\n## Key Insights:",
        ]
        for ins in report.insights:
            md_lines.append(f"- **[{ins.category}]** {ins.insight} (Evidence: {ins.evidence})")

        if report.failed_hypotheses:
            md_lines.append("\n## Failed / Rejected Hypotheses:")
            for fh in report.failed_hypotheses:
                md_lines.append(f"- {fh}")

        md_path.write_text("\n".join(md_lines), encoding="utf-8")
        return json_path

    def load_reflection(self, path: Path) -> ReflectionReport:
        content = Path(path).read_text(encoding="utf-8")
        return ReflectionReport.model_validate_json(content)

    def to_prior_knowledge_package(
        self,
        report: ReflectionReport,
        source_profile: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Convert a reflection into the prior-knowledge package contract."""
        return {
            "source_iteration_id": report.iteration_id,
            "source_profile": source_profile or {"metric": report.metric_name},
            "champion_model_family": report.best_family,
            "champion_score": report.champion_score,
            "insights": [ins.model_dump() for ins in report.insights],
            "failed_hypotheses": report.failed_hypotheses,
        }
