from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ReflectionInsight(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    category: str
    insight: str
    evidence: str
    transferable: bool = True


class ReflectionReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    iteration_id: str
    champion_experiment_id: str | None = None
    champion_score: float | None = None
    metric_name: str
    best_family: str | None = None
    insights: list[ReflectionInsight] = Field(default_factory=list)
    failed_hypotheses: list[str] = Field(default_factory=list)


class IterationReflector:
    """Reflects on finished experiment trajectories to formulate transferable meta-memory."""

    def generate_reflection(
        self,
        iteration_id: str,
        completed_results: list[dict[str, Any]],
        metric_name: str = "accuracy",
        higher_is_better: bool = True,
        ensemble_assessment: dict[str, Any] | None = None,
    ) -> ReflectionReport:
        if not completed_results:
            return ReflectionReport(
                iteration_id=iteration_id,
                metric_name=metric_name,
                insights=[
                    ReflectionInsight(
                        category="execution",
                        insight="No completed experiments were recorded for this iteration.",
                        evidence="Empty completed results queue.",
                        transferable=False,
                    )
                ],
            )

        # Separate succeeded vs failed
        succeeded = [r for r in completed_results if r.get("status") == "SUCCEEDED"]
        failed = [r for r in completed_results if r.get("status") != "SUCCEEDED"]

        insights: list[ReflectionInsight] = []
        champion_id = None
        champion_score = None
        best_family = None

        if succeeded:
            sorted_succeeded = sorted(
                succeeded,
                key=lambda r: r.get("metrics", {}).get(
                    metric_name, -float("inf") if higher_is_better else float("inf")
                ),
                reverse=higher_is_better,
            )
            champ = sorted_succeeded[0]
            champion_id = champ.get("experiment_id")
            champion_score = champ.get("metrics", {}).get(metric_name)
            best_family = champ.get("model_family")

            insights.append(
                ReflectionInsight(
                    category="champion",
                    insight=f"Model family '{best_family}' delivered the top validation score.",
                    evidence=(
                        f"Experiment '{champion_id}' achieved {metric_name}={champion_score:.4f}."
                    ),
                    transferable=True,
                )
            )

            family_scores: dict[str, list[float]] = {}
            for res in succeeded:
                fam = res.get("model_family", "unknown")
                score = res.get("metrics", {}).get(metric_name)
                if score is not None:
                    family_scores.setdefault(fam, []).append(score)

            if len(family_scores) >= 2:
                summary = ", ".join(
                    f"{k}: avg={sum(v) / len(v):.4f}" for k, v in family_scores.items()
                )
                insights.append(
                    ReflectionInsight(
                        category="family_comparison",
                        insight="Family performance hierarchy observed during validation.",
                        evidence=summary,
                        transferable=True,
                    )
                )

        if ensemble_assessment:
            decision = ensemble_assessment.get("decision")
            gain = ensemble_assessment.get("ensemble_gain")
            if decision == "SUPPORTED" and gain is not None:
                insights.append(
                    ReflectionInsight(
                        category="ensemble",
                        insight=(
                            "Ensemble blending improved performance by "
                            f"+{gain:.4f} over best base candidate."
                        ),
                        evidence="Ensemble gain confirmed on identical holdout/OOF.",
                        transferable=True,
                    )
                )

        failed_hypotheses = [
            str(f.get("experiment_id", "unknown")) + ": " + str(f.get("error", "Failed execution"))
            for f in failed
        ]

        return ReflectionReport(
            iteration_id=iteration_id,
            champion_experiment_id=champion_id,
            champion_score=champion_score,
            metric_name=metric_name,
            best_family=best_family,
            insights=insights,
            failed_hypotheses=failed_hypotheses,
        )
