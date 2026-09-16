from collections import defaultdict
from typing import Any


class AdaptiveScheduler:
    """Dynamic exploration vs exploitation scheduler for candidate experiment queues."""

    def __init__(self, exploration_weight: float = 0.2) -> None:
        self.exploration_weight = exploration_weight

    def prioritize_queue(
        self,
        queue: list[dict[str, Any]],
        completed_results: list[dict[str, Any]],
        metric_name: str = "accuracy",
        higher_is_better: bool = True,
    ) -> list[dict[str, Any]]:
        if not completed_results or len(queue) <= 1:
            return list(queue)

        # Aggregate stats per model family
        family_scores: dict[str, list[float]] = defaultdict(list)
        for res in completed_results:
            fam = res.get("model_family")
            metrics = res.get("metrics", {})
            if fam and metric_name in metrics:
                family_scores[fam].append(metrics[metric_name])

        family_avg_score: dict[str, float] = {}
        for fam, scores in family_scores.items():
            family_avg_score[fam] = sum(scores) / len(scores)

        # Best overall score
        all_scores = [s for scs in family_scores.values() for s in scs]
        best_score = max(all_scores) if higher_is_better else min(all_scores)

        def score_candidate(cand: dict[str, Any]) -> float:
            fam = cand.get("model_family", "")
            trials = len(family_scores.get(fam, []))

            if trials == 0:
                # Untested family -> high exploration bonus
                return best_score if higher_is_better else -best_score

            avg_score = family_avg_score[fam]
            # UCB-like exploration term: bonus diminishes with more trials
            bonus = self.exploration_weight / (1.0 + trials)
            if higher_is_better:
                return avg_score + bonus
            else:
                return -avg_score + bonus

        # Sort descending by priority score
        sorted_queue = sorted(queue, key=score_candidate, reverse=True)
        return sorted_queue
