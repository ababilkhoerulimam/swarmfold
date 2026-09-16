from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, Protocol

from swarmfold.technique_cards import TECHNIQUE_CORPUS, TechniqueCard


class RerankerProtocol(Protocol):
    def rerank(self, query: str, cards: list[TechniqueCard]) -> list[TechniqueCard]:
        ...


class ScoreReranker:
    def rerank(self, query: str, cards: list[TechniqueCard]) -> list[TechniqueCard]:
        # Default pass-through deterministic reranker
        return cards


class TechniqueRetriever:
    def __init__(
        self,
        corpus: Sequence[TechniqueCard] | None = None,
        reranker: RerankerProtocol | None = None,
    ) -> None:
        self.corpus = list(corpus or TECHNIQUE_CORPUS)
        self.reranker = reranker or ScoreReranker()

    def retrieve(
        self,
        *,
        role: str | None = None,
        category: str | None = None,
        task_type: str = "binary_classification",
        dataset_traits: list[str] | None = None,
        query: str | None = None,
        top_k: int = 3,
        role_scoped_only: bool = False,
    ) -> list[TechniqueCard]:
        traits = set(trait.lower() for trait in (dataset_traits or []))

        # Tokenize query, discarding very short stopwords
        raw_tokens = re.findall(r"\w+", (query or "").lower())
        stopwords = {
            "and", "or", "the", "in", "on", "at", "to",
            "for", "with", "a", "an", "is", "of", "by",
        }
        query_tokens = set(t for t in raw_tokens if len(t) > 1 and t not in stopwords)

        scored_cards: list[tuple[float, TechniqueCard]] = []
        for card in self.corpus:
            # If role_scoped_only is requested, enforce strict role ownership
            if (
                role
                and role_scoped_only
                and card.canonical_role != role
                and role not in card.secondary_roles
            ):
                continue

            # Task type compatibility filter: allow if task matches or category is task-agnostic
            agnostic_cats = {"eda", "operations", "robustness"}
            if task_type not in card.task_types and card.category not in agnostic_cats:
                continue

            score = 1.0

            # Role relevance boost (canonical owner gets higher boost than secondary)
            if role:
                if card.canonical_role == role:
                    score += 5.0
                elif role in card.secondary_roles:
                    score += 3.0

            # Category preference boost or match
            if category and card.category == category:
                score += 3.0

            # Dataset traits relevance
            if traits:
                card_prereqs = set(p.lower() for p in card.prerequisites)
                matched_prereqs = traits.intersection(card_prereqs)
                if matched_prereqs:
                    score += 8.0 * len(matched_prereqs)

                prereq_str = " ".join(card.prerequisites).lower()
                strengths_str = " ".join(card.strengths).lower()
                card_trait_text = (
                    f"{card.card_id} {card.title} {card.when_to_use} {card.rationale} "
                    f"{card.assumptions} {prereq_str} {strengths_str}"
                ).lower()

                for trait in traits:
                    clean_trait = trait.lower().strip()
                    trait_phrase = clean_trait.replace("_", " ")
                    if clean_trait in card_trait_text or trait_phrase in card_trait_text:
                        score += 5.0
                    else:
                        stop_traits = {
                            "features", "columns", "data", "dataset", "split", "values",
                        }
                        distinct_words = [
                            w for w in clean_trait.split("_")
                            if w not in stop_traits and len(w) > 2
                        ]
                        if distinct_words and any(w in card_trait_text for w in distinct_words):
                            score += 2.5 * sum(1 for w in distinct_words if w in card_trait_text)

            # Multi-field query token matching
            if query_tokens:
                title_words = set(re.findall(r"\w+", card.title.lower()))
                pkg_words = set(re.findall(r"\w+", card.exact_package.lower()))
                card_id_words = set(re.findall(r"\w+", card.card_id.lower()))
                aliases_words = set(re.findall(r"\w+", " ".join(card.aliases).lower()))
                desc_text = f"{card.definition} {card.when_to_use} {card.rationale}".lower()
                desc_words = set(re.findall(r"\w+", desc_text))

                # Exact card_id / title / alias match has highest weight
                target_identifiers = title_words.union(card_id_words).union(aliases_words)
                title_overlap = len(query_tokens.intersection(target_identifiers))
                score += 4.0 * title_overlap

                # Exact package match
                pkg_overlap = len(query_tokens.intersection(pkg_words))
                score += 2.5 * pkg_overlap

                # Description / indication match
                desc_overlap = len(query_tokens.intersection(desc_words))
                score += 1.2 * desc_overlap

            scored_cards.append((score, card))

        scored_cards.sort(key=lambda item: item[0], reverse=True)
        selected = [card for _, card in scored_cards[:top_k]]
        return self.reranker.rerank(query or "", selected)

    def format_for_prompt(self, cards: Sequence[TechniqueCard]) -> str:
        if not cards:
            return "No specific technique cards retrieved."

        lines = ["### Retrieved SwarmFold Technique Knowledge:"]
        for card in cards:
            lines.append(f"\n#### [{card.card_id}] {card.title}")
            lines.append(
                f"- **Category:** {card.category} | **Canonical Role:** {card.canonical_role}"
            )
            if card.secondary_roles:
                lines.append(f"- **Secondary Roles:** `{', '.join(card.secondary_roles)}`")
            if card.aliases:
                lines.append(f"- **Aliases:** `{', '.join(card.aliases)}`")
            if card.exact_package:
                lines.append(f"- **Implementation Package:** `{card.exact_package}`")
            lines.append(f"- **Rationale & Definition:** {card.rationale}")
            if card.when_to_use:
                lines.append(f"- **When to Use:** {card.when_to_use}")
            if card.when_not_to_use:
                lines.append(f"- **When NOT to Use:** {card.when_not_to_use}")
            bound_rule = (
                f"- **Leakage Boundary Rule:** {card.fold_boundary_rule} "
                f"(Risk: {card.leakage_risk})"
            )
            lines.append(bound_rule)
            if card.recommended_parameters:
                params_str = ", ".join(
                    f"{k}={v}" for k, v in card.recommended_parameters.items()
                )
                lines.append(f"- **Recommended Defaults:** `{params_str}`")
            elif card.key_hyperparameters:
                lines.append(f"- **Key Hyperparameters:** {card.key_hyperparameters}")
            if card.example_hypothesis:
                lines.append(f"- **Hypothesis Template:** \"{card.example_hypothesis}\"")
            if card.acceptance_criteria:
                lines.append(f"- **Acceptance Criteria:** {card.acceptance_criteria}")
        return "\n".join(lines)

    def record_provenance(
        self,
        cards: Sequence[TechniqueCard],
        query: str | None = None,
        role: str | None = None,
    ) -> dict[str, Any]:
        return {
            "query": query,
            "role": role,
            "selected_cards": [
                {
                    "card_id": c.card_id,
                    "title": c.title,
                    "category": c.category,
                    "canonical_role": c.canonical_role,
                    "exact_package": c.exact_package,
                    "leakage_risk": c.leakage_risk,
                    "content_hash": c.content_hash,
                }
                for c in cards
            ],
        }
