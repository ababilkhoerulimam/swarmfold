import os
import re
from typing import Literal, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field

from swarmfold.schemas import ContractBundle, ExperimentState


class LiteratureSource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str = Field(min_length=1)
    provider: Literal["openalex", "crossref"]
    title: str = Field(min_length=1)
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    url: str
    cited_by_count: int = Field(default=0, ge=0)


class LiteratureReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["COMPLETED", "PARTIAL", "UNAVAILABLE"]
    queries: list[str]
    sources: list[LiteratureSource]
    handling_recommendations: list[str]
    limitations: list[str]


class LiteratureSearchTool(Protocol):
    name: str

    def search(self, query: str, *, limit: int) -> list[LiteratureSource]: ...


class OpenAlexSearchTool:
    name = "openalex"

    def __init__(self, client: httpx.Client | None = None) -> None:
        self.client = client or httpx.Client(timeout=12.0, follow_redirects=True)

    def search(self, query: str, *, limit: int) -> list[LiteratureSource]:
        params: dict[str, str | int] = {
            "search": query,
            "filter": "type:article",
            "per_page": limit,
        }
        if api_key := os.getenv("OPENALEX_API_KEY"):
            params["api_key"] = api_key
        response = self.client.get("https://api.openalex.org/works", params=params)
        response.raise_for_status()
        records = response.json().get("results", [])
        return [_openalex_source(item) for item in records if item.get("title")]


class CrossrefSearchTool:
    name = "crossref"

    def __init__(self, client: httpx.Client | None = None) -> None:
        self.client = client or httpx.Client(timeout=12.0, follow_redirects=True)

    def search(self, query: str, *, limit: int) -> list[LiteratureSource]:
        params: dict[str, str | int] = {
            "query.bibliographic": query,
            "filter": "type:journal-article",
            "rows": limit,
        }
        headers = {"User-Agent": "SwarmFold/0.1 literature-research-agent"}
        if mailto := os.getenv("CROSSREF_MAILTO"):
            params["mailto"] = mailto
        response = self.client.get(
            "https://api.crossref.org/works", params=params, headers=headers
        )
        response.raise_for_status()
        records = response.json().get("message", {}).get("items", [])
        return [_crossref_source(item) for item in records if item.get("title")]


class LiteratureResearchAgent:
    def __init__(
        self,
        tools: list[LiteratureSearchTool] | None = None,
        *,
        results_per_tool: int = 3,
    ) -> None:
        self.tools = tools or [OpenAlexSearchTool(), CrossrefSearchTool()]
        self.results_per_tool = results_per_tool

    def research(self, state: ExperimentState) -> LiteratureReview:
        queries = _research_queries(state)
        sources: list[LiteratureSource] = []
        limitations: list[str] = []
        successful_tools = 0
        for tool in self.tools:
            try:
                tool_results: list[LiteratureSource] = []
                for query in queries:
                    tool_results.extend(tool.search(query, limit=self.results_per_tool))
                sources.extend(tool_results)
                successful_tools += 1
            except (httpx.HTTPError, ValueError, TypeError, KeyError) as exc:
                limitations.append(f"{tool.name} search failed with {type(exc).__name__}")
        unique_sources = _deduplicate_sources(sources)
        if successful_tools == len(self.tools) and unique_sources:
            status = "COMPLETED"
        elif successful_tools and unique_sources:
            status = "PARTIAL"
        else:
            status = "UNAVAILABLE"
        if not unique_sources:
            limitations.append("No citable literature metadata was retrieved")
        limitations.append(
            "Literature supports hypotheses only; dataset-specific claims require local validation"
        )
        return LiteratureReview(
            status=status,
            queries=queries,
            sources=unique_sources[:12],
            handling_recommendations=_handling_recommendations(state, unique_sources),
            limitations=limitations,
        )


class DisabledLiteratureResearchAgent:
    def research(self, state: ExperimentState) -> LiteratureReview:
        return LiteratureReview(
            status="UNAVAILABLE",
            queries=_research_queries(state),
            sources=[],
            handling_recommendations=[],
            limitations=[
                "Online literature research is disabled; set SWARMFOLD_LITERATURE_RESEARCH=true"
            ],
        )


def create_literature_research_agent(
    enabled: bool | None = None,
) -> LiteratureResearchAgent | DisabledLiteratureResearchAgent:
    if enabled is None:
        enabled = os.getenv("SWARMFOLD_LITERATURE_RESEARCH", "false").casefold() in {
            "1",
            "true",
            "yes",
            "on",
        }
    return LiteratureResearchAgent() if enabled else DisabledLiteratureResearchAgent()


def _research_queries(state: ExperimentState) -> list[str]:
    contracts = ContractBundle.model_validate(state["contracts"])
    columns = " ".join(contracts.data.feature_columns[:8])
    domain = _search_terms(state.get("project_context", ""), limit=10)
    eda = state.get("role_reports", {}).get("eda_agent", {})
    handling_terms: list[str] = []
    if isinstance(eda, dict):
        if int(eda.get("categorical_features", 0)):
            handling_terms.append("categorical encoding")
        if int(eda.get("missing_values", 0)):
            handling_terms.append("missing value imputation")
        counts = eda.get("class_counts", {})
        if isinstance(counts, dict) and counts:
            values = [int(value) for value in counts.values()]
            if min(values) and max(values) / min(values) >= 2:
                handling_terms.append("class imbalance")
    task_query = (
        f"machine learning {contracts.competition.task_type} {columns} "
        f"{' '.join(handling_terms)} preprocessing validation"
    )
    queries = [" ".join(task_query.split())[:500]]
    if domain:
        queries.append(f"{domain} prediction machine learning methodology"[:500])
    return queries


def _search_terms(value: str, *, limit: int) -> str:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", value.casefold())
    blocked = {
        "and",
        "are",
        "data",
        "dataset",
        "for",
        "from",
        "project",
        "the",
        "this",
        "with",
    }
    selected: list[str] = []
    for token in tokens:
        if token not in blocked and token not in selected:
            selected.append(token)
        if len(selected) == limit:
            break
    return " ".join(selected)


def _handling_recommendations(
    state: ExperimentState, sources: list[LiteratureSource]
) -> list[str]:
    contracts = ContractBundle.model_validate(state["contracts"])
    recommendations = [
        "Fit every learned preprocessing step inside each training fold.",
        "Compare at least one linear and one nonlinear model on the same validation split.",
    ]
    eda = state.get("role_reports", {}).get("eda_agent", {})
    if isinstance(eda, dict) and int(eda.get("categorical_features", 0)):
        recommendations.append(
            "Compare fold-safe categorical encodings and verify unseen-category behavior."
        )
    if isinstance(eda, dict) and int(eda.get("missing_values", 0)):
        recommendations.append(
            "Treat imputation strategy as an experiment and fit it within each fold."
        )
    if contracts.metric.metric_name == "roc_auc":
        recommendations.append(
            "Evaluate probability ranking and calibration separately before threshold selection."
        )
    if sources:
        recommendations.append(
            "Translate cited methods into falsifiable local experiments before promotion."
        )
    return recommendations


def _openalex_source(item: dict[str, object]) -> LiteratureSource:
    location = item.get("primary_location") or {}
    source = location.get("source") or {} if isinstance(location, dict) else {}
    doi = _normalize_doi(item.get("doi"))
    source_id = str(item.get("id") or doi or item["title"])
    return LiteratureSource(
        source_id=source_id,
        provider="openalex",
        title=str(item["title"]),
        year=_optional_int(item.get("publication_year")),
        venue=str(source.get("display_name")) if source.get("display_name") else None,
        doi=doi,
        url=str(doi or item.get("id")),
        cited_by_count=max(0, _optional_int(item.get("cited_by_count")) or 0),
    )


def _crossref_source(item: dict[str, object]) -> LiteratureSource:
    title_value = item["title"]
    title = str(title_value[0] if isinstance(title_value, list) else title_value)
    venue_value = item.get("container-title")
    venue = venue_value[0] if isinstance(venue_value, list) and venue_value else None
    doi = _normalize_doi(item.get("DOI"))
    return LiteratureSource(
        source_id=str(doi or item.get("URL") or title),
        provider="crossref",
        title=title,
        year=_crossref_year(item),
        venue=str(venue) if venue else None,
        doi=doi,
        url=str(doi or item.get("URL") or "https://api.crossref.org/works"),
        cited_by_count=max(0, _optional_int(item.get("is-referenced-by-count")) or 0),
    )


def _normalize_doi(value: object) -> str | None:
    if not value:
        return None
    doi = str(value).removeprefix("https://doi.org/").removeprefix("http://doi.org/")
    return f"https://doi.org/{doi}"


def _crossref_year(item: dict[str, object]) -> int | None:
    for field in ("published", "published-print", "published-online", "issued"):
        value = item.get(field)
        if not isinstance(value, dict):
            continue
        parts = value.get("date-parts")
        if isinstance(parts, list) and parts and isinstance(parts[0], list) and parts[0]:
            return _optional_int(parts[0][0])
    return None


def _optional_int(value: object) -> int | None:
    return int(value) if isinstance(value, int | float) else None


def _deduplicate_sources(sources: list[LiteratureSource]) -> list[LiteratureSource]:
    unique: dict[str, LiteratureSource] = {}
    for source in sorted(sources, key=lambda item: item.cited_by_count, reverse=True):
        key = (source.doi or source.title).casefold()
        unique.setdefault(key, source)
    return list(unique.values())
