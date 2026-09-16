from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def clean_text(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9_-]", "_", text)
    text = re.sub(r"[\s_-]+", "_", text).strip("_")
    if not text:
        return "card_unknown"
    if not text[0].isalnum():
        text = f"c_{text}".strip("_")
    return text[:64]


def map_leakage(risk_str: str | None) -> str:
    risk = (risk_str or "").lower()
    if any(term in risk for term in ["tinggi", "high", "kritis", "severe"]):
        return "HIGH"
    if any(term in risk for term in ["sedang", "medium", "moderat", "moderate"]):
        return "MEDIUM"
    if any(term in risk for term in ["rendah", "low", "minimal"]):
        return "LOW"
    return "NONE"


def get_canonical_doc_key(filename: str) -> str:
    fn = filename.lower().strip()
    if "domain" in fn or "docum" in fn:
        return "domain_target_decomposition"
    if "eda" in fn:
        return "eda"
    if "ensemble" in fn:
        return "ensemble"
    if "feature" in fn:
        return "feature_engineering"
    if "modeling" in fn:
        return "modeling"
    if "operational" in fn:
        return "operational"
    if "problem" in fn or "probem" in fn:
        return "problem_formulation"
    if "robustness" in fn:
        return "robustness_uncertainty"
    if "temporal" in fn:
        return "temporal_validation"
    if "val" in fn:
        return "validation"
    return slugify(fn)


def map_category_role(doc_key: str) -> tuple[str, str]:
    if doc_key == "feature_engineering":
        return "feature_engineering", "feature_engineer"
    if doc_key == "modeling":
        return "model", "model_engineer"
    if doc_key == "ensemble":
        return "ensemble", "oof_ensemble_scientist"
    if doc_key == "temporal_validation":
        return "validation", "temporal_validation_leakage_auditor"
    if doc_key == "validation":
        return "validation", "validation_agent"
    if doc_key == "eda":
        return "eda", "eda_specialist"
    if doc_key == "robustness_uncertainty":
        return "robustness", "robustness_auditor"
    if doc_key == "problem_formulation":
        return "problem_formulation", "problem_formulator"
    if doc_key == "domain_target_decomposition":
        return "target_decomposition", "target_decomposition_specialist"
    if doc_key == "operational":
        return "operations", "pod_supervisor"
    return "preprocessing", "model_scout"


def map_task_types(problem_str: str | None) -> list[str]:
    p = (problem_str or "").lower()
    tasks: list[str] = []
    if any(k in p for k in ["klasifikasi biner", "binary", "biner"]):
        tasks.append("binary_classification")
    elif "klasifikasi" in p or "classification" in p:
        tasks.append("binary_classification")
        tasks.append("multiclass_classification")
    if any(k in p for k in ["multikelas", "multiclass", "multi-class"]):
        tasks.append("multiclass_classification")
    if any(k in p for k in ["regresi", "regression"]):
        tasks.append("regression")
    if any(k in p for k in ["semua", "all", "agnostik", "tabular"]) or not tasks:
        tasks = ["binary_classification", "multiclass_classification", "regression"]
    return list(dict.fromkeys(tasks))


OWNERSHIP_PRIORITY: dict[str, str] = {
    "temporal leakage checks (validation gates)": "temporal_validation",
    "walk forward / expanding window cross validation": "temporal_validation",
    "purged k fold & embargo": "temporal_validation",
    "rotation forest": "ensemble",
    "competing risks": "target_decomposition",
    "multi state survival": "target_decomposition",
    "interval censored survival": "target_decomposition",
    "quantile regression": "model",
    "tabpfn 2.5": "model",
    "tabiclv2": "model",
}


def parse_knowledge_doc(filepath: Path | str) -> list[dict[str, Any]]:
    path = Path(filepath)
    if not path.exists():
        return []

    content = path.read_text(encoding="utf-8", errors="ignore")
    doc_key = get_canonical_doc_key(path.name)
    category, canonical_role = map_category_role(doc_key)
    sections = re.split(r"\n(?=#{1,4}\s+)", content)

    cards: list[dict[str, Any]] = []

    # 1. Parse 20-field specification tables
    for sec in sections:
        if "| Field |" not in sec and "| field |" not in sec.lower():
            continue

        header_match = re.match(r"^(#{1,4})\s+(.+)", sec.strip())
        raw_title = header_match.group(2).strip() if header_match else "Unknown"
        title = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", raw_title)
        title = re.sub(r"^[0-9.]+\s*", "", title).strip()

        table_match = re.search(
            r"\|\s*Field\s*\|\s*(?:Detail|Deskripsi)\s*\|.*?\n((?:\|.*?\n)+)",
            sec,
            re.DOTALL | re.IGNORECASE,
        )
        fields: dict[str, str] = {}
        if table_match:
            for row in table_match.group(1).strip().split("\n"):
                cols = [c.strip() for c in row.split("|")]
                if len(cols) >= 3:
                    fn = cols[1].replace("**", "").strip().lower()
                    fv = cols[2].strip()
                    if fn and not set(fn).issubset({"-", " "}):
                        fields[fn] = fv

        if not fields:
            continue

        definition = fields.get("definisi", "")
        when_to_use = fields.get("kapan digunakan", "")
        when_not_to_use = fields.get("kapan tidak digunakan", "")
        leakage_text = fields.get("risiko leakage", "")
        leakage_risk = map_leakage(leakage_text)
        default_rule = (
            "Fit strictly inside training fold; transform validation/test out-of-fold."
        )
        fold_rule = leakage_text or default_rule
        task_types = map_task_types(fields.get("problem yang didukung", ""))
        exact_pkg = fields.get("exact package", "") or fields.get("exact implementation", "")

        prereqs = [
            clean_text(p)
            for p in fields.get("data preprocessing", "").split(";")
            if clean_text(p) and clean_text(p) != "Standar"
        ][:5]
        if not prereqs:
            prereqs = ["tabular"]

        ens_compat = (
            fields.get("ensemble compatibility", "")
            or fields.get("agent compatibility", "")
        )

        card_dict = {
            "title": title,
            "category": category,
            "canonical_role": canonical_role,
            "task_types": task_types,
            "prerequisites": prereqs,
            "leakage_risk": leakage_risk,
            "fold_boundary_rule": fold_rule,
            "rationale": clean_text(definition)[:280] or f"Technique {title} for {category}",
            "definition": definition,
            "when_to_use": when_to_use,
            "when_not_to_use": when_not_to_use,
            "assumptions": fields.get("asumsi", ""),
            "preprocessing": fields.get("data preprocessing", ""),
            "dataset_scale": fields.get("dataset scale", ""),
            "metric_compatibility": fields.get("metric compatibility", ""),
            "key_hyperparameters": fields.get("hyperparameter penting", ""),
            "initial_parameter_ranges": fields.get("kisaran parameter awal", ""),
            "compute_profile": fields.get("cpu/gpu/ram", ""),
            "failure_modes": fields.get("failure modes", ""),
            "comparison_baselines": fields.get("teknik pembanding", ""),
            "ensemble_compatibility": ens_compat,
            "exact_package": exact_pkg,
            "example_hypothesis": fields.get("contoh hypothesis", ""),
            "acceptance_criteria": fields.get("acceptance criteria", ""),
            "rejection_criteria": fields.get("rejection criteria", ""),
            "source": fields.get("sumber", ""),
            "aliases": [],
            "secondary_roles": [],
        }
        cards.append(card_dict)

    # 2. Parse overview tables in modeling documents if present
    if doc_key == "modeling":
        overview_tables = re.findall(
            r"##\s+\d+\.\s+(Classification|Regression).*?\n(\|.*?)(?=\n##|\n#|\Z)",
            content,
            re.DOTALL,
        )
        for sec_type, tbl in overview_tables:
            rows = [r.strip() for r in tbl.strip().split("\n") if r.strip().startswith("|")]
            if len(rows) < 3:
                continue
            headers = [c.strip().lower() for c in rows[0].split("|")[1:-1]]
            task_type = (
                "binary_classification"
                if sec_type.lower() == "classification"
                else "regression"
            )
            for row in rows[2:]:
                cols = [c.strip() for c in row.split("|")[1:-1]]
                if len(cols) >= 6:
                    m = dict(zip(headers, cols, strict=False))
                    model_name = m.get("technique", "")
                    if not model_name:
                        continue
                    m_lower = model_name.lower()
                    is_boost_or_forest = "boost" in m_lower or "forest" in m_lower
                    c_role = "model_engineer" if is_boost_or_forest else "model_scout"
                    supported_tasks = (
                        [task_type, "multiclass_classification"]
                        if task_type == "binary_classification"
                        else [task_type]
                    )
                    fold_boundary = (
                        m.get("risk", "") or "Fit exclusively on training fold."
                    )
                    rat = clean_text(m.get("fit", ""))[:280] or f"Model family {model_name}"
                    cards.append(
                        {
                            "title": model_name,
                            "category": "model",
                            "canonical_role": c_role,
                            "task_types": supported_tasks,
                            "prerequisites": ["tabular"],
                            "leakage_risk": map_leakage(m.get("risk", "")),
                            "fold_boundary_rule": fold_boundary,
                            "rationale": rat,
                            "definition": m.get("fit", ""),
                            "when_to_use": clean_text(m.get("fit", "")),
                            "when_not_to_use": "",
                            "assumptions": "",
                            "preprocessing": "",
                            "dataset_scale": "",
                            "metric_compatibility": "",
                            "key_hyperparameters": m.get("tune/compute", ""),
                            "initial_parameter_ranges": m.get("tune/compute", ""),
                            "compute_profile": m.get("tune/compute", ""),
                            "failure_modes": m.get("risk", ""),
                            "comparison_baselines": m.get("compare/ensemble", ""),
                            "ensemble_compatibility": m.get("compare/ensemble", ""),
                            "exact_package": m.get("implementation", ""),
                            "example_hypothesis": m.get("experiment gate", ""),
                            "acceptance_criteria": m.get("experiment gate", ""),
                            "rejection_criteria": "",
                            "source": m.get("evidence", ""),
                            "aliases": [],
                            "secondary_roles": [],
                        }
                    )

    return cards


def merge_card_cluster(cluster: list[dict[str, Any]]) -> dict[str, Any]:
    norm_title = re.sub(r"[\s_-]+", " ", cluster[0]["title"].strip().lower())
    pref_cat = OWNERSHIP_PRIORITY.get(norm_title)

    def sort_key(c: dict[str, Any]) -> int:
        score = 0
        if pref_cat and pref_cat in c["category"]:
            score += 10
        score += sum(1 for v in c.values() if v)
        return score

    sorted_cluster = sorted(cluster, key=sort_key, reverse=True)
    primary = dict(sorted_cluster[0])

    aliases = set(primary.get("aliases", []))
    secondary_roles = set(primary.get("secondary_roles", []))

    for other in sorted_cluster[1:]:
        aliases.add(other["card_id"])
        aliases.add(other["title"])
        if other["canonical_role"] != primary["canonical_role"]:
            secondary_roles.add(other["canonical_role"])

        for k, v in other.items():
            if v and not primary.get(k):
                primary[k] = v
            elif isinstance(v, list) and isinstance(primary.get(k), list):
                primary[k] = list(dict.fromkeys(primary[k] + v))

    primary["aliases"] = sorted(list(aliases))
    primary["secondary_roles"] = sorted(list(secondary_roles))

    h_data = (
        f"{primary['title']}|{primary['category']}|{primary['canonical_role']}|"
        f"{primary['definition']}|{primary.get('exact_package', '')}"
    )
    primary["content_hash"] = hashlib.sha256(h_data.encode("utf-8")).hexdigest()[:16]

    return primary


def parse_all_knowledge(
    knowledge_dir: Path | str | None = None,
    downloads_dir: Path | str | None = None,
) -> list[dict[str, Any]]:
    # Repository-local is the authoritative source
    if knowledge_dir:
        target_dir = Path(knowledge_dir)
    else:
        repo_root = Path(__file__).resolve().parent.parent.parent
        target_dir = repo_root / "knowledge" / "agents"

    if not target_dir.exists() and downloads_dir:
        target_dir = Path(downloads_dir)

    if not target_dir.exists() or not target_dir.is_dir():
        raise FileNotFoundError(f"Authoritative knowledge directory not found: {target_dir}")

    parsed_raw_cards: list[dict[str, Any]] = []
    seen_doc_keys: set[str] = set()

    for f in sorted(target_dir.iterdir()):
        if f.is_file() and (f.suffix in [".md", ".txt"]) and f.name != "CORPUS_AUDIT.md":
            doc_key = get_canonical_doc_key(f.name)
            if doc_key in seen_doc_keys:
                continue
            seen_doc_keys.add(doc_key)
            doc_cards = parse_knowledge_doc(f)
            parsed_raw_cards.extend(doc_cards)

    # Core anchor ID mappings to maintain 100% backward compatibility
    cb_long_alias = "catboost ordered boosting with native categorical handling"
    core_aliases = {
        "lightgbm histogram-based gradient boosting": "lightgbm_tabular_boosting",
        "lightgbm classifier": "lightgbm_tabular_boosting",
        "xgboost regularized tree boosting": "xgboost_regularized_boosting",
        "xgboost classifier": "xgboost_regularized_boosting",
        cb_long_alias: "catboost_categorical_boosting",
        "catboost classifier": "catboost_categorical_boosting",
        "random forest bagged decision trees": "random_forest_robust_nonlinear",
        "random forest classifier": "random_forest_robust_nonlinear",
        "regularized linear / logistic baseline": "logistic_regression_regularized",
        "logistic regression": "logistic_regression_regularized",
        "caruana stepwise greedy ensemble selection": "caruana_greedy_ensemble_selection",
        "greedy ensemble selection klasik": "caruana_greedy_ensemble_selection",
        "greedy ensemble selection (caruana) & sub-varian": "caruana_greedy_ensemble_selection",
        "out-of-fold regularized target encoding": "kfold_out_of_fold_target_encoding",
        "target encoding": "kfold_out_of_fold_target_encoding",
        "target encoding (mean encoding)": "kfold_out_of_fold_target_encoding",
        "leakage-safe target encoding": "kfold_out_of_fold_target_encoding",
        "purged walk-forward time-series split": "purged_walk_forward_temporal_split",
        "walk-forward / expanding window cross-validation": "purged_walk_forward_temporal_split",
    }

    seen_ids: dict[str, int] = {}
    assigned_cards: list[dict[str, Any]] = []

    for raw in parsed_raw_cards:
        t_clean = raw["title"].lower().strip()
        matched_anchor = core_aliases.get(t_clean)

        if matched_anchor and matched_anchor not in seen_ids:
            card_id = matched_anchor
        else:
            base_id = slugify(f"{raw['category']}_{raw['title']}")
            if base_id in seen_ids:
                seen_ids[base_id] += 1
                card_id = f"{base_id}_{seen_ids[base_id]}"
            else:
                seen_ids[base_id] = 1
                card_id = base_id

        seen_ids[card_id] = seen_ids.get(card_id, 1)
        raw["card_id"] = card_id
        assigned_cards.append(raw)

    # Semantic deduplication by normalized title
    by_title: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in assigned_cards:
        norm_t = re.sub(r"[\s_-]+", " ", c["title"].strip().lower())
        by_title[norm_t].append(c)

    canonical_cards: list[dict[str, Any]] = []
    for _norm_t, clist in by_title.items():
        if len(clist) == 1:
            c = dict(clist[0])
            h_data = (
                f"{c['title']}|{c['category']}|{c['canonical_role']}|"
                f"{c['definition']}|{c.get('exact_package', '')}"
            )
            c["content_hash"] = hashlib.sha256(h_data.encode("utf-8")).hexdigest()[:16]
            c["aliases"] = []
            c["secondary_roles"] = []
            canonical_cards.append(c)
        else:
            canonical_cards.append(merge_card_cluster(clist))

    # Sort deterministically by card_id
    canonical_cards.sort(key=lambda item: item["card_id"])
    return canonical_cards


def compile_corpus_to_json(
    output_path: Path | str | None = None,
    knowledge_dir: Path | str | None = None,
) -> Path:
    if output_path is None:
        output_path = Path(__file__).resolve().parent / "data" / "compiled_technique_corpus.json"
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    cards = parse_all_knowledge(knowledge_dir=knowledge_dir)
    json_bytes = json.dumps(cards, indent=2, ensure_ascii=False, sort_keys=True)
    dest.write_text(json_bytes, encoding="utf-8")
    return dest


if __name__ == "__main__":
    out = compile_corpus_to_json()
    print(f"Compiled corpus written to {out}")
