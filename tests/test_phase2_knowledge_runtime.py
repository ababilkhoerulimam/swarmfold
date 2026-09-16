from collections import Counter
from pathlib import Path

from swarmfold.corpus_audit import run_corpus_audit
from swarmfold.knowledge_parser import compile_corpus_to_json
from swarmfold.model_registry import executable_model_families
from swarmfold.retrieval import TechniqueRetriever
from swarmfold.technique_cards import TECHNIQUE_CORPUS
from swarmfold.technique_plan import build_technique_plan, validate_technique_plan


def test_recompilation_is_deterministic_and_byte_stable(tmp_path: Path) -> None:
    # Compile twice to two separate files from repository-local documents
    out1 = tmp_path / "corpus1.json"
    out2 = tmp_path / "corpus2.json"

    compile_corpus_to_json(output_path=out1)
    compile_corpus_to_json(output_path=out2)

    bytes1 = out1.read_bytes()
    bytes2 = out2.read_bytes()

    assert bytes1 == bytes2, "Recompiling source documents must produce byte-stable canonical cards"
    assert len(bytes1) > 500_000


def test_zero_duplicate_titles_in_canonical_corpus() -> None:
    assert len(TECHNIQUE_CORPUS) >= 400
    title_counts = Counter(c.title.strip().lower() for c in TECHNIQUE_CORPUS)
    duplicate_titles = {t: cnt for t, cnt in title_counts.items() if cnt > 1}
    assert len(duplicate_titles) == 0, f"No duplicate titles allowed, found: {duplicate_titles}"


def test_merged_clusters_preserve_aliases_and_secondary_roles() -> None:
    # 1. Rotation forest should have ensemble owner and model secondary role
    rot_forest = next((c for c in TECHNIQUE_CORPUS if "rotation_forest" in c.card_id), None)
    assert rot_forest is not None
    assert rot_forest.canonical_role == "oof_ensemble_scientist"
    assert "model_engineer" in rot_forest.secondary_roles
    assert len(rot_forest.aliases) > 0
    assert rot_forest.content_hash

    # 2. Purged walk forward should have temporal validation owner and validation secondary role
    walk_fwd = next((c for c in TECHNIQUE_CORPUS if "purged_walk_forward" in c.card_id), None)
    assert walk_fwd is not None
    assert walk_fwd.canonical_role == "temporal_validation_leakage_auditor"
    assert "validation_agent" in walk_fwd.secondary_roles


def test_corpus_audit_generation(tmp_path: Path) -> None:
    json_path = tmp_path / "audit.json"
    md_path = tmp_path / "audit.md"

    audit_res = run_corpus_audit(
        output_json_path=json_path,
        output_md_path=md_path,
    )

    assert audit_res["total_canonical_cards"] == len(TECHNIQUE_CORPUS)
    assert audit_res["duplicate_title_count"] == 0
    assert audit_res["merged_clusters_count"] > 5
    assert json_path.exists()
    assert md_path.exists()

    md_content = md_path.read_text(encoding="utf-8")
    assert "Agent Knowledge Corpus Audit" in md_content
    assert "Total Canonical Cards" in md_content
    assert "Registered Executable Families" in md_content


def test_role_scoped_hybrid_retrieval() -> None:
    retriever = TechniqueRetriever()

    # Strict role scoping
    fe_cards = retriever.retrieve(role="feature_engineer", top_k=5, role_scoped_only=True)
    assert len(fe_cards) > 0
    for card in fe_cards:
        assert (
            card.canonical_role == "feature_engineer"
            or "feature_engineer" in card.secondary_roles
        )

    # Multi-role card matching via secondary roles
    ensemble_scout_cards = retriever.retrieve(role="oof_ensemble_scientist", top_k=5)
    card_ids = {c.card_id for c in ensemble_scout_cards}
    assert any("rotation_forest" in cid or "caruana" in cid for cid in card_ids)


def test_dataset_traits_deterministically_alter_plan() -> None:
    retriever = TechniqueRetriever()

    # Trait 1: High-cardinality categorical
    cat_cards = retriever.retrieve(
        role="model_engineer",
        dataset_traits=["categorical_features", "high_cardinality"],
        top_k=2,
    )
    plan_cat = build_technique_plan(
        cat_cards, role="model_engineer", task_type="binary_classification"
    )
    assert any("catboost" in cid for cid in plan_cat.source_card_ids)

    # Trait 2: Temporal drift / time-series
    time_cards = retriever.retrieve(
        role="temporal_validation_leakage_auditor",
        dataset_traits=["temporal_ordering", "time_series"],
        top_k=2,
    )
    plan_time = build_technique_plan(
        time_cards,
        role="temporal_validation_leakage_auditor",
        task_type="binary_classification",
    )
    assert any("walk_forward" in cid or "purged" in cid for cid in plan_time.source_card_ids)
    assert plan_time.validation_strategy != ""


def test_technique_plan_execution_gate() -> None:
    # 1. Executable model (LightGBM)
    lgb_card = next(c for c in TECHNIQUE_CORPUS if c.card_id == "lightgbm_tabular_boosting")
    plan_lgb = build_technique_plan([lgb_card], role="model_engineer", model_family="lightgbm")
    assert plan_lgb.is_executable is True
    assert plan_lgb.execution_block_reason is None

    # 2. Declared but uninstalled/untested model family (TabNet)
    tabnet_card = next((c for c in TECHNIQUE_CORPUS if "tabnet" in c.card_id), None)
    assert tabnet_card is not None
    plan_tabnet = build_technique_plan([tabnet_card], role="model_engineer", model_family="tabnet")
    assert plan_tabnet.is_executable is False
    assert "DECLARED" in plan_tabnet.execution_block_reason
    assert "EXECUTABLE" in plan_tabnet.execution_block_reason


def test_removing_adapter_makes_technique_ineligible_for_execution() -> None:
    lgb_card = next(c for c in TECHNIQUE_CORPUS if c.card_id == "lightgbm_tabular_boosting")
    plan = build_technique_plan([lgb_card], role="model_engineer", model_family="lightgbm")
    assert plan.is_executable is True

    # Simulate adapter uninstallation/removal from executable registry
    mock_executables = set(executable_model_families()) - {"lightgbm"}
    blocked_plan = validate_technique_plan(plan, executable_families=mock_executables)
    assert blocked_plan.is_executable is False
    assert "lightgbm" in blocked_plan.execution_block_reason
