import time
from pathlib import Path

from swarmfold.planner import DeterministicPlanner
from swarmfold.retrieval import TechniqueRetriever
from swarmfold.technique_cards import TECHNIQUE_CORPUS


def test_technique_card_corpus_integrity() -> None:
    # Verify comprehensive 400+ card corpus
    assert len(TECHNIQUE_CORPUS) >= 400

    card_ids = {card.card_id for card in TECHNIQUE_CORPUS}
    assert "lightgbm_tabular_boosting" in card_ids
    assert "xgboost_regularized_boosting" in card_ids
    assert "catboost_categorical_boosting" in card_ids
    assert "caruana_greedy_ensemble_selection" in card_ids
    assert "kfold_out_of_fold_target_encoding" in card_ids
    assert "purged_walk_forward_temporal_split" in card_ids
    assert "random_forest_robust_nonlinear" in card_ids
    assert "logistic_regression_regularized" in card_ids

    # Verify 8+ categories are actively represented
    categories = {card.category for card in TECHNIQUE_CORPUS}
    expected_categories = {
        "model",
        "feature_engineering",
        "ensemble",
        "validation",
        "eda",
        "robustness",
        "problem_formulation",
        "target_decomposition",
    }
    assert expected_categories.issubset(categories)

    # Verify standard safety fields on every single card
    for card in TECHNIQUE_CORPUS:
        assert card.title
        assert card.rationale
        assert card.fold_boundary_rule
        assert card.leakage_risk in {"NONE", "LOW", "MEDIUM", "HIGH"}


def test_20_field_coverage_and_metadata() -> None:
    cards_with_package = [c for c in TECHNIQUE_CORPUS if c.exact_package]
    assert len(cards_with_package) > 100

    cards_with_hypothesis = [c for c in TECHNIQUE_CORPUS if c.example_hypothesis]
    assert len(cards_with_hypothesis) > 100

    cards_with_when_to_use = [c for c in TECHNIQUE_CORPUS if c.when_to_use]
    assert len(cards_with_when_to_use) > 200


def test_technique_retriever_role_and_trait_filtering() -> None:
    retriever = TechniqueRetriever()

    # Role matching for ensemble
    ensemble_cards = retriever.retrieve(role="oof_ensemble_scientist", top_k=2)
    assert len(ensemble_cards) > 0
    assert any(card.category == "ensemble" for card in ensemble_cards)

    # Trait matching for categorical features
    cat_cards = retriever.retrieve(
        role="model_engineer",
        dataset_traits=["categorical_features"],
        top_k=3,
    )
    assert any("catboost" in card.card_id for card in cat_cards)

    # Query matching for target encoding
    feat_cards = retriever.retrieve(
        query="smooth target posterior encoding for high-cardinality columns",
        top_k=2,
    )
    assert any("target_encoding" in card.card_id for card in feat_cards)


def test_multi_category_domain_queries() -> None:
    retriever = TechniqueRetriever()

    # 1. Temporal validation query
    temp_cards = retriever.retrieve(query="purged walk forward embargo time series", top_k=3)
    assert any(c.category == "validation" for c in temp_cards)
    assert any("walk_forward" in c.card_id or "purged" in c.card_id for c in temp_cards)

    # 2. Robustness query
    rob_cards = retriever.retrieve(query="multi seed retraining stochastic stability", top_k=3)
    assert any(c.category == "robustness" for c in rob_cards)

    # 3. EDA query
    eda_cards = retriever.retrieve(query="missing value pattern analysis mcar mar mnar", top_k=3)
    assert any(c.category == "eda" for c in eda_cards)

    # 4. Target decomposition query
    decomp_cards = retriever.retrieve(
        query="frequency severity decomposition zero inflated hurdle", top_k=3
    )
    assert any(c.category == "target_decomposition" for c in decomp_cards)


def test_retriever_prompt_formatting() -> None:
    retriever = TechniqueRetriever()
    cards = retriever.retrieve(role="model_engineer", top_k=2)
    prompt_str = retriever.format_for_prompt(cards)

    assert "Retrieved SwarmFold Technique Knowledge" in prompt_str
    assert "Leakage Boundary Rule" in prompt_str
    for card in cards:
        assert card.card_id in prompt_str


def test_retrieval_latency_benchmark() -> None:
    retriever = TechniqueRetriever()
    t0 = time.perf_counter()
    for _ in range(20):
        _ = retriever.retrieve(
            role="model_engineer",
            dataset_traits=["categorical_features", "high_cardinality"],
            query="gradient boosting native categorical tree interaction",
            top_k=5,
        )
    elapsed_avg = (time.perf_counter() - t0) / 20
    # Each retrieval over 470+ cards must take less than 35ms
    assert elapsed_avg < 0.035, f"Retrieval too slow: {elapsed_avg * 1000:.2f} ms"


def test_planner_integrates_technique_retrieval(tmp_path) -> None:
    from swarmfold.contracts import freeze_contracts

    fixture = Path(__file__).parent / "fixtures" / "demo_classification.csv"
    contracts_state = freeze_contracts(
        {
            "dataset_path": str(fixture.resolve()),
            "target_column": "target",
            "seed": 42,
            "test_size": 0.25,
            "output_dir": str(tmp_path / "contracts"),
        }
    )
    state = {
        **contracts_state,
        "dataset_path": str(fixture.resolve()),
        "target_column": "target",
        "output_dir": str(tmp_path / "output"),
    }

    planner = DeterministicPlanner()
    proposal = planner.propose(state)

    assert proposal.hypothesis is not None
    assert len(proposal.hypothesis) > 10
