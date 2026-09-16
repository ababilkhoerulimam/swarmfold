import csv
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from swarmfold.ensemble import (
    PredictionRegistry,
    StackingEnsemble,
    UnifiedEnsembleEngine,
    reconstruct_ensemble_submission,
)


def test_stacking_ensemble_regression_and_classification() -> None:
    rng = np.random.default_rng(42)
    n = 100
    # Regression
    y_reg = rng.normal(10, 2, n)
    p1 = y_reg + rng.normal(0, 0.5, n)
    p2 = y_reg + rng.normal(0, 0.5, n)
    stacker_reg = StackingEnsemble(task_type="regression")
    stacker_reg.fit([p1, p2], y_reg)
    preds = stacker_reg.predict([p1, p2])
    assert len(preds) == n
    params = stacker_reg.get_meta_params()
    assert "coef" in params

    # Classification
    y_clf = rng.choice([0, 1], size=n)
    p1_clf = np.clip(y_clf + rng.normal(0, 0.4, n), 0, 1)
    p2_clf = np.clip(y_clf + rng.normal(0, 0.4, n), 0, 1)
    stacker_clf = StackingEnsemble(task_type="binary_classification")
    stacker_clf.fit([p1_clf, p2_clf], y_clf)
    probs = stacker_clf.predict_proba([p1_clf, p2_clf])
    assert probs.shape == (n, 2)



def test_prediction_registry_alignment_validation() -> None:
    registry = PredictionRegistry()
    registry.add_member(
        member_id="m1",
        row_ids=["0", "1", "2"],
        predictions=[0.1, 0.2, 0.3],
        targets=[0, 1, 0],
        metric_score=0.8,
    )
    # Different row count
    registry.add_member(
        member_id="m2",
        row_ids=["0", "1"],
        predictions=[0.1, 0.2],
        targets=[0, 1],
        metric_score=0.7,
    )
    with pytest.raises(ValueError, match="Row count mismatch"):
        registry.validate_alignment()

    # Different targets
    registry2 = PredictionRegistry()
    registry2.add_member(
        member_id="m1",
        row_ids=["0", "1", "2"],
        predictions=[0.1, 0.2, 0.3],
        targets=[0, 1, 0],
    )
    registry2.add_member(
        member_id="m2",
        row_ids=["0", "1", "2"],
        predictions=[0.1, 0.2, 0.3],
        targets=[0, 0, 1],
    )
    with pytest.raises(ValueError, match="Target values mismatch"):
        registry2.validate_alignment()


def test_prediction_registry_diversity_filtering() -> None:
    registry = PredictionRegistry()
    rng = np.random.default_rng(42)
    n = 100
    targets = rng.choice([0, 1], size=n)
    p1 = targets * 0.7 + rng.normal(0, 0.1, n)
    p2 = p1.copy()  # Identical / collinear
    p3 = targets * 0.5 + rng.normal(0, 0.3, n)

    registry.add_member("m1", list(range(n)), p1, targets, metric_score=0.85)
    registry.add_member("m2", list(range(n)), p2, targets, metric_score=0.84)
    registry.add_member("m3", list(range(n)), p3, targets, metric_score=0.75)

    diverse = registry.filter_diverse_members(metric_name="accuracy", max_correlation=0.99)
    assert "m1" in diverse
    assert "m2" not in diverse  # Collinear duplicate filtered out
    assert "m3" in diverse


def test_unified_ensemble_engine_equal_average_and_rank_average() -> None:
    rng = np.random.default_rng(123)
    n = 200
    targets = rng.choice([0, 1], size=n)
    p1 = np.clip(targets * 0.6 + rng.normal(0, 0.4, n), 0, 1)
    p2 = np.clip(targets * 0.5 + rng.normal(0, 0.5, n), 0, 1)

    registry = PredictionRegistry()
    registry.add_member("m1", list(range(n)), p1, targets, metric_score=0.72)
    registry.add_member("m2", list(range(n)), p2, targets, metric_score=0.68)

    engine = UnifiedEnsembleEngine(
        metric_name="roc_auc",
        task_type="binary_classification",
        methods=["equal_average", "rank_average"],
    )
    decision, _ = engine.evaluate_registry(registry)

    assert decision.status in {"ADOPTED", "REJECTED"}
    assert decision.ensemble_score > 0.5
    assert len(decision.weights) == 2


def test_unified_ensemble_engine_constrained_blend_and_caruana() -> None:
    rng = np.random.default_rng(999)
    n = 150
    targets = rng.choice([0, 1], size=n)
    p1 = np.clip(targets * 0.7 + rng.normal(0, 0.3, n), 0, 1)
    p2 = np.clip(targets * 0.6 + rng.normal(0, 0.35, n), 0, 1)
    p3 = np.clip(targets * 0.5 + rng.normal(0, 0.4, n), 0, 1)

    registry = PredictionRegistry()
    registry.add_member("m1", list(range(n)), p1, targets, metric_score=0.75)
    registry.add_member("m2", list(range(n)), p2, targets, metric_score=0.73)
    registry.add_member("m3", list(range(n)), p3, targets, metric_score=0.70)

    engine = UnifiedEnsembleEngine(
        metric_name="roc_auc",
        task_type="binary_classification",
        methods=["constrained_weighted_blend", "caruana"],
    )
    decision, _ = engine.evaluate_registry(registry)

    assert decision.champion_lane == "m1"
    assert decision.champion_score == 0.75
    assert decision.ensemble_score >= 0.75 - 1e-6
    assert abs(sum(decision.weights.values()) - 1.0) < 1e-4


def test_unified_ensemble_engine_lower_is_better_rejection_and_adoption() -> None:
    targets = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    # m1 is perfect (RMSE = 0.0)
    p1 = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    # m2 is terrible (RMSE ~ 100.0)
    p2 = np.array([110.0, 120.0, 130.0, 140.0, 150.0])

    registry = PredictionRegistry()
    registry.add_member("m1", list(range(5)), p1, targets, metric_score=0.0)
    registry.add_member("m2", list(range(5)), p2, targets, metric_score=100.0)

    engine = UnifiedEnsembleEngine(
        metric_name="rmse",
        task_type="regression",
        methods=["equal_average"],
    )
    decision, _ = engine.evaluate_registry(registry)

    # Blending terrible model with perfect model makes RMSE worse (50 > 0)
    # So ensemble must be REJECTED!
    assert decision.status == "REJECTED"
    assert decision.adopted is False
    assert decision.improved_over_best_single is False
    assert decision.champion_lane == "m1"
    assert decision.champion_score == 0.0


def test_unified_ensemble_engine_majority_voting() -> None:
    targets = np.array(["cat", "dog", "dog", "cat", "cat"])
    p1 = np.array(["cat", "dog", "cat", "cat", "cat"])  # 4/5
    p2 = np.array(["dog", "dog", "dog", "cat", "cat"])  # 4/5
    p3 = np.array(["cat", "dog", "dog", "dog", "cat"])  # 4/5

    registry = PredictionRegistry()
    registry.add_member("m1", list(range(5)), p1, targets, metric_score=0.8)
    registry.add_member("m2", list(range(5)), p2, targets, metric_score=0.8)
    registry.add_member("m3", list(range(5)), p3, targets, metric_score=0.8)

    engine = UnifiedEnsembleEngine(
        metric_name="accuracy",
        task_type="multiclass_classification",
        methods=["majority_vote"],
    )
    decision, _ = engine.evaluate_registry(registry)

    # Majority vote:
    # 0: cat, dog, cat -> cat (match)
    # 1: dog, dog, dog -> dog (match)
    # 2: cat, dog, dog -> dog (match)
    # 3: cat, cat, dog -> cat (match)
    # 4: cat, cat, cat -> cat (match)
    # Accuracy = 5/5 = 1.0!
    assert decision.status == "ADOPTED"
    assert decision.adopted is True
    assert decision.ensemble_score == 1.0


def test_reconstruct_ensemble_submission_roundtrip(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    m1_dir = artifacts_dir / "m1"
    m2_dir = artifacts_dir / "m2"
    m1_dir.mkdir(parents=True)
    m2_dir.mkdir(parents=True)

    # Member 1 test predictions
    pq.write_table(
        pa.table({"row": ["0", "1", "2"], "prediction": ["0.2", "0.4", "0.8"]}),
        m1_dir / "test_predictions.parquet",
    )
    # Member 2 test predictions
    pq.write_table(
        pa.table({"row": ["0", "1", "2"], "prediction": ["0.4", "0.6", "0.6"]}),
        m2_dir / "test_predictions.parquet",
    )

    # Champion template submission
    template_sub = m1_dir / "submission.csv"
    with template_sub.open("w", newline="", encoding="utf-8") as h:
        writer = csv.DictWriter(h, fieldnames=["id", "target"])
        writer.writeheader()
        writer.writerow({"id": "0", "target": "0.2"})
        writer.writerow({"id": "1", "target": "0.4"})
        writer.writerow({"id": "2", "target": "0.8"})

    contracts_file = m1_dir / "contracts.json"
    contracts_file.write_text(
        json.dumps({"competition": {"prediction_column": "target"}}),
        encoding="utf-8",
    )

    ensemble_spec = {
        "status": "ADOPTED",
        "adopted": True,
        "method": "equal_average",
        "champion_lane": "m1",
        "weights": {"m1": 0.5, "m2": 0.5},
        "rule": "equal-weight average",
    }
    spec_path = tmp_path / "ensemble_spec.json"
    spec_path.write_text(json.dumps(ensemble_spec), encoding="utf-8")

    out_csv = tmp_path / "reconstructed_submission.csv"
    reconstruct_ensemble_submission(
        ensemble_spec_path=spec_path,
        member_artifacts_dir=artifacts_dir,
        output_path=out_csv,
        template_submission_path=template_sub,
        contracts_path=contracts_file,
    )

    assert out_csv.is_file()
    with out_csv.open(newline="", encoding="utf-8") as h:
        rows = list(csv.DictReader(h))

    # 0.5 * [0.2, 0.4, 0.8] + 0.5 * [0.4, 0.6, 0.6] = [0.3, 0.5, 0.7]
    assert [float(r["target"]) for r in rows] == pytest.approx([0.3, 0.5, 0.7])
