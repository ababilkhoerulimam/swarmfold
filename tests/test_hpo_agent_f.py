import csv
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from swarmfold.ensemble import (
    PredictionRegistry,
    UnifiedEnsembleEngine,
    optimize_classification_threshold,
    reconstruct_ensemble_submission,
)
from swarmfold.optimization import HPOConfig, HPOController


def test_hpo_controller_is_seeded_persistent_and_prunes_same_fold(tmp_path: Path) -> None:
    config = HPOConfig(
        study_id="boosting-study",
        direction="maximize",
        seed=17,
        min_trials_before_pruning=2,
    )
    controller = HPOController(tmp_path / "hpo.sqlite", config)
    first = controller.suggest("hist_gradient_boosting", 0)
    assert controller.suggest("hist_gradient_boosting", 0) == first

    for trial_number, score in ((1, 0.82), (2, 0.79)):
        controller.suggest("hist_gradient_boosting", trial_number)
        decision = controller.report_fold(trial_number, 0, score)
        assert decision.should_prune is False
        controller.complete(trial_number, score)

    controller.suggest("hist_gradient_boosting", 3)
    pruned = controller.report_fold(3, 0, 0.50)
    assert pruned.should_prune is True
    assert pruned.reference_median == pytest.approx(0.805)

    resumed = HPOController(tmp_path / "hpo.sqlite", config)
    assert resumed.trials()[0]["parameters"] == first


def test_threshold_optimizer_rejects_probability_metrics() -> None:
    targets = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.45, 0.9])

    result = optimize_classification_threshold(
        targets,
        probabilities,
        metric_name="f1_macro",
    )
    assert 0.0 < result.threshold < 1.0
    assert result.optimized_score >= result.baseline_score

    with pytest.raises(ValueError, match="not supported"):
        optimize_classification_threshold(targets, probabilities, metric_name="roc_auc")


def test_agent_f_stacking_uses_cross_fit_and_reconstructs_submission(tmp_path: Path) -> None:
    rng = np.random.default_rng(43)
    targets = rng.integers(0, 2, size=80)
    first = np.clip(targets * 0.65 + rng.normal(0.2, 0.30, size=80), 0.0, 1.0)
    second = np.clip(targets * 0.75 + rng.normal(0.1, 0.22, size=80), 0.0, 1.0)
    first_test = np.array([0.12, 0.61, 0.86])
    second_test = np.array([0.18, 0.73, 0.80])

    registry = PredictionRegistry()
    registry.add_member(
        "ai-a--one",
        list(range(len(targets))),
        first,
        targets,
        metric_score=0.50,
        test_predictions=first_test,
    )
    registry.add_member(
        "ai-b--two",
        list(range(len(targets))),
        second,
        targets,
        metric_score=0.51,
        test_predictions=second_test,
    )

    engine = UnifiedEnsembleEngine(
        metric_name="roc_auc",
        task_type="binary_classification",
        methods=["stacking"],
        artifact_directory=tmp_path / "ensemble",
    )
    decision, test_predictions = engine.evaluate_registry(registry)

    assert decision.method == "stacking"
    assert decision.adopted is True
    assert test_predictions is not None
    assert decision.reconstruction_spec["extra"]["reconstruction_passed"] is True
    assert (tmp_path / "ensemble" / "meta-model.joblib").is_file()
    assert (tmp_path / "ensemble" / "meta-oof-predictions.parquet").is_file()

    member_root = tmp_path / "ensemble-members"
    for member_id, predictions in {
        "ai-a--one": first_test,
        "ai-b--two": second_test,
    }.items():
        member_dir = member_root / member_id
        member_dir.mkdir(parents=True)
        pq.write_table(
            pa.table(
                {"row": ["0", "1", "2"], "prediction": predictions.tolist()}
            ),
            member_dir / "test_predictions.parquet",
        )

    template = tmp_path / "submission.csv"
    with template.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "prediction"])
        writer.writeheader()
        writer.writerows({"id": index, "prediction": 0.0} for index in range(3))
    contracts = tmp_path / "contracts.json"
    contracts.write_text(
        json.dumps({"competition": {"prediction_column": "prediction"}}),
        encoding="utf-8",
    )
    spec = tmp_path / "ensemble_spec.json"
    spec.write_text(json.dumps(decision.model_dump()), encoding="utf-8")

    reconstructed = tmp_path / "reconstructed.csv"
    reconstruct_ensemble_submission(
        spec,
        tmp_path,
        reconstructed,
        template_submission_path=template,
        contracts_path=contracts,
    )
    with reconstructed.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [float(row["prediction"]) for row in rows] == pytest.approx(test_predictions)
