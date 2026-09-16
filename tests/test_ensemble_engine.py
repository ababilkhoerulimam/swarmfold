import numpy as np

from swarmfold.ensemble import CaruanaEnsembleSelector, StackingEnsemble


def test_caruana_ensemble_selection_improves_or_matches_baseline() -> None:
    rng = np.random.default_rng(42)
    n_samples = 200

    # True target binary
    targets = rng.choice([0, 1], size=n_samples, p=[0.5, 0.5])

    # Model 1: 75% accurate
    p1 = targets.astype(float) + rng.normal(0, 0.5, n_samples)
    p1 = np.clip((p1 - p1.min()) / (p1.max() - p1.min()), 0.0, 1.0)

    # Model 2: 70% accurate with different errors
    p2 = targets.astype(float) + rng.normal(0, 0.6, n_samples)
    p2 = np.clip((p2 - p2.min()) / (p2.max() - p2.min()), 0.0, 1.0)

    # Model 3: 65% accurate
    p3 = targets.astype(float) + rng.normal(0, 0.8, n_samples)
    p3 = np.clip((p3 - p3.min()) / (p3.max() - p3.min()), 0.0, 1.0)

    selector = CaruanaEnsembleSelector(
        iterations=30,
        metric_name="roc_auc",
        higher_is_better=True,
    )
    selector.fit([p1, p2, p3], targets)

    assert selector.best_score_ is not None
    assert len(selector.selected_indices) > 0
    assert len(selector.weights_) > 0
    assert abs(sum(selector.weights_.values()) - 1.0) < 1e-6

    # Ensemble ROC-AUC should be >= best single model ROC-AUC
    from sklearn.metrics import roc_auc_score

    score1 = roc_auc_score(targets, p1)
    score2 = roc_auc_score(targets, p2)
    score3 = roc_auc_score(targets, p3)
    best_single = max(score1, score2, score3)

    assert selector.best_score_ >= best_single - 1e-9

    # Test predict
    test_preds = selector.predict([p1, p2, p3])
    assert len(test_preds) == n_samples
    assert np.all(test_preds >= 0.0) and np.all(test_preds <= 1.0)


def test_stacking_ensemble() -> None:
    rng = np.random.default_rng(42)
    n_samples = 150
    targets = rng.choice([0, 1], size=n_samples)

    p1 = np.clip(targets.astype(float) + rng.normal(0, 0.4, n_samples), 0, 1)
    p2 = np.clip(targets.astype(float) + rng.normal(0, 0.4, n_samples), 0, 1)

    stacker = StackingEnsemble(task_type="binary_classification", seed=42)
    stacker.fit([p1, p2], targets)

    preds = stacker.predict([p1, p2])
    assert len(preds) == n_samples

    probs = stacker.predict_proba([p1, p2])
    assert probs.shape == (n_samples, 2)
