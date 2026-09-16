from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from swarmfold.features import (
    CyclicTimeEncoder,
    DateDecomposer,
    FeatureAblationReport,
    FeaturePipelineExecutor,
    FeatureRegistry,
    FeatureTransformerSpec,
    FrequencyEncoder,
    InteractionPolynomial,
    MissingIndicatorAdder,
    QuantileBinner,
    TargetEncoderFoldSafe,
    compute_feature_ablation,
    get_feature_registry,
)
from swarmfold.graph import run_experiment

FIXTURE = Path(__file__).parent / "fixtures" / "demo_classification.csv"


def test_feature_registry_lifecycle() -> None:
    registry = get_feature_registry()
    assert registry.is_registered("standard_scaler")
    assert registry.is_registered("target_encoder")
    assert registry.is_registered("date_decomposer")
    assert registry.is_registered("interaction_polynomial")

    # List specs
    scaling_specs = registry.list_specs(family="scaling")
    assert any(s.name == "standard_scaler" for s in scaling_specs)
    assert any(s.name == "robust_scaler" for s in scaling_specs)

    # Create transformer
    scaler = registry.create("standard_scaler")
    assert scaler is not None

    # Custom spec registration
    custom_spec = FeatureTransformerSpec(
        name="custom_dummy",
        family="transformation",
        description="A test transformer spec",
    )
    custom_registry = FeatureRegistry()
    custom_registry.register(custom_spec, lambda **kw: "dummy_instance")
    assert custom_registry.is_registered("custom_dummy")
    assert custom_registry.create("custom_dummy") == "dummy_instance"

    with pytest.raises(KeyError):
        custom_registry.get_spec("non_existent")


def test_target_encoder_fold_safe_isolation() -> None:
    # 20 samples: "A" associated with 0, "B" associated with 1
    categories = ["A"] * 10 + ["B"] * 10
    targets = [0] * 10 + [1] * 10
    df_train = pd.DataFrame({"cat": categories})
    y_train = pd.Series(targets)

    encoder = TargetEncoderFoldSafe(columns=["cat"], cv_folds=2, smoothing=5.0)

    # fit_transform on train fold
    transformed_train = encoder.fit_transform(df_train, y_train)
    assert transformed_train.shape == (20, 1)
    assert not transformed_train.isna().any().any()

    # The mean of A should be lower than B in transformed values
    mean_a = transformed_train.iloc[:10, 0].mean()
    mean_b = transformed_train.iloc[10:, 0].mean()
    assert mean_a < mean_b

    # Transform unseen test data
    df_test = pd.DataFrame({"cat": ["A", "B", "UNKNOWN_NEW_CAT"]})
    transformed_test = encoder.transform(df_test)
    assert transformed_test.shape == (3, 1)
    assert not transformed_test.isna().any().any()
    # UNKNOWN should be filled with global mean
    global_mean = encoder.global_means_["cat"]
    assert np.isclose(transformed_test.iloc[2, 0], global_mean)


def test_target_encoder_regression_and_multiclass() -> None:
    df = pd.DataFrame({"cat": ["x"] * 10 + ["y"] * 10 + ["z"] * 10})
    y_reg = pd.Series(
        [10.0 + i for i in range(10)]
        + [50.0 + i for i in range(10)]
        + [100.0 + i for i in range(10)]
    )

    encoder_reg = TargetEncoderFoldSafe(
        columns=["cat"], cv_folds=2, task_type="regression", smoothing=1.0
    )
    res_reg = encoder_reg.fit_transform(df, y_reg)
    assert res_reg.shape == (30, 1)
    mean_x = res_reg.iloc[:10, 0].mean()
    mean_y = res_reg.iloc[10:20, 0].mean()
    mean_z = res_reg.iloc[20:, 0].mean()
    assert mean_x < mean_y < mean_z

    y_multi = pd.Series(["low"] * 10 + ["medium"] * 10 + ["high"] * 10)
    encoder_multi = TargetEncoderFoldSafe(
        columns=["cat"], cv_folds=2, task_type="multiclass_classification"
    )
    res_multi = encoder_multi.fit_transform(df, y_multi)
    assert res_multi.shape == (30, 1)
    assert not res_multi.isna().any().any()


def test_frequency_encoder() -> None:
    df = pd.DataFrame({"cat": ["apple", "apple", "apple", "banana", "banana", "cherry"]})
    encoder = FrequencyEncoder(columns=["cat"], normalize=True)
    encoder.fit(df)
    transformed = encoder.transform(df)

    assert transformed.shape == (6, 1)
    assert np.isclose(transformed.iloc[0, 0], 3.0 / 6.0)
    assert np.isclose(transformed.iloc[3, 0], 2.0 / 6.0)
    assert np.isclose(transformed.iloc[5, 0], 1.0 / 6.0)

    # Test unseen category
    test_df = pd.DataFrame({"cat": ["orange"]})
    assert np.isclose(encoder.transform(test_df).iloc[0, 0], 0.0)


def test_date_decomposer_and_cyclic_encoding() -> None:
    df = pd.DataFrame(
        {
            "event_time": [
                "2023-01-15",
                "2023-06-20",
                "2023-12-31",
            ]
        }
    )
    decomposer = DateDecomposer(date_columns=["event_time"])
    decomposer.fit(df)
    date_parts = decomposer.transform(df)

    assert "event_time_year" in date_parts.columns
    assert "event_time_month" in date_parts.columns
    assert "event_time_day" in date_parts.columns
    assert "event_time_dayofweek" in date_parts.columns
    assert "event_time_is_weekend" in date_parts.columns
    assert "event_time_quarter" in date_parts.columns
    assert date_parts["event_time_month"].tolist() == [1.0, 6.0, 12.0]

    # Cyclic time encoder on the date parts
    cyclic = CyclicTimeEncoder(periods={"event_time_month": 12.0})
    cyclic.fit(date_parts)
    cyclic_feats = cyclic.transform(date_parts)

    assert "event_time_month_sin" in cyclic_feats.columns
    assert "event_time_month_cos" in cyclic_feats.columns
    assert (cyclic_feats >= -1.0).all().all() and (cyclic_feats <= 1.0).all().all()


def test_missing_indicator_and_quantile_binning() -> None:
    df = pd.DataFrame(
        {
            "num1": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0, np.nan, 8.0, 9.0, 10.0],
            "num2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
        }
    )
    miss = MissingIndicatorAdder()
    miss.fit(df)
    indicators = miss.transform(df)
    assert "num1_isna" in indicators.columns
    assert indicators["num1_isna"].sum() == 2.0

    binner = QuantileBinner(n_bins=3)
    binner.fit(df)
    binned = binner.transform(df)
    assert "num1_bin" in binned.columns
    assert "num2_bin" in binned.columns
    assert set(binned["num2_bin"].unique()).issubset({0.0, 1.0, 2.0})


def test_interaction_polynomial_capping() -> None:
    # 6 numeric columns -> 15 pairwise interaction combinations
    data = {f"c{i}": np.random.default_rng(42).normal(size=30) for i in range(6)}
    df = pd.DataFrame(data)

    # Cap at 5 interactions
    poly = InteractionPolynomial(max_features=5, interaction_only=True)
    poly.fit(df)
    interactions = poly.transform(df)
    assert interactions.shape == (30, 5)


def test_feature_pipeline_executor_end_to_end() -> None:
    df = pd.DataFrame(
        {
            "feat_num": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
            "feat_cat": ["A", "B", "A", "B", "A", "B", "A", "B"],
            "feat_date": [
                "2023-01-01",
                "2023-01-02",
                "2023-01-03",
                "2023-01-04",
                "2023-01-05",
                "2023-01-06",
                "2023-01-07",
                "2023-01-08",
            ],
        }
    )
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])

    strategies = [
        "raw_numeric",
        "scaled_numeric",
        "target_encoded",
        "power_transformed",
        "date_expanded",
        "interaction_pairs",
        "frequency_encoded",
        "selected_50",
    ]
    for strategy in strategies:
        executor = FeaturePipelineExecutor(
            strategy=strategy,
            task_type="binary_classification",
        )
        transformed = executor.fit_transform(df, y)
        assert isinstance(transformed, np.ndarray)
        assert transformed.shape[0] == len(df)
        assert transformed.shape[1] > 0
        assert executor.input_feature_count == 3
        assert executor.output_feature_count == transformed.shape[1]


def test_feature_pipeline_serialization_and_replay(tmp_path: Path) -> None:
    df_train = pd.DataFrame(
        {
            "num": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "cat": ["x", "y", "x", "y", "x", "y"],
        }
    )
    y_train = np.array([0, 1, 0, 1, 0, 1])
    df_test = pd.DataFrame(
        {
            "num": [1.5, 4.5],
            "cat": ["x", "y"],
        }
    )

    executor = FeaturePipelineExecutor(
        strategy="target_encoded",
        task_type="binary_classification",
    )
    executor.fit(df_train, y_train)
    original_test_preds = executor.transform(df_test)

    # Save and reload
    save_path = tmp_path / "pipeline.joblib"
    joblib.dump(executor, save_path)
    loaded = joblib.load(save_path)

    replayed_test_preds = loaded.transform(df_test)
    assert np.allclose(original_test_preds, replayed_test_preds)


def test_feature_ablation_report() -> None:
    report = compute_feature_ablation(
        strategy_name="target_encoded",
        baseline_strategy="raw_numeric",
        input_feature_count=10,
        output_feature_count=15,
        memory_bytes_before=1000,
        memory_bytes_after=1500,
        train_time_seconds=0.25,
        baseline_metric_score=0.82,
        candidate_metric_score=0.86,
        higher_is_better=True,
    )
    assert isinstance(report, FeatureAblationReport)
    assert report.feature_count_delta == 5
    assert np.isclose(report.metric_delta or 0.0, 0.04)
    assert report.is_worthwhile is True
    assert "improved" in report.rationale


def test_graph_execution_with_new_feature_strategy(tmp_path: Path) -> None:
    # Run graph experiment using target_encoded strategy
    result = run_experiment(
        FIXTURE,
        seed=42,
        output_dir=tmp_path / "target_enc_run",
        max_experiments=1,
    )
    assert result["status"] == "ACCEPTED"
    assert result["verification"]["accepted"] is True

    # In local execution, run_path is in runs_dir
    runs_dir = tmp_path / "target_enc_run" / "runs"
    subdirs = list(runs_dir.iterdir())
    assert len(subdirs) == 1
    run_path = subdirs[0]

    # Verify feature_pipeline.joblib exists in the run directory
    assert (run_path / "feature_pipeline.joblib").is_file()
    assert (run_path / "model.joblib").is_file()

    # Verify metrics record feature counts
    metrics = result["run_result"]["metrics"]
    assert "feature_count_in" in metrics
    assert "feature_count_out" in metrics
    assert metrics["feature_count_in"] > 0
    assert metrics["feature_count_out"] > 0
