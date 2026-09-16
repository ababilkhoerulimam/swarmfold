from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from swarmfold.contracts import freeze_contracts
from swarmfold.dataset_traits import DatasetTraits, extract_dataset_traits
from swarmfold.graph import run_experiment
from swarmfold.schemas import ContractBundle
from swarmfold.validation_splitters import (
    ExpandingTimeSeriesSplitter,
    GroupKFoldSplitter,
    PurgedTimeSeriesSplitter,
    RepeatedStratifiedKFoldSplitter,
    StratifiedGroupKFoldSplitter,
    recommend_validation_strategy,
)


def test_dataset_traits_detection() -> None:
    # 1. Temporal data detection
    df_temporal = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=100, freq="D"),
            "feat1": np.random.randn(100),
            "target": np.random.randint(0, 2, 100),
        }
    )
    traits_temporal = extract_dataset_traits(df_temporal, "target")
    assert traits_temporal.is_temporal is True
    assert traits_temporal.primary_time_column == "timestamp"
    assert traits_temporal.time_columns == ["timestamp"]

    # 2. Grouped entity data detection
    df_grouped = pd.DataFrame(
        {
            "customer_id": [f"cust_{i % 10}" for i in range(100)],
            "feat1": np.random.randn(100),
            "target": np.random.randint(0, 2, 100),
        }
    )
    traits_grouped = extract_dataset_traits(df_grouped, "target")
    assert traits_grouped.has_repeated_entities is True
    assert traits_grouped.primary_group_column == "customer_id"
    assert traits_grouped.group_columns == ["customer_id"]

    # 3. Class imbalance detection
    df_imbalanced = pd.DataFrame(
        {
            "feat1": np.random.randn(100),
            "target": [1] * 5 + [0] * 95,
        }
    )
    traits_imbalanced = extract_dataset_traits(df_imbalanced, "target")
    assert traits_imbalanced.class_imbalance is True
    assert traits_imbalanced.rare_classes == ["1"]


def test_validation_recommendations() -> None:
    # Temporal recommendation
    traits_temp = DatasetTraits(
        row_count=200,
        feature_count=4,
        task_type="binary_classification",
        is_temporal=True,
        primary_time_column="timestamp",
        time_columns=["timestamp"],
    )
    rec_temp = recommend_validation_strategy(traits_temp)
    assert rec_temp.recommended_strategy == "purged_time_series"
    assert rec_temp.time_column == "timestamp"
    assert rec_temp.confidence > 0.85

    # Grouped recommendation
    traits_group = DatasetTraits(
        row_count=300,
        feature_count=4,
        task_type="binary_classification",
        has_repeated_entities=True,
        primary_group_column="store_id",
        group_columns=["store_id"],
    )
    rec_group = recommend_validation_strategy(traits_group)
    assert rec_group.recommended_strategy == "stratified_group_kfold"
    assert rec_group.group_column == "store_id"

    # Small imbalanced dataset -> repeated stratified kfold
    traits_small = DatasetTraits(
        row_count=150,
        feature_count=5,
        task_type="binary_classification",
        class_imbalance=True,
    )
    rec_small = recommend_validation_strategy(traits_small)
    assert rec_small.recommended_strategy == "repeated_stratified_kfold"
    assert rec_small.n_repeats == 2


def test_group_kfold_splitter_zero_leakage() -> None:
    groups = pd.Series([f"entity_{i % 8}" for i in range(120)])
    df = pd.DataFrame(
        {"feat": np.arange(120), "group": groups, "target": np.random.randint(0, 2, 120)}
    )
    splitter = GroupKFoldSplitter(n_splits=4)
    splits = list(splitter.split(df, groups=groups))
    assert len(splits) == 4

    for train_indices, val_indices in splits:
        train_groups = set(df.iloc[train_indices]["group"])
        val_groups = set(df.iloc[val_indices]["group"])
        # Crucial check: zero entity overlap
        assert len(train_groups.intersection(val_groups)) == 0


def test_stratified_group_kfold_splitter_zero_leakage() -> None:
    groups = pd.Series([f"entity_{i % 12}" for i in range(120)])
    targets = pd.Series([0, 1] * 60)
    df = pd.DataFrame({"feat": np.arange(120), "group": groups, "target": targets})
    splitter = StratifiedGroupKFoldSplitter(n_splits=3, random_state=42)
    splits = list(splitter.split(df, y=targets, groups=groups))
    assert len(splits) == 3

    for train_indices, val_indices in splits:
        train_groups = set(df.iloc[train_indices]["group"])
        val_groups = set(df.iloc[val_indices]["group"])
        assert len(train_groups.intersection(val_groups)) == 0
        train_targets = df.iloc[train_indices]["target"]
        val_targets = df.iloc[val_indices]["target"]
        assert len(np.unique(train_targets)) == 2
        assert len(np.unique(val_targets)) == 2


def test_expanding_time_series_splitter_causality() -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=60, freq="D"),
            "feat": np.arange(60),
            "target": np.random.randint(0, 2, 60),
        }
    )
    splitter = ExpandingTimeSeriesSplitter(n_splits=3, time_column="timestamp")
    splits = list(splitter.split(df))
    assert len(splits) == 3

    for train_indices, val_indices in splits:
        max_train_time = df.iloc[train_indices]["timestamp"].max()
        min_val_time = df.iloc[val_indices]["timestamp"].min()
        assert max_train_time < min_val_time
        assert len(train_indices) > 0


def test_purged_time_series_splitter_buffer() -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=50, freq="D"),
            "feat": np.arange(50),
            "target": np.random.randint(0, 2, 50),
        }
    )
    purge = 2
    embargo = 3
    splitter = PurgedTimeSeriesSplitter(
        n_splits=3,
        time_column="timestamp",
        purge_window=purge,
        embargo_window=embargo,
    )
    splits = list(splitter.split(df))
    assert len(splits) == 3

    for train_indices, val_indices in splits:
        val_min = val_indices.min()
        val_max = val_indices.max()
        for tr in train_indices:
            assert not (val_min - purge <= tr <= val_max + embargo)


def test_repeated_stratified_kfold_splitter() -> None:
    targets = pd.Series([0, 1] * 30)
    df = pd.DataFrame(
        {
            "feat": np.arange(60),
            "target": targets,
        }
    )
    splitter = RepeatedStratifiedKFoldSplitter(n_splits=3, n_repeats=2, random_state=42)
    splits = list(splitter.split(df, y=targets))
    assert len(splits) == 6


def test_freeze_contracts_validation_binding(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "group_id": [f"grp_{i % 5}" for i in range(50)],
            "timestamp": pd.date_range("2025-01-01", periods=50, freq="D"),
            "feat": np.arange(50),
            "target": [0, 1] * 25,
        }
    )
    dataset_csv = tmp_path / "train.csv"
    df.to_csv(dataset_csv, index=False)

    state = {
        "dataset_path": str(dataset_csv),
        "target_column": "target",
        "output_dir": str(tmp_path / "out"),
        "validation_strategy": "group_kfold",
        "group_column": "group_id",
        "splits": 4,
        "seed": 42,
    }
    result = freeze_contracts(state)
    contracts = ContractBundle.model_validate(result["contracts"])
    assert contracts.validation.strategy == "group_kfold"
    assert contracts.validation.group_column == "group_id"
    assert contracts.validation.splits == 4
    assert contracts.validation.split_id is not None

    bad_state = dict(state, group_column="non_existent_column")
    with pytest.raises(ValueError, match="requires group_column in dataset features"):
        freeze_contracts(bad_state)


def test_row_id_preservation_and_group_split_execution(tmp_path: Path) -> None:
    n = 60
    custom_ids = [f"sample_{1000 + i}" for i in range(n)]
    df = pd.DataFrame(
        {
            "entity": [f"entity_{i % 6}" for i in range(n)],
            "x1": np.linspace(0, 10, n),
            "x2": np.random.randn(n),
            "target": [0, 1] * (n // 2),
        },
        index=custom_ids,
    )
    dataset_path = tmp_path / "group_data.csv"
    df.to_csv(dataset_path, index=True, index_label="row_id")

    out_dir = tmp_path / "run_group"
    contracts_state = freeze_contracts(
        {
            "dataset_path": str(dataset_path),
            "target_column": "target",
            "id_column": "row_id",
            "output_dir": str(out_dir),
            "validation_strategy": "group_kfold",
            "group_column": "entity",
            "splits": 3,
            "seed": 42,
            "metric_name": "roc_auc",
        }
    )
    contracts = ContractBundle.model_validate(contracts_state["contracts"])

    result = run_experiment(
        dataset_path=dataset_path,
        contracts=contracts,
        metric_name="roc_auc",
        output_dir=out_dir,
        max_experiments=1,
    )

    assert result["status"] == "ACCEPTED"
    assert result["verification"]["accepted"] is True

    # Find validation_predictions artifact
    val_pred_artifact = next(
        a for a in result["run_result"]["artifacts"] if "validation_predictions.csv" in a["path"]
    )
    pred_path = Path(val_pred_artifact["path"])
    assert pred_path.is_file()
    preds_df = pd.read_csv(pred_path)
    assert "row_id" in preds_df.columns
    # Row IDs should match original custom_ids
    assert set(preds_df["row_id"].astype(str)).issubset(set(custom_ids))

    # Verify oof_predictions.parquet exists
    oof_path = pred_path.parent / "oof_predictions.parquet"
    assert oof_path.is_file()
    oof_df = pd.read_parquet(oof_path)
    assert "row_id" in oof_df.columns
    assert set(oof_df["row_id"].astype(str)).issubset(set(custom_ids))


def test_temporal_split_graph_execution(tmp_path: Path) -> None:
    n = 60
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=n, freq="D"),
            "x1": np.linspace(0, 10, n),
            "x2": np.random.randn(n),
            "target": [0, 1] * (n // 2),
        }
    )
    dataset_path = tmp_path / "temporal_data.csv"
    df.to_csv(dataset_path, index=False)

    out_dir = tmp_path / "run_temporal"
    contracts_state = freeze_contracts(
        {
            "dataset_path": str(dataset_path),
            "target_column": "target",
            "output_dir": str(out_dir),
            "validation_strategy": "expanding_walk_forward",
            "time_column": "timestamp",
            "splits": 3,
            "seed": 42,
            "metric_name": "roc_auc",
        }
    )
    contracts = ContractBundle.model_validate(contracts_state["contracts"])

    result = run_experiment(
        dataset_path=dataset_path,
        contracts=contracts,
        metric_name="roc_auc",
        output_dir=out_dir,
        max_experiments=1,
    )

    assert result["status"] == "ACCEPTED"
    assert result["verification"]["accepted"] is True
