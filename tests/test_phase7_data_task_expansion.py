import io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from swarmfold.competition_bundle import inspect_competition_bundle, load_test_features
from swarmfold.coordinator_api import CoordinatorSettings, create_coordinator_app
from swarmfold.dashboard import DashboardRunManager
from swarmfold.data import (
    infer_task_type_dataset,
    inspect_training_dataset,
    load_training_dataset,
)
from swarmfold.features.executor import FeaturePipelineExecutor
from swarmfold.features.registry import get_feature_registry
from swarmfold.features.transformers import TextTfidfEncoder
from swarmfold.intake import (
    CompetitionProfile,
    FileUnderstanding,
    detect_unsupported_modalities,
)
from swarmfold.metrics import default_metric_for_task, get_metric_spec
from swarmfold.tasks import (
    MultilabelClassificationAdapter,
    TimeSeriesForecastingAdapter,
    get_task_adapter,
    get_task_registry,
)

ADMIN_TOKEN = "dashboard-admin-token-123"
POD_TOKENS = {
    "pod-a": "dashboard-pod-a-token-123",
    "pod-b": "dashboard-pod-b-token-123",
    "pod-c": "dashboard-pod-c-token-123",
}


def _auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {ADMIN_TOKEN}"}


def _dummy_runner(*args, **kwargs) -> dict[str, object]:
    return {"status": "ACCEPTED", "completed_results": [], "critiques": []}


class _ParquetFakeProfiler:
    def profile(self, files: list[Path]) -> CompetitionProfile:
        names = {p.name for p in files}
        roles = {
            "train.parquet": "training_data",
            "test.parquet": "inference_data",
            "sample_submission.parquet": "submission_template",
            "project.md": "competition_brief",
        }
        return CompetitionProfile(
            summary="Parquet tabular classification benchmark.",
            task_type="binary_classification",
            metric="accuracy",
            training_file="train.parquet",
            inference_files=["test.parquet"],
            submission_template="sample_submission.parquet",
            target_column="target",
            prediction_columns=["target"],
            id_columns=["id"],
            validation_strategy="stratified_kfold",
            files=[
                FileUnderstanding(filename=name, role=roles[name], reasoning="Parquet bundle match")
                for name in sorted(names)
            ],
            execution_ready=True,
            blockers=[],
            questions=[],
            confidence=0.95,
        )


def test_parquet_loading_and_bundle_inspection(tmp_path: Path) -> None:
    # 1. Create synthetic Parquet bundle
    train_df = pd.DataFrame({
        "id": [f"row_{i}" for i in range(12)],
        "feature_1": [float(i * 1.5) for i in range(12)],
        "feature_2": [float(i % 3) for i in range(12)],
        "target": ["class_a" if i % 2 == 0 else "class_b" for i in range(12)],
    })
    test_df = pd.DataFrame({
        "id": [f"test_{i}" for i in range(4)],
        "feature_1": [10.0, 11.5, 13.0, 14.5],
        "feature_2": [0.0, 1.0, 2.0, 0.0],
    })
    sub_df = pd.DataFrame({
        "id": [f"test_{i}" for i in range(4)],
        "target": ["class_a"] * 4,
    })

    train_path = tmp_path / "train.parquet"
    test_path = tmp_path / "test.parquet"
    sub_path = tmp_path / "sample_submission.parquet"
    brief_path = tmp_path / "project.md"

    train_df.to_parquet(train_path)
    test_df.to_parquet(test_path)
    sub_df.to_parquet(sub_path)
    brief_path.write_text("# Benchmark Brief\n\ntarget_column: target\n", encoding="utf-8")

    # 2. Inspect bundle
    spec = inspect_competition_bundle(train_path, test_path, sub_path, brief_path)
    assert spec.target_column == "target"
    assert spec.feature_columns == ["id", "feature_1", "feature_2"]
    assert spec.id_columns == ["id"]
    assert spec.train_rows == 12
    assert spec.test_rows == 4

    # 3. Load dataset
    features, target = load_training_dataset(train_path, "target", "binary_classification")
    assert len(features) == 12
    assert "target" not in features.columns
    assert list(target.unique()) == ["class_a", "class_b"]

    # 4. Load test features
    test_loaded = load_test_features(test_path, ["id", "feature_1", "feature_2"])
    assert len(test_loaded) == 4
    assert list(test_loaded.columns) == ["id", "feature_1", "feature_2"]

    # 5. Inspection and task inference
    n_rows, feat_cols, classes = inspect_training_dataset(
        train_path, "target", "binary_classification"
    )
    assert n_rows == 12
    assert classes == ["class_a", "class_b"]
    assert infer_task_type_dataset(train_path, "target") == "binary_classification"


def test_dashboard_parquet_upload_and_profiling(tmp_path: Path) -> None:
    manager = DashboardRunManager(
        tmp_path / "runs", runner=_dummy_runner, profiler=_ParquetFakeProfiler()
    )
    app = create_coordinator_app(
        CoordinatorSettings(
            database_url=f"sqlite:///{(tmp_path / 'dash1.db').as_posix()}",
            artifact_root=tmp_path / "artifacts1",
            admin_token=ADMIN_TOKEN,
            pod_tokens=POD_TOKENS,
        ),
        run_manager=manager,
    )

    train_df = pd.DataFrame({
        "id": [f"r_{i}" for i in range(12)],
        "feat": [float(i) for i in range(12)],
        "target": [i % 2 for i in range(12)],
    })
    test_df = pd.DataFrame({
        "id": [f"t_{i}" for i in range(3)],
        "feat": [1.0, 2.0, 3.0],
    })
    sub_df = pd.DataFrame({
        "id": [f"t_{i}" for i in range(3)],
        "target": [0, 0, 0],
    })

    train_bytes = io.BytesIO()
    test_bytes = io.BytesIO()
    sub_bytes = io.BytesIO()
    train_df.to_parquet(train_bytes)
    test_df.to_parquet(test_bytes)
    sub_df.to_parquet(sub_bytes)

    uploads = [
        ("files", ("train.parquet", train_bytes.getvalue(), "application/octet-stream")),
        ("files", ("test.parquet", test_bytes.getvalue(), "application/octet-stream")),
        ("files", ("sample_submission.parquet", sub_bytes.getvalue(), "application/octet-stream")),
        ("files", ("project.md", b"# Parquet Run\ntarget_column: target\n", "text/markdown")),
    ]

    with TestClient(app) as client:
        response = client.post("/v1/ui/runs", headers=_auth(), files=uploads)
        assert response.status_code == 202
        run_id = response.json()["run_id"]

        profiled = manager.wait(run_id, timeout=5)
        assert profiled["status"] == "AWAITING_CONFIRMATION"
        assert profiled["profile"]["training_file"] == "train.parquet"

        confirmed = client.post(
            f"/v1/ui/runs/{run_id}/confirm",
            headers=_auth(),
            json=profiled["profile"],
        )
        assert confirmed.status_code == 202
        assert confirmed.json()["status"] == "READY_TO_TRAIN"
        assert confirmed.json()["execution_inputs"]["training_file"] == "train.parquet"


def test_modality_profiling_and_blocking(tmp_path: Path) -> None:
    # 1. Create unsupported modality files
    audio_file = tmp_path / "recording.wav"
    audio_file.write_bytes(b"RIFFdummywavecontent")
    video_file = tmp_path / "clip.mp4"
    video_file.write_bytes(b"dummyvideocontent")
    npy_file = tmp_path / "embeddings.npy"
    npy_file.write_bytes(b"dummynpysavedarray")
    tabular_file = tmp_path / "tabular.csv"
    tabular_file.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

    blockers = detect_unsupported_modalities([audio_file, video_file, npy_file, tabular_file])
    assert len(blockers) == 3
    assert any("audio" in b and "recording.wav" in b for b in blockers)
    assert any("video" in b and "clip.mp4" in b for b in blockers)
    assert any("unsupported_binary" in b and "embeddings.npy" in b for b in blockers)

    # 2. Test blocking in DashboardRunManager
    class _ProfilerWithUnsupported:
        def profile(self, files: list[Path]) -> CompetitionProfile:
            return CompetitionProfile(
                summary="Bundle with audio and video.",
                task_type="binary_classification",
                metric="accuracy",
                training_file="tabular.csv",
                inference_files=["tabular.csv"],
                submission_template="tabular.csv",
                target_column="a",
                prediction_columns=["a"],
                id_columns=["b"],
                validation_strategy="holdout",
                files=[
                    FileUnderstanding(filename=p.name, role="unsupported", reasoning="Unsupported")
                    for p in files
                ],
                execution_ready=True,
                blockers=[],
                questions=[],
                confidence=0.5,
            )

    manager = DashboardRunManager(
        tmp_path / "runs", runner=_dummy_runner, profiler=_ProfilerWithUnsupported()
    )
    app = create_coordinator_app(
        CoordinatorSettings(
            database_url=f"sqlite:///{(tmp_path / 'dash2.db').as_posix()}",
            artifact_root=tmp_path / "artifacts2",
            admin_token=ADMIN_TOKEN,
            pod_tokens=POD_TOKENS,
        ),
        run_manager=manager,
    )

    tabular_csv = b"a,b\n0,1\n1,2\n0,3\n1,4\n0,5\n1,6\n0,7\n1,8\n0,9\n1,10\n"
    uploads = [
        ("files", ("tabular.csv", tabular_csv, "text/csv")),
        ("files", ("sound.wav", b"RIFFaudio", "application/octet-stream")),
        ("files", ("project.md", b"# Brief\ntarget_column: a\n", "text/markdown")),
    ]

    with TestClient(app) as client:
        response = client.post("/v1/ui/runs", headers=_auth(), files=uploads)
        assert response.status_code == 202
        run_id = response.json()["run_id"]

        profiled = manager.wait(run_id, timeout=5)
        assert profiled["status"] == "AWAITING_CONFIRMATION"
        assert profiled["profile"]["execution_ready"] is False
        assert any("sound.wav" in b for b in profiled["profile"]["blockers"])

        # 1. Attempt to confirm with unresolved blocker must fail
        confirmed = client.post(
            f"/v1/ui/runs/{run_id}/confirm",
            headers=_auth(),
            json=profiled["profile"],
        )
        assert confirmed.status_code == 409
        assert "blocker" in confirmed.text.casefold()

        # 2. Attempt to bypass blocker with a resolution must still fail at bundle inspection
        profile_resolved = dict(profiled["profile"])
        profile_resolved["resolutions"] = [
            {"issue": b, "response": "attempt to bypass", "status": "resolved"}
            for b in profiled["profile"]["blockers"]
        ]
        confirmed_resolved = client.post(
            f"/v1/ui/runs/{run_id}/confirm",
            headers=_auth(),
            json=profile_resolved,
        )
        assert confirmed_resolved.status_code == 409
        assert "unsupported modality" in confirmed_resolved.text.casefold()


def test_text_feature_pipeline_fold_safety(tmp_path: Path) -> None:
    train_texts = [
        "deep neural network architecture for image recognition",
        "gradient boosting decision trees for tabular prediction",
        "transformers and language models for text classification",
        "linear models with ridge and lasso regularization",
        "ensemble stacking and blended model predictions",
        "random forests with bagging and bootstrap samples",
        "recurrent networks with long short term memory",
        "support vector machines with radial basis kernel",
        "logistic regression for binary probabilistic targets",
        "k-means clustering and unsupervised representations",
    ]
    test_texts = [
        "transformers and neural models for tabular representations",
        "novel quantum computing algorithm for optimization",
    ]

    train_df = pd.DataFrame({"text": train_texts, "target": [i % 2 for i in range(10)]})
    test_df = pd.DataFrame({"text": test_texts})

    # 1. Direct TextTfidfEncoder test
    encoder = TextTfidfEncoder(columns=["text"], max_features=20, ngram_range=(1, 2), min_df=1)
    encoder.fit(train_df)
    train_feat = encoder.transform(train_df)
    test_feat = encoder.transform(test_df)

    assert isinstance(train_feat, pd.DataFrame)
    assert train_feat.shape[0] == 10
    assert train_feat.shape[1] > 0
    assert all(col.startswith("tfidf_text_") for col in train_feat.columns)
    assert not train_feat.isna().any().any()

    # Verify no test vocabulary leaked (e.g. 'quantum' should not be in features)
    assert not any("quantum" in col for col in train_feat.columns)
    assert test_feat.shape[0] == 2
    assert test_feat.shape[1] == train_feat.shape[1]

    # 2. Test FeaturePipelineExecutor with text_tfidf strategy
    train_features = train_df[["text"]]
    executor = FeaturePipelineExecutor(strategy="text_tfidf")
    executor.fit(train_features, train_df["target"])
    train_mat = executor.transform(train_features)
    test_mat = executor.transform(test_df)

    assert train_mat.shape[0] == 10
    assert train_mat.shape[1] > 0
    assert test_mat.shape[0] == 2
    assert test_mat.shape[1] == train_mat.shape[1]

    # 3. Serialization roundtrip parity
    dump_path = tmp_path / "text_pipeline.joblib"
    joblib.dump(executor, dump_path)
    loaded_executor = joblib.load(dump_path)
    reloaded_test_mat = loaded_executor.transform(test_df)
    np.testing.assert_allclose(test_mat, reloaded_test_mat)

    # 4. Verify FeatureRegistry contains text_tfidf
    reg = get_feature_registry()
    assert reg.is_registered("text_tfidf")
    spec = reg.get_spec("text_tfidf")
    assert spec.family == "text"
    created_enc = reg.create("text_tfidf", max_features=15)
    assert isinstance(created_enc, TextTfidfEncoder)
    assert created_enc.max_features == 15


def test_multilabel_classification_adapter_and_flow() -> None:
    adapter = get_task_adapter("multilabel_classification")
    assert isinstance(adapter, MultilabelClassificationAdapter)
    assert adapter.default_metric == "multilabel_f1"
    assert "multilabel_f1" in adapter.supported_metrics

    # Target validation
    valid_target = pd.DataFrame({
        "label_a": [0, 1, 1, 0, 1, 0, 1, 0, 1, 0],
        "label_b": [1, 0, 1, 1, 0, 0, 1, 1, 0, 1],
    })
    adapter.validate_target(valid_target)

    # Baselines
    sanity = adapter.sanity_baseline()
    strong = adapter.strong_baseline()
    assert sanity.model_family == "logistic_regression"
    assert sanity.parameters.get("multi_output") is True
    assert strong.model_family == "hist_gradient_boosting"
    assert strong.parameters.get("multi_output") is True

    # Prediction shape & format submission
    assert adapter.prediction_shape(10, n_classes=2) == (10, 2)
    id_df = pd.DataFrame({"id": [f"row_{i}" for i in range(10)]})
    preds = np.array([[0.8, 0.2]] * 10)
    sub = adapter.format_submission(id_df, preds, ["label_a", "label_b"])
    assert "label_a" in sub.columns and "label_b" in sub.columns
    assert sub.shape == (10, 3)

    # Metric validation
    metric = get_metric_spec("multilabel_f1")
    assert metric.higher_is_better
    assert "multilabel_classification" in metric.task_types
    assert default_metric_for_task("multilabel_classification") == "multilabel_f1"


def test_time_series_forecasting_adapter_and_flow() -> None:
    adapter = get_task_adapter("time_series_forecasting")
    assert isinstance(adapter, TimeSeriesForecastingAdapter)
    assert adapter.default_metric == "rmse"
    assert adapter.recommended_validation_strategy == "purged_time_series"

    # Target validation
    ts_target = pd.Series([10.5 + i * 0.3 for i in range(15)])
    adapter.validate_target(ts_target)

    # Baselines
    sanity = adapter.sanity_baseline()
    strong = adapter.strong_baseline()
    assert sanity.model_family == "logistic_regression"
    assert strong.model_family == "hist_gradient_boosting"

    # Prediction shape & submission formatting
    assert adapter.prediction_shape(15) == (15,)
    id_df = pd.DataFrame({"timestamp": [f"2026-09-{i+1:02d}" for i in range(15)]})
    preds = np.linspace(10.0, 14.5, 15)
    sub = adapter.format_submission(id_df, preds, ["predicted_value"])
    assert "predicted_value" in sub.columns
    assert len(sub) == 15

    # Metrics
    rmse_spec = get_metric_spec("rmse")
    assert "time_series_forecasting" in rmse_spec.task_types
    assert default_metric_for_task("time_series_forecasting") == "rmse"


def test_task_adapter_registry_completeness() -> None:
    registry = get_task_registry()
    adapters = registry.list_adapters()
    assert len(adapters) >= 5

    task_types = {a.task_type for a in adapters}
    expected_tasks = {
        "binary_classification",
        "multiclass_classification",
        "regression",
        "multilabel_classification",
        "time_series_forecasting",
    }
    assert expected_tasks.issubset(task_types)

    for a in adapters:
        assert a.default_metric in a.supported_metrics
        assert a.sanity_baseline() is not None
        assert a.strong_baseline() is not None
        assert len(a.compatible_ensemble_strategies()) > 0
        assert a.recommended_validation_strategy
