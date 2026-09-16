import json
from pathlib import Path

from fastapi.testclient import TestClient

from swarmfold.coordinator_api import CoordinatorSettings, create_coordinator_app
from swarmfold.dashboard import DashboardRunManager
from swarmfold.intake import CompetitionProfile, FileUnderstanding

ADMIN_TOKEN = "dashboard-admin-token-123"
POD_TOKENS = {
    "pod-a": "dashboard-pod-a-token-123",
    "pod-b": "dashboard-pod-b-token-123",
    "pod-c": "dashboard-pod-c-token-123",
}


def test_dashboard_serves_control_room_and_redacts_provider_key(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-appear-in-dashboard-response")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:20129/v1")
    monkeypatch.setenv("SWARMFOLD_PLANNER", "live")
    monkeypatch.setenv("SWARMFOLD_MODEL", "openai:agy/gemini-3.8-flash-high")
    app = _app(
        tmp_path,
        DashboardRunManager(tmp_path / "runs", runner=_fake_runner, profiler=_FakeProfiler()),
    )

    with TestClient(app) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "Experiment control room" in page.text
        assert "Analisis bundle" in page.text
        assert "Apakah pemahaman ini sudah benar?" in page.text
        assert client.get("/v1/ui/config").status_code == 401

        config = client.get("/v1/ui/config", headers=_auth())
        assert config.status_code == 200
        assert config.json() == {
            "planner": "live",
            "model": "openai:agy/gemini-3.8-flash-high",
            "base_url": "http://localhost:20129/v1",
            "api_key_configured": True,
        }
        assert "must-not-appear" not in config.text
    app.state.store.dispose()


def test_dashboard_profiles_then_requires_confirmation_before_training(tmp_path: Path) -> None:
    manager = DashboardRunManager(tmp_path / "runs", runner=_fake_runner, profiler=_FakeProfiler())
    app = _app(tmp_path, manager)

    fixture_root = Path(__file__).parent / "fixtures"
    upload_specs = [
        ("observations.csv", fixture_root / "bundle_learn.csv", "text/csv"),
        ("future_rows.csv", fixture_root / "bundle_score.csv", "text/csv"),
        ("answer_shape.csv", fixture_root / "bundle_template.csv", "text/csv"),
        ("competition.md", fixture_root / "bundle_brief.md", "text/markdown"),
    ]
    handles = [path.open("rb") for _, path, _ in upload_specs]
    try:
        uploads = [
            ("files", (name, handle, content_type))
            for (name, _, content_type), handle in zip(upload_specs, handles, strict=True)
        ]
        with TestClient(app) as client:
            response = client.post(
                "/v1/ui/runs",
                headers=_auth(),
                data={
                    "max_experiments": "5",
                    "max_training_seconds": "30",
                    "iteration_id": "ui-test",
                    "lane_id": "ai-b",
                },
                files=uploads,
            )
            assert response.status_code == 202, response.text
            run_id = response.json()["run_id"]

            profiled = manager.wait(run_id, timeout=3)
            assert profiled["status"] == "AWAITING_CONFIRMATION"
            assert profiled["experiments"] == []

            confirmed = client.post(
                f"/v1/ui/runs/{run_id}/confirm",
                headers=_auth(),
                json=profiled["profile"],
            )
            assert confirmed.status_code == 202, confirmed.text
            assert confirmed.json()["status"] == "READY_TO_TRAIN"
            started = client.post(
                f"/v1/ui/runs/{run_id}/start",
                headers=_auth(),
            )
            assert started.status_code == 202, started.text
            manager.wait(run_id, timeout=3)
            record = client.get(f"/v1/ui/runs/{run_id}", headers=_auth()).json()

            assert record["status"] == "ACCEPTED"
            assert record["target_column"] == "outcome"
            assert record["lane_id"] == "ai-b"
            assert record["experiments"][0]["accuracy"] == 0.75
            assert record["full_package"]["path"].endswith("full-package.zip")
            assert (tmp_path / "runs" / run_id / "full-package.zip").is_file()
            assert (tmp_path / "runs" / run_id / "status.json").is_file()
            listing = client.get("/v1/ui/runs", headers=_auth()).json()
            assert [item["run_id"] for item in listing] == [run_id]
            assert (
                client.get(
                    f"/v1/ui/runs/{run_id}/artifacts/unknown.txt", headers=_auth()
                ).status_code
                == 404
            )
    finally:
        for handle in handles:
            handle.close()
    app.state.store.dispose()


def test_dashboard_requires_a_recorded_resolution_for_each_issue(tmp_path: Path) -> None:
    manager = DashboardRunManager(
        tmp_path / "runs", runner=_fake_runner, profiler=_BlockingProfiler()
    )
    app = _app(tmp_path, manager)
    fixture_root = Path(__file__).parent / "fixtures"
    upload_specs = [
        ("observations.csv", fixture_root / "bundle_learn.csv", "text/csv"),
        ("future_rows.csv", fixture_root / "bundle_score.csv", "text/csv"),
        ("answer_shape.csv", fixture_root / "bundle_template.csv", "text/csv"),
        ("competition.md", fixture_root / "bundle_brief.md", "text/markdown"),
    ]
    handles = [path.open("rb") for _, path, _ in upload_specs]
    try:
        uploads = [
            ("files", (name, handle, content_type))
            for (name, _, content_type), handle in zip(upload_specs, handles, strict=True)
        ]
        with TestClient(app) as client:
            response = client.post("/v1/ui/runs", headers=_auth(), files=uploads)
            run_id = response.json()["run_id"]
            profiled = manager.wait(run_id, timeout=3)

            unresolved = client.post(
                f"/v1/ui/runs/{run_id}/confirm",
                headers=_auth(),
                json=profiled["profile"],
            )
            assert unresolved.status_code == 409

            issue = profiled["profile"]["blockers"][0]
            resolved_profile = {
                **profiled["profile"],
                "resolutions": [
                    {
                        "issue": issue,
                        "response": "Use the mixed categorical preprocessing adapter.",
                        "status": "resolved",
                    }
                ],
            }
            confirmed = client.post(
                f"/v1/ui/runs/{run_id}/confirm",
                headers=_auth(),
                json=resolved_profile,
            )
            assert confirmed.status_code == 202, confirmed.text
            assert confirmed.json()["profile"]["execution_ready"] is True
    finally:
        for handle in handles:
            handle.close()
    app.state.store.dispose()


def test_dashboard_runs_selected_ai_lanes_as_one_portfolio(tmp_path: Path) -> None:
    manager = DashboardRunManager(tmp_path / "runs", runner=_fake_runner, profiler=_FakeProfiler())
    app = _app(tmp_path, manager)
    fixture_root = Path(__file__).parent / "fixtures"
    specs = [
        ("observations.csv", fixture_root / "bundle_learn.csv", "text/csv"),
        ("future_rows.csv", fixture_root / "bundle_score.csv", "text/csv"),
        ("answer_shape.csv", fixture_root / "bundle_template.csv", "text/csv"),
        ("competition.md", fixture_root / "bundle_brief.md", "text/markdown"),
    ]
    handles = [path.open("rb") for _, path, _ in specs]
    lanes = [
        {"lane_id": lane_id, "enabled": lane_id in {"ai-a", "ai-d"}, "max_experiments": 5}
        for lane_id in ("ai-a", "ai-b", "ai-c", "ai-d", "ai-e")
    ]
    try:
        uploads = [
            ("files", (name, handle, content_type))
            for (name, _, content_type), handle in zip(specs, handles, strict=True)
        ]
        with TestClient(app) as client:
            response = client.post(
                "/v1/ui/runs",
                headers=_auth(),
                data={
                    "lane_configs": json.dumps(lanes),
                    "max_concurrent_lanes": "2",
                    "iteration_id": "portfolio-test",
                },
                files=uploads,
            )
            run_id = response.json()["run_id"]
            profiled = manager.wait(run_id, timeout=3)
            client.post(f"/v1/ui/runs/{run_id}/confirm", headers=_auth(), json=profiled["profile"])
            client.post(f"/v1/ui/runs/{run_id}/start", headers=_auth())
            result = manager.wait(run_id, timeout=5)

        assert result["status"] == "ACCEPTED"
        assert {item["lane_id"] for item in result["experiments"]} == {"ai-a", "ai-d"}
        assert {item["status"] for item in result["lanes"] if item["enabled"]} == {"ACCEPTED"}
        assert result["max_experiments"] == 10
        assert Path(result["full_package"]["path"]).is_file()
    finally:
        for handle in handles:
            handle.close()
    app.state.store.dispose()


def test_dashboard_rejects_invalid_dataset_before_queueing(tmp_path: Path) -> None:
    app = _app(
        tmp_path,
        DashboardRunManager(tmp_path / "runs", runner=_fake_runner, profiler=_FakeProfiler()),
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/ui/runs",
            headers=_auth(),
            files=[("files", ("notes.txt", b"not a dataset", "text/plain"))],
        )
        assert response.status_code == 422
        assert not list((tmp_path / "runs").glob("*/status.json"))
    app.state.store.dispose()


def _app(tmp_path: Path, manager: DashboardRunManager):
    return create_coordinator_app(
        CoordinatorSettings(
            database_url=f"sqlite:///{(tmp_path / 'dashboard.db').as_posix()}",
            artifact_root=tmp_path / "artifacts",
            admin_token=ADMIN_TOKEN,
            pod_tokens=POD_TOKENS,
        ),
        run_manager=manager,
    )


def _fake_runner(dataset_path: Path, **kwargs):
    assert dataset_path.is_file()
    assert kwargs["thread_id"]
    assert kwargs["test_dataset_path"].name == "future_rows.csv"
    assert kwargs["sample_submission_path"].name == "answer_shape.csv"
    assert kwargs["project_path"].name == "competition.md"
    return {
        "status": "ACCEPTED",
        "stop_reason": "experiment budget exhausted",
        "completed_results": [
            {
                "experiment_id": "ui-logreg",
                "model_family": "logistic_regression",
                "status": "SUCCEEDED",
                "metrics": {"accuracy": 0.75},
                "duration_seconds": 0.5,
                "resource_usage": {"peak_rss_mb": 128.0},
            }
        ],
        "critiques": [{"experiment_id": "ui-logreg", "verdict": "IMPROVED"}],
        "resource_usage": {"training_seconds": 0.5},
        "verification": {"accepted": True, "reasons": []},
        "packet": None,
    }


def _auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {ADMIN_TOKEN}"}


class _FakeProfiler:
    def profile(self, files: list[Path]) -> CompetitionProfile:
        names = {path.name for path in files}
        assert names == {
            "observations.csv",
            "future_rows.csv",
            "answer_shape.csv",
            "competition.md",
        }
        roles = {
            "observations.csv": "training_data",
            "future_rows.csv": "inference_data",
            "answer_shape.csv": "submission_template",
            "competition.md": "competition_brief",
        }
        return CompetitionProfile(
            summary="Binary outcome prediction with a separate scoring set.",
            task_type="binary_classification",
            metric="accuracy",
            training_file="observations.csv",
            inference_files=["future_rows.csv"],
            submission_template="answer_shape.csv",
            target_column="outcome",
            prediction_columns=["outcome"],
            id_columns=["record_id"],
            validation_strategy="stratified holdout",
            files=[
                FileUnderstanding(filename=name, role=roles[name], reasoning="Schema match")
                for name in sorted(names)
            ],
            execution_ready=True,
            blockers=[],
            questions=[],
            confidence=0.98,
        )


class _BlockingProfiler(_FakeProfiler):
    def profile(self, files: list[Path]) -> CompetitionProfile:
        profile = super().profile(files)
        return profile.model_copy(
            update={
                "execution_ready": False,
                "blockers": ["Categorical string features require preprocessing."],
            }
        )
