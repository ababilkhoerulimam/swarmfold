import zipfile
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from swarmfold.backup import create_local_backup, restore_local_backup
from swarmfold.contracts import freeze_contracts
from swarmfold.coordinator_api import CoordinatorSettings, create_coordinator_app
from swarmfold.coordinator_client import CoordinatorClient
from swarmfold.coordinator_models import IterationCreate, LaneCharter
from swarmfold.coordinator_store import CoordinatorStore
from swarmfold.graph import run_experiment
from swarmfold.packet_archive import create_packet_archive, extract_and_validate_archive
from swarmfold.schemas import ContractBundle

FIXTURE = Path(__file__).parent / "fixtures" / "demo_classification.csv"
ADMIN_TOKEN = "admin-token-for-tests-123"
POD_TOKENS = {
    "pod-a": "pod-a-token-for-tests-123",
    "pod-b": "pod-b-token-for-tests-123",
    "pod-c": "pod-c-token-for-tests-123",
}


def _charters() -> list[LaneCharter]:
    return [
        LaneCharter(
            lane_id="ai-a",
            objective="Produce a stable local validation champion.",
            model_focus="stable baselines",
            risk_level="low",
            expected_diversity="Low-variance predictions from conservative models.",
        ),
        LaneCharter(
            lane_id="ai-b",
            objective="Explore alternative nonlinear model mechanisms.",
            model_focus="tree models",
            risk_level="medium",
            expected_diversity="Nonlinear partitions unlike the linear baseline.",
        ),
        LaneCharter(
            lane_id="ai-c",
            objective="Challenge the leaders with diverse ensemble candidates.",
            model_focus="ensembles",
            risk_level="high",
            expected_diversity="Blended probabilities across distinct model families.",
        ),
    ]


@pytest.fixture(scope="module")
def phase3_environment(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, object]]:
    root = tmp_path_factory.mktemp("phase3")
    tracking_uri = f"sqlite:///{(root / 'shared-mlflow.db').as_posix()}"
    contract_state = freeze_contracts(
        {
            "dataset_path": str(FIXTURE.resolve()),
            "target_column": "target",
            "seed": 42,
            "test_size": 0.25,
            "output_dir": str(root / "contract-source"),
        }
    )
    contracts = ContractBundle.model_validate(contract_state["contracts"])

    database_path = root / "coordinator.db"
    artifact_root = root / "coordinator-artifacts"
    settings = CoordinatorSettings(
        database_url=f"sqlite:///{database_path.as_posix()}",
        artifact_root=artifact_root,
        admin_token=ADMIN_TOKEN,
        pod_tokens=POD_TOKENS,
    )
    app = create_coordinator_app(settings)
    client = TestClient(app)
    iteration = IterationCreate(
        iteration_id="iteration-1",
        contracts=contracts,
        lane_charters=_charters(),
    )
    response = client.post(
        "/v1/iterations",
        json=iteration.model_dump(),
        headers=_auth(ADMIN_TOKEN),
    )
    assert response.status_code == 201
    shared = client.get("/v1/iterations/iteration-1", headers=_auth(POD_TOKENS["pod-a"]))
    assert shared.status_code == 200
    shared_contracts = ContractBundle.model_validate(shared.json()["contracts"])
    packets: dict[str, Path] = {}
    consumed_hashes: set[str] = set()
    for pod_id, lane_id in (("pod-a", "ai-a"), ("pod-b", "ai-b"), ("pod-c", "ai-c")):
        result = run_experiment(
            FIXTURE,
            output_dir=root / pod_id,
            tracking_uri=tracking_uri,
            max_experiments=1,
            thread_id=f"iteration-1-{pod_id}",
            iteration_id="iteration-1",
            lane_id=lane_id,
            contracts=shared_contracts,
        )
        packets[lane_id] = Path(result["packet"]["packet_path"])
        consumed_hashes.add(result["contracts"]["contract_sha256"])
    assert consumed_hashes == {shared_contracts.contract_sha256}
    try:
        yield {
            "root": root,
            "database_path": database_path,
            "artifact_root": artifact_root,
            "client": client,
            "packets": packets,
            "iteration": iteration,
        }
    finally:
        client.close()
        app.state.store.dispose()


def test_three_pods_register_publish_and_open_barrier(
    phase3_environment: dict[str, object],
) -> None:
    client: TestClient = phase3_environment["client"]
    packets: dict[str, Path] = phase3_environment["packets"]
    root: Path = phase3_environment["root"]

    forbidden = client.post(
        "/v1/iterations",
        json=phase3_environment["iteration"].model_dump(),
        headers=_auth(POD_TOKENS["pod-a"]),
    )
    assert forbidden.status_code == 403

    for index, (pod_id, lane_id) in enumerate(
        (("pod-a", "ai-a"), ("pod-b", "ai-b"), ("pod-c", "ai-c")), start=1
    ):
        registered = client.post(
            "/v1/pods/register",
            json={
                "pod_id": pod_id,
                "iteration_id": "iteration-1",
                "lane_id": lane_id,
                "capabilities": {"cpu_count": 4},
            },
            headers=_auth(POD_TOKENS[pod_id]),
        )
        assert registered.status_code == 200
        heartbeat = client.post(
            f"/v1/pods/{pod_id}/heartbeat",
            json={"status": "RUNNING"},
            headers=_auth(POD_TOKENS[pod_id]),
        )
        assert heartbeat.status_code == 200

        archive_path = root / f"{lane_id}.zip"
        packet_sha256 = create_packet_archive(packets[lane_id], archive_path)
        with archive_path.open("rb") as archive:
            uploaded = client.post(
                f"/v1/iterations/iteration-1/lanes/{lane_id}/packet",
                data={"packet_sha256": packet_sha256},
                files={"packet": (archive_path.name, archive, "application/zip")},
                headers=_auth(POD_TOKENS[pod_id]),
            )
        assert uploaded.status_code == 200, uploaded.text
        assert Path(uploaded.json()["artifact_path"], "COMPLETED.json").is_file()

        barrier = client.get(
            "/v1/iterations/iteration-1/barrier", headers=_auth(POD_TOKENS[pod_id])
        )
        assert barrier.status_code == 200
        expected_status = "READY" if index == 3 else "PENDING"
        assert barrier.json()["status"] == expected_status

    assert barrier.json()["received_lanes"] == ["ai-a", "ai-b", "ai-c"]
    replay_archive = root / "ai-a.zip"
    replay_sha256 = create_packet_archive(packets["ai-a"], replay_archive)
    with replay_archive.open("rb") as archive:
        replay = client.post(
            "/v1/iterations/iteration-1/lanes/ai-a/packet",
            data={"packet_sha256": replay_sha256},
            files={"packet": ("ai-a.zip", archive, "application/zip")},
            headers=_auth(POD_TOKENS["pod-a"]),
        )
    assert replay.status_code == 200
    assert replay.json()["idempotent_replay"] is True


def test_iteration_contract_and_charters_are_immutable(
    phase3_environment: dict[str, object],
) -> None:
    client: TestClient = phase3_environment["client"]
    iteration: IterationCreate = phase3_environment["iteration"]
    changed = iteration.model_dump()
    changed["lane_charters"][0]["objective"] = (
        "A changed objective must not overwrite shared truth."
    )

    response = client.post("/v1/iterations", json=changed, headers=_auth(ADMIN_TOKEN))

    assert response.status_code == 409


def test_tokens_are_role_and_pod_scoped(phase3_environment: dict[str, object]) -> None:
    client: TestClient = phase3_environment["client"]
    packets: dict[str, Path] = phase3_environment["packets"]
    root: Path = phase3_environment["root"]

    unauthorized = client.get("/v1/iterations/iteration-1", headers=_auth("invalid-token-value"))
    assert unauthorized.status_code == 401
    wrong_identity = client.post(
        "/v1/pods/register",
        json={
            "pod_id": "pod-b",
            "iteration_id": "iteration-1",
            "lane_id": "ai-b",
        },
        headers=_auth(POD_TOKENS["pod-a"]),
    )
    assert wrong_identity.status_code == 403

    archive_path = root / "wrong-lane.zip"
    packet_sha256 = create_packet_archive(packets["ai-a"], archive_path)
    with archive_path.open("rb") as archive:
        wrong_lane = client.post(
            "/v1/iterations/iteration-1/lanes/ai-b/packet",
            data={"packet_sha256": packet_sha256},
            files={"packet": (archive_path.name, archive, "application/zip")},
            headers=_auth(POD_TOKENS["pod-b"]),
        )
    assert wrong_lane.status_code == 422


def test_shared_contract_rejects_different_local_split(
    phase3_environment: dict[str, object], tmp_path: Path
) -> None:
    iteration: IterationCreate = phase3_environment["iteration"]

    with pytest.raises(ValueError, match="Local data does not match shared contracts"):
        run_experiment(
            FIXTURE,
            output_dir=tmp_path,
            seed=43,
            max_experiments=1,
            iteration_id="iteration-1",
            lane_id="ai-a",
            contracts=iteration.contracts,
        )

    assert not (tmp_path / "runs").exists()


def test_packet_archive_rejects_path_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("packet/../../outside.txt", "unsafe")

    with pytest.raises(ValueError, match="unsafe path"):
        extract_and_validate_archive(archive_path, tmp_path / "extracted")


def test_network_failure_retains_outbox_packet(
    phase3_environment: dict[str, object], tmp_path: Path
) -> None:
    packets: dict[str, Path] = phase3_environment["packets"]
    attempts = 0

    def transport(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ConnectError("offline", request=request)
        return httpx.Response(200, json={"accepted": True})

    http_client = httpx.Client(
        base_url="http://coordinator.test",
        transport=httpx.MockTransport(transport),
    )
    client = CoordinatorClient(
        "http://coordinator.test",
        POD_TOKENS["pod-a"],
        tmp_path / "outbox",
        client=http_client,
    )
    item = client.queue_packet(
        iteration_id="iteration-1", lane_id="ai-a", packet_path=packets["ai-a"]
    )

    assert client.flush_outbox() == (0, 1)
    assert item.is_dir()
    assert client.flush_outbox() == (1, 0)
    assert not item.exists()
    http_client.close()


def test_coordinator_backup_restores_metadata_and_artifacts(
    phase3_environment: dict[str, object], tmp_path: Path
) -> None:
    backup = create_local_backup(
        phase3_environment["database_path"],
        phase3_environment["artifact_root"],
        tmp_path / "coordinator.tar.gz",
    )
    restored_database = tmp_path / "restored" / "coordinator.db"
    restored_artifacts = tmp_path / "restored" / "artifacts"

    restore_local_backup(backup, restored_database, restored_artifacts)

    store = CoordinatorStore(f"sqlite:///{restored_database.as_posix()}")
    try:
        assert store.get_iteration("iteration-1")["iteration_id"] == "iteration-1"
        received, missing = store.barrier("iteration-1")
        assert received == ["ai-a", "ai-b", "ai-c"]
        assert missing == []
    finally:
        store.dispose()
    assert len(list(restored_artifacts.glob("iterations/iteration-1/ai-*/*/COMPLETED.json"))) == 3


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}
