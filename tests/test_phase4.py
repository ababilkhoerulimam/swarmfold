import json
from pathlib import Path

from fastapi.testclient import TestClient

from swarmfold.contracts import freeze_contracts
from swarmfold.coordinator_api import CoordinatorSettings, create_coordinator_app
from swarmfold.coordinator_models import IterationCreate, LaneCharter
from swarmfold.graph import run_experiment
from swarmfold.human_packets import (
    HumanCandidateRequest,
    HumanFailureRequest,
    HumanWaiverRequest,
    publish_human_packet,
)
from swarmfold.knowledge import validate_knowledge_snapshot
from swarmfold.packet_archive import create_packet_archive
from swarmfold.packets import read_packet_manifest
from swarmfold.schemas import ContractBundle, KnowledgeFinding

FIXTURE = Path(__file__).parent / "fixtures" / "demo_classification.csv"
ADMIN_TOKEN = "phase-four-admin-token-123"
POD_TOKENS = {
    "pod-a": "phase-four-pod-a-token-123",
    "pod-b": "phase-four-pod-b-token-123",
    "pod-c": "phase-four-pod-c-token-123",
}


def test_six_lane_knowledge_loop_and_next_iteration(tmp_path: Path) -> None:
    contracts = _contracts(tmp_path)
    settings = CoordinatorSettings(
        database_url=f"sqlite:///{(tmp_path / 'coordinator.db').as_posix()}",
        artifact_root=tmp_path / "artifacts",
        admin_token=ADMIN_TOKEN,
        pod_tokens=POD_TOKENS,
    )
    app = create_coordinator_app(settings)
    with TestClient(app) as client:
        iteration = IterationCreate(
            iteration_id="iteration-six",
            contracts=contracts,
            lane_charters=_six_charters(),
            submission_quota=1,
        )
        created = client.post(
            "/v1/iterations", json=iteration.model_dump(), headers=_auth(ADMIN_TOKEN)
        )
        assert created.status_code == 201

        ai_packets = _ai_packets(tmp_path, contracts)
        assert read_packet_manifest(ai_packets["ai-a"]).resource_usage["wall_seconds"] > 0
        for pod_id, lane_id in (("pod-a", "ai-a"), ("pod-b", "ai-b"), ("pod-c", "ai-c")):
            registered = client.post(
                "/v1/pods/register",
                json={
                    "pod_id": pod_id,
                    "iteration_id": "iteration-six",
                    "lane_id": lane_id,
                },
                headers=_auth(POD_TOKENS[pod_id]),
            )
            assert registered.status_code == 200
            _upload_packet(
                client,
                tmp_path,
                ai_packets[lane_id],
                lane_id,
                POD_TOKENS[pod_id],
            )

        assert client.get(
            "/v1/iterations/iteration-six/barrier", headers=_auth(ADMIN_TOKEN)
        ).json()["status"] == "PENDING"

        human_packets = _human_packets(tmp_path, contracts, ai_packets["ai-a"])
        forbidden = _upload_packet(
            client,
            tmp_path,
            human_packets["human-1"],
            "human-1",
            POD_TOKENS["pod-a"],
            expected_status=403,
        )
        assert forbidden.status_code == 403
        for lane_id, packet in human_packets.items():
            _upload_packet(client, tmp_path, packet, lane_id, ADMIN_TOKEN)

        barrier = client.get(
            "/v1/iterations/iteration-six/barrier", headers=_auth(ADMIN_TOKEN)
        )
        assert barrier.json() == {
            "iteration_id": "iteration-six",
            "status": "READY",
            "received_lanes": ["ai-a", "ai-b", "ai-c", "human-1", "human-2", "human-3"],
            "missing_lanes": [],
        }

        submitted = client.post(
            "/v1/iterations/iteration-six/submission-decisions",
            json={
                "lane_id": "ai-a",
                "decision": "SUBMITTED",
                "approver": "team-owner",
                "reason": "Best verified candidate within the declared quota",
            },
            headers=_auth(ADMIN_TOKEN),
        )
        assert submitted.status_code == 200
        quota_rejection = client.post(
            "/v1/iterations/iteration-six/submission-decisions",
            json={
                "lane_id": "ai-b",
                "decision": "SUBMITTED",
                "approver": "team-owner",
                "reason": "Attempt to exceed the declared quota",
            },
            headers=_auth(ADMIN_TOKEN),
        )
        assert quota_rejection.status_code == 409
        for lane_id in ("ai-b", "ai-c", "human-1"):
            response = client.post(
                "/v1/iterations/iteration-six/submission-decisions",
                json={
                    "lane_id": lane_id,
                    "decision": "WITHHELD",
                    "approver": "team-owner",
                    "reason": f"Recorded team decision for {lane_id}",
                },
                headers=_auth(ADMIN_TOKEN),
            )
            assert response.status_code == 200, response.text
        gate = client.get(
            "/v1/iterations/iteration-six/submission-gate", headers=_auth(ADMIN_TOKEN)
        )
        assert gate.json()["status"] == "READY"
        assert gate.json()["submitted_lanes"] == ["ai-a"]

        leaderboard = client.post(
            "/v1/iterations/iteration-six/leaderboard-results",
            json={
                "lane_id": "ai-a",
                "score": 0.81,
                "rank": 12,
                "recorded_by": "team-owner",
                "local_supported": False,
                "notes": "Public score retained separately from local validation.",
            },
            headers=_auth(ADMIN_TOKEN),
        )
        assert leaderboard.status_code == 200

        merged = client.post(
            "/v1/iterations/iteration-six/knowledge-snapshots",
            json={"snapshot_id": "v001"},
            headers=_auth(ADMIN_TOKEN),
        )
        assert merged.status_code == 200, merged.text
        snapshot_path = Path(merged.json()["artifact_path"])
        assert validate_knowledge_snapshot(snapshot_path)["snapshot_id"] == "v001"
        assert _read_jsonl(snapshot_path / "rejected-findings.jsonl")
        assert _read_jsonl(snapshot_path / "conflicts.jsonl")
        assert any(
            item["status"] == "LEADERBOARD_ONLY"
            for item in _read_jsonl(snapshot_path / "accepted-findings.jsonl")
        )
        effectiveness = client.post(
            "/v1/iterations/iteration-six/effectiveness",
            headers=_auth(ADMIN_TOKEN),
        )
        assert effectiveness.status_code == 200
        assert effectiveness.json()["metrics"]["terminal_lane_count"] == 6
        assert effectiveness.json()["metrics"]["reproducible_candidate_share"] > 0

        replay = client.post(
            "/v1/iterations/iteration-six/knowledge-snapshots",
            json={"snapshot_id": "v001"},
            headers=_auth(ADMIN_TOKEN),
        )
        assert replay.json()["idempotent_replay"] is True

        next_iteration = IterationCreate(
            iteration_id="iteration-next",
            contracts=contracts,
            lane_charters=_six_charters(),
            knowledge_snapshot_id="v001",
        )
        next_created = client.post(
            "/v1/iterations", json=next_iteration.model_dump(), headers=_auth(ADMIN_TOKEN)
        )
        assert next_created.status_code == 201
        fetched = client.get(
            "/v1/iterations/iteration-next", headers=_auth(POD_TOKENS["pod-a"])
        )
        assert fetched.json()["knowledge_snapshot_id"] == "v001"
        assert fetched.json()["knowledge_snapshot"]["snapshot_id"] == "v001"
        assert fetched.json()["knowledge_snapshot"]["contents"]["hypothesis-backlog.json"]
    app.state.store.dispose()


def test_local_only_merge_requires_explicit_leaderboard_close(tmp_path: Path) -> None:
    contracts = _contracts(tmp_path)
    settings = CoordinatorSettings(
        database_url=f"sqlite:///{(tmp_path / 'local-only.db').as_posix()}",
        artifact_root=tmp_path / "local-artifacts",
        admin_token=ADMIN_TOKEN,
        pod_tokens=POD_TOKENS,
    )
    app = create_coordinator_app(settings)
    with TestClient(app) as client:
        iteration = IterationCreate(
            iteration_id="local-only",
            contracts=contracts,
            lane_charters=_six_charters(),
            submission_quota=0,
        )
        assert client.post(
            "/v1/iterations", json=iteration.model_dump(), headers=_auth(ADMIN_TOKEN)
        ).status_code == 201
        ai_packets = _ai_packets(tmp_path, contracts, prefix="local")
        for pod_id, lane_id in (("pod-a", "ai-a"), ("pod-b", "ai-b"), ("pod-c", "ai-c")):
            client.post(
                "/v1/pods/register",
                json={"pod_id": pod_id, "iteration_id": "local-only", "lane_id": lane_id},
                headers=_auth(POD_TOKENS[pod_id]),
            )
            _upload_packet(
                client,
                tmp_path,
                ai_packets[lane_id],
                lane_id,
                POD_TOKENS[pod_id],
                iteration_id="local-only",
            )
        waivers = _human_waivers(tmp_path, contracts, iteration_id="local-only")
        for lane_id, packet in waivers.items():
            _upload_packet(
                client,
                tmp_path,
                packet,
                lane_id,
                ADMIN_TOKEN,
                iteration_id="local-only",
            )
        for lane_id in ("ai-a", "ai-b", "ai-c"):
            assert client.post(
                "/v1/iterations/local-only/submission-decisions",
                json={
                    "lane_id": lane_id,
                    "decision": "WITHHELD",
                    "approver": "team-owner",
                    "reason": "No external upload expected for this iteration",
                },
                headers=_auth(ADMIN_TOKEN),
            ).status_code == 200
        closed = client.post(
            "/v1/iterations/local-only/leaderboard/close", headers=_auth(ADMIN_TOKEN)
        )
        assert closed.status_code == 200
        merged = client.post(
            "/v1/iterations/local-only/knowledge-snapshots",
            json={"snapshot_id": "v010"},
            headers=_auth(ADMIN_TOKEN),
        )
        assert merged.status_code == 200
    app.state.store.dispose()


def _contracts(tmp_path: Path) -> ContractBundle:
    state = freeze_contracts(
        {
            "dataset_path": str(FIXTURE.resolve()),
            "target_column": "target",
            "seed": 42,
            "test_size": 0.25,
            "output_dir": str(tmp_path / "contracts"),
        }
    )
    return ContractBundle.model_validate(state["contracts"])


def _six_charters() -> list[LaneCharter]:
    lanes = ("ai-a", "ai-b", "ai-c", "human-1", "human-2", "human-3")
    return [
        LaneCharter(
            lane_id=lane,
            objective=f"Produce independent verified evidence for lane {lane}.",
            model_focus="independent mechanism",
            risk_level="medium",
            expected_diversity=f"Lane {lane} explores a distinct prediction mechanism.",
        )
        for lane in lanes
    ]


def _ai_packets(
    tmp_path: Path, contracts: ContractBundle, prefix: str = "main"
) -> dict[str, Path]:
    packets = {}
    iteration_id = "local-only" if prefix == "local" else "iteration-six"
    for lane_id in ("ai-a", "ai-b", "ai-c"):
        result = run_experiment(
            FIXTURE,
            output_dir=tmp_path / f"{prefix}-{lane_id}",
            tracking_uri=f"sqlite:///{(tmp_path / f'{prefix}-mlflow.db').as_posix()}",
            max_experiments=1,
            thread_id=f"{prefix}-{lane_id}",
            iteration_id=iteration_id,
            lane_id=lane_id,
            contracts=contracts,
        )
        packets[lane_id] = Path(result["packet"]["packet_path"])
    return packets


def _human_packets(
    tmp_path: Path, contracts: ContractBundle, ai_packet: Path
) -> dict[str, Path]:
    inputs = tmp_path / "human-inputs"
    inputs.mkdir()
    submission = inputs / "submission.csv"
    submission.write_bytes((ai_packet / "submission.csv").read_bytes())
    oof = inputs / "oof.parquet"
    test = inputs / "test.parquet"
    ledger = inputs / "ledger.jsonl"
    findings = inputs / "findings.jsonl"
    oof.write_bytes((ai_packet / "oof_predictions.parquet").read_bytes())
    test.write_bytes((ai_packet / "test_predictions.parquet").read_bytes())
    ledger.write_text(json.dumps({"run_id": "human-run-1", "status": "SUCCEEDED"}) + "\n")
    finding = _finding(contracts, "human-1", "human-run-1", "VALIDATED")
    findings.write_text(json.dumps(finding.model_dump()) + "\n", encoding="utf-8")
    candidate = publish_human_packet(
        HumanCandidateRequest(
            packet_type="candidate",
            iteration_id="iteration-six",
            lane_id="human-1",
            candidate_id="human-candidate-1",
            run_id="human-run-1",
            metrics={"accuracy": 0.75},
            decision="Publish verified human candidate for team review.",
        ),
        contracts,
        tmp_path / "human-1",
        submission=submission,
        oof_predictions=oof,
        test_predictions=test,
        experiment_ledger=ledger,
        findings=findings,
    )

    failure_inputs = tmp_path / "failure-inputs"
    failure_inputs.mkdir()
    failure_ledger = failure_inputs / "ledger.jsonl"
    failure_findings = failure_inputs / "findings.jsonl"
    failure_ledger.write_text(
        json.dumps({"run_id": "human-run-2", "status": "FAILED"}) + "\n"
    )
    opposed = _finding(contracts, "human-2", "human-run-2", "REJECTED")
    failure_findings.write_text(json.dumps(opposed.model_dump()) + "\n", encoding="utf-8")
    failure = publish_human_packet(
        HumanFailureRequest(
            packet_type="failure",
            iteration_id="iteration-six",
            lane_id="human-2",
            experiment_id="human-experiment-2",
            run_id="human-run-2",
            terminal_reason="The tested feature mechanism did not survive validation.",
            missing_deliverables=["submission.csv"],
            retry_useful=False,
            decision="Preserve this failure to prevent duplicate future work.",
        ),
        contracts,
        tmp_path / "human-2",
        experiment_ledger=failure_ledger,
        findings=failure_findings,
    )
    waiver = publish_human_packet(
        HumanWaiverRequest(
            packet_type="waiver",
            iteration_id="iteration-six",
            lane_id="human-3",
            run_id="human-waiver-3",
            reason="The team approved a waiver after the declared lane deadline.",
            approver="team-owner",
            incomplete_work=["manual blend"],
            decision="Waive this lane and preserve its incomplete-work description.",
        ),
        contracts,
        tmp_path / "human-3",
    )
    return {
        "human-1": Path(candidate.packet_path),
        "human-2": Path(failure.packet_path),
        "human-3": Path(waiver.packet_path),
    }


def _human_waivers(
    tmp_path: Path, contracts: ContractBundle, *, iteration_id: str
) -> dict[str, Path]:
    packets = {}
    for lane_id in ("human-1", "human-2", "human-3"):
        packet = publish_human_packet(
            HumanWaiverRequest(
                packet_type="waiver",
                iteration_id=iteration_id,
                lane_id=lane_id,
                run_id=f"{iteration_id}-{lane_id}-waiver",
                reason=f"The team approved the local-only waiver for {lane_id}.",
                approver="team-owner",
                decision=f"Close {lane_id} without an external submission candidate.",
            ),
            contracts,
            tmp_path / f"{iteration_id}-{lane_id}",
        )
        packets[lane_id] = Path(packet.packet_path)
    return packets


def _finding(
    contracts: ContractBundle, lane_id: str, run_id: str, status: str
) -> KnowledgeFinding:
    return KnowledgeFinding(
        finding_id=f"iteration-six-{lane_id}-finding",
        iteration_id="iteration-six",
        source_lane=lane_id,
        status=status,
        claim="Manual frequency encoding improves rare category handling",
        mechanism="Count information reduces sparse category fragmentation",
        source_runs=[run_id],
        data_version=contracts.data.data_version,
        split_id=contracts.validation.split_id,
        metric_name=contracts.metric.metric_name,
    )


def _upload_packet(
    client: TestClient,
    tmp_path: Path,
    packet: Path,
    lane_id: str,
    token: str,
    *,
    iteration_id: str = "iteration-six",
    expected_status: int = 200,
):
    archive = tmp_path / f"{iteration_id}-{lane_id}-{token[:4]}.zip"
    digest = create_packet_archive(packet, archive)
    segment = "human-lanes" if lane_id.startswith("human-") else "lanes"
    with archive.open("rb") as handle:
        response = client.post(
            f"/v1/iterations/{iteration_id}/{segment}/{lane_id}/packet",
            data={"packet_sha256": digest},
            files={"packet": (archive.name, handle, "application/zip")},
            headers=_auth(token),
        )
    assert response.status_code == expected_status, response.text
    return response


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}
