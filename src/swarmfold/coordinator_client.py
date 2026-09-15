import json
import shutil
from pathlib import Path

import httpx

from swarmfold.packet_archive import create_packet_archive


class CoordinatorClient:
    def __init__(
        self,
        base_url: str,
        token: str,
        outbox_dir: str | Path,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        self.outbox_dir = Path(outbox_dir)
        self.outbox_dir.mkdir(parents=True, exist_ok=True)
        self.client = client or httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=30.0,
        )

    def create_iteration(self, payload: dict[str, object]) -> dict[str, object]:
        response = self.client.post("/v1/iterations", json=payload)
        response.raise_for_status()
        return response.json()

    def register_pod(
        self,
        *,
        pod_id: str,
        iteration_id: str,
        lane_id: str,
        capabilities: dict[str, str | int | float | bool] | None = None,
    ) -> dict[str, object]:
        response = self.client.post(
            "/v1/pods/register",
            json={
                "pod_id": pod_id,
                "iteration_id": iteration_id,
                "lane_id": lane_id,
                "capabilities": capabilities or {},
            },
        )
        response.raise_for_status()
        return response.json()

    def heartbeat(self, pod_id: str, status: str) -> dict[str, object]:
        response = self.client.post(f"/v1/pods/{pod_id}/heartbeat", json={"status": status})
        response.raise_for_status()
        return response.json()

    def get_iteration(self, iteration_id: str) -> dict[str, object]:
        response = self.client.get(f"/v1/iterations/{iteration_id}")
        response.raise_for_status()
        return response.json()

    def submission_gate(self, iteration_id: str) -> dict[str, object]:
        response = self.client.get(f"/v1/iterations/{iteration_id}/submission-gate")
        response.raise_for_status()
        return response.json()

    def decide_submission(self, iteration_id: str, payload: dict[str, object]) -> dict[str, object]:
        response = self.client.post(
            f"/v1/iterations/{iteration_id}/submission-decisions", json=payload
        )
        response.raise_for_status()
        return response.json()

    def record_leaderboard(
        self, iteration_id: str, payload: dict[str, object]
    ) -> dict[str, object]:
        response = self.client.post(
            f"/v1/iterations/{iteration_id}/leaderboard-results", json=payload
        )
        response.raise_for_status()
        return response.json()

    def close_leaderboard(self, iteration_id: str) -> dict[str, object]:
        response = self.client.post(f"/v1/iterations/{iteration_id}/leaderboard/close")
        response.raise_for_status()
        return response.json()

    def merge_knowledge(self, iteration_id: str, snapshot_id: str) -> dict[str, object]:
        response = self.client.post(
            f"/v1/iterations/{iteration_id}/knowledge-snapshots",
            json={"snapshot_id": snapshot_id},
        )
        response.raise_for_status()
        return response.json()

    def get_knowledge_snapshot(self, snapshot_id: str) -> dict[str, object]:
        response = self.client.get(f"/v1/knowledge-snapshots/{snapshot_id}")
        response.raise_for_status()
        return response.json()

    def evaluate_effectiveness(self, iteration_id: str) -> dict[str, object]:
        response = self.client.post(f"/v1/iterations/{iteration_id}/effectiveness")
        response.raise_for_status()
        return response.json()

    def get_audit_events(self, limit: int = 100) -> dict[str, object]:
        response = self.client.get("/v1/audit-events", params={"limit": limit})
        response.raise_for_status()
        return response.json()

    def queue_packet(self, *, iteration_id: str, lane_id: str, packet_path: str | Path) -> Path:
        temporary_archive = self.outbox_dir / ".packet.zip"
        packet_sha256 = create_packet_archive(packet_path, temporary_archive)
        item = self.outbox_dir / packet_sha256
        item.mkdir(exist_ok=True)
        archive_path = item / "packet.zip"
        temporary_archive.replace(archive_path)
        (item / "metadata.json").write_text(
            json.dumps(
                {
                    "iteration_id": iteration_id,
                    "lane_id": lane_id,
                    "packet_sha256": packet_sha256,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return item

    def flush_outbox(self) -> tuple[int, int]:
        published = 0
        retained = 0
        for item in sorted(path for path in self.outbox_dir.iterdir() if path.is_dir()):
            metadata = json.loads((item / "metadata.json").read_text(encoding="utf-8"))
            try:
                with (item / "packet.zip").open("rb") as archive:
                    lane_segment = (
                        f"human-lanes/{metadata['lane_id']}"
                        if str(metadata["lane_id"]).startswith("human-")
                        else f"lanes/{metadata['lane_id']}"
                    )
                    response = self.client.post(
                        f"/v1/iterations/{metadata['iteration_id']}/{lane_segment}/packet",
                        data={"packet_sha256": metadata["packet_sha256"]},
                        files={"packet": ("packet.zip", archive, "application/zip")},
                    )
                response.raise_for_status()
            except (httpx.RequestError, httpx.HTTPStatusError):
                retained += 1
                continue
            shutil.rmtree(item)
            published += 1
        return published, retained
