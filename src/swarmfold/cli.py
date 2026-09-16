import json
import os
from pathlib import Path
from typing import Annotated, Literal

import typer
from dotenv import load_dotenv

from swarmfold.graph import run_experiment
from swarmfold.packets import validate_terminal_packet

app = typer.Typer(help="Run evidence-first data science experiments.", no_args_is_help=True)


@app.callback()
def main() -> None:
    """SwarmFold command-line interface."""


@app.command("run-demo")
def run_demo(
    dataset: Annotated[
        Path,
        typer.Option(exists=True, file_okay=True, dir_okay=False, readable=True),
    ] = Path("tests/fixtures/demo_classification.csv"),
    target: Annotated[str, typer.Option()] = "target",
    seed: Annotated[int, typer.Option(min=0)] = 42,
    output_dir: Annotated[Path, typer.Option()] = Path(".swarmfold"),
    planner: Annotated[Literal["mock", "live"] | None, typer.Option()] = None,
    model: Annotated[str | None, typer.Option()] = None,
    tracking_uri: Annotated[str | None, typer.Option()] = None,
    max_experiments: Annotated[int, typer.Option(min=1, max=200)] = 4,
    max_training_seconds: Annotated[float | None, typer.Option(min=0.001)] = None,
    thread_id: Annotated[str | None, typer.Option()] = None,
    resume: Annotated[bool, typer.Option()] = False,
    interrupt_before: Annotated[str | None, typer.Option(hidden=True)] = None,
    iteration_id: Annotated[str, typer.Option()] = "local-iteration",
    lane_id: Annotated[str, typer.Option()] = "ai-a",
    contracts_file: Annotated[
        Path | None,
        typer.Option(exists=True, file_okay=True, dir_okay=False, readable=True),
    ] = None,
) -> None:
    """Run or resume the local LangGraph pod and publish one terminal packet."""
    load_dotenv()
    shared_contracts = None
    if contracts_file:
        contract_payload = json.loads(contracts_file.read_text(encoding="utf-8"))
        shared_contracts = contract_payload.get("contracts", contract_payload)
    try:
        result = run_experiment(
            dataset,
            target_column=target,
            seed=seed,
            output_dir=output_dir,
            planner_mode=planner,
            model=model,
            tracking_uri=tracking_uri,
            max_experiments=max_experiments,
            max_training_seconds=max_training_seconds,
            thread_id=thread_id,
            resume=resume,
            interrupt_before=interrupt_before,
            iteration_id=iteration_id,
            lane_id=lane_id,
            contracts=shared_contracts,
        )
    except (RuntimeError, ValueError) as exc:
        typer.echo(f"Configuration error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] not in {"ACCEPTED", "INTERRUPTED"}:
        raise typer.Exit(code=1)


@app.command("validate-packet")
def validate_packet(
    packet: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True, readable=True),
    ],
) -> None:
    """Validate a completed candidate, failure, or waiver packet."""
    try:
        result = validate_terminal_packet(packet)
    except ValueError as exc:
        typer.echo(f"Packet validation failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(result.model_dump(), indent=2, sort_keys=True))


@app.command("serve-coordinator")
def serve_coordinator(
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option(min=1, max=65535)] = 8000,
) -> None:
    """Serve the authenticated coordinator API using environment configuration."""
    import uvicorn

    from swarmfold.coordinator_api import CoordinatorSettings, create_coordinator_app

    load_dotenv()
    try:
        settings = CoordinatorSettings.from_env()
    except (json.JSONDecodeError, ValueError) as exc:
        typer.echo(f"Configuration error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    uvicorn.run(create_coordinator_app(settings), host=host, port=port)


@app.command("queue-packet")
def queue_packet(
    packet: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True, readable=True),
    ],
    iteration_id: Annotated[str, typer.Option()],
    lane_id: Annotated[str, typer.Option()],
    outbox_dir: Annotated[Path, typer.Option()] = Path(".swarmfold/outbox"),
) -> None:
    """Create a verified portable archive in the durable publication outbox."""
    from swarmfold.coordinator_client import CoordinatorClient

    client = CoordinatorClient("http://unused", "not-used-for-queueing", outbox_dir)
    item = client.queue_packet(iteration_id=iteration_id, lane_id=lane_id, packet_path=packet)
    typer.echo(str(item.resolve()))


@app.command("flush-outbox")
def flush_outbox(
    coordinator_url: Annotated[str, typer.Option()],
    outbox_dir: Annotated[Path, typer.Option()] = Path(".swarmfold/outbox"),
    admin: Annotated[bool, typer.Option()] = False,
) -> None:
    """Publish queued packets, retaining every item that cannot be delivered."""
    from swarmfold.coordinator_client import CoordinatorClient

    load_dotenv()
    token_name = "SWARMFOLD_COORDINATOR_TOKEN" if admin else "SWARMFOLD_POD_TOKEN"
    token = os.getenv(token_name)
    if not token:
        typer.echo(f"Configuration error: {token_name} is required", err=True)
        raise typer.Exit(code=2)
    client = CoordinatorClient(coordinator_url, token, outbox_dir)
    published, retained = client.flush_outbox()
    typer.echo(json.dumps({"published": published, "retained": retained}, sort_keys=True))
    if retained:
        raise typer.Exit(code=1)


@app.command("backup-coordinator")
def backup_coordinator(
    database: Annotated[Path, typer.Option(exists=True, file_okay=True, dir_okay=False)],
    artifacts: Annotated[Path, typer.Option(exists=True, file_okay=False, dir_okay=True)],
    output: Annotated[Path, typer.Option()],
) -> None:
    """Create a consistent backup of local coordinator metadata and artifacts."""
    from swarmfold.backup import create_local_backup

    try:
        backup = create_local_backup(database, artifacts, output)
    except ValueError as exc:
        typer.echo(f"Backup failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(str(backup))


@app.command("restore-coordinator")
def restore_coordinator(
    backup: Annotated[Path, typer.Option(exists=True, file_okay=True, dir_okay=False)],
    database: Annotated[Path, typer.Option()],
    artifacts: Annotated[Path, typer.Option()],
) -> None:
    """Restore coordinator state into new, empty local targets."""
    from swarmfold.backup import restore_local_backup

    try:
        restore_local_backup(backup, database, artifacts)
    except ValueError as exc:
        typer.echo(f"Restore failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo("Coordinator backup restored")


@app.command("verify-backup")
def verify_backup(
    backup: Annotated[Path, typer.Argument(exists=True, file_okay=True, dir_okay=False)],
) -> None:
    """Run archive hash and SQLite integrity checks without restoring live state."""
    from swarmfold.backup import verify_local_backup

    try:
        result = verify_local_backup(backup)
    except ValueError as exc:
        typer.echo(f"Backup verification failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@app.command("prune-backups")
def prune_backups(
    directory: Annotated[Path, typer.Argument(exists=True, file_okay=False, dir_okay=True)],
    keep_last: Annotated[int, typer.Option(min=1)] = 7,
    max_age_days: Annotated[int, typer.Option(min=1)] = 30,
) -> None:
    """Apply count and age retention to local coordinator backup archives."""
    from swarmfold.backup import apply_backup_retention

    removed = apply_backup_retention(
        directory,
        keep_last=keep_last,
        max_age_days=max_age_days,
    )
    typer.echo(json.dumps({"removed": [str(path) for path in removed]}, indent=2))


@app.command("freeze-contracts")
def freeze_contract_bundle(
    dataset: Annotated[
        Path,
        typer.Option(exists=True, file_okay=True, dir_okay=False, readable=True),
    ] = Path("tests/fixtures/demo_classification.csv"),
    target: Annotated[str, typer.Option()] = "target",
    seed: Annotated[int, typer.Option(min=0)] = 42,
    test_size: Annotated[float, typer.Option(min=0.001, max=0.999)] = 0.25,
    output_dir: Annotated[Path, typer.Option()] = Path(".swarmfold/contract-source"),
) -> None:
    """Freeze a portable contract bundle before starting a distributed iteration."""
    from swarmfold.contracts import freeze_contracts

    result = freeze_contracts(
        {
            "dataset_path": str(dataset.resolve()),
            "target_column": target,
            "seed": seed,
            "test_size": test_size,
            "output_dir": str(output_dir.resolve()),
        }
    )
    typer.echo(result["contracts_path"])


@app.command("create-iteration")
def create_iteration(
    coordinator_url: Annotated[str, typer.Option()],
    iteration_id: Annotated[str, typer.Option()],
    contracts: Annotated[Path, typer.Option(exists=True, file_okay=True, dir_okay=False)],
    charters: Annotated[Path, typer.Option(exists=True, file_okay=True, dir_okay=False)],
    submission_quota: Annotated[int, typer.Option(min=0, max=6)] = 3,
    knowledge_snapshot_id: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Publish immutable contracts and three-lane or six-lane charters."""
    from swarmfold.coordinator_client import CoordinatorClient
    from swarmfold.coordinator_models import IterationCreate

    load_dotenv()
    token = os.getenv("SWARMFOLD_COORDINATOR_TOKEN")
    if not token:
        typer.echo("Configuration error: SWARMFOLD_COORDINATOR_TOKEN is required", err=True)
        raise typer.Exit(code=2)
    payload = IterationCreate.model_validate(
        {
            "iteration_id": iteration_id,
            "contracts": json.loads(contracts.read_text(encoding="utf-8")),
            "lane_charters": json.loads(charters.read_text(encoding="utf-8")),
            "submission_quota": submission_quota,
            "knowledge_snapshot_id": knowledge_snapshot_id,
        }
    )
    client = CoordinatorClient(coordinator_url, token, ".swarmfold/admin-outbox")
    result = client.create_iteration(payload.model_dump())
    typer.echo(json.dumps(result, sort_keys=True))


@app.command("register-pod")
def register_pod(
    coordinator_url: Annotated[str, typer.Option()],
    pod_id: Annotated[str, typer.Option()],
    iteration_id: Annotated[str, typer.Option()],
    lane_id: Annotated[str, typer.Option()],
) -> None:
    """Register this authenticated laptop pod for its assigned lane."""
    from swarmfold.coordinator_client import CoordinatorClient

    load_dotenv()
    token = os.getenv("SWARMFOLD_POD_TOKEN")
    if not token:
        typer.echo("Configuration error: SWARMFOLD_POD_TOKEN is required", err=True)
        raise typer.Exit(code=2)
    client = CoordinatorClient(coordinator_url, token, ".swarmfold/outbox")
    result = client.register_pod(
        pod_id=pod_id,
        iteration_id=iteration_id,
        lane_id=lane_id,
    )
    typer.echo(json.dumps(result, sort_keys=True))


@app.command("fetch-iteration")
def fetch_iteration(
    coordinator_url: Annotated[str, typer.Option()],
    iteration_id: Annotated[str, typer.Option()],
    output: Annotated[Path, typer.Option()] = Path(".swarmfold/shared-iteration.json"),
) -> None:
    """Fetch immutable contracts and lane charters for a laptop pod."""
    from swarmfold.artifacts import write_json
    from swarmfold.coordinator_client import CoordinatorClient

    load_dotenv()
    token = os.getenv("SWARMFOLD_POD_TOKEN")
    if not token:
        typer.echo("Configuration error: SWARMFOLD_POD_TOKEN is required", err=True)
        raise typer.Exit(code=2)
    client = CoordinatorClient(coordinator_url, token, ".swarmfold/outbox")
    result = client.get_iteration(iteration_id)
    path = write_json(output, result)
    typer.echo(str(path.resolve()))


@app.command("publish-human-packet")
def publish_human_lane_packet(
    metadata: Annotated[Path, typer.Option(exists=True, file_okay=True, dir_okay=False)],
    contracts: Annotated[Path, typer.Option(exists=True, file_okay=True, dir_okay=False)],
    output_dir: Annotated[Path, typer.Option()] = Path(".swarmfold/human"),
    submission: Annotated[Path | None, typer.Option()] = None,
    oof_predictions: Annotated[Path | None, typer.Option()] = None,
    test_predictions: Annotated[Path | None, typer.Option()] = None,
    experiment_ledger: Annotated[Path | None, typer.Option()] = None,
    findings: Annotated[Path | None, typer.Option()] = None,
    artifact: Annotated[list[Path] | None, typer.Option()] = None,
) -> None:
    """Build and verify one candidate, failure, or waiver packet for a human lane."""
    from swarmfold.human_packets import load_human_packet_request, publish_human_packet
    from swarmfold.schemas import ContractBundle

    request = load_human_packet_request(metadata)
    contract_payload = json.loads(contracts.read_text(encoding="utf-8"))
    bundle = ContractBundle.model_validate(contract_payload.get("contracts", contract_payload))
    try:
        result = publish_human_packet(
            request,
            bundle,
            output_dir,
            submission=submission,
            oof_predictions=oof_predictions,
            test_predictions=test_predictions,
            experiment_ledger=experiment_ledger,
            findings=findings,
            artifacts=artifact,
        )
    except ValueError as exc:
        typer.echo(f"Human packet publication failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(result.model_dump(), indent=2, sort_keys=True))


@app.command("submission-gate")
def show_submission_gate(
    coordinator_url: Annotated[str, typer.Option()],
    iteration_id: Annotated[str, typer.Option()],
) -> None:
    """Show candidate upload decisions without exposing competition credentials."""
    client = _admin_client(coordinator_url)
    typer.echo(json.dumps(client.submission_gate(iteration_id), indent=2, sort_keys=True))


@app.command("decide-submission")
def decide_submission(
    coordinator_url: Annotated[str, typer.Option()],
    iteration_id: Annotated[str, typer.Option()],
    lane_id: Annotated[str, typer.Option()],
    decision: Annotated[Literal["SUBMITTED", "WITHHELD"], typer.Option()],
    approver: Annotated[str, typer.Option()],
    reason: Annotated[str, typer.Option()],
) -> None:
    """Record an immutable human upload decision for a verified candidate."""
    client = _admin_client(coordinator_url)
    result = client.decide_submission(
        iteration_id,
        {"lane_id": lane_id, "decision": decision, "approver": approver, "reason": reason},
    )
    typer.echo(json.dumps(result, sort_keys=True))


@app.command("record-leaderboard")
def record_leaderboard(
    coordinator_url: Annotated[str, typer.Option()],
    iteration_id: Annotated[str, typer.Option()],
    lane_id: Annotated[str, typer.Option()],
    score: Annotated[float, typer.Option()],
    recorded_by: Annotated[str, typer.Option()],
    rank: Annotated[int | None, typer.Option(min=1)] = None,
    local_supported: Annotated[bool, typer.Option()] = False,
    notes: Annotated[str, typer.Option()] = "",
) -> None:
    """Record leaderboard evidence for a candidate approved and uploaded by a human."""
    client = _admin_client(coordinator_url)
    result = client.record_leaderboard(
        iteration_id,
        {
            "lane_id": lane_id,
            "score": score,
            "rank": rank,
            "recorded_by": recorded_by,
            "local_supported": local_supported,
            "notes": notes,
        },
    )
    typer.echo(json.dumps(result, sort_keys=True))


@app.command("close-leaderboard")
def close_leaderboard(
    coordinator_url: Annotated[str, typer.Option()],
    iteration_id: Annotated[str, typer.Option()],
) -> None:
    """Close delayed leaderboard intake so local evidence can be merged."""
    client = _admin_client(coordinator_url)
    typer.echo(json.dumps(client.close_leaderboard(iteration_id), sort_keys=True))


@app.command("merge-knowledge")
def merge_iteration_knowledge(
    coordinator_url: Annotated[str, typer.Option()],
    iteration_id: Annotated[str, typer.Option()],
    snapshot_id: Annotated[str, typer.Option()],
) -> None:
    """Reconcile all terminal lanes into one immutable knowledge snapshot."""
    client = _admin_client(coordinator_url)
    result = client.merge_knowledge(iteration_id, snapshot_id)
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@app.command("fetch-knowledge")
def fetch_knowledge(
    coordinator_url: Annotated[str, typer.Option()],
    snapshot_id: Annotated[str, typer.Option()],
    output: Annotated[Path, typer.Option()] = Path(".swarmfold/knowledge-snapshot.json"),
) -> None:
    """Fetch a verified snapshot for portfolio planning in the next iteration."""
    from swarmfold.artifacts import write_json

    client = _admin_client(coordinator_url)
    path = write_json(output, client.get_knowledge_snapshot(snapshot_id))
    typer.echo(str(path.resolve()))


@app.command("evaluate-effectiveness")
def evaluate_effectiveness(
    coordinator_url: Annotated[str, typer.Option()],
    iteration_id: Annotated[str, typer.Option()],
) -> None:
    """Create evidence-derived swarm effectiveness metrics for an iteration."""
    client = _admin_client(coordinator_url)
    result = client.evaluate_effectiveness(iteration_id)
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@app.command("audit-log")
def audit_log(
    coordinator_url: Annotated[str, typer.Option()],
    limit: Annotated[int, typer.Option(min=1, max=1000)] = 100,
) -> None:
    """Read the hash-chained coordinator access log as an admin or auditor."""
    from swarmfold.coordinator_client import CoordinatorClient

    load_dotenv()
    token = os.getenv("SWARMFOLD_AUDITOR_TOKEN") or os.getenv("SWARMFOLD_COORDINATOR_TOKEN")
    if not token:
        typer.echo(
            "Configuration error: SWARMFOLD_AUDITOR_TOKEN or "
            "SWARMFOLD_COORDINATOR_TOKEN is required",
            err=True,
        )
        raise typer.Exit(code=2)
    client = CoordinatorClient(coordinator_url, token, ".swarmfold/audit-client")
    typer.echo(json.dumps(client.get_audit_events(limit), indent=2, sort_keys=True))


@app.command("migrate-record")
def migrate_versioned_record(
    record_type: Annotated[
        Literal["packet_manifest", "packet_completion", "knowledge_snapshot", "prompt"],
        typer.Option(),
    ],
    source: Annotated[Path, typer.Argument(exists=True, file_okay=True, dir_okay=False)],
    output: Annotated[Path, typer.Option()],
) -> None:
    """Migrate one versioned JSON record without modifying its source."""
    from swarmfold.artifacts import write_json
    from swarmfold.versioning import migrate_record

    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        typer.echo("Migration input must be a JSON object", err=True)
        raise typer.Exit(code=1)
    try:
        migrated = migrate_record(record_type, value)
    except ValueError as exc:
        typer.echo(f"Record migration failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(str(write_json(output, migrated).resolve()))


@app.command("capability-matrix")
def capability_matrix_command(
    format_type: Annotated[
        Literal["markdown", "json"], typer.Option("--format", "-f")
    ] = "markdown",
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
) -> None:
    """Generate or export the multi-dimensional SwarmFold capability matrix."""
    from swarmfold.capability_matrix import (
        format_json_matrix,
        format_markdown_matrix,
    )

    if format_type == "json":
        data = format_json_matrix()
        rendered = json.dumps(data, indent=2, sort_keys=True)
    else:
        rendered = format_markdown_matrix()

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
        typer.echo(f"Capability matrix written to {output.resolve()}")
    else:
        typer.echo(rendered)


@app.command("verify-package")
def verify_package_command(
    package: Annotated[
        Path,
        typer.Argument(exists=True, readable=True, help="Path to full-package.zip or package dir"),
    ],
) -> None:
    """Programmatically verify the integrity, portability, and safety of a full package."""
    from swarmfold.package_verifier import verify_portfolio_package

    try:
        report = verify_portfolio_package(package)
    except Exception as exc:
        typer.echo(f"Verification failed with error: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    typer.echo(json.dumps(report, indent=2, sort_keys=True))
    if not report.get("valid"):
        raise typer.Exit(code=1)


def _admin_client(coordinator_url: str):
    from swarmfold.coordinator_client import CoordinatorClient

    load_dotenv()
    token = os.getenv("SWARMFOLD_COORDINATOR_TOKEN")
    if not token:
        typer.echo("Configuration error: SWARMFOLD_COORDINATOR_TOKEN is required", err=True)
        raise typer.Exit(code=2)
    return CoordinatorClient(coordinator_url, token, ".swarmfold/admin-outbox")


if __name__ == "__main__":
    app()
