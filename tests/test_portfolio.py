import csv
import json
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from swarmfold.portfolio import LaneConfig, PortfolioConfig, build_full_package


def test_portfolio_requires_five_unique_lanes_and_one_enabled() -> None:
    lanes = [
        LaneConfig(lane_id=lane_id, enabled=False)
        for lane_id in ("ai-a", "ai-b", "ai-c", "ai-d", "ai-e")
    ]
    with pytest.raises(ValueError, match="Enable at least one"):
        PortfolioConfig(lanes=lanes)

    configured = PortfolioConfig(
        lanes=[
            lane.model_copy(update={"enabled": lane.lane_id in {"ai-b", "ai-d"}}) for lane in lanes
        ]
    )
    assert [lane.lane_id for lane in configured.enabled_lanes] == ["ai-b", "ai-d"]
    assert configured.total_experiment_budget == 10


def test_full_package_combines_lanes_and_adopts_better_oof_ensemble(tmp_path: Path) -> None:
    run_directory = tmp_path / "run"
    bundle = run_directory / "bundle"
    bundle.mkdir(parents=True)
    (bundle / "project.md").write_text("Competition brief", encoding="utf-8")

    lane_results = {
        "ai-a": _candidate_result(run_directory, "ai-a", 0.70, [0.1, 0.2, 0.6, 0.7], [0.2, 0.8]),
        "ai-b": _candidate_result(run_directory, "ai-b", 0.75, [0.2, 0.8, 0.4, 0.9], [0.4, 0.6]),
    }
    record = {
        "run_id": "a" * 32,
        "iteration_id": "iteration-1",
        "profile": {
            "task_type": "binary_classification",
            "metric": "roc_auc",
            "target_column": "target",
        },
        "max_experiments": 4,
        "lanes": [
            {
                "lane_id": lane_id,
                "name": lane_id.upper(),
                "enabled": lane_id in lane_results,
                "status": "ACCEPTED" if lane_id in lane_results else "DISABLED",
                "max_experiments": 2,
            }
            for lane_id in ("ai-a", "ai-b", "ai-c", "ai-d", "ai-e")
        ],
    }

    package = build_full_package(run_directory, record, lane_results)

    assert package["ensemble_adopted"] is True
    assert package["champion_lane"] == "ai-b"
    assert Path(package["path"]).is_file()
    with (run_directory / "full-package" / "submission.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert [float(row["prediction"]) for row in rows] == pytest.approx([0.3, 0.7])
    knowledge = (run_directory / "full-package" / "experiment-knowledge.md").read_text(
        encoding="utf-8"
    )
    assert "ai-a" in knowledge and "ai-b" in knowledge
    with zipfile.ZipFile(run_directory / "full-package.zip") as archive:
        names = set(archive.namelist())
    assert {
        "manifest.json",
        "experiment-knowledge.md",
        "submission.csv",
        "experiments/ai-a/packet/oof_predictions.parquet",
        "experiments/ai-b/packet/oof_predictions.parquet",
    }.issubset(names)


def test_full_package_rejects_ensemble_that_does_not_beat_champion(tmp_path: Path) -> None:
    run_directory = tmp_path / "run"
    (run_directory / "bundle").mkdir(parents=True)
    lane_results = {
        "ai-a": _candidate_result(
            run_directory,
            "ai-a",
            1.0,
            [0.1, 0.2, 0.8, 0.9],
            [0.2, 0.8],
        ),
        "ai-b": _candidate_result(
            run_directory,
            "ai-b",
            0.0,
            [0.9, 0.8, 0.2, 0.1],
            [0.8, 0.2],
        ),
    }
    record = {
        "run_id": "b" * 32,
        "iteration_id": "iteration-2",
        "profile": {
            "task_type": "binary_classification",
            "metric": "roc_auc",
            "target_column": "target",
        },
        "max_experiments": 4,
        "lanes": [
            {
                "lane_id": lane_id,
                "name": lane_id.upper(),
                "enabled": lane_id in lane_results,
                "status": "ACCEPTED" if lane_id in lane_results else "DISABLED",
                "max_experiments": 2,
            }
            for lane_id in ("ai-a", "ai-b", "ai-c", "ai-d", "ai-e")
        ],
    }

    package = build_full_package(run_directory, record, lane_results)
    assessment = json.loads(
        (run_directory / "full-package" / "ensemble-assessment.json").read_text(encoding="utf-8")
    )

    assert package["ensemble_adopted"] is False
    assert assessment["status"] == "REJECTED"
    assert assessment["improved_over_best_single"] is False
    assert (run_directory / "full-package" / "submission.csv").read_text(encoding="utf-8") == (
        run_directory / "source-packets" / "ai-a" / "submission.csv"
    ).read_text(encoding="utf-8")


def _candidate_result(
    run_directory: Path,
    lane_id: str,
    score: float,
    oof_predictions: list[float],
    test_predictions: list[float],
) -> dict[str, object]:
    packet = run_directory / "source-packets" / lane_id
    packet.mkdir(parents=True)
    targets = ["No", "No", "Yes", "Yes"]
    pq.write_table(
        pa.table(
            {
                "row": [str(index) for index in range(4)],
                "prediction": [str(value) for value in oof_predictions],
                "target": targets,
            }
        ),
        packet / "oof_predictions.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "row": ["0", "1"],
                "prediction": [str(value) for value in test_predictions],
            }
        ),
        packet / "test_predictions.parquet",
    )
    (packet / "submission.csv").write_text("record_id,prediction\n10,0\n11,0\n", encoding="utf-8")
    (packet / "contracts.json").write_text(
        json.dumps({"competition": {"prediction_column": "prediction"}}), encoding="utf-8"
    )
    (packet / "end-to-end.ipynb").write_text("{}", encoding="utf-8")
    (packet / "experiment-ledger.jsonl").write_text(
        json.dumps(
            {
                "experiment_id": f"{lane_id}-1",
                "model_family": "hist_gradient_boosting",
                "metrics": {"roc_auc": score},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (packet / "findings.jsonl").write_text(
        json.dumps(
            {
                "experiment_id": f"{lane_id}-1",
                "verdict": "IMPROVED",
                "recommendation": "Keep for ensemble assessment.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "status": "ACCEPTED",
        "completed_results": [
            {
                "experiment_id": f"{lane_id}-1",
                "model_family": "hist_gradient_boosting",
                "metrics": {"roc_auc": score},
            }
        ],
        "critiques": [],
        "packet": {
            "packet_type": "candidate",
            "packet_path": str(packet),
        },
    }
