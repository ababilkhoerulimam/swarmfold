from pathlib import Path

import pytest

from swarmfold.contracts import freeze_contracts
from swarmfold.quality import audit_temporal_validation, formulate_problem


def test_problem_formulation_detects_temporal_intent(tmp_path: Path) -> None:
    dataset = tmp_path / "history.csv"
    dataset.write_text(
        "month,value,target\n"
        + "".join(
            f"2025-{month:02d}-01,{month},{month % 2}\n" for month in range(1, 13)
        ),
        encoding="utf-8",
    )
    state = {
        "dataset_path": str(dataset),
        "target_column": "target",
        "project_context": "Forecast future demand as a time series.",
        "output_dir": str(tmp_path / "output"),
        "role_reports": {},
    }

    result = formulate_problem(state)

    assert result["problem_formulation"]["prediction_structure"] == "temporal"
    assert result["problem_formulation"]["recommended_validation_strategy"] == (
        "rolling_origin"
    )
    assert result["problem_formulation"]["execution_supported"] is False


def test_temporal_auditor_blocks_mismatched_holdout(tmp_path: Path) -> None:
    dataset = tmp_path / "history.csv"
    dataset.write_text(
        "month,value,target\n"
        + "".join(
            f"2025-{month:02d}-01,{month},{month % 2}\n" for month in range(1, 13)
        ),
        encoding="utf-8",
    )
    state = {
        "dataset_path": str(dataset),
        "target_column": "target",
        "project_context": "Forecast future demand as a time series.",
        "output_dir": str(tmp_path / "output"),
        "role_reports": {},
        "seed": 42,
        "test_size": 0.25,
        "metric_name": "accuracy",
    }
    state.update(formulate_problem(state))
    state.update(freeze_contracts(state))

    with pytest.raises(ValueError, match="does not match the formulated problem"):
        audit_temporal_validation(state)
