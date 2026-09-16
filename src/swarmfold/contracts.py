import hashlib
from pathlib import Path

from swarmfold.artifacts import sha256_file, sha256_json, write_json
from swarmfold.competition_bundle import inspect_competition_bundle
from swarmfold.data import infer_task_type_csv, inspect_training_csv
from swarmfold.metrics import default_metric_for_task, get_metric_spec
from swarmfold.schemas import (
    CompetitionContract,
    ContractBundle,
    DataContract,
    ExperimentState,
    MetricContract,
    ValidationContract,
)
from swarmfold.tasks import get_task_adapter


def contract_bundle_sha256(bundle: ContractBundle) -> str:
    payload = bundle.model_dump(
        exclude={
            "contract_sha256": True,
            "data": {
                "dataset_path",
                "test_dataset_path",
                "sample_submission_path",
                "project_path",
            },
        }
    )
    return sha256_json(payload)


def freeze_contracts(state: ExperimentState) -> ExperimentState:
    dataset_path = Path(state["dataset_path"]).resolve()
    target_column = state.get("target_column", "target")
    seed = state.get("seed", 42)
    test_size = state.get("test_size", 0.25)
    formulation = state.get("problem_formulation", {})
    task_type = str(
        formulation.get("task_type")
        if isinstance(formulation, dict) and formulation.get("task_type")
        else infer_task_type_csv(dataset_path, target_column)
    )
    metric_name = str(state.get("metric_name") or default_metric_for_task(task_type))
    metric_spec = get_metric_spec(metric_name)
    metric_spec.validate_task(task_type)
    row_count, feature_columns, target_classes = inspect_training_csv(
        dataset_path, target_column, task_type
    )
    shared_contracts = None
    if "contracts" in state:
        shared_contracts = ContractBundle.model_validate(state["contracts"])

    custom_id_col = state.get("id_column") or (
        state.get("id_columns", [None])[0] if isinstance(state.get("id_columns"), list) else None
    ) or (
        shared_contracts.competition.id_columns[0]
        if shared_contracts
        and shared_contracts.competition.id_columns
        and shared_contracts.competition.id_columns != ["row"]
        else None
    )
    if custom_id_col and custom_id_col in feature_columns:
        feature_columns = [c for c in feature_columns if c != custom_id_col]
    task_adapter = get_task_adapter(task_type)
    task_adapter.validate_target_classes(target_classes)

    dataset_sha256 = sha256_file(dataset_path)
    bundle_spec = None
    test_dataset_path = None
    sample_submission_path = None
    project_path = None
    bundle_hashes: dict[str, str] = {}
    if all(
        state.get(name) for name in ("test_dataset_path", "sample_submission_path", "project_path")
    ):
        test_dataset_path = Path(state["test_dataset_path"]).resolve()
        sample_submission_path = Path(state["sample_submission_path"]).resolve()
        project_path = Path(state["project_path"]).resolve()
        bundle_spec = inspect_competition_bundle(
            dataset_path,
            test_dataset_path,
            sample_submission_path,
            project_path,
            target_column=target_column,
        )
        if bundle_spec.target_column != target_column:
            raise ValueError("Inferred bundle target does not match the requested target")
        bundle_hashes = {
            "test_dataset_sha256": sha256_file(test_dataset_path),
            "sample_submission_sha256": sha256_file(sample_submission_path),
            "project_sha256": sha256_file(project_path),
        }
    if "contracts" in state:
        shared = ContractBundle.model_validate(state["contracts"])
        if shared.contract_sha256 != contract_bundle_sha256(shared):
            raise ValueError("Shared contract bundle hash is invalid")
        mismatches = []
        if shared.data.dataset_sha256 != dataset_sha256:
            mismatches.append("dataset_sha256")
        if shared.data.row_count != row_count:
            mismatches.append("row_count")
        if shared.data.feature_columns != feature_columns:
            mismatches.append("feature_columns")
        if shared.data.target_classes != target_classes:
            mismatches.append("target_classes")
        if shared.data.target_column != target_column:
            mismatches.append("target_column")
        if shared.competition.task_type != task_type:
            mismatches.append("task_type")
        if shared.metric.metric_name != metric_name:
            mismatches.append("metric_name")
        for field, value in bundle_hashes.items():
            if getattr(shared.data, field) != value:
                mismatches.append(field)
        if shared.validation.seed != seed or shared.validation.test_size != test_size:
            mismatches.append("validation")
        if mismatches:
            raise ValueError(f"Local data does not match shared contracts: {', '.join(mismatches)}")
        bound_data = shared.data.model_copy(
            update={
                "dataset_path": str(dataset_path),
                "test_dataset_path": str(test_dataset_path) if test_dataset_path else None,
                "sample_submission_path": (
                    str(sample_submission_path) if sample_submission_path else None
                ),
                "project_path": str(project_path) if project_path else None,
            }
        )
        return _write_contracts(state, shared.model_copy(update={"data": bound_data}))

    data_fingerprint = sha256_json({"train": dataset_sha256, **bundle_hashes})

    val_conf = state.get("validation", {}) if isinstance(state.get("validation"), dict) else {}
    strategy = (
        state.get("validation_strategy")
        or val_conf.get("strategy")
        or ("holdout" if task_type == "regression" else "stratified_holdout")
    )
    strategy_aliases = {
        "rolling_origin": "expanding_walk_forward",
        "walk_forward": "expanding_walk_forward",
        "purged_walk_forward": "purged_time_series",
        "purged_kfold_embargo": "purged_time_series",
    }
    validation_strategy = strategy_aliases.get(strategy, strategy)

    splits = int(state.get("splits") or val_conf.get("splits") or 5)
    repeats = int(state.get("repeats") or val_conf.get("repeats") or 1)
    group_col = (
        state.get("group_column")
        or val_conf.get("group_column")
        or (formulation.get("group_column") if isinstance(formulation, dict) else None)
    )
    time_col = (
        state.get("time_column")
        or val_conf.get("time_column")
        or (formulation.get("time_column") if isinstance(formulation, dict) else None)
    )
    purge_win = int(
        state.get("purge_window")
        or val_conf.get("purge_window")
        or (formulation.get("purge_window") if isinstance(formulation, dict) else 0)
        or 0
    )
    embargo_win = int(
        state.get("embargo_window")
        or val_conf.get("embargo_window")
        or (formulation.get("embargo_window") if isinstance(formulation, dict) else 0)
        or 0
    )

    if validation_strategy in ("group_kfold", "stratified_group_kfold") and (
        not group_col or group_col not in feature_columns
    ):
        raise ValueError(
            f"Strategy '{validation_strategy}' requires group_column in dataset features: "
            f"'{group_col}' not found"
        )

    if (
        validation_strategy in ("expanding_walk_forward", "purged_time_series")
        and time_col
        and time_col not in feature_columns
    ):
        raise ValueError(
            f"Strategy '{validation_strategy}' time_column '{time_col}' not found in features"
        )

    split_material = (
        f"{data_fingerprint}:{target_column}:{seed}:{test_size:.12g}:"
        f"{validation_strategy}:{splits}:{repeats}:{group_col}:{time_col}:"
        f"{purge_win}:{embargo_win}"
    )
    strat_slug = validation_strategy.replace("_", "-")
    split_id = f"{strat_slug}-{hashlib.sha256(split_material.encode()).hexdigest()[:12]}"

    bundle = ContractBundle(
        competition=CompetitionContract(
            competition_id="swarmfold-fixture",
            task_type=task_type,
            target_column=target_column,
            submission_columns=(
                tuple(bundle_spec.submission_columns) if bundle_spec else ("row", "prediction")
            ),
            prediction_column=(bundle_spec.prediction_column if bundle_spec else "prediction"),
            id_columns=(
                [custom_id_col]
                if custom_id_col
                else (bundle_spec.id_columns if bundle_spec else ["row"])
            ),
        ),
        data=DataContract(
            data_version=f"sha256:{data_fingerprint if bundle_spec else dataset_sha256}",
            dataset_path=str(dataset_path),
            dataset_sha256=dataset_sha256,
            row_count=row_count,
            feature_columns=feature_columns,
            target_column=target_column,
            target_classes=target_classes,
            test_dataset_path=str(test_dataset_path) if test_dataset_path else None,
            test_dataset_sha256=bundle_hashes.get("test_dataset_sha256"),
            sample_submission_path=(
                str(sample_submission_path) if sample_submission_path else None
            ),
            sample_submission_sha256=bundle_hashes.get("sample_submission_sha256"),
            project_path=str(project_path) if project_path else None,
            project_sha256=bundle_hashes.get("project_sha256"),
            test_row_count=bundle_spec.test_rows if bundle_spec else None,
        ),
        metric=MetricContract(
            metric_name=metric_name,
            metric_version=metric_spec.version,
            higher_is_better=metric_spec.higher_is_better,
            minimum=metric_spec.minimum,
            maximum=metric_spec.maximum,
        ),
        validation=ValidationContract(
            split_id=split_id,
            strategy=validation_strategy,
            seed=seed,
            test_size=test_size,
            splits=splits,
            repeats=repeats,
            group_column=group_col,
            time_column=time_col,
            purge_window=purge_win,
            embargo_window=embargo_win,
        ),
    )
    return _write_contracts(state, bundle)


def _write_contracts(state: ExperimentState, bundle: ContractBundle) -> ExperimentState:
    output_root = Path(state["output_dir"])
    unsigned_path = output_root / "contracts" / "contracts.json"
    unsigned_contracts = bundle.model_dump(exclude={"contract_sha256"})
    write_json(unsigned_path, unsigned_contracts)
    contract_sha256 = bundle.contract_sha256 or contract_bundle_sha256(bundle)
    signed_bundle = bundle.model_copy(update={"contract_sha256": contract_sha256})
    contracts_path = write_json(
        output_root / "contracts" / "bundle.json", signed_bundle.model_dump()
    )
    return {
        "contracts": signed_bundle.model_dump(),
        "contracts_path": str(contracts_path.resolve()),
    }
