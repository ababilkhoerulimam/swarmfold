import hashlib
from pathlib import Path

from swarmfold.artifacts import sha256_file, sha256_json, write_json
from swarmfold.competition_bundle import inspect_competition_bundle
from swarmfold.data import inspect_classification_csv
from swarmfold.schemas import (
    CompetitionContract,
    ContractBundle,
    DataContract,
    ExperimentState,
    MetricContract,
    ValidationContract,
)


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
    metric_name = state.get("metric_name", "accuracy")
    metric_version = "roc-auc-v1" if metric_name == "roc_auc" else "accuracy-v1"
    row_count, feature_columns, target_classes = inspect_classification_csv(
        dataset_path, target_column
    )
    if len(target_classes) != 2:
        raise ValueError("The Phase 1 fixture contract requires exactly two target classes")

    dataset_sha256 = sha256_file(dataset_path)
    bundle_spec = None
    test_dataset_path = None
    sample_submission_path = None
    project_path = None
    bundle_hashes: dict[str, str] = {}
    if all(
        state.get(name)
        for name in ("test_dataset_path", "sample_submission_path", "project_path")
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
    split_material = f"{data_fingerprint}:{target_column}:{seed}:{test_size:.12g}"
    split_id = f"holdout-{hashlib.sha256(split_material.encode()).hexdigest()[:12]}"
    bundle = ContractBundle(
        competition=CompetitionContract(
            competition_id="swarmfold-fixture",
            task_type="binary_classification",
            target_column=target_column,
            submission_columns=(
                tuple(bundle_spec.submission_columns)
                if bundle_spec
                else ("row", "prediction")
            ),
            prediction_column=(bundle_spec.prediction_column if bundle_spec else "prediction"),
            id_columns=(bundle_spec.id_columns if bundle_spec else ["row"]),
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
        metric=MetricContract(metric_name=metric_name, metric_version=metric_version),
        validation=ValidationContract(split_id=split_id, seed=seed, test_size=test_size),
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
