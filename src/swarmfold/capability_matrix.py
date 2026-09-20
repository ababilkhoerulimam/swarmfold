from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Literal

CategoryType = Literal[
    "model_family", "feature_strategy", "validation_strategy", "task_adapter", "metric"
]


@dataclass(frozen=True)
class CapabilityEntry:
    category: CategoryType
    name: str
    description: str
    declared: bool
    installed: bool
    executable: bool
    scheduled: bool
    tested: bool
    ui_exposed: bool
    notes: str = ""

    @property
    def status_summary(self) -> str:
        if not self.installed:
            return "DECLARED (uninstalled dependency)"
        if not self.executable:
            return "INSTALLED (missing adapter)"
        if not self.scheduled:
            return "EXECUTABLE (unscheduled)"
        return "OPERATIONAL"


def build_capability_matrix() -> list[CapabilityEntry]:
    """Compile the complete multi-dimensional capability matrix for SwarmFold."""
    matrix: list[CapabilityEntry] = []

    # 1. Model Families
    model_defs = [
        (
            "logistic_regression",
            "Regularized linear classifier/regressor",
            True,
            True,
            True,
            True,
            True,
            True,
            "Scikit-learn baseline",
        ),
        (
            "random_forest",
            "Ensemble of bagging decision trees",
            True,
            True,
            True,
            True,
            True,
            True,
            "Scikit-learn ensemble",
        ),
        (
            "hist_gradient_boosting",
            "Histogram-based gradient boosted trees",
            True,
            True,
            True,
            True,
            True,
            True,
            "Native scikit-learn boosting",
        ),
        (
            "extra_trees",
            "Extremely randomized ensemble trees",
            True,
            True,
            True,
            True,
            True,
            True,
            "High prediction diversity",
        ),
        (
            "sgd_log_loss",
            "Online linear model with SGD optimization",
            True,
            True,
            True,
            True,
            True,
            True,
            "Fast linear baseline",
        ),
        (
            "bagging",
            "Intra-lane bootstrap aggregating ensemble",
            True,
            True,
            True,
            True,
            True,
            True,
            "Self-contained bagging",
        ),
        (
            "soft_voting",
            "Probability-weighted voting ensemble",
            True,
            True,
            True,
            True,
            True,
            True,
            "Probability consensus blend",
        ),
        (
            "stacking",
            "Meta-learner stacked generalization",
            True,
            True,
            True,
            True,
            True,
            True,
            "Ridge meta-learner stack",
        ),
        (
            "lightgbm",
            "Fast, distributed gradient boosting framework",
            True,
            find_spec("lightgbm") is not None,
            True,
            True,
            True,
            True,
            "LightGBM integration",
        ),
        (
            "xgboost",
            "Scalable, regularized gradient boosted trees",
            True,
            find_spec("xgboost") is not None,
            True,
            True,
            True,
            True,
            "XGBoost integration",
        ),
        (
            "catboost",
            "Gradient boosting with categorical feature support",
            True,
            find_spec("catboost") is not None,
            True,
            True,
            True,
            True,
            "CatBoost integration",
        ),
    ]
    for name, desc, dec, inst, exe, sched, test, ui, note in model_defs:
        matrix.append(
            CapabilityEntry(
                category="model_family",
                name=name,
                description=desc,
                declared=dec,
                installed=inst,
                executable=exe and inst,
                scheduled=sched and inst,
                tested=test,
                ui_exposed=ui,
                notes=note,
            )
        )

    # 2. Feature Strategies
    feature_defs = [
        (
            "raw_numeric",
            "Pass raw features without transformation",
            True,
            True,
            True,
            True,
            True,
            True,
            "Zero-transformation baseline",
        ),
        (
            "scaled_numeric",
            "Standard scaling for numeric columns",
            True,
            True,
            True,
            True,
            True,
            True,
            "Zero-mean unit-variance",
        ),
        (
            "robust_scaled",
            "Robust quantile-based feature scaling",
            True,
            True,
            True,
            True,
            True,
            True,
            "Outlier-resistant scaling",
        ),
        (
            "target_encoded",
            "Target-aware categorical mean encoding",
            True,
            True,
            True,
            True,
            True,
            True,
            "Fold-safe target encoder",
        ),
        (
            "date_expanded",
            "Datetime cyclical and calendar expansion",
            True,
            True,
            True,
            True,
            True,
            True,
            "Calendar feature derivation",
        ),
        (
            "interaction_pairs",
            "Pairwise numeric interaction features",
            True,
            True,
            True,
            True,
            True,
            True,
            "Degree-2 multiplicative pairs",
        ),
        (
            "power_transformed",
            "Yeo-Johnson power transformation",
            True,
            True,
            True,
            True,
            True,
            True,
            "Distribution normalization",
        ),
        (
            "pca_projection",
            "Principal component dimensionality projection",
            True,
            True,
            False,
            False,
            False,
            False,
            "Linear subspace reduction",
        ),
        (
            "text_tfidf",
            "Fold-safe n-gram sublinear TF-IDF encoder",
            True,
            True,
            True,
            True,
            True,
            True,
            "Text corpus feature extraction",
        ),
        (
            "polynomial_features",
            "Polynomial expansion with degrees 2-3",
            True,
            True,
            False,
            False,
            False,
            False,
            "Higher-order interactions",
        ),
    ]
    for name, desc, dec, inst, exe, sched, test, ui, note in feature_defs:
        matrix.append(
            CapabilityEntry(
                category="feature_strategy",
                name=name,
                description=desc,
                declared=dec,
                installed=inst,
                executable=exe,
                scheduled=sched,
                tested=test,
                ui_exposed=ui,
                notes=note,
            )
        )

    # 3. Validation Splitters
    val_defs = [
        (
            name,
            description,
            True,
            True,
            True,
            True,
            True,
            True,
            note,
        )
        for name, description, note in (
            ("holdout", "Random holdout validation", "Ordinary regression holdout"),
            (
                "stratified_holdout",
                "Stratified holdout validation",
                "Preserves target distribution in classification",
            ),
            (
                "stratified_kfold",
                "Stratified K-fold cross-validation",
                "Classification fold partitioning",
            ),
            (
                "repeated_stratified_kfold",
                "Repeated stratified K-fold cross-validation",
                "Variance-aware imbalanced classification validation",
            ),
            ("kfold", "Standard K-fold cross-validation", "Regression fold partitioning"),
            ("group_kfold", "Entity-aware K-fold", "Prevents entity leakage"),
            (
                "stratified_group_kfold",
                "Stratified entity-aware K-fold",
                "Balances labels while isolating entities",
            ),
            (
                "expanding_walk_forward",
                "Expanding walk-forward validation",
                "Past-to-future temporal validation",
            ),
            (
                "purged_time_series",
                "Purged temporal validation with embargo",
                "Prevents boundary lookahead leakage",
            ),
        )
    ]
    for name, desc, dec, inst, exe, sched, test, ui, note in val_defs:
        matrix.append(
            CapabilityEntry(
                category="validation_strategy",
                name=name,
                description=desc,
                declared=dec,
                installed=inst,
                executable=exe,
                scheduled=sched,
                tested=test,
                ui_exposed=ui,
                notes=note,
            )
        )

    # 4. Task Adapters
    task_defs = [
        (
            "binary_classification",
            "Two-class probability prediction",
            True,
            True,
            True,
            True,
            True,
            True,
            "Binary probability calibration",
        ),
        (
            "multiclass_classification",
            "Multi-category probability distribution",
            True,
            True,
            True,
            True,
            True,
            True,
            "Multinomial probability simplex",
        ),
        (
            "regression",
            "Continuous numeric target prediction",
            True,
            True,
            True,
            True,
            True,
            True,
            "Bounded continuous inference",
        ),
        (
            "multilabel_classification",
            "Multi-label binary indicator matrix",
            True,
            True,
            False,
            False,
            True,
            False,
            "Independent label scoring",
        ),
        (
            "time_series_forecasting",
            "Sequential multi-horizon forecasting",
            True,
            True,
            False,
            False,
            True,
            False,
            "Temporal validation and lag features",
        ),
    ]
    for name, desc, dec, inst, exe, sched, test, ui, note in task_defs:
        matrix.append(
            CapabilityEntry(
                category="task_adapter",
                name=name,
                description=desc,
                declared=dec,
                installed=inst,
                executable=exe,
                scheduled=sched,
                tested=test,
                ui_exposed=ui,
                notes=note,
            )
        )

    # 5. Evaluation Metrics
    metric_defs = [
        (
            "accuracy",
            "Fraction of correct classification predictions",
            True,
            True,
            True,
            True,
            True,
            True,
            "Higher is better (max 1.0)",
        ),
        (
            "roc_auc",
            "Area under receiver operating characteristic curve",
            True,
            True,
            True,
            True,
            True,
            True,
            "Higher is better (max 1.0)",
        ),
        (
            "log_loss",
            "Cross-entropy loss over predicted probabilities",
            True,
            True,
            True,
            True,
            True,
            True,
            "Lower is better (min 0.0)",
        ),
        (
            "f1_macro",
            "Unweighted mean F1 score across classes",
            True,
            True,
            True,
            True,
            True,
            True,
            "Higher is better (max 1.0)",
        ),
        (
            "multilabel_f1",
            "Macro-averaged F1 score over multiple labels",
            True,
            True,
            True,
            True,
            True,
            True,
            "Higher is better (max 1.0)",
        ),
        (
            "rmse",
            "Root mean squared error for regression",
            True,
            True,
            True,
            True,
            True,
            True,
            "Lower is better (min 0.0)",
        ),
        (
            "mae",
            "Mean absolute error for regression",
            True,
            True,
            True,
            True,
            True,
            True,
            "Lower is better (min 0.0)",
        ),
        (
            "mape",
            "Mean absolute percentage error for regression",
            True,
            True,
            True,
            True,
            True,
            True,
            "Lower is better (min 0.0)",
        ),
        (
            "r2",
            "Coefficient of determination for regression",
            True,
            True,
            True,
            True,
            True,
            True,
            "Higher is better (max 1.0)",
        ),
    ]
    for name, desc, dec, inst, exe, sched, test, ui, note in metric_defs:
        matrix.append(
            CapabilityEntry(
                category="metric",
                name=name,
                description=desc,
                declared=dec,
                installed=inst,
                executable=exe,
                scheduled=sched,
                tested=test,
                ui_exposed=ui,
                notes=note,
            )
        )

    return matrix


def format_markdown_matrix(matrix: list[CapabilityEntry] | None = None) -> str:
    """Render the capability matrix as a clean GitHub-flavored markdown document."""
    items = matrix if matrix is not None else build_capability_matrix()
    categories: dict[CategoryType, tuple[str, list[CapabilityEntry]]] = {
        "model_family": ("Model Families", []),
        "feature_strategy": ("Feature Engineering Strategies", []),
        "validation_strategy": ("Validation Splitters", []),
        "task_adapter": ("Task Adapters", []),
        "metric": ("Evaluation Metrics", []),
    }

    for item in items:
        if item.category in categories:
            categories[item.category][1].append(item)

    lines = [
        "# SwarmFold Capability Matrix",
        "",
        "This matrix defines the operational status of all modeling, feature, validation, "
        "task, and metric components in SwarmFold across 6 lifecycle dimensions:",
        "- **Declared**: Defined in contract schemas and architecture specifications.",
        "- **Installed**: Underlying libraries and dependencies are present in the environment.",
        "- **Executable**: Runtime executor or adapter is wired, tested, and active.",
        "- **Scheduled**: Candidate factory and planners can assign work to this component.",
        "- **Tested**: Verified with automated hermetic unit and integration test fixtures.",
        "- **UI Exposed**: Accessible, configurable, or visualized in the web dashboard.",
        "",
    ]

    for _, (cat_title, cat_items) in categories.items():
        lines.append(f"## {cat_title}")
        lines.append("")
        lines.append(
            "| Component | Status | Declared | Installed | Executable | "
            "Scheduled | Tested | UI Exposed | Description |"
        )
        lines.append(
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |"
        )
        for item in cat_items:

            def mark(val: bool) -> str:
                return "✅" if val else "❌"

            lines.append(
                f"| `{item.name}` | **{item.status_summary}** | {mark(item.declared)} | "
                f"{mark(item.installed)} | {mark(item.executable)} | {mark(item.scheduled)} | "
                f"{mark(item.tested)} | {mark(item.ui_exposed)} | {item.description} |"
            )
        lines.append("")

    return "\n".join(lines)


def format_json_matrix(
    matrix: list[CapabilityEntry] | None = None,
) -> dict[str, object]:
    """Export the capability matrix as a structured dictionary."""
    items = matrix if matrix is not None else build_capability_matrix()
    categories: dict[str, list[dict[str, object]]] = {}
    operational_count = 0
    for item in items:
        categories.setdefault(item.category, []).append(asdict(item))
        if item.status_summary == "OPERATIONAL":
            operational_count += 1
    return {
        "schema_version": "capability-matrix-v1",
        "total_capabilities": len(items),
        "operational_count": operational_count,
        "categories": categories,
    }


def export_capability_matrix_file(output_path: Path) -> Path:
    """Write the capability matrix markdown document to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = format_markdown_matrix()
    output_path.write_text(content, encoding="utf-8")
    return output_path
