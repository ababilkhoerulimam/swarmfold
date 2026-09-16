from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from sklearn.feature_selection import SelectPercentile, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)

from swarmfold.features.transformers import (
    CyclicTimeEncoder,
    DateDecomposer,
    FrequencyEncoder,
    InteractionPolynomial,
    MissingIndicatorAdder,
    QuantileBinner,
    TargetEncoderFoldSafe,
    TextTfidfEncoder,
)


class FeatureTransformerSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    family: str
    description: str
    supported_dtypes: list[str] = Field(default_factory=lambda: ["numeric"])
    requires_target: bool = False
    leakage_boundary: str = "Fit strictly inside training fold."
    default_parameters: dict[str, Any] = Field(default_factory=dict)


class FeatureRegistry:
    """Registry cataloging available fold-safe feature transformers."""

    def __init__(self) -> None:
        self._specs: dict[str, FeatureTransformerSpec] = {}
        self._builders: dict[str, Callable[..., Any]] = {}
        self._register_defaults()

    def register(
        self,
        spec: FeatureTransformerSpec,
        builder: Callable[..., Any],
    ) -> None:
        self._specs[spec.name] = spec
        self._builders[spec.name] = builder

    def get_spec(self, name: str) -> FeatureTransformerSpec:
        if name not in self._specs:
            raise KeyError(f"Feature transformer '{name}' is not registered in FeatureRegistry")
        return self._specs[name]

    def is_registered(self, name: str) -> bool:
        return name in self._specs

    def create(self, name: str, **kwargs: Any) -> Any:
        if name not in self._builders:
            raise KeyError(f"Feature transformer '{name}' is not registered in FeatureRegistry")
        spec = self._specs[name]
        params = dict(spec.default_parameters)
        params.update(kwargs)
        return self._builders[name](**params)

    def list_specs(self, family: str | None = None) -> list[FeatureTransformerSpec]:
        specs = list(self._specs.values())
        if family is not None:
            specs = [s for s in specs if s.family == family]
        return sorted(specs, key=lambda s: (s.family, s.name))

    def _register_defaults(self) -> None:
        # 1. Imputation
        self.register(
            FeatureTransformerSpec(
                name="simple_imputer_median",
                family="imputation",
                description="Median imputation for numeric features.",
                supported_dtypes=["numeric"],
                default_parameters={"strategy": "median"},
            ),
            lambda **params: SimpleImputer(**params, keep_empty_features=True),
        )
        self.register(
            FeatureTransformerSpec(
                name="simple_imputer_constant",
                family="imputation",
                description="Constant '__missing__' imputation for categoricals.",
                supported_dtypes=["categorical"],
                default_parameters={"strategy": "constant", "fill_value": "__missing__"},
            ),
            lambda **params: SimpleImputer(**params),
        )
        self.register(
            FeatureTransformerSpec(
                name="missing_indicator",
                family="imputation",
                description="Binary indicator columns for missing values.",
                supported_dtypes=["all"],
            ),
            lambda **params: MissingIndicatorAdder(**params),
        )

        # 2. Scaling & Transformations
        self.register(
            FeatureTransformerSpec(
                name="standard_scaler",
                family="scaling",
                description="Standardize features to zero mean and unit variance.",
                supported_dtypes=["numeric"],
            ),
            lambda **params: StandardScaler(**params),
        )
        self.register(
            FeatureTransformerSpec(
                name="robust_scaler",
                family="scaling",
                description="Scale features using statistics robust to outliers.",
                supported_dtypes=["numeric"],
            ),
            lambda **params: RobustScaler(**params),
        )
        self.register(
            FeatureTransformerSpec(
                name="power_transformer",
                family="transformation",
                description="Yeo-Johnson parametric power transformation to stabilize variance.",
                supported_dtypes=["numeric"],
                default_parameters={"method": "yeo-johnson", "standardize": True},
            ),
            lambda **params: PowerTransformer(**params),
        )
        self.register(
            FeatureTransformerSpec(
                name="quantile_binner",
                family="transformation",
                description="Non-linear discretization into quantile bins.",
                supported_dtypes=["numeric"],
                default_parameters={"n_bins": 5},
            ),
            lambda **params: QuantileBinner(**params),
        )

        # 3. Categorical Encodings
        self.register(
            FeatureTransformerSpec(
                name="one_hot_encoder",
                family="encoding",
                description="One-hot dummy encoding with missing frequency threshold.",
                supported_dtypes=["categorical"],
                default_parameters={
                    "handle_unknown": "ignore",
                    "sparse_output": False,
                    "min_frequency": 2,
                },
            ),
            lambda **params: OneHotEncoder(**params),
        )
        self.register(
            FeatureTransformerSpec(
                name="ordinal_encoder",
                family="encoding",
                description="Ordinal integer encoding preserving unknown and missing values.",
                supported_dtypes=["categorical"],
                default_parameters={
                    "handle_unknown": "use_encoded_value",
                    "unknown_value": -1,
                    "encoded_missing_value": -2,
                },
            ),
            lambda **params: OrdinalEncoder(**params),
        )
        self.register(
            FeatureTransformerSpec(
                name="target_encoder",
                family="encoding",
                description="Fold-safe smoothed out-of-fold target encoding.",
                supported_dtypes=["categorical"],
                requires_target=True,
                leakage_boundary="Inner fold CV on train fold only; validation targets isolated.",
                default_parameters={"smoothing": 10.0, "cv_folds": 5},
            ),
            lambda **params: TargetEncoderFoldSafe(**params),
        )
        self.register(
            FeatureTransformerSpec(
                name="frequency_encoder",
                family="encoding",
                description="Normalized value frequency categorical encoding.",
                supported_dtypes=["categorical"],
                default_parameters={"normalize": True},
            ),
            lambda **params: FrequencyEncoder(**params),
        )

        # 4. Temporal Features
        self.register(
            FeatureTransformerSpec(
                name="date_decomposer",
                family="temporal",
                description="Extracts year, month, day, dayofweek, is_weekend, quarter.",
                supported_dtypes=["datetime", "categorical"],
            ),
            lambda **params: DateDecomposer(**params),
        )
        self.register(
            FeatureTransformerSpec(
                name="cyclic_time_encoder",
                family="temporal",
                description="Sin/Cos cyclical projections for periodic calendar features.",
                supported_dtypes=["numeric", "temporal"],
            ),
            lambda **params: CyclicTimeEncoder(**params),
        )

        # 5. Interactions
        self.register(
            FeatureTransformerSpec(
                name="interaction_polynomial",
                family="interaction",
                description="Pairwise 2nd-degree polynomial interactions with feature capping.",
                supported_dtypes=["numeric"],
                default_parameters={"max_features": 30, "interaction_only": True},
            ),
            lambda **params: InteractionPolynomial(**params),
        )

        # 6. Feature Selection
        self.register(
            FeatureTransformerSpec(
                name="univariate_selector",
                family="selection",
                description="Univariate ANOVA F-value feature percentile selection.",
                supported_dtypes=["numeric"],
                requires_target=True,
                leakage_boundary="Score function evaluated strictly on training fold.",
                default_parameters={"percentile": 50, "task_type": "binary_classification"},
            ),
            lambda **params: self._build_univariate_selector(**params),
        )
        # 8. Text Features
        self.register(
            FeatureTransformerSpec(
                name="text_tfidf",
                family="text",
                description="Fold-safe TF-IDF text feature extraction.",
                supported_dtypes=["text"],
                requires_target=False,
                leakage_boundary="Vocabulary fitted strictly inside training fold.",
                default_parameters={"max_features": 50, "ngram_range": (1, 2), "min_df": 2},
            ),
            lambda **params: TextTfidfEncoder(**params),
        )

    def _build_univariate_selector(
        self,
        percentile: int = 50,
        task_type: str = "binary_classification",
        **kwargs: Any,
    ) -> SelectPercentile:
        score_func = f_regression if task_type == "regression" else f_classif
        return SelectPercentile(score_func=score_func, percentile=percentile)


_GLOBAL_FEATURE_REGISTRY: FeatureRegistry | None = None


def get_feature_registry() -> FeatureRegistry:
    global _GLOBAL_FEATURE_REGISTRY
    if _GLOBAL_FEATURE_REGISTRY is None:
        _GLOBAL_FEATURE_REGISTRY = FeatureRegistry()
    return _GLOBAL_FEATURE_REGISTRY
