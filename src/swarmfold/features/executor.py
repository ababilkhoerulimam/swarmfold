from __future__ import annotations

import contextlib
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectPercentile, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
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
    TargetEncoderFoldSafe,
    TextTfidfEncoder,
)


class FeaturePipelineSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    strategy: str = "raw_numeric"
    task_type: str = "binary_classification"
    use_one_hot: bool = False
    percentile: int | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class FeaturePipelineExecutor(BaseEstimator, TransformerMixin):
    """Executes fold-safe feature pipelines with zero validation leakage."""

    def __init__(
        self,
        strategy: str = "raw_numeric",
        task_type: str = "binary_classification",
        use_one_hot: bool = False,
        percentile: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> None:
        self.strategy = strategy
        self.task_type = task_type
        self.use_one_hot = use_one_hot
        self.percentile = percentile
        self.options = options or {}

        self.input_feature_count: int = 0
        self.output_feature_count: int = 0
        self.feature_names_in_: list[str] = []
        self.feature_names_out_: list[str] = []
        self.numeric_columns_: list[str] = []
        self.categorical_columns_: list[str] = []
        self.datetime_columns_: list[str] = []
        self.text_columns_: list[str] = []

        # Internal fitted components
        self.pipeline_: ColumnTransformer | None = None
        self.target_encoder_: TargetEncoderFoldSafe | None = None
        self.date_decomposer_: DateDecomposer | None = None
        self.cyclic_encoder_: CyclicTimeEncoder | None = None
        self.freq_encoder_: FrequencyEncoder | None = None
        self.interaction_poly_: InteractionPolynomial | None = None
        self.text_encoder_: TextTfidfEncoder | None = None
        self.selector_: SelectPercentile | None = None
        self.selected_mask_: np.ndarray | None = None

    def _prepare_df(self, X: Any) -> pd.DataFrame:
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        for col in self.numeric_columns_:
            if col not in df.columns:
                df[col] = np.nan
        for col in self.categorical_columns_:
            if col not in df.columns:
                df[col] = "__missing__"
            elif df[col].dtype != object:
                df[col] = df[col].astype(object)
        for col in self.text_columns_:
            if col not in df.columns:
                df[col] = ""
        return df

    def _inspect_columns(self, X: pd.DataFrame) -> None:
        self.feature_names_in_ = [str(c) for c in X.columns]
        self.input_feature_count = len(self.feature_names_in_)

        dt_cols = []
        for col in X.columns:
            if pd.api.types.is_datetime64_any_dtype(X[col]):
                dt_cols.append(col)
            elif self.strategy == "date_expanded" and (
                X[col].dtype == object or str(X[col].dtype) == "string"
            ):
                sample = X[col].dropna().head(10)
                if not sample.empty:
                    first_str = str(sample.iloc[0])
                    if len(first_str) >= 8 and first_str[:4].isdigit() and "-" in first_str:
                        try:
                            parsed = pd.to_datetime(sample, errors="coerce")
                            if parsed.notna().mean() >= 0.8:
                                dt_cols.append(col)
                        except Exception:
                            pass
        self.datetime_columns_ = dt_cols

        num_cols = X.select_dtypes(include="number").columns.tolist()
        self.numeric_columns_ = [c for c in num_cols if c not in dt_cols]
        cat_candidates = [
            c for c in X.columns if c not in self.numeric_columns_ and c not in dt_cols
        ]
        text_cols = []
        regular_cat_cols = []
        for col in cat_candidates:
            if self.strategy == "text_tfidf":
                text_cols.append(col)
            else:
                sample = X[col].dropna().astype(str).head(20)
                if not sample.empty and (
                    sample.str.len().mean() > 25 or sample.str.contains(" ").mean() > 0.3
                ):
                    text_cols.append(col)
                else:
                    regular_cat_cols.append(col)
        self.text_columns_ = text_cols
        self.categorical_columns_ = regular_cat_cols

    def fit(self, X: Any, y: Any = None) -> FeaturePipelineExecutor:
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self._inspect_columns(df)
        df = self._prepare_df(df)

        if (
            self.strategy in {"text_tfidf", "full_pipeline"} or self.text_columns_
        ) and self.text_columns_:
            self.text_encoder_ = TextTfidfEncoder(
                columns=self.text_columns_,
                **self.options.get("text_tfidf", {}),
            )
            self.text_encoder_.fit(df)

        if self.strategy == "target_encoded" and self.categorical_columns_ and y is not None:
            self.target_encoder_ = TargetEncoderFoldSafe(
                columns=self.categorical_columns_,
                task_type=self.task_type,
                **self.options.get("target_encoder", {}),
            )
            self.target_encoder_.fit(df[self.categorical_columns_], y)

        elif self.strategy == "frequency_encoded" and self.categorical_columns_:
            self.freq_encoder_ = FrequencyEncoder(
                columns=self.categorical_columns_,
                **self.options.get("frequency_encoder", {}),
            )
            self.freq_encoder_.fit(df[self.categorical_columns_])

        elif self.strategy == "date_expanded" and self.datetime_columns_:
            self.date_decomposer_ = DateDecomposer(date_columns=self.datetime_columns_)
            self.date_decomposer_.fit(df)
            self.cyclic_encoder_ = CyclicTimeEncoder()
            date_parts = self.date_decomposer_.transform(df)
            self.cyclic_encoder_.fit(date_parts)

        # Standard ColumnTransformer for numeric and remaining categorical
        transformers = []
        if self.numeric_columns_:
            numeric_steps = [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True))
            ]
            if self.strategy in {"scaled_numeric", "interaction_pairs", "full_pipeline"} or (
                self.use_one_hot and self.strategy == "raw_numeric"
            ):
                numeric_steps.append(("scaler", StandardScaler()))
            elif self.strategy == "power_transformed":
                numeric_steps.append(
                    ("power", PowerTransformer(method="yeo-johnson", standardize=True))
                )
            elif self.strategy == "robust_scaled":
                numeric_steps.append(("robust", RobustScaler()))
            transformers.append(("numeric", Pipeline(numeric_steps), self.numeric_columns_))

        cat_cols = [c for c in self.categorical_columns_]
        if self.strategy not in {"target_encoded", "frequency_encoded"} and cat_cols:
            if self.use_one_hot:
                encoder = OneHotEncoder(
                    handle_unknown="ignore", sparse_output=False, min_frequency=2
                )
            else:
                encoder = OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-2,
                )
            transformers.append(
                (
                    "categorical",
                    Pipeline(
                        [
                            (
                                "imputer",
                                SimpleImputer(strategy="constant", fill_value="__missing__"),
                            ),
                            ("encoder", encoder),
                        ]
                    ),
                    cat_cols,
                )
            )

        if transformers:
            self.pipeline_ = ColumnTransformer(transformers, remainder="drop")
            transformed_num_cat = self.pipeline_.fit_transform(df)
        else:
            self.pipeline_ = None
            transformed_num_cat = np.empty((len(df), 0))

        # Interaction polynomial on numeric if requested
        if self.strategy in {"interaction_pairs", "full_pipeline"} and self.numeric_columns_:
            self.interaction_poly_ = InteractionPolynomial(
                columns=self.numeric_columns_,
                max_features=self.options.get("max_interaction_features", 25),
            )
            self.interaction_poly_.fit(df[self.numeric_columns_])

        # Selection step
        percentile = self.percentile
        if self.strategy == "selected_50":
            percentile = 50
        elif self.strategy == "selected_75":
            percentile = 75

        if percentile is not None and y is not None:
            inter_mat = self._assemble_transformed(df, transformed_num_cat, is_fit=True, y=y)
            score_func = f_regression if self.task_type == "regression" else f_classif
            self.selector_ = SelectPercentile(score_func=score_func, percentile=percentile)
            target_arr = y.to_numpy() if isinstance(y, (pd.Series, pd.DataFrame)) else np.asarray(y)
            if self.task_type == "regression":
                target_arr = target_arr.astype(float)
            elif len(np.unique(target_arr)) == 2:
                classes = sorted(np.unique(target_arr))
                target_arr = (target_arr == classes[1]).astype(float)

            self.selector_.fit(inter_mat, target_arr)
            support = self.selector_.get_support()
            if not np.any(support):
                scores = np.nan_to_num(getattr(self.selector_, "scores_", np.array([])), nan=-1.0)
                best_idx = int(np.argmax(scores)) if len(scores) > 0 else 0
                support = np.zeros(inter_mat.shape[1], dtype=bool)
                support[min(best_idx, inter_mat.shape[1] - 1)] = True
            self.selected_mask_ = support

        final_mat = self.transform(df)
        self.output_feature_count = final_mat.shape[1]
        self.feature_names_out_ = [f"feat_{i}" for i in range(self.output_feature_count)]

        return self

    def _assemble_transformed(
        self,
        df: pd.DataFrame,
        base_transformed: np.ndarray,
        is_fit: bool = False,
        y: Any = None,
    ) -> np.ndarray:
        parts: list[np.ndarray] = []

        if base_transformed.shape[1] > 0:
            parts.append(
                base_transformed.toarray()
                if hasattr(base_transformed, "toarray")
                else np.asarray(base_transformed, dtype=float)
            )

        # Target encoding
        if self.strategy == "target_encoded" and self.target_encoder_ is not None:
            cat_df = df[self.categorical_columns_]
            if is_fit and y is not None:
                te_df = self.target_encoder_.fit_transform(cat_df, y)
            else:
                te_df = self.target_encoder_.transform(cat_df)
            parts.append(te_df.to_numpy(dtype=float))

        # Frequency encoding
        if self.strategy == "frequency_encoded" and self.freq_encoder_ is not None:
            fe_df = self.freq_encoder_.transform(df[self.categorical_columns_])
            parts.append(fe_df.to_numpy(dtype=float))

        # Date decomposition & cyclic
        if self.strategy == "date_expanded" and self.date_decomposer_ is not None:
            date_df = self.date_decomposer_.transform(df)
            parts.append(date_df.to_numpy(dtype=float))
            if self.cyclic_encoder_ is not None:
                cyclic_df = self.cyclic_encoder_.transform(date_df)
                parts.append(cyclic_df.to_numpy(dtype=float))

        # Interaction polynomial
        if self.interaction_poly_ is not None and self.numeric_columns_:
            inter_df = self.interaction_poly_.transform(df[self.numeric_columns_])
            if not inter_df.empty:
                parts.append(inter_df.to_numpy(dtype=float))

        # Text TF-IDF features
        if self.text_encoder_ is not None and self.text_columns_:
            text_df = self.text_encoder_.transform(df)
            if not text_df.empty:
                parts.append(text_df.to_numpy(dtype=float))

        if not parts:
            return np.zeros((len(df), 1), dtype=float)

        concatenated = np.hstack(parts)
        return np.nan_to_num(concatenated, nan=0.0, posinf=0.0, neginf=0.0)

    def transform(self, X: Any) -> np.ndarray:
        df = self._prepare_df(X)
        if self.pipeline_ is not None:
            base_mat = self.pipeline_.transform(df)
        else:
            base_mat = np.empty((len(df), 0))

        matrix = self._assemble_transformed(df, base_mat, is_fit=False)
        if self.selected_mask_ is not None and len(self.selected_mask_) == matrix.shape[1]:
            matrix = matrix[:, self.selected_mask_]
        elif self.selector_ is not None:
            with contextlib.suppress(Exception):
                matrix = self.selector_.transform(matrix)
        return matrix

    def fit_transform(self, X: Any, y: Any = None, **fit_params: Any) -> np.ndarray:
        df = self._prepare_df(X)
        self.fit(df, y)

        if self.pipeline_ is not None:
            base_mat = self.pipeline_.transform(df)
        else:
            base_mat = np.empty((len(df), 0))

        matrix = self._assemble_transformed(df, base_mat, is_fit=True, y=y)
        if self.selected_mask_ is not None and len(self.selected_mask_) == matrix.shape[1]:
            matrix = matrix[:, self.selected_mask_]
        elif self.selector_ is not None:
            with contextlib.suppress(Exception):
                matrix = self.selector_.transform(matrix)
        return matrix

    def get_feature_names_out(self) -> list[str]:
        return self.feature_names_out_
