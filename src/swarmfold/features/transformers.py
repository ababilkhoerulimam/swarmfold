from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer, PolynomialFeatures


class TargetEncoderFoldSafe(BaseEstimator, TransformerMixin):
    """Fold-safe out-of-fold smoothed target encoder.

    Uses inner cross-validation during fit_transform on the training fold to
    guarantee that training targets are never used in-sample without isolation.
    Uses smoothed empirical Bayes / m-estimate weighting:
        encoded = (count * mean + m * global_mean) / (count + m)
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        smoothing: float = 10.0,
        cv_folds: int = 5,
        task_type: str = "binary_classification",
    ) -> None:
        self.columns = columns
        self.smoothing = float(max(smoothing, 1e-6))
        self.cv_folds = int(max(cv_folds, 2))
        self.task_type = task_type
        self.encodings_: dict[str, dict[Any, float]] = {}
        self.global_means_: dict[str, float] = {}
        self.feature_names_out_: list[str] = []

    def _prepare_target(self, y: Any) -> np.ndarray:
        vals = y.to_numpy() if isinstance(y, (pd.Series, pd.DataFrame)) else np.asarray(y)

        if self.task_type == "regression":
            return vals.astype(float)

        # For classification, encode binary labels to 0.0 and 1.0
        unique_labels = sorted(set(vals.ravel()))
        if len(unique_labels) == 2:
            pos_label = unique_labels[1]
            return (vals == pos_label).astype(float).ravel()

        # For multiclass, convert to integer codes
        label_map = {lbl: float(idx) for idx, lbl in enumerate(unique_labels)}
        return np.vectorize(label_map.get)(vals).astype(float).ravel()

    def fit(self, X: Any, y: Any) -> TargetEncoderFoldSafe:
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        cols = self.columns or df.columns.tolist()
        y_arr = self._prepare_target(y)
        global_mean = float(np.nanmean(y_arr))

        self.encodings_ = {}
        self.global_means_ = {}
        self.feature_names_out_ = [f"{col}_te" for col in cols]

        for col in cols:
            series = df[col].astype(str)
            col_df = pd.DataFrame({"val": series, "target": y_arr})
            grouped = col_df.groupby("val", dropna=False)["target"]
            counts = grouped.count()
            means = grouped.mean()

            # Smoothed mean
            smoothed = (counts * means + self.smoothing * global_mean) / (counts + self.smoothing)
            self.encodings_[col] = smoothed.to_dict()
            self.global_means_[col] = global_mean

        return self

    def transform(self, X: Any) -> pd.DataFrame:
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        cols = self.columns or [c for c in self.encodings_ if c in df.columns]
        result = pd.DataFrame(index=df.index)

        for col in cols:
            mapping = self.encodings_.get(col, {})
            global_mean = self.global_means_.get(col, 0.0)
            series = df[col].astype(str)
            result[f"{col}_te"] = series.map(mapping).fillna(global_mean).astype(float)

        return result

    def fit_transform(self, X: Any, y: Any = None, **fit_params: Any) -> pd.DataFrame:
        if y is None:
            raise ValueError("TargetEncoderFoldSafe requires target 'y' for fit_transform")

        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        cols = self.columns or df.columns.tolist()
        y_arr = self._prepare_target(y)
        n_samples = len(df)

        # Fit global statistics for later transform calls (e.g. test or validation)
        self.fit(X, y)

        # Inner OOF transformation on the training fold
        oof_result = pd.DataFrame(index=df.index)
        for col in cols:
            oof_result[f"{col}_te"] = np.nan

        k = min(self.cv_folds, max(2, n_samples // 2))
        if self.task_type == "binary_classification" and len(np.unique(y_arr)) == 2:
            splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
            splits = list(splitter.split(df, y_arr))
        else:
            splitter = KFold(n_splits=k, shuffle=True, random_state=42)
            splits = list(splitter.split(df))

        for col in cols:
            out_col = f"{col}_te"
            col_series = df[col].astype(str)
            for inner_tr_idx, inner_val_idx in splits:
                tr_series = col_series.iloc[inner_tr_idx]
                tr_target = y_arr[inner_tr_idx]
                tr_global_mean = float(np.nanmean(tr_target))

                temp_df = pd.DataFrame({"val": tr_series, "target": tr_target})
                grp = temp_df.groupby("val", dropna=False)["target"]
                counts = grp.count()
                means = grp.mean()
                smoothed = (counts * means + self.smoothing * tr_global_mean) / (
                    counts + self.smoothing
                )
                mapping = smoothed.to_dict()

                val_series = col_series.iloc[inner_val_idx]
                encoded_vals = val_series.map(mapping).fillna(tr_global_mean).astype(float)
                oof_result.iloc[inner_val_idx, oof_result.columns.get_loc(out_col)] = (
                    encoded_vals.values
                )

            # Any residual NaNs get filled with global mean
            oof_result[out_col] = oof_result[out_col].fillna(self.global_means_[col])

        return oof_result


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Categorical frequency / count encoder.

    Replaces categorical values with their frequency proportion in the dataset.
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        normalize: bool = True,
        unseen_value: float = 0.0,
    ) -> None:
        self.columns = columns
        self.normalize = normalize
        self.unseen_value = unseen_value
        self.frequencies_: dict[str, dict[Any, float]] = {}
        self.feature_names_out_: list[str] = []

    def fit(self, X: Any, y: Any = None) -> FrequencyEncoder:
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        cols = self.columns or df.columns.tolist()
        self.frequencies_ = {}
        self.feature_names_out_ = [f"{c}_freq" for c in cols]

        for col in cols:
            series = df[col].astype(str)
            freqs = series.value_counts(normalize=self.normalize).to_dict()
            self.frequencies_[col] = freqs

        return self

    def transform(self, X: Any) -> pd.DataFrame:
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        cols = self.columns or [c for c in self.frequencies_ if c in df.columns]
        result = pd.DataFrame(index=df.index)

        for col in cols:
            freq_map = self.frequencies_.get(col, {})
            series = df[col].astype(str)
            result[f"{col}_freq"] = (
                series.map(freq_map).fillna(self.unseen_value).astype(float)
            )

        return result


class DateDecomposer(BaseEstimator, TransformerMixin):
    """Extracts date components from timestamp columns.

    Generates: year, month, day, dayofweek, is_weekend, quarter.
    """

    def __init__(
        self,
        date_columns: list[str] | None = None,
        drop_original: bool = True,
    ) -> None:
        self.date_columns = date_columns
        self.drop_original = drop_original
        self.detected_columns_: list[str] = []
        self.feature_names_out_: list[str] = []

    def fit(self, X: Any, y: Any = None) -> DateDecomposer:
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        if self.date_columns:
            self.detected_columns_ = [c for c in self.date_columns if c in df.columns]
        else:
            self.detected_columns_ = []
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    self.detected_columns_.append(col)
                elif df[col].dtype == object or str(df[col].dtype) == "string":
                    sample = df[col].dropna().head(20)
                    if not sample.empty:
                        try:
                            parsed = pd.to_datetime(sample, errors="coerce")
                            if parsed.notna().mean() >= 0.8:
                                self.detected_columns_.append(col)
                        except Exception:
                            pass

        out_names = []
        for col in self.detected_columns_:
            for part in ("year", "month", "day", "dayofweek", "is_weekend", "quarter"):
                out_names.append(f"{col}_{part}")
        self.feature_names_out_ = out_names
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        result = pd.DataFrame(index=df.index)

        for col in self.detected_columns_:
            if col in df.columns:
                parsed = pd.to_datetime(df[col], errors="coerce")
                result[f"{col}_year"] = parsed.dt.year.fillna(2000).astype(float)
                result[f"{col}_month"] = parsed.dt.month.fillna(1).astype(float)
                result[f"{col}_day"] = parsed.dt.day.fillna(1).astype(float)
                result[f"{col}_dayofweek"] = parsed.dt.dayofweek.fillna(0).astype(float)
                result[f"{col}_is_weekend"] = (
                    parsed.dt.dayofweek.isin([5, 6]).fillna(False).astype(float)
                )
                result[f"{col}_quarter"] = parsed.dt.quarter.fillna(1).astype(float)

        return result


class CyclicTimeEncoder(BaseEstimator, TransformerMixin):
    """Encodes cyclic time features using sine and cosine projections.

    Maps values into periodic circle coordinates: sin(2*pi*x/period), cos(2*pi*x/period).
    """

    PERIOD_MAP = {
        "month": 12.0,
        "dayofweek": 7.0,
        "hour": 24.0,
        "day": 31.0,
        "quarter": 4.0,
        "minute": 60.0,
    }

    def __init__(self, periods: dict[str, float] | None = None) -> None:
        self.periods = periods or {}
        self.resolved_periods_: dict[str, float] = {}
        self.feature_names_out_: list[str] = []

    def fit(self, X: Any, y: Any = None) -> CyclicTimeEncoder:
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.resolved_periods_ = {}
        out_names = []

        for col in df.columns:
            period = None
            if col in self.periods:
                period = float(self.periods[col])
            else:
                col_lower = str(col).lower()
                for key, val in self.PERIOD_MAP.items():
                    if key in col_lower:
                        period = val
                        break
            if period is not None:
                self.resolved_periods_[str(col)] = period
                out_names.append(f"{col}_sin")
                out_names.append(f"{col}_cos")

        self.feature_names_out_ = out_names
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        result = pd.DataFrame(index=df.index)

        for col, period in self.resolved_periods_.items():
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
                radians = 2.0 * math.pi * vals / period
                result[f"{col}_sin"] = np.sin(radians)
                result[f"{col}_cos"] = np.cos(radians)

        return result


class MissingIndicatorAdder(BaseEstimator, TransformerMixin):
    """Appends binary indicator columns for missing values."""

    def __init__(self, columns: list[str] | None = None) -> None:
        self.columns = columns
        self.indicator_columns_: list[str] = []
        self.feature_names_out_: list[str] = []

    def fit(self, X: Any, y: Any = None) -> MissingIndicatorAdder:
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        cols = self.columns or df.columns.tolist()
        self.indicator_columns_ = [c for c in cols if df[c].isna().any()]
        self.feature_names_out_ = [f"{c}_isna" for c in self.indicator_columns_]
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        result = pd.DataFrame(index=df.index)
        for col in self.indicator_columns_:
            if col in df.columns:
                result[f"{col}_isna"] = df[col].isna().astype(float)
            else:
                result[f"{col}_isna"] = 0.0
        return result


class QuantileBinner(BaseEstimator, TransformerMixin):
    """Discretizes numeric features into quantile bins."""

    def __init__(self, n_bins: int = 5, columns: list[str] | None = None) -> None:
        self.n_bins = int(max(2, n_bins))
        self.columns = columns
        self.fitted_columns_: list[str] = []
        self.discretizer_: KBinsDiscretizer | None = None
        self.feature_names_out_: list[str] = []

    def fit(self, X: Any, y: Any = None) -> QuantileBinner:
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        cols = self.columns or df.select_dtypes(include="number").columns.tolist()
        self.fitted_columns_ = cols
        self.feature_names_out_ = [f"{c}_bin" for c in cols]

        if not cols:
            return self

        matrix = df[cols].fillna(df[cols].median()).to_numpy()
        self.discretizer_ = KBinsDiscretizer(
            n_bins=self.n_bins,
            encode="ordinal",
            strategy="quantile",
            subsample=None,
        )
        self.discretizer_.fit(matrix)
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        if not self.fitted_columns_ or self.discretizer_ is None:
            return pd.DataFrame(index=df.index)

        matrix = df[self.fitted_columns_].fillna(0.0).to_numpy()
        binned = self.discretizer_.transform(matrix)
        return pd.DataFrame(binned, index=df.index, columns=self.feature_names_out_)


class InteractionPolynomial(BaseEstimator, TransformerMixin):
    """Pairwise feature interactions and polynomial terms with feature count capping."""

    def __init__(
        self,
        max_features: int = 30,
        columns: list[str] | None = None,
        interaction_only: bool = True,
    ) -> None:
        self.max_features = max_features
        self.columns = columns
        self.interaction_only = interaction_only
        self.poly_: PolynomialFeatures | None = None
        self.selected_indices_: list[int] = []
        self.fitted_columns_: list[str] = []
        self.feature_names_out_: list[str] = []

    def fit(self, X: Any, y: Any = None) -> InteractionPolynomial:
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        cols = self.columns or df.select_dtypes(include="number").columns.tolist()
        self.fitted_columns_ = cols

        if len(cols) < 2:
            self.feature_names_out_ = []
            return self

        matrix = df[cols].fillna(0.0).to_numpy()
        self.poly_ = PolynomialFeatures(
            degree=2,
            interaction_only=self.interaction_only,
            include_bias=False,
        )
        self.poly_.fit(matrix)
        names = self.poly_.get_feature_names_out(cols)

        interaction_idx = [i for i, name in enumerate(names) if " " in name or "^2" in name]
        if len(interaction_idx) > self.max_features:
            transformed = self.poly_.transform(matrix)[:, interaction_idx]
            variances = np.nanvar(transformed, axis=0)
            top_ranked = np.argsort(variances)[::-1][: self.max_features]
            self.selected_indices_ = [interaction_idx[i] for i in top_ranked]
        else:
            self.selected_indices_ = interaction_idx

        self.feature_names_out_ = [
            names[i].replace(" ", "_x_") for i in self.selected_indices_
        ]
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        if not self.fitted_columns_ or self.poly_ is None or not self.selected_indices_:
            return pd.DataFrame(index=df.index)

        matrix = df[self.fitted_columns_].fillna(0.0).to_numpy()
        transformed = self.poly_.transform(matrix)[:, self.selected_indices_]
        return pd.DataFrame(transformed, index=df.index, columns=self.feature_names_out_)


class TextTfidfEncoder(BaseEstimator, TransformerMixin):
    """Fold-safe TF-IDF extraction for freeform text features.

    Fitted strictly on the training fold, safely handling missing values,
    unseen terms, and emitting clean deterministic column names.
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        max_features: int = 50,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 2,
    ) -> None:
        self.columns = columns
        self.max_features = int(max(max_features, 1))
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.vectorizers_: dict[str, Any] = {}
        self.feature_names_out_: list[str] = []
        self.fitted_columns_: list[str] = []

    def fit(self, X: Any, y: Any = None) -> TextTfidfEncoder:
        from sklearn.feature_extraction.text import TfidfVectorizer

        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        cols = self.columns or [
            c for c in df.columns if df[c].dtype == object or str(df[c].dtype) == "string"
        ]
        self.fitted_columns_ = []
        self.vectorizers_ = {}
        self.feature_names_out_ = []

        for col in cols:
            series = df[col].fillna("").astype(str)
            if series.str.strip().ne("").any():
                actual_min_df = min(self.min_df, max(1, len(series)))
                vec = TfidfVectorizer(
                    max_features=self.max_features,
                    ngram_range=self.ngram_range,
                    min_df=actual_min_df,
                    token_pattern=r"(?u)\b\w+\b",
                )
                try:
                    vec.fit(series)
                    terms = vec.get_feature_names_out()
                    if len(terms) > 0:
                        self.vectorizers_[col] = vec
                        self.fitted_columns_.append(col)
                        col_feature_names = [
                            f"tfidf_{col}_{t.replace(' ', '_')}" for t in terms
                        ]
                        self.feature_names_out_.extend(col_feature_names)
                except ValueError:
                    pass
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        if not self.vectorizers_:
            return pd.DataFrame(index=df.index)

        dfs = []
        for col in self.fitted_columns_:
            vec = self.vectorizers_[col]
            series = (
                df[col].fillna("").astype(str)
                if col in df.columns
                else pd.Series("", index=df.index)
            )
            mat = vec.transform(series).toarray()
            terms = vec.get_feature_names_out()
            col_names = [f"tfidf_{col}_{t.replace(' ', '_')}" for t in terms]
            col_df = pd.DataFrame(mat, index=df.index, columns=col_names)
            dfs.append(col_df)

        if not dfs:
            return pd.DataFrame(index=df.index)
        return pd.concat(dfs, axis=1)
