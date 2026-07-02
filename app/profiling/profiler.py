from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import numpy as np
from loguru import logger


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    inferred_type: str  # numeric | categorical | datetime | text | boolean
    null_count: int
    null_pct: float
    unique_count: int
    unique_pct: float
    sample_values: list[Any]

    # numeric
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    median: float | None = None
    std: float | None = None

    # categorical
    top_value: str | None = None
    top_freq: int | None = None
    top_pct: float | None = None

    # datetime
    date_min: str | None = None
    date_max: str | None = None


@dataclass
class DatasetProfile:
    filename: str
    shape: tuple[int, int]
    memory_usage_mb: float
    duplicate_rows: int
    duplicate_pct: float
    columns: list[ColumnProfile] = field(default_factory=list)
    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    datetime_columns: list[str] = field(default_factory=list)
    text_columns: list[str] = field(default_factory=list)
    boolean_columns: list[str] = field(default_factory=list)
    correlation: dict[str, dict[str, float]] | None = None

    def model_dump(self) -> dict:
        return {
            "filename": self.filename,
            "shape": list(self.shape),
            "memory_usage_mb": round(self.memory_usage_mb, 2),
            "duplicate_rows": self.duplicate_rows,
            "duplicate_pct": round(self.duplicate_pct, 2),
            "column_count": len(self.columns),
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "datetime_columns": self.datetime_columns,
            "text_columns": self.text_columns,
            "boolean_columns": self.boolean_columns,
            "columns": [c.__dict__ for c in self.columns],
        }


class DataProfiler:

    def __init__(self, df: pd.DataFrame):
        self._df = df

    def profile(self, filename: str = "unknown") -> DatasetProfile:
        logger.info("Profiling dataset")
        df = self._df

        profile = DatasetProfile(
            filename=filename,
            shape=df.shape,
            memory_usage_mb=df.memory_usage(deep=True).sum() / (1024 * 1024),
            duplicate_rows=int(df.duplicated().sum()),
            duplicate_pct=round(float(df.duplicated().mean() * 100), 2),
        )

        for col in df.columns:
            col_profile = self._profile_column(df, col)
            profile.columns.append(col_profile)

            it = col_profile.inferred_type
            if it == "numeric":
                profile.numeric_columns.append(col)
            elif it == "categorical":
                profile.categorical_columns.append(col)
            elif it == "datetime":
                profile.datetime_columns.append(col)
            elif it == "text":
                profile.text_columns.append(col)
            elif it == "boolean":
                profile.boolean_columns.append(col)

        if len(profile.numeric_columns) >= 2:
            try:
                corr = df[profile.numeric_columns].corr().round(3)
                profile.correlation = corr.to_dict()
            except Exception:
                pass

        logger.info(f"Profile done — {profile.shape[0]} rows, {profile.shape[1]} cols, {len(profile.numeric_columns)} numeric")
        return profile

    def _profile_column(self, df: pd.DataFrame, col: str) -> ColumnProfile:
        s = df[col]
        null_count = int(s.isnull().sum())
        null_pct = round(float(s.isnull().mean() * 100), 2)
        unique_count = int(s.nunique())
        unique_pct = round(unique_count / max(len(s), 1) * 100, 2)
        sample = s.dropna().head(3).tolist()

        inferred = self._infer_type(s)

        cp = ColumnProfile(
            name=col,
            dtype=str(s.dtype),
            inferred_type=inferred,
            null_count=null_count,
            null_pct=null_pct,
            unique_count=unique_count,
            unique_pct=unique_pct,
            sample_values=sample,
        )

        if inferred == "numeric":
            cp.min = float(s.min()) if pd.notna(s.min()) else None
            cp.max = float(s.max()) if pd.notna(s.max()) else None
            cp.mean = round(float(s.mean()), 2) if pd.notna(s.mean()) else None
            cp.median = round(float(s.median()), 2) if pd.notna(s.median()) else None
            cp.std = round(float(s.std()), 2) if pd.notna(s.std()) else None

        elif inferred == "categorical":
            if not s.dropna().empty:
                top = s.value_counts().head(1)
                cp.top_value = str(top.index[0])
                cp.top_freq = int(top.iloc[0])
                cp.top_pct = round(float(top.iloc[0] / max(len(s.dropna()), 1) * 100), 2)

        elif inferred == "datetime":
            cp.date_min = str(s.min()) if pd.notna(s.min()) else None
            cp.date_max = str(s.max()) if pd.notna(s.max()) else None

        return cp

    @staticmethod
    def _infer_type(s: pd.Series) -> str:
        dtype = s.dtype
        name = str(dtype).lower()

        if pd.api.types.is_bool_dtype(s):
            return "boolean"
        if pd.api.types.is_datetime64_any_dtype(s):
            return "datetime"
        if pd.api.types.is_numeric_dtype(s):
            return "numeric"

        non_null = s.dropna()
        if len(non_null) == 0:
            return "text"

        if pd.api.types.is_string_dtype(s) or s.dtype == object:
            avg_len = non_null.astype(str).str.len().mean()
            unique_ratio = non_null.nunique() / max(len(non_null), 1)

            if unique_ratio < 0.05 and len(non_null) > 10:
                return "categorical"
            if avg_len > 50:
                return "text"
            if unique_ratio < 0.5:
                return "categorical"

            try:
                pd.to_datetime(non_null.head(100))
                return "datetime"
            except (ValueError, TypeError):
                pass

            return "categorical"

        return "text"
