import io
import re
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from loguru import logger

from app.profiling.profiler import DatasetProfile


sns.set_theme(style="whitegrid")


class VisualizationService:

    def __init__(self, df: pd.DataFrame, profile: DatasetProfile):
        self._df = df
        self._profile = profile

    def generate(self, query: str) -> tuple[str | None, str | None]:
        q = query.lower()

        if "histogram" in q or "distribution" in q:
            return self._histogram(query)
        elif "bar" in q:
            return self._bar_chart(query)
        elif "pie" in q:
            return self._pie_chart(query)
        elif "scatter" in q or "plot" in q:
            return self._scatter(query)
        elif "line" in q or "trend" in q:
            return self._line_chart(query)
        elif "box" in q:
            return self._box_plot(query)
        elif "correlation" in q or "heatmap" in q:
            return self._correlation_heatmap()

        return None, "I can generate bar, pie, scatter, histogram, line, box, and correlation charts. Specify which type."

    def _histogram(self, query: str) -> tuple[str | None, str | None]:
        col = self._find_col(query)
        if not col:
            return None, f"Numeric columns available: {', '.join(self._profile.numeric_columns[:6])}"
        fig, ax = plt.subplots(figsize=(8, 4))
        self._df[col].hist(bins=20, ax=ax, color="#d97706", edgecolor="white")
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        buf = self._fig_to_buffer(fig)
        explanation = f"Histogram showing the distribution of **{col}** across {len(self._df)} rows."
        return buf, explanation

    def _bar_chart(self, query: str) -> tuple[str | None, str | None]:
        col = self._find_col(query)
        if not col:
            cat_cols = self._profile.categorical_columns
            if not cat_cols:
                return None, "No categorical columns found for bar chart."
            col = cat_cols[0]
        top = self._df[col].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(8, 4))
        top.plot(kind="bar", ax=ax, color="#d97706")
        ax.set_title(f"Top 10 values in {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        buf = self._fig_to_buffer(fig)
        explanation = f"Bar chart of top 10 values in **{col}**. Most frequent: **{top.index[0]}** ({top.iloc[0]} rows)."
        return buf, explanation

    def _pie_chart(self, query: str) -> tuple[str | None, str | None]:
        col = self._find_col(query)
        if not col:
            cat_cols = self._profile.categorical_columns
            if not cat_cols:
                return None, "No categorical columns found for pie chart."
            col = cat_cols[0]
        top = self._df[col].value_counts().head(6)
        fig, ax = plt.subplots(figsize=(6, 6))
        top.plot(kind="pie", ax=ax, autopct="%1.1f%%", colors=sns.color_palette("husl", len(top)))
        ax.set_title(f"Distribution of {col}")
        ax.set_ylabel("")
        buf = self._fig_to_buffer(fig)
        explanation = f"Pie chart showing distribution of **{col}** across {len(top)} categories."
        return buf, explanation

    def _scatter(self, query: str) -> tuple[str | None, str | None]:
        cols = self._find_two_cols(query)
        if not cols:
            nums = self._profile.numeric_columns[:2]
            if len(nums) < 2:
                return None, "Need at least 2 numeric columns for a scatter plot."
            cols = nums
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(self._df[cols[0]], self._df[cols[1]], alpha=0.5, color="#d97706")
        ax.set_xlabel(cols[0])
        ax.set_ylabel(cols[1])
        ax.set_title(f"{cols[0]} vs {cols[1]}")
        buf = self._fig_to_buffer(fig)
        explanation = f"Scatter plot of **{cols[0]}** vs **{cols[1]}** ({len(self._df)} points)."
        return buf, explanation

    def _line_chart(self, query: str) -> tuple[str | None, str | None]:
        cols = self._find_two_cols(query)
        if not cols:
            dt_cols = self._profile.datetime_columns
            if dt_cols:
                x = dt_cols[0]
            else:
                x = self._df.index
            nums = self._profile.numeric_columns
            if not nums:
                return None, "No numeric column for line chart."
            cols = (x, nums[0])
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(self._df[cols[0]], self._df[cols[1]], color="#d97706", linewidth=1)
        ax.set_title(f"{cols[1]} over {cols[0]}")
        ax.set_xlabel(cols[0])
        ax.set_ylabel(cols[1])
        plt.xticks(rotation=45, ha="right")
        buf = self._fig_to_buffer(fig)
        explanation = f"Line chart of **{cols[1]}** over **{cols[0]}**."
        return buf, explanation

    def _box_plot(self, query: str) -> tuple[str | None, str | None]:
        col = self._find_col(query)
        if not col:
            nums = self._profile.numeric_columns
            if not nums:
                return None, "No numeric columns for box plot."
            col = nums[0]
        fig, ax = plt.subplots(figsize=(8, 4))
        self._df.boxplot(column=col, ax=ax)
        ax.set_title(f"Box plot of {col}")
        ax.set_ylabel(col)
        buf = self._fig_to_buffer(fig)
        explanation = f"Box plot of **{col}** — median: **{self._df[col].median():.2f}**, IQR spread visible."
        return buf, explanation

    def _correlation_heatmap(self) -> tuple[str | None, str | None]:
        nums = self._profile.numeric_columns
        if len(nums) < 2:
            return None, "Need at least 2 numeric columns for correlation heatmap."
        fig, ax = plt.subplots(figsize=(max(6, len(nums)), max(4, len(nums) * 0.7)))
        sns.heatmap(self._df[nums].corr(), annot=True, cmap="YlOrRd", ax=ax, fmt=".2f")
        ax.set_title("Correlation Heatmap")
        buf = self._fig_to_buffer(fig)
        explanation = f"Correlation heatmap for {len(nums)} numeric columns."
        return buf, explanation

    def _find_col(self, query: str) -> str | None:
        for col in self._df.columns:
            if col.lower() in query.lower():
                return col
        return None

    def _find_two_cols(self, query: str) -> tuple | None:
        found = [c for c in self._df.columns if c.lower() in query.lower()]
        if len(found) >= 2:
            return found[0], found[1]
        return None

    @staticmethod
    def _fig_to_buffer(fig: plt.Figure) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        import base64
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode()
        return f"data:image/png;base64,{b64}"
