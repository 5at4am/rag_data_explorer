import re
from typing import Any

import pandas as pd
from loguru import logger

from app.profiling.profiler import DatasetProfile
from app.rag.vectorstore import VectorStoreService
from app.memory.memory import ConversationMemory
from app.services.query_service import QueryService


class HybridQueryEngine:

    def __init__(
        self,
        df: pd.DataFrame,
        profile: DatasetProfile,
        summary: str,
        vectorstore: VectorStoreService,
        memory: ConversationMemory,
    ):
        self._df = df
        self._profile = profile
        self._summary = summary
        self._vectorstore = vectorstore
        self._memory = memory
        self._query_service = QueryService(summary, vectorstore, memory)

    def answer(self, question: str) -> str:
        logger.info(f"Query: {question[:80]}")

        if self._is_pandas_query(question):
            logger.debug("Routing to Pandas engine")
            try:
                return self._pandas_answer(question)
            except Exception as e:
                logger.warning(f"Pandas engine failed: {e}, falling back to RAG")
                return self._query_service.answer(question)

        logger.debug("Routing to RAG engine")
        return self._query_service.answer(question)

    def _is_pandas_query(self, question: str) -> bool:
        q = question.lower().strip()
        pandas_keywords = [
            "count", "average", "sum", "total", "mean", "median",
            "minimum", "maximum", "highest", "lowest", "top",
            "how many", "how much",
            "filter", "where", "sort", "group",
            "older than", "younger than", "greater than", "less than",
            "above", "below", "between",
            "unique", "distinct",
        ]
        return any(kw in q for kw in pandas_keywords)

    def _pandas_answer(self, question: str) -> str:
        q = question.lower().strip()
        df = self._df
        numeric_cols = self._profile.numeric_columns
        result = ""

        avg_match = re.search(r"average (?:of |)(\w+)", q)
        if avg_match:
            col = self._resolve_col(avg_match.group(1))
            if col and col in numeric_cols:
                val = df[col].mean()
                return f"Average **{col}**: **{val:.2f}**"

        sum_match = re.search(r"(?:sum|total) (?:of |)(\w+)", q)
        if sum_match:
            col = self._resolve_col(sum_match.group(1))
            if col and col in numeric_cols:
                val = df[col].sum()
                return f"Total **{col}**: **{val:,.2f}**"

        count_match = re.search(r"(?:count|how many)(?: rows|)(?:\s*$)", q)
        if count_match:
            return f"**{len(df)}** rows"

        max_match = re.search(r"(?:highest|maximum|max|top) (\w+)", q)
        if max_match:
            col = self._resolve_col(max_match.group(1))
            if col and col in numeric_cols:
                val = df[col].max()
                return f"Highest **{col}**: **{val:,.2f}**"

        min_match = re.search(r"(?:lowest|minimum|min) (\w+)", q)
        if min_match:
            col = self._resolve_col(min_match.group(1))
            if col and col in numeric_cols:
                val = df[col].min()
                return f"Lowest **{col}**: **{val:,.2f}**"

        gt_match = re.search(r"(\w+)\s*(?:>|greater than|above)\s*(\d+\.?\d*)", q)
        if gt_match:
            col = self._resolve_col(gt_match.group(1))
            val = float(gt_match.group(2))
            if col:
                filtered = df[df[col] > val]
                n = len(filtered)
                sample = filtered.head(5).to_string(index=False)
                return f"**{n}** rows where {col} > {val}\n\n```\n{sample}\n```"

        lt_match = re.search(r"(\w+)\s*(?:<|less than|below|under)\s*(\d+\.?\d*)", q)
        if lt_match:
            col = self._resolve_col(lt_match.group(1))
            val = float(lt_match.group(2))
            if col:
                filtered = df[df[col] < val]
                n = len(filtered)
                sample = filtered.head(5).to_string(index=False)
                return f"**{n}** rows where {col} < {val}\n\n```\n{sample}\n```"

        raise ValueError("Could not parse Pandas query")

    def _resolve_col(self, name: str) -> str | None:
        for col in self._df.columns:
            if name.lower() in col.lower():
                return col
        return None
