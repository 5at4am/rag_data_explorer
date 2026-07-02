from typing import Any

import pandas as pd
from langchain_core.documents import Document
from loguru import logger


class DocumentCreator:

    def create(self, df: pd.DataFrame, filename: str, sheet_name: str | None = None) -> list[Document]:
        logger.info(f"Creating {len(df)} documents from {filename}")
        docs = []

        for idx, row in df.iterrows():
            page_content = self._row_to_text(row)
            metadata = {
                "row": int(idx),
                "filename": filename,
                "source": "upload",
            }
            if sheet_name:
                metadata["sheet"] = sheet_name
            docs.append(Document(page_content=page_content, metadata=metadata))

        logger.debug(f"Created {len(docs)} documents")
        return docs

    @staticmethod
    def _row_to_text(row: pd.Series) -> str:
        parts = []
        for col, val in row.items():
            if pd.notna(val):
                parts.append(f"{col}:\n{val}")
        return "\n\n".join(parts)
