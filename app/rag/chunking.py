from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from app.config.settings import settings


class ChunkingService:

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        strategy: str | None = None,
    ):
        self._chunk_size = chunk_size or settings.chunk_size
        self._chunk_overlap = chunk_overlap or settings.chunk_overlap
        self._strategy = strategy or settings.chunk_strategy

    def split(self, documents: list[Document]) -> list[Document]:
        logger.info(
            f"Splitting {len(documents)} docs — size={self._chunk_size}, overlap={self._chunk_overlap}, strategy={self._strategy}"
        )

        if self._strategy == "recursive":
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
                separators=["\n\n", "\n", ".", " ", ""],
            )
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
            )

        chunks = splitter.split_documents(documents)
        logger.debug(f"Created {len(chunks)} chunks")
        return chunks
