import os
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from loguru import logger

from app.config.settings import settings
from app.rag.embeddings import EmbeddingService


class VectorStoreService:

    def __init__(self, embedder: EmbeddingService):
        self._embedder = embedder
        self._store: Any = None
        self._store_type = settings.vector_store
        self._index_dir = Path(settings.data_dir) / "vectorstore"
        self._index_dir.mkdir(parents=True, exist_ok=True)

    def build(self, documents: list[Document]) -> None:
        logger.info(f"Building {self._store_type} index with {len(documents)} documents")
        embeddings = self._embedder.get_client()

        if self._store_type == "faiss":
            from langchain_community.vectorstores import FAISS
            self._store = FAISS.from_documents(documents, embeddings)
        elif self._store_type == "chroma":
            from langchain_chroma import Chroma
            persist_dir = str(self._index_dir / "chroma")
            self._store = Chroma.from_documents(
                documents, embeddings, persist_directory=persist_dir
            )
        else:
            raise ValueError(f"Unsupported vector store: {self._store_type}")

        self._persist()
        logger.info("Vector store built and persisted")

    def load(self) -> bool:
        index_path = self._index_dir / "faiss_index"
        if not index_path.exists():
            logger.debug("No persisted index found")
            return False

        logger.info(f"Loading persisted {self._store_type} index")
        embeddings = self._embedder.get_client()

        try:
            if self._store_type == "faiss":
                from langchain_community.vectorstores import FAISS
                self._store = FAISS.load_local(
                    str(index_path), embeddings, allow_dangerous_deserialization=True
                )
            elif self._store_type == "chroma":
                from langchain_chroma import Chroma
                persist_dir = str(self._index_dir / "chroma")
                self._store = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
            return True
        except Exception as e:
            logger.warning(f"Failed to load index: {e}")
            return False

    def get_retriever(self, k: int | None = None) -> VectorStoreRetriever:
        if self._store is None:
            raise RuntimeError("Vector store not built or loaded")
        return self._store.as_retriever(
            search_kwargs={"k": k or settings.retrieval_k}
        )

    def _persist(self) -> None:
        if self._store_type == "faiss" and hasattr(self._store, "save_local"):
            self._store.save_local(str(self._index_dir / "faiss_index"))
