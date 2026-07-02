from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger

from app.config.settings import settings


class EmbeddingService:

    def __init__(self, provider: str | None = None, model: str | None = None):
        self._provider = provider or settings.embedding_provider
        self._model = model or settings.embedding_model
        self._client: Embeddings | None = None

    def get_client(self) -> Embeddings:
        if self._client is not None:
            return self._client

        logger.info(f"Initializing embeddings — provider={self._provider}, model={self._model}")

        if self._provider == "huggingface":
            self._client = HuggingFaceEmbeddings(model_name=self._model)
        elif self._provider == "openai":
            from langchain_openai import OpenAIEmbeddings
            self._client = OpenAIEmbeddings(
                model=self._model or settings.openai_embedding_model,
                api_key=settings.openai_api_key,
            )
        elif self._provider == "ollama":
            from langchain_ollama import OllamaEmbeddings
            self._client = OllamaEmbeddings(model=self._model)
        else:
            raise ValueError(f"Unknown embedding provider: {self._provider}")

        return self._client
