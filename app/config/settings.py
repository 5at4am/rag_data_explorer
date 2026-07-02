from pydantic_settings import BaseSettings
from typing import Literal
import os


class Settings(BaseSettings):
    app_name: str = "Dataset Chat (RAG)"
    debug: bool = True
    upload_dir: str = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "uploads")
    data_dir: str = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")

    max_upload_size_mb: int = 50
    allowed_extensions: list[str] = [".csv", ".xlsx", ".json", ".parquet"]

    llm_provider: Literal["gemini", "openai", "ollama"] = "gemini"
    llm_model: str = "gemini-2.0-flash-lite"
    google_api_key: str | None = None
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"

    embedding_provider: Literal["openai", "huggingface", "ollama"] = "huggingface"
    embedding_model: str = "all-MiniLM-L6-v2"
    openai_embedding_model: str = "text-embedding-3-small"

    vector_store: Literal["faiss", "chroma"] = "faiss"
    chunk_size: int = 512
    chunk_overlap: int = 64
    chunk_strategy: Literal["recursive", "semantic"] = "recursive"

    retrieval_k: int = 4
    temperature: float = 0.3
    max_tokens: int = 2048

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
