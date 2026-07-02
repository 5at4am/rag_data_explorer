import os
from pathlib import Path

from loguru import logger

from app.config.settings import settings


class FileValidationError(Exception):
    pass


def validate_file_size(file_path: str | Path) -> None:
    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if size_mb > settings.max_upload_size_mb:
        raise FileValidationError(
            f"File size ({size_mb:.1f} MB) exceeds limit of {settings.max_upload_size_mb} MB"
        )
    logger.debug(f"File size OK: {size_mb:.1f} MB")


def validate_extension(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in settings.allowed_extensions:
        raise FileValidationError(
            f"Unsupported format '{ext}'. Allowed: {', '.join(settings.allowed_extensions)}"
        )
    logger.debug(f"Extension OK: {ext}")
    return ext


def validate_not_empty(file_path: str | Path) -> None:
    if os.path.getsize(file_path) == 0:
        raise FileValidationError("File is empty")


def validate_file(file_path: str | Path) -> str:
    path = Path(file_path)
    ext = validate_extension(path.name)
    validate_not_empty(path)
    validate_file_size(path)
    return ext
