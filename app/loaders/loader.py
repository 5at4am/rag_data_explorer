from pathlib import Path
from typing import Any
from dataclasses import dataclass

import pandas as pd
from loguru import logger

from app.utils.validators import validate_file, FileValidationError


class DataLoadError(Exception):
    pass


@dataclass
class LoadResult:
    df: pd.DataFrame
    filename: str
    sheet_name: str | None = None
    file_size_bytes: int = 0
    format: str = ""


class DataLoader:

    SUPPORTED_FORMATS = {
        ".csv": "csv",
        ".xlsx": "excel",
        ".json": "json",
        ".parquet": "parquet",
    }

    def __init__(self, encoding: str | None = None):
        self._encoding = encoding

    def load(self, path: str | Path) -> pd.DataFrame:
        path = Path(path)
        validate_file(path)
        ext = path.suffix.lower()

        handler = self._get_handler(ext)
        logger.info(f"Loading {ext} file: {path.name}")

        try:
            df = handler(path)
            if df.empty:
                raise DataLoadError("Dataset has no rows after loading")
            logger.debug(f"Loaded {df.shape[0]} rows, {df.shape[1]} cols")
            return df
        except pd.errors.EmptyDataError:
            raise DataLoadError("File contains no data")
        except pd.errors.ParserError as e:
            raise DataLoadError(f"Could not parse file: {e}")
        except UnicodeDecodeError as e:
            raise DataLoadError(f"Encoding issue: {e}. Try a different encoding.")
        except Exception as e:
            raise DataLoadError(str(e))

    def _get_handler(self, ext: str):
        handlers = {
            ".csv": self._load_csv,
            ".xlsx": self._load_excel,
            ".json": self._load_json,
            ".parquet": self._load_parquet,
        }
        handler = handlers.get(ext)
        if not handler:
            raise FileValidationError(f"Unsupported format: {ext}")
        return handler

    def _load_csv(self, path: Path) -> pd.DataFrame:
        encodings = [self._encoding, "utf-8", "latin-1", "cp1252", "ISO-8859-1"]
        for enc in encodings:
            if enc is None:
                continue
            try:
                return pd.read_csv(path, encoding=enc, low_memory=False)
            except (UnicodeDecodeError, UnicodeError):
                continue
        raise DataLoadError("Could not read CSV with any common encoding")

    def _load_excel(self, path: Path) -> pd.DataFrame:
        try:
            xls = pd.ExcelFile(path, engine="openpyxl")
            if len(xls.sheet_names) > 1:
                logger.info(f"Multiple sheets found: {xls.sheet_names}, using first")
            return pd.read_excel(path, engine="openpyxl")
        except ValueError as e:
            raise DataLoadError(f"Corrupted Excel file: {e}")

    def _load_json(self, path: Path) -> pd.DataFrame:
        try:
            return pd.read_json(path, encoding="utf-8")
        except ValueError:
            try:
                return pd.read_json(path, lines=True, encoding="utf-8")
            except ValueError as e:
                raise DataLoadError(f"Invalid JSON format: {e}")

    def _load_parquet(self, path: Path) -> pd.DataFrame:
        try:
            return pd.read_parquet(path, engine="pyarrow")
        except ImportError:
            raise DataLoadError("Install pyarrow: pip install pyarrow")
        except Exception as e:
            raise DataLoadError(f"Corrupted Parquet file: {e}")
