"""
Data loading utilities for the EDA pipeline.
Supports CSV, Excel, JSON, and Parquet formats.
"""

import os
import pandas as pd
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Cache loaded dataframes to avoid re-reading
_df_cache: dict = {}


def load_dataset(path: str) -> dict:
    """
    Load a dataset and return a summary profile.
    Supports: .csv, .xlsx, .xls, .json, .parquet
    """
    path = str(path)
    ext = Path(path).suffix.lower()

    logger.info(f"Loading dataset from: {path}")

    if path in _df_cache:
        df = _df_cache[path]
    else:
        if ext == ".csv":
            df = pd.read_csv(path)
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(path)
        elif ext == ".json":
            df = pd.read_json(path)
        elif ext == ".parquet":
            df = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported file format: {ext}. Use CSV, Excel, JSON, or Parquet.")

        _df_cache[path] = df

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()

    summary = {
        "shape": list(df.shape),
        "columns": df.columns.tolist(),
        "numeric_cols": len(numeric_cols),
        "categorical_cols": len(categorical_cols),
        "datetime_cols": len(datetime_cols),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "sample": df.head(3).to_dict(orient="records"),
        "duplicate_rows": int(df.duplicated().sum()),
    }

    logger.info(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} cols")
    return summary


def get_dataframe(path: str) -> pd.DataFrame:
    """Get cached dataframe or load it"""
    if path not in _df_cache:
        load_dataset(path)
    return _df_cache[path]
