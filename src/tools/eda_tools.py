"""
Core EDA analysis tools: missing values, outliers, statistics, correlations.
"""

import numpy as np
import pandas as pd
from scipy import stats
from src.tools.data_loader import get_dataframe
from src.utils.logger import get_logger

logger = get_logger(__name__)


def analyze_missing_values(path: str) -> dict:
    """Analyze missing values across all columns"""
    df = get_dataframe(path)

    missing_counts = df.isnull().sum().to_dict()
    missing_percentages = (df.isnull().mean() * 100).to_dict()

    # Identify columns by missingness severity
    high_missing = [col for col, pct in missing_percentages.items() if pct > 50]
    medium_missing = [col for col, pct in missing_percentages.items() if 10 < pct <= 50]
    low_missing = [col for col, pct in missing_percentages.items() if 0 < pct <= 10]

    # Missing patterns - which rows have multiple missing values
    row_missing = df.isnull().sum(axis=1)
    rows_with_missing = int((row_missing > 0).sum())

    return {
        "missing_counts": {k: int(v) for k, v in missing_counts.items()},
        "missing_percentages": {k: round(float(v), 2) for k, v in missing_percentages.items()},
        "total_missing_cells": int(df.isnull().sum().sum()),
        "total_cells": int(df.size),
        "overall_missing_pct": round(float(df.isnull().mean().mean() * 100), 2),
        "rows_with_missing": rows_with_missing,
        "high_missing_cols": high_missing,      # >50%
        "medium_missing_cols": medium_missing,  # 10-50%
        "low_missing_cols": low_missing,        # <10%
        "complete_columns": [col for col, count in missing_counts.items() if count == 0],
    }


def detect_outliers(path: str) -> dict:
    """
    Detect outliers using IQR and Z-score methods.
    Returns per-column outlier info for numeric columns only.
    """
    df = get_dataframe(path)
    numeric_df = df.select_dtypes(include="number")
    result = {}

    for col in numeric_df.columns:
        series = numeric_df[col].dropna()
        if len(series) < 4:
            continue

        # IQR method
        Q1 = float(series.quantile(0.25))
        Q3 = float(series.quantile(0.75))
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        iqr_outliers = series[(series < lower_bound) | (series > upper_bound)]

        # Z-score method (|z| > 3)
        z_scores = np.abs(stats.zscore(series))
        zscore_outliers = series[z_scores > 3]

        result[col] = {
            "iqr_outliers": int(len(iqr_outliers)),
            "zscore_outliers": int(len(zscore_outliers)),
            "iqr_lower_bound": round(lower_bound, 4),
            "iqr_upper_bound": round(upper_bound, 4),
            "outlier_pct_iqr": round(len(iqr_outliers) / len(series) * 100, 2),
            "Q1": round(Q1, 4),
            "Q3": round(Q3, 4),
            "IQR": round(IQR, 4),
        }

    return result


def compute_statistics(path: str) -> dict:
    """Compute comprehensive descriptive statistics"""
    df = get_dataframe(path)
    numeric_df = df.select_dtypes(include="number")
    result = {}

    for col in numeric_df.columns:
        series = numeric_df[col].dropna()
        if len(series) == 0:
            continue

        try:
            skewness = float(series.skew())
            kurtosis = float(series.kurtosis())
        except Exception:
            skewness = kurtosis = 0.0

        result[col] = {
            "count": int(series.count()),
            "mean": round(float(series.mean()), 4),
            "median": round(float(series.median()), 4),
            "std": round(float(series.std()), 4),
            "variance": round(float(series.var()), 4),
            "min": round(float(series.min()), 4),
            "max": round(float(series.max()), 4),
            "range": round(float(series.max() - series.min()), 4),
            "q1": round(float(series.quantile(0.25)), 4),
            "q3": round(float(series.quantile(0.75)), 4),
            "skewness": round(skewness, 4),
            "kurtosis": round(kurtosis, 4),
            "skew_type": (
                "right-skewed" if skewness > 0.5
                else "left-skewed" if skewness < -0.5
                else "symmetric"
            ),
        }

    # Also add categorical stats
    cat_df = df.select_dtypes(include=["object", "category"])
    cat_stats = {}
    for col in cat_df.columns:
        vc = df[col].value_counts()
        cat_stats[col] = {
            "unique_values": int(df[col].nunique()),
            "top_value": str(vc.index[0]) if len(vc) > 0 else "N/A",
            "top_frequency": int(vc.iloc[0]) if len(vc) > 0 else 0,
            "top_pct": round(float(vc.iloc[0] / len(df) * 100), 2) if len(vc) > 0 else 0,
        }

    return {"numeric": result, "categorical": cat_stats}


def analyze_correlations(path: str) -> dict:
    """Compute Pearson correlation matrix and identify top correlations"""
    df = get_dataframe(path)
    numeric_df = df.select_dtypes(include="number")

    if numeric_df.shape[1] < 2:
        return {"matrix": {}, "top_correlations": [], "message": "Need at least 2 numeric columns"}

    corr_matrix = numeric_df.corr(method="pearson")

    # Find top N correlations (excluding self-correlations)
    corr_pairs = []
    cols = corr_matrix.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr_matrix.iloc[i, j]
            if not np.isnan(val):
                corr_pairs.append({
                    "col1": cols[i],
                    "col2": cols[j],
                    "correlation": round(float(val), 4),
                    "strength": (
                        "strong" if abs(val) > 0.7
                        else "moderate" if abs(val) > 0.4
                        else "weak"
                    ),
                    "direction": "positive" if val > 0 else "negative"
                })

    corr_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)

    return {
        "matrix": {col: {c: round(float(v), 4) for c, v in row.items()}
                   for col, row in corr_matrix.to_dict().items()},
        "top_correlations": corr_pairs[:10],
        "columns_analyzed": cols,
    }


def get_data_types(path: str) -> dict:
    """Return column data types with classification"""
    df = get_dataframe(path)
    return {
        col: {
            "dtype": str(df[col].dtype),
            "category": (
                "numeric" if pd.api.types.is_numeric_dtype(df[col])
                else "datetime" if pd.api.types.is_datetime64_any_dtype(df[col])
                else "categorical" if df[col].nunique() / len(df) < 0.05
                else "text"
            ),
            "unique_values": int(df[col].nunique()),
            "unique_pct": round(df[col].nunique() / len(df) * 100, 2),
        }
        for col in df.columns
    }
