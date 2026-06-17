"""
Tests for the Agentic EDA Pipeline
"""
import os
import pytest
import pandas as pd
import numpy as np
import tempfile

# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_csv(tmp_path):
    """Create a sample CSV with known properties"""
    np.random.seed(42)
    df = pd.DataFrame({
        "age": np.random.randint(18, 80, 200).astype(float),
        "income": np.random.exponential(50000, 200),
        "score": np.random.normal(75, 15, 200),
        "category": np.random.choice(["A", "B", "C", "D"], 200),
        "city": np.random.choice(["NY", "LA", "Chicago"], 200),
    })
    # Inject missing values
    df.loc[0:19, "age"] = np.nan      # 10% missing
    df.loc[0:4,  "income"] = np.nan   # 2.5% missing
    # Inject outliers
    df.loc[0, "score"] = 500  # extreme outlier
    df.loc[1, "score"] = -200
    path = str(tmp_path / "test_data.csv")
    df.to_csv(path, index=False)
    return path


# ─── Data Loader Tests ────────────────────────────────────────────────────────

class TestDataLoader:
    def test_load_csv(self, sample_csv):
        from src.tools.data_loader import load_dataset
        summary = load_dataset(sample_csv)
        assert summary["shape"] == [200, 5]
        assert "age" in summary["columns"]
        assert summary["numeric_cols"] == 3
        assert summary["categorical_cols"] == 2

    def test_unsupported_format(self, tmp_path):
        from src.tools.data_loader import load_dataset
        bad_file = str(tmp_path / "test.txt")
        with open(bad_file, "w") as f:
            f.write("data")
        with pytest.raises(ValueError, match="Unsupported"):
            load_dataset(bad_file)

    def test_get_dataframe(self, sample_csv):
        from src.tools.data_loader import get_dataframe
        df = get_dataframe(sample_csv)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (200, 5)


# ─── EDA Tools Tests ─────────────────────────────────────────────────────────

class TestMissingValues:
    def test_detects_missing(self, sample_csv):
        from src.tools.eda_tools import analyze_missing_values
        result = analyze_missing_values(sample_csv)
        assert result["missing_counts"]["age"] == 20
        assert result["missing_counts"]["income"] == 5
        assert result["missing_percentages"]["age"] == pytest.approx(10.0, abs=0.1)

    def test_complete_columns(self, sample_csv):
        from src.tools.eda_tools import analyze_missing_values
        result = analyze_missing_values(sample_csv)
        assert "score" in result["complete_columns"]
        assert "category" in result["complete_columns"]

    def test_total_missing(self, sample_csv):
        from src.tools.eda_tools import analyze_missing_values
        result = analyze_missing_values(sample_csv)
        assert result["total_missing_cells"] == 25  # 20 + 5


class TestOutlierDetection:
    def test_detects_outliers(self, sample_csv):
        from src.tools.eda_tools import detect_outliers
        result = detect_outliers(sample_csv)
        assert "score" in result
        assert result["score"]["iqr_outliers"] >= 2  # at least our 2 injected outliers

    def test_has_bounds(self, sample_csv):
        from src.tools.eda_tools import detect_outliers
        result = detect_outliers(sample_csv)
        for col, info in result.items():
            assert "iqr_lower_bound" in info
            assert "iqr_upper_bound" in info
            assert info["iqr_upper_bound"] > info["iqr_lower_bound"]


class TestStatistics:
    def test_numeric_stats(self, sample_csv):
        from src.tools.eda_tools import compute_statistics
        result = compute_statistics(sample_csv)
        assert "numeric" in result
        assert "categorical" in result
        assert "age" in result["numeric"]
        stats = result["numeric"]["age"]
        assert all(k in stats for k in ["mean", "median", "std", "min", "max", "skewness"])

    def test_categorical_stats(self, sample_csv):
        from src.tools.eda_tools import compute_statistics
        result = compute_statistics(sample_csv)
        assert "category" in result["categorical"]
        cat = result["categorical"]["category"]
        assert cat["unique_values"] == 4


class TestCorrelations:
    def test_correlation_matrix(self, sample_csv):
        from src.tools.eda_tools import analyze_correlations
        result = analyze_correlations(sample_csv)
        assert "matrix" in result
        assert "top_correlations" in result
        assert len(result["top_correlations"]) > 0

    def test_correlation_bounds(self, sample_csv):
        from src.tools.eda_tools import analyze_correlations
        result = analyze_correlations(sample_csv)
        for pair in result["top_correlations"]:
            assert -1.0 <= pair["correlation"] <= 1.0
            assert pair["strength"] in ["strong", "moderate", "weak"]


# ─── Visualization Tests ──────────────────────────────────────────────────────

class TestVisualizations:
    def test_creates_charts(self, sample_csv, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from src.tools.data_loader import load_dataset
        from src.tools.eda_tools import (
            analyze_missing_values, detect_outliers,
            compute_statistics, analyze_correlations
        )
        from src.tools.visualization import create_visualizations

        load_dataset(sample_csv)
        paths = create_visualizations(
            dataset_path=sample_csv,
            missing_analysis=analyze_missing_values(sample_csv),
            outlier_analysis=detect_outliers(sample_csv),
            statistics=compute_statistics(sample_csv),
            correlations=analyze_correlations(sample_csv),
        )
        assert len(paths) >= 4
        for p in paths:
            if p:
                assert os.path.exists(p)


# ─── Pipeline Integration Test ────────────────────────────────────────────────

class TestPipeline:
    def test_full_pipeline_no_llm(self, sample_csv, tmp_path, monkeypatch):
        """Test pipeline runs end to end without Groq key"""
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        from src.agents.eda_agent import run_eda_pipeline

        state = run_eda_pipeline(sample_csv)
        assert state["df_summary"] is not None
        assert state["missing_analysis"] is not None
        assert state["outlier_analysis"] is not None
        assert state["statistics"] is not None
        assert state["report"] is not None
        assert state["current_step"] == "complete"

    def test_pipeline_produces_report(self, sample_csv, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        from src.agents.eda_agent import run_eda_pipeline

        state = run_eda_pipeline(sample_csv)
        assert state["report"] is not None
        assert "Dataset Overview" in state["report"]
        assert "Missing Values" in state["report"]
