"""
Automated visualization generation for the EDA pipeline.
Creates publication-quality charts using matplotlib and seaborn.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for servers
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from src.tools.data_loader import get_dataframe
from src.utils.logger import get_logger

warnings.filterwarnings("ignore")
logger = get_logger(__name__)

# ─── Style Configuration ──────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor": "#1e2130",
    "axes.edgecolor": "#444",
    "axes.labelcolor": "#e0e0e0",
    "text.color": "#e0e0e0",
    "xtick.color": "#aaa",
    "ytick.color": "#aaa",
    "grid.color": "#333",
    "grid.linewidth": 0.5,
    "font.family": "DejaVu Sans",
})

ACCENT_COLORS = ["#6c63ff", "#ff6584", "#43b89c", "#f9a825", "#29b6f6", "#ef5350"]
OUTPUT_DIR = "outputs/charts"


def _ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_visualizations(
    dataset_path: str,
    missing_analysis: dict,
    outlier_analysis: dict,
    statistics: dict,
    correlations: dict,
) -> list:
    """Create all EDA visualizations and return list of saved paths"""
    _ensure_output_dir()
    df = get_dataframe(dataset_path)
    paths = []

    try:
        p = plot_overview_dashboard(df, missing_analysis, statistics)
        paths.append(p)
    except Exception as e:
        logger.warning(f"Overview dashboard failed: {e}")

    try:
        p = plot_missing_values(df, missing_analysis)
        paths.append(p)
    except Exception as e:
        logger.warning(f"Missing values chart failed: {e}")

    try:
        p = plot_distributions(df, statistics)
        paths.append(p)
    except Exception as e:
        logger.warning(f"Distributions chart failed: {e}")

    try:
        p = plot_correlation_heatmap(df, correlations)
        paths.append(p)
    except Exception as e:
        logger.warning(f"Correlation heatmap failed: {e}")

    try:
        p = plot_outlier_boxplots(df, outlier_analysis)
        paths.append(p)
    except Exception as e:
        logger.warning(f"Boxplot chart failed: {e}")

    try:
        p = plot_categorical_distributions(df)
        paths.append(p)
    except Exception as e:
        logger.warning(f"Categorical chart failed: {e}")

    logger.info(f"Created {len(paths)} visualizations")
    return [p for p in paths if p]


def plot_overview_dashboard(df: pd.DataFrame, missing: dict, stats: dict) -> str:
    """Overview dashboard: shape, types, missing summary"""
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor("#0f1117")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.4)

    # Title
    fig.suptitle("📊 Dataset Overview Dashboard", fontsize=20, color="#6c63ff", y=0.98, fontweight="bold")

    # 1. Data types pie
    ax1 = fig.add_subplot(gs[0, 0])
    dtype_counts = df.dtypes.apply(lambda x: "Numeric" if pd.api.types.is_numeric_dtype(x)
                                    else "Datetime" if pd.api.types.is_datetime64_any_dtype(x)
                                    else "Categorical").value_counts()
    ax1.pie(dtype_counts.values, labels=dtype_counts.index,
            colors=ACCENT_COLORS[:len(dtype_counts)],
            autopct="%1.0f%%", startangle=90,
            textprops={"color": "#e0e0e0", "fontsize": 10})
    ax1.set_title("Column Types", color="#e0e0e0", fontsize=11)

    # 2. Missing values bar
    ax2 = fig.add_subplot(gs[0, 1])
    miss_pct = {k: v for k, v in missing.get("missing_percentages", {}).items() if v > 0}
    if miss_pct:
        cols = list(miss_pct.keys())[:10]
        vals = [miss_pct[c] for c in cols]
        bars = ax2.barh(cols, vals, color=ACCENT_COLORS[1], alpha=0.85)
        ax2.set_xlabel("Missing %", color="#aaa")
        ax2.axvline(x=50, color="#f9a825", linestyle="--", linewidth=1, label="50%")
        ax2.legend(fontsize=8)
        for bar, val in zip(bars, vals):
            ax2.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                     f"{val:.1f}%", va="center", color="#e0e0e0", fontsize=8)
    else:
        ax2.text(0.5, 0.5, "✅ No Missing\nValues!", ha="center", va="center",
                 fontsize=14, color="#43b89c", transform=ax2.transAxes)
    ax2.set_title("Missing Data", color="#e0e0e0", fontsize=11)

    # 3. Dataset stats summary
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis("off")
    rows_val = df.shape[0]
    cols_val = df.shape[1]
    dup_val = int(df.duplicated().sum())
    total_missing = missing.get("total_missing_cells", 0)
    summary_text = [
        ("Rows", f"{rows_val:,}"),
        ("Columns", f"{cols_val}"),
        ("Duplicates", f"{dup_val:,}"),
        ("Missing Cells", f"{total_missing:,}"),
        ("Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB"),
        ("Complete Rows", f"{int((df.notna().all(axis=1)).sum()):,}"),
    ]
    for i, (label, value) in enumerate(summary_text):
        y_pos = 0.85 - i * 0.15
        ax3.text(0.05, y_pos, f"{label}:", color="#aaa", fontsize=11, transform=ax3.transAxes)
        ax3.text(0.6, y_pos, value, color="#6c63ff", fontsize=11,
                 fontweight="bold", transform=ax3.transAxes)
    ax3.set_title("Quick Stats", color="#e0e0e0", fontsize=11)

    # 4. Numeric column skewness
    ax4 = fig.add_subplot(gs[1, :2])
    num_stats = stats.get("numeric", {})
    if num_stats:
        cols_s = list(num_stats.keys())[:12]
        skews = [num_stats[c]["skewness"] for c in cols_s]
        colors = [ACCENT_COLORS[0] if s >= 0 else ACCENT_COLORS[1] for s in skews]
        ax4.bar(cols_s, skews, color=colors, alpha=0.85)
        ax4.axhline(y=0, color="#fff", linewidth=0.8)
        ax4.axhline(y=0.5, color="#f9a825", linewidth=0.7, linestyle="--", label="Skew threshold")
        ax4.axhline(y=-0.5, color="#f9a825", linewidth=0.7, linestyle="--")
        ax4.set_xticklabels(cols_s, rotation=30, ha="right", fontsize=8)
        ax4.set_ylabel("Skewness", color="#aaa")
        ax4.legend(fontsize=8)
    ax4.set_title("Column Skewness", color="#e0e0e0", fontsize=11)

    # 5. Heatmap of nullity
    ax5 = fig.add_subplot(gs[1, 2])
    null_matrix = df.isnull().astype(int)
    if null_matrix.sum().sum() > 0:
        sample = null_matrix.sample(min(50, len(null_matrix))).T
        sns.heatmap(sample, ax=ax5, cmap="RdYlGn_r", cbar=False, linewidths=0)
        ax5.set_xticklabels([])
        ax5.set_yticklabels(ax5.get_yticklabels(), fontsize=7)
    else:
        ax5.text(0.5, 0.5, "✅ Complete\nDataset", ha="center", va="center",
                 color="#43b89c", fontsize=13, transform=ax5.transAxes)
    ax5.set_title("Null Pattern Map", color="#e0e0e0", fontsize=11)

    path = f"{OUTPUT_DIR}/01_overview_dashboard.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    return path


def plot_missing_values(df: pd.DataFrame, missing: dict) -> str:
    """Detailed missing values visualization"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle("🔍 Missing Values Analysis", fontsize=16, color="#6c63ff", fontweight="bold")

    miss_pct = missing.get("missing_percentages", {})
    all_cols = list(miss_pct.keys())
    all_vals = [miss_pct[c] for c in all_cols]

    # Left: full missing % bar chart
    ax = axes[0]
    colors = ["#ef5350" if v > 50 else "#f9a825" if v > 10 else "#43b89c" if v > 0 else "#444"
              for v in all_vals]
    ax.barh(all_cols[:20], all_vals[:20], color=colors[:20], alpha=0.9)
    ax.set_xlabel("Missing Percentage (%)", color="#aaa")
    ax.axvline(x=50, color="#ef5350", linestyle="--", linewidth=1.2, label=">50% (Drop)")
    ax.axvline(x=10, color="#f9a825", linestyle="--", linewidth=1.2, label=">10% (Impute)")
    ax.legend(fontsize=8)
    ax.set_title("Missing % per Column", color="#e0e0e0")

    # Right: missing severity donut
    ax2 = axes[1]
    high = len(missing.get("high_missing_cols", []))
    med = len(missing.get("medium_missing_cols", []))
    low = len(missing.get("low_missing_cols", []))
    complete = len(missing.get("complete_columns", []))
    sizes = [high, med, low, complete]
    labels = [f"High >50%\n({high})", f"Medium 10-50%\n({med})",
               f"Low <10%\n({low})", f"Complete\n({complete})"]
    explode = [0.05] * 4
    wedge_colors = ["#ef5350", "#f9a825", "#43b89c", "#6c63ff"]
    non_zero = [(s, l, c) for s, l, c in zip(sizes, labels, wedge_colors) if s > 0]
    if non_zero:
        s, l, c = zip(*non_zero)
        wedges, texts, autotexts = ax2.pie(
            s, labels=l, colors=c, autopct="%1.0f%%",
            startangle=90, textprops={"color": "#e0e0e0", "fontsize": 9},
        )
    ax2.set_title("Missing Severity Distribution", color="#e0e0e0")

    path = f"{OUTPUT_DIR}/02_missing_values.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    return path


def plot_distributions(df: pd.DataFrame, stats: dict) -> str:
    """Distribution plots for all numeric columns"""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        return None

    n_cols = min(len(numeric_cols), 12)
    ncols = 3
    nrows = (n_cols + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle("📈 Feature Distributions", fontsize=16, color="#6c63ff",
                 y=1.01, fontweight="bold")
    axes = axes.flatten() if nrows > 1 else [axes] if ncols == 1 else axes.flatten()

    for i, col in enumerate(numeric_cols[:n_cols]):
        ax = axes[i]
        series = df[col].dropna()
        ax.hist(series, bins=30, color=ACCENT_COLORS[i % len(ACCENT_COLORS)],
                alpha=0.8, edgecolor="none")
        ax.axvline(series.mean(), color="#fff", linewidth=1.2, linestyle="--", label="Mean")
        ax.axvline(series.median(), color="#f9a825", linewidth=1.2, linestyle=":", label="Median")
        ax.set_title(col, color="#e0e0e0", fontsize=9)
        ax.legend(fontsize=7)
        if col in stats.get("numeric", {}):
            s = stats["numeric"][col]
            ax.text(0.97, 0.97, f"skew: {s['skewness']:.2f}", transform=ax.transAxes,
                    ha="right", va="top", color="#aaa", fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/03_distributions.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    return path


def plot_correlation_heatmap(df: pd.DataFrame, correlations: dict) -> str:
    """Correlation heatmap"""
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        return None

    corr = numeric_df.corr()
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle("🔗 Correlation Analysis", fontsize=16, color="#6c63ff", fontweight="bold")

    # Full heatmap
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, ax=axes[0], mask=mask, annot=len(corr) <= 12,
                cmap="coolwarm", center=0, fmt=".2f",
                annot_kws={"size": 7}, square=True,
                linewidths=0.3, linecolor="#333",
                cbar_kws={"shrink": 0.8})
    axes[0].set_title("Correlation Matrix", color="#e0e0e0")
    axes[0].tick_params(colors="#aaa", labelsize=8)

    # Top correlations bar chart
    top = correlations.get("top_correlations", [])[:10]
    if top:
        labels = [f"{p['col1']} × {p['col2']}" for p in top]
        vals = [p["correlation"] for p in top]
        colors = [ACCENT_COLORS[0] if v > 0 else ACCENT_COLORS[1] for v in vals]
        axes[1].barh(labels, vals, color=colors, alpha=0.85)
        axes[1].axvline(x=0, color="#fff", linewidth=0.8)
        axes[1].axvline(x=0.7, color="#43b89c", linestyle="--", linewidth=1, label="Strong +")
        axes[1].axvline(x=-0.7, color="#43b89c", linestyle="--", linewidth=1, label="Strong -")
        axes[1].set_xlabel("Pearson r", color="#aaa")
        axes[1].legend(fontsize=8)
        axes[1].set_title("Top 10 Correlations", color="#e0e0e0")

    path = f"{OUTPUT_DIR}/04_correlations.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    return path


def plot_outlier_boxplots(df: pd.DataFrame, outlier_analysis: dict) -> str:
    """Box plots for outlier visualization"""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        return None

    n_cols = min(len(numeric_cols), 12)
    ncols = 3
    nrows = (n_cols + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle("🎯 Outlier Analysis (Box Plots)", fontsize=16, color="#6c63ff",
                 y=1.01, fontweight="bold")
    axes = axes.flatten() if nrows > 1 else [axes] if ncols == 1 else axes.flatten()

    for i, col in enumerate(numeric_cols[:n_cols]):
        ax = axes[i]
        series = df[col].dropna()
        bp = ax.boxplot(series, patch_artist=True, vert=True,
                        medianprops={"color": "#fff", "linewidth": 2},
                        flierprops={"marker": "o", "markerfacecolor": ACCENT_COLORS[1],
                                    "markersize": 4, "alpha": 0.6},
                        whiskerprops={"color": "#aaa"},
                        capprops={"color": "#aaa"})
        bp["boxes"][0].set_facecolor(ACCENT_COLORS[i % len(ACCENT_COLORS)])
        bp["boxes"][0].set_alpha(0.7)

        outlier_info = outlier_analysis.get(col, {})
        n_out = outlier_info.get("iqr_outliers", 0)
        ax.set_title(f"{col}\n({n_out} outliers)", color="#e0e0e0", fontsize=9)
        ax.set_xticks([])

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/05_outlier_boxplots.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    return path


def plot_categorical_distributions(df: pd.DataFrame) -> str:
    """Bar charts for categorical columns"""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    # Filter to reasonable cardinality
    cat_cols = [c for c in cat_cols if df[c].nunique() <= 30][:9]
    if not cat_cols:
        return None

    ncols = 3
    nrows = (len(cat_cols) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle("🏷️ Categorical Distributions", fontsize=16, color="#6c63ff",
                 y=1.01, fontweight="bold")
    axes = axes.flatten() if nrows > 1 else [axes] if ncols == 1 else axes.flatten()

    for i, col in enumerate(cat_cols):
        ax = axes[i]
        vc = df[col].value_counts().head(10)
        ax.bar(range(len(vc)), vc.values,
               color=[ACCENT_COLORS[j % len(ACCENT_COLORS)] for j in range(len(vc))],
               alpha=0.85)
        ax.set_xticks(range(len(vc)))
        ax.set_xticklabels(vc.index, rotation=35, ha="right", fontsize=7)
        ax.set_title(f"{col} (top {len(vc)})", color="#e0e0e0", fontsize=9)
        ax.set_ylabel("Count", color="#aaa", fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/06_categorical.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    return path
