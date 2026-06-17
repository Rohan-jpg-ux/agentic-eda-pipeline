"""
Streamlit UI for the Agentic EDA Pipeline
Run: streamlit run app.py
"""

import os
import sys
import json
import time
import tempfile
from pathlib import Path

import streamlit as st
import pandas as pd

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agentic EDA Pipeline",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stApp { background-color: #0f1117; }
    
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6c63ff, #43b89c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .hero-sub {
        color: #888;
        font-size: 1.05rem;
        margin-top: 0.2rem;
        margin-bottom: 1.5rem;
    }
    .step-card {
        background: #1e2130;
        border: 1px solid #2d3148;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 8px 0;
    }
    .step-active { border-left: 4px solid #6c63ff; }
    .step-done   { border-left: 4px solid #43b89c; }
    .step-error  { border-left: 4px solid #ef5350; }

    .metric-card {
        background: #1e2130;
        border: 1px solid #2d3148;
        border-radius: 10px;
        padding: 14px 18px;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #6c63ff; }
    .metric-label { font-size: 0.8rem; color: #888; margin-top: 2px; }

    .insight-box {
        background: linear-gradient(135deg, #1a1d2e, #1e2538);
        border: 1px solid #6c63ff44;
        border-radius: 12px;
        padding: 20px 24px;
        margin: 12px 0;
    }

    div[data-testid="stSidebar"] { background-color: #151825; }
    .stButton>button {
        background: linear-gradient(135deg, #6c63ff, #43b89c);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        width: 100%;
    }
    .stButton>button:hover { opacity: 0.9; transform: translateY(-1px); }
</style>
""", unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔬 EDA Agent Config")
    st.markdown("---")

    groq_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Get your free key at console.groq.com",
    )
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key

    st.markdown("---")
    st.markdown("### 🔗 Pipeline Steps")
    steps = [
        ("📥", "Load Dataset"),
        ("🔍", "Missing Values"),
        ("🎯", "Outlier Detection"),
        ("📊", "Statistics"),
        ("🎨", "Visualizations"),
        ("🧠", "AI Insights (Llama 3)"),
        ("📝", "Generate Report"),
    ]
    for icon, name in steps:
        st.markdown(f"{icon} {name}")

    st.markdown("---")
    st.markdown("**Stack**")
    st.markdown("🦜 LangGraph · 🦙 Llama 3 · ⚡ Groq")
    st.markdown("📊 Pandas · Seaborn · Matplotlib")


# ─── Main UI ─────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🔬 Agentic EDA Pipeline</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Automated Exploratory Data Analysis powered by LangGraph + Llama 3</div>', unsafe_allow_html=True)

# Upload section
col1, col2 = st.columns([2, 1])
with col1:
    uploaded_file = st.file_uploader(
        "Upload your dataset",
        type=["csv", "xlsx", "xls", "json", "parquet"],
        help="Supports CSV, Excel, JSON, and Parquet",
    )

with col2:
    st.markdown("#### 📁 Or use a sample dataset")
    sample = st.selectbox(
        "Sample datasets",
        ["None", "Titanic", "Iris", "Tips", "Diamonds"],
        label_visibility="collapsed",
    )

# Handle sample datasets
dataset_path = None
df_preview = None

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as f:
        f.write(uploaded_file.getbuffer())
        dataset_path = f.name

    if uploaded_file.name.endswith(".csv"):
        df_preview = pd.read_csv(dataset_path)
    elif uploaded_file.name.endswith((".xlsx", ".xls")):
        df_preview = pd.read_excel(dataset_path)
    elif uploaded_file.name.endswith(".json"):
        df_preview = pd.read_json(dataset_path)

elif sample != "None":
    import seaborn as sns
    try:
        df_preview = sns.load_dataset(sample.lower())
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w") as f:
            df_preview.to_csv(f.name, index=False)
            dataset_path = f.name
        st.info(f"✅ Loaded sample dataset: **{sample}** ({df_preview.shape[0]:,} rows × {df_preview.shape[1]} cols)")
    except Exception as e:
        st.error(f"Could not load sample dataset: {e}")

# Preview
if df_preview is not None:
    with st.expander("👁️ Dataset Preview", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{df_preview.shape[0]:,}")
        c2.metric("Columns", df_preview.shape[1])
        c3.metric("Missing %", f"{df_preview.isnull().mean().mean()*100:.1f}%")
        c4.metric("Duplicates", df_preview.duplicated().sum())
        st.dataframe(df_preview.head(8), use_container_width=True)

# Run button
if dataset_path:
    st.markdown("---")
    run_col, _ = st.columns([1, 3])
    with run_col:
        run_clicked = st.button("🚀 Run EDA Pipeline", use_container_width=True)

    if run_clicked:
        if not os.getenv("GROQ_API_KEY"):
            st.error("⚠️ Please enter your Groq API key in the sidebar to enable AI insights.")
            st.info("The pipeline will run without AI insights if you proceed.")

        # Progress tracking
        progress_bar = st.progress(0)
        status_area = st.empty()
        results_container = st.container()

        steps_status = {
            "load_data": "⏳", "missing_values": "⏳", "outlier_detection": "⏳",
            "statistics": "⏳", "visualization": "⏳", "insights": "⏳", "report": "⏳",
        }

        def update_status(step, icon, msg):
            steps_status[step] = icon
            lines = []
            for s, i in steps_status.items():
                lines.append(f"{i} **{s.replace('_', ' ').title()}**")
            status_area.markdown("  |  ".join(lines))
            progress_bar.progress(
                list(steps_status.keys()).index(step) / len(steps_status)
            )

        try:
            with st.spinner("Running EDA pipeline..."):
                from src.agents.eda_agent import run_eda_pipeline

                update_status("load_data", "🔄", "Loading...")
                state = run_eda_pipeline(dataset_path)

                for step in list(steps_status.keys()):
                    steps_status[step] = "✅"
                progress_bar.progress(1.0)
                status_area.markdown("  |  ".join(f"✅ **{s.replace('_', ' ').title()}**" for s in steps_status))

            st.success("🎉 EDA Pipeline completed successfully!")

            # ─── Results Tabs ──────────────────────────────────────────────
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Overview", "🔍 Analysis", "📈 Visualizations", "🧠 AI Insights", "📝 Report"
            ])

            with tab1:
                st.markdown("### Dataset Summary")
                summary = state.get("df_summary", {})
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Rows", f"{summary.get('shape', [0,0])[0]:,}")
                c2.metric("Columns", summary.get('shape', [0,0])[1])
                c3.metric("Numeric Cols", summary.get('numeric_cols', 0))
                c4.metric("Categorical Cols", summary.get('categorical_cols', 0))

                missing = state.get("missing_analysis", {})
                overall_miss = missing.get("overall_missing_pct", 0)
                dups = summary.get("duplicate_rows", 0)
                c5, c6, c7, c8 = st.columns(4)
                c5.metric("Missing %", f"{overall_miss:.1f}%")
                c6.metric("Duplicate Rows", dups)
                c7.metric("Memory", f"{summary.get('memory_mb', 0):.1f} MB")
                c8.metric("Complete Rows", f"{missing.get('rows_with_missing', 0):,} affected")

            with tab2:
                st.markdown("### Missing Values")
                miss_data = state.get("missing_analysis", {})
                miss_counts = miss_data.get("missing_counts", {})
                miss_pct = miss_data.get("missing_percentages", {})
                if any(v > 0 for v in miss_counts.values()):
                    miss_df = pd.DataFrame({
                        "Column": list(miss_counts.keys()),
                        "Missing Count": list(miss_counts.values()),
                        "Missing %": [miss_pct.get(k, 0) for k in miss_counts],
                    }).sort_values("Missing %", ascending=False)
                    st.dataframe(miss_df[miss_df["Missing Count"] > 0], use_container_width=True)
                else:
                    st.success("✅ No missing values found!")

                st.markdown("### Outlier Summary")
                outlier_data = state.get("outlier_analysis", {})
                if outlier_data:
                    out_df = pd.DataFrame([
                        {"Column": col, "IQR Outliers": v["iqr_outliers"],
                         "Z-Score Outliers": v["zscore_outliers"],
                         "Outlier %": f"{v['outlier_pct_iqr']:.1f}%"}
                        for col, v in outlier_data.items()
                    ]).sort_values("IQR Outliers", ascending=False)
                    st.dataframe(out_df, use_container_width=True)

                st.markdown("### Top Correlations")
                corrs = state.get("correlations", {}).get("top_correlations", [])
                if corrs:
                    corr_df = pd.DataFrame(corrs[:10])
                    st.dataframe(corr_df, use_container_width=True)

            with tab3:
                st.markdown("### Generated Visualizations")
                viz_paths = state.get("visualizations", [])
                if viz_paths:
                    for path in viz_paths:
                        if os.path.exists(path):
                            st.image(path, use_column_width=True)
                            st.markdown("---")
                else:
                    st.warning("No visualizations generated")

            with tab4:
                st.markdown("### 🧠 AI Insights — Llama 3 via Groq")
                insights = state.get("insights", "")
                if insights and "unavailable" not in insights.lower():
                    st.markdown(f'<div class="insight-box">{insights}</div>', unsafe_allow_html=True)
                else:
                    st.warning("AI insights require a valid GROQ_API_KEY. Add it in the sidebar and re-run.")

            with tab5:
                st.markdown("### 📝 Full EDA Report")
                report = state.get("report", "")
                if report:
                    st.markdown(report)
                    st.download_button(
                        "⬇️ Download Report (Markdown)",
                        data=report,
                        file_name="eda_report.md",
                        mime="text/markdown",
                    )

        except Exception as e:
            st.error(f"Pipeline error: {str(e)}")
            st.exception(e)
else:
    st.markdown("""
    <div class="step-card step-active">
    <b>👆 Get Started</b><br>
    Upload a CSV, Excel, JSON, or Parquet dataset — or pick a sample from the dropdown.
    Then hit <b>Run EDA Pipeline</b> to let the agent do its magic.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🤖 What this agent does automatically:")
    cols = st.columns(3)
    features = [
        ("📥", "Load & Profile", "Reads any tabular dataset, detects types, shapes, duplicates"),
        ("🔍", "Missing Values", "Maps missing data with severity levels and imputation suggestions"),
        ("🎯", "Outlier Detection", "IQR + Z-score methods across all numeric columns"),
        ("📊", "Statistics", "Full descriptive stats: mean, std, skewness, kurtosis, quantiles"),
        ("🔗", "Correlations", "Pearson matrix + top correlated pairs ranked by strength"),
        ("🎨", "Visualizations", "6 publication-quality charts auto-generated"),
        ("🧠", "AI Insights", "Llama 3 analyzes results and gives actionable recommendations"),
        ("📝", "Report", "Full Markdown report downloadable with all findings"),
        ("🦜", "LangGraph", "Each step is a node in a directed agent graph"),
    ]
    for i, (icon, title, desc) in enumerate(features):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="step-card">
            <b>{icon} {title}</b><br>
            <span style="color:#888;font-size:0.85rem">{desc}</span>
            </div>
            """, unsafe_allow_html=True)
