"""
Agentic AI Pipeline for Automated EDA
Uses LangGraph for orchestration + Groq (Llama 3) for intelligence
"""

import os
import json
from typing import TypedDict, Annotated, List, Optional
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool

from src.tools.data_loader import load_dataset
from src.tools.eda_tools import (
    analyze_missing_values,
    detect_outliers,
    compute_statistics,
    analyze_correlations,
    get_data_types,
)
from src.tools.visualization import create_visualizations
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─── State Definition ────────────────────────────────────────────────────────

class EDAState(TypedDict):
    messages: Annotated[list, add_messages]
    dataset_path: str
    df_summary: Optional[dict]
    missing_analysis: Optional[dict]
    outlier_analysis: Optional[dict]
    statistics: Optional[dict]
    correlations: Optional[dict]
    visualizations: Optional[List[str]]
    insights: Optional[str]
    report: Optional[str]
    current_step: str
    errors: List[str]


# ─── LLM Setup ────────────────────────────────────────────────────────────────

def get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set")
    return ChatGroq(
        model="llama3-70b-8192",
        temperature=0.1,
        max_tokens=4096,
        api_key=api_key,
    )


# ─── Agent Nodes ─────────────────────────────────────────────────────────────

def load_data_node(state: EDAState) -> EDAState:
    """Node: Load and profile the dataset"""
    logger.info("📥 Loading dataset...")
    try:
        summary = load_dataset(state["dataset_path"])
        state["df_summary"] = summary
        state["current_step"] = "data_loaded"
        state["messages"].append(
            AIMessage(content=f"✅ Dataset loaded: {summary['shape'][0]} rows × {summary['shape'][1]} columns")
        )
    except Exception as e:
        state["errors"].append(f"Data loading error: {str(e)}")
        state["current_step"] = "error"
    return state


def missing_values_node(state: EDAState) -> EDAState:
    """Node: Analyze missing values"""
    logger.info("🔍 Analyzing missing values...")
    try:
        result = analyze_missing_values(state["dataset_path"])
        state["missing_analysis"] = result
        state["current_step"] = "missing_analyzed"
        missing_count = sum(1 for v in result["missing_counts"].values() if v > 0)
        state["messages"].append(
            AIMessage(content=f"✅ Missing value analysis complete: {missing_count} columns have missing data")
        )
    except Exception as e:
        state["errors"].append(f"Missing values error: {str(e)}")
    return state


def outlier_detection_node(state: EDAState) -> EDAState:
    """Node: Detect outliers using IQR and Z-score"""
    logger.info("🎯 Detecting outliers...")
    try:
        result = detect_outliers(state["dataset_path"])
        state["outlier_analysis"] = result
        state["current_step"] = "outliers_detected"
        total_outliers = sum(v["iqr_outliers"] for v in result.values())
        state["messages"].append(
            AIMessage(content=f"✅ Outlier detection complete: {total_outliers} total outliers found across numeric columns")
        )
    except Exception as e:
        state["errors"].append(f"Outlier detection error: {str(e)}")
    return state


def statistics_node(state: EDAState) -> EDAState:
    """Node: Compute descriptive statistics"""
    logger.info("📊 Computing statistics...")
    try:
        stats = compute_statistics(state["dataset_path"])
        corr = analyze_correlations(state["dataset_path"])
        dtypes = get_data_types(state["dataset_path"])
        state["statistics"] = stats
        state["correlations"] = corr
        state["df_summary"]["dtypes"] = dtypes
        state["current_step"] = "statistics_computed"
        state["messages"].append(
            AIMessage(content="✅ Statistical analysis complete: descriptive stats and correlations computed")
        )
    except Exception as e:
        state["errors"].append(f"Statistics error: {str(e)}")
    return state


def visualization_node(state: EDAState) -> EDAState:
    """Node: Create visualizations"""
    logger.info("🎨 Creating visualizations...")
    try:
        viz_paths = create_visualizations(
            dataset_path=state["dataset_path"],
            missing_analysis=state["missing_analysis"],
            outlier_analysis=state["outlier_analysis"],
            statistics=state["statistics"],
            correlations=state["correlations"],
        )
        state["visualizations"] = viz_paths
        state["current_step"] = "visualized"
        state["messages"].append(
            AIMessage(content=f"✅ Created {len(viz_paths)} visualizations")
        )
    except Exception as e:
        state["errors"].append(f"Visualization error: {str(e)}")
    return state


def insights_node(state: EDAState) -> EDAState:
    """Node: Generate AI insights using Llama 3 via Groq"""
    logger.info("🧠 Generating AI insights with Llama 3...")
    try:
        llm = get_llm()

        context = f"""
You are an expert data scientist. Analyze this dataset profile and provide actionable insights.

DATASET SUMMARY:
- Shape: {state['df_summary']['shape']}
- Columns: {state['df_summary']['columns']}
- Data Types: {json.dumps(state['df_summary'].get('dtypes', {}), indent=2)}

MISSING VALUES:
{json.dumps(state['missing_analysis'], indent=2)}

OUTLIER ANALYSIS:
{json.dumps(state['outlier_analysis'], indent=2)}

DESCRIPTIVE STATISTICS (sample):
{json.dumps({k: v for k, v in list(state['statistics'].items())[:5]}, indent=2)}

CORRELATION HIGHLIGHTS:
{json.dumps(state['correlations'].get('top_correlations', []), indent=2)}

Please provide:
1. **Data Quality Assessment** - Issues found and severity
2. **Key Statistical Patterns** - Notable distributions and trends
3. **Outlier Insights** - Which outliers matter and why
4. **Correlation Findings** - Important relationships between variables
5. **Missing Data Strategy** - Recommended imputation approaches
6. **Modeling Readiness** - What preprocessing is needed before ML
7. **Top 3 Actionable Recommendations** for next steps
"""

        response = llm.invoke([
            SystemMessage(content="You are an expert data scientist specializing in exploratory data analysis. Be specific, actionable, and concise."),
            HumanMessage(content=context)
        ])

        state["insights"] = response.content
        state["current_step"] = "insights_generated"
        state["messages"].append(
            AIMessage(content="✅ AI insights generated using Llama 3")
        )
    except Exception as e:
        state["errors"].append(f"Insights generation error: {str(e)}")
        state["insights"] = "Insights unavailable - check GROQ_API_KEY"
    return state


def report_node(state: EDAState) -> EDAState:
    """Node: Compile final markdown report"""
    logger.info("📝 Compiling final report...")
    try:
        missing_df = state.get("missing_analysis", {})
        outlier_df = state.get("outlier_analysis", {})
        stats = state.get("statistics", {})
        viz = state.get("visualizations", [])

        # Build missing value table
        missing_rows = ""
        for col, count in missing_df.get("missing_counts", {}).items():
            pct = missing_df.get("missing_percentages", {}).get(col, 0)
            if count > 0:
                missing_rows += f"| {col} | {count} | {pct:.1f}% |\n"

        # Build outlier table
        outlier_rows = ""
        for col, info in outlier_df.items():
            outlier_rows += f"| {col} | {info['iqr_outliers']} | {info['zscore_outliers']} |\n"

        report = f"""# 🔬 Automated EDA Report
**Dataset:** `{state['dataset_path'].split('/')[-1]}`  
**Generated by:** Agentic AI Pipeline (LangGraph + Llama 3)

---

## 📋 Dataset Overview
| Property | Value |
|----------|-------|
| Rows | {state['df_summary']['shape'][0]:,} |
| Columns | {state['df_summary']['shape'][1]} |
| Numeric Columns | {state['df_summary'].get('numeric_cols', 'N/A')} |
| Categorical Columns | {state['df_summary'].get('categorical_cols', 'N/A')} |

---

## 🚨 Missing Values
| Column | Missing Count | Missing % |
|--------|--------------|-----------|
{missing_rows if missing_rows else "| No missing values found | - | - |"}

---

## 🎯 Outlier Detection (IQR & Z-Score)
| Column | IQR Outliers | Z-Score Outliers |
|--------|-------------|-----------------|
{outlier_rows if outlier_rows else "| No numeric columns | - | - |"}

---

## 📊 Descriptive Statistics
{_format_stats_table(stats)}

---

## 📈 Visualizations Generated
{chr(10).join(f'- `{v}`' for v in viz)}

---

## 🧠 AI-Generated Insights (Llama 3)

{state.get('insights', 'No insights generated')}

---

## ⚠️ Errors Encountered
{chr(10).join(f'- {e}' for e in state['errors']) if state['errors'] else 'None'}

---
*Report generated by Agentic EDA Pipeline | LangGraph + Llama 3 via Groq*
"""
        state["report"] = report

        # Save report
        os.makedirs("outputs", exist_ok=True)
        report_path = "outputs/eda_report.md"
        with open(report_path, "w") as f:
            f.write(report)

        state["current_step"] = "complete"
        state["messages"].append(
            AIMessage(content=f"✅ Report saved to `{report_path}`")
        )
        logger.info("✅ EDA Pipeline complete!")
    except Exception as e:
        state["errors"].append(f"Report generation error: {str(e)}")
    return state


def _format_stats_table(stats: dict) -> str:
    if not stats:
        return "No statistics available"
    rows = []
    for col, s in list(stats.items())[:8]:
        rows.append(
            f"| {col} | {s.get('mean', 'N/A'):.3f} | {s.get('std', 'N/A'):.3f} | "
            f"{s.get('min', 'N/A'):.3f} | {s.get('max', 'N/A'):.3f} |"
            if all(isinstance(s.get(k), (int, float)) for k in ['mean','std','min','max'])
            else f"| {col} | N/A | N/A | N/A | N/A |"
        )
    header = "| Column | Mean | Std | Min | Max |\n|--------|------|-----|-----|-----|"
    return header + "\n" + "\n".join(rows)


# ─── Routing Logic ────────────────────────────────────────────────────────────

def should_continue(state: EDAState) -> str:
    if state["current_step"] == "error":
        return "end"
    return "continue"


# ─── Graph Construction ───────────────────────────────────────────────────────

def build_eda_graph() -> StateGraph:
    graph = StateGraph(EDAState)

    # Add nodes
    graph.add_node("load_data", load_data_node)
    graph.add_node("missing_values", missing_values_node)
    graph.add_node("outlier_detection", outlier_detection_node)
    graph.add_node("statistics", statistics_node)
    graph.add_node("visualization", visualization_node)
    graph.add_node("insights", insights_node)
    graph.add_node("report", report_node)

    # Define flow
    graph.set_entry_point("load_data")
    graph.add_edge("load_data", "missing_values")
    graph.add_edge("missing_values", "outlier_detection")
    graph.add_edge("outlier_detection", "statistics")
    graph.add_edge("statistics", "visualization")
    graph.add_edge("visualization", "insights")
    graph.add_edge("insights", "report")
    graph.add_edge("report", END)

    return graph.compile()


def run_eda_pipeline(dataset_path: str) -> EDAState:
    """Main entry point to run the full EDA pipeline"""
    graph = build_eda_graph()

    initial_state: EDAState = {
        "messages": [HumanMessage(content=f"Run EDA on: {dataset_path}")],
        "dataset_path": dataset_path,
        "df_summary": None,
        "missing_analysis": None,
        "outlier_analysis": None,
        "statistics": None,
        "correlations": None,
        "visualizations": None,
        "insights": None,
        "report": None,
        "current_step": "start",
        "errors": [],
    }

    final_state = graph.invoke(initial_state)
    return final_state
