# 🔬 Agentic AI Pipeline for Automated EDA

> **An intelligent data analysis agent** that automatically explores your datasets using LangGraph orchestration + Llama 3 (via Groq) for AI-powered insights.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-purple?logo=langchain)](https://langchain-ai.github.io/langgraph/)
[![Llama 3](https://img.shields.io/badge/Llama_3-70B-orange)](https://groq.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?logo=streamlit)](https://streamlit.io)
[![CI](https://github.com/YOUR_USERNAME/agentic-eda-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/agentic-eda-pipeline/actions)

---

## 🎯 What It Does

Upload any tabular dataset and the agent **automatically**:

| Step | Agent Node | Description |
|------|-----------|-------------|
| 1 | 📥 **Load Data** | Reads CSV/Excel/JSON/Parquet, detects types, shape |
| 2 | 🔍 **Missing Values** | Maps nulls with severity levels, imputation suggestions |
| 3 | 🎯 **Outlier Detection** | IQR + Z-score across all numeric columns |
| 4 | 📊 **Statistics** | Mean, std, skewness, kurtosis, quartiles, correlations |
| 5 | 🎨 **Visualizations** | 6 publication-quality auto-generated charts |
| 6 | 🧠 **AI Insights** | Llama 3 analyzes results → actionable recommendations |
| 7 | 📝 **Report** | Full Markdown report, downloadable |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph Pipeline                        │
│                                                              │
│  load_data → missing_values → outlier_detection             │
│      ↓                                                       │
│  statistics → visualization → insights (Llama 3) → report   │
└─────────────────────────────────────────────────────────────┘
        ↑
   Streamlit UI (app.py)
```

Each step is a **LangGraph node** with typed state passing between them. Llama 3 runs on Groq's ultra-fast inference API.

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/YOUR_USERNAME/agentic-eda-pipeline.git
cd agentic-eda-pipeline
pip install -r requirements.txt
```

### 2. Set Up API Key
```bash
cp .env.example .env
# Edit .env and add your Groq API key
# Get one free at: https://console.groq.com
```

### 3. Run the App
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) and upload your dataset!

---

## 🌐 Live Demo

**Deployed on Streamlit Cloud:** [your-app.streamlit.app](https://your-app.streamlit.app)

---

## 📂 Project Structure

```
agentic-eda-pipeline/
├── app.py                      # Streamlit UI
├── src/
│   ├── agents/
│   │   └── eda_agent.py        # LangGraph pipeline (main brain)
│   ├── tools/
│   │   ├── data_loader.py      # Multi-format dataset loading
│   │   ├── eda_tools.py        # Missing values, outliers, stats, correlations
│   │   └── visualization.py    # 6 auto-generated chart types
│   └── utils/
│       └── logger.py
├── tests/
│   └── test_pipeline.py        # Pytest test suite
├── .github/
│   └── workflows/ci.yml        # GitHub Actions CI
├── .streamlit/config.toml      # Streamlit dark theme
├── requirements.txt
└── .env.example
```

---

## 🛠️ Tech Stack

- **[LangGraph](https://langchain-ai.github.io/langgraph/)** — Agent orchestration with typed state graph
- **[Llama 3 70B](https://groq.com)** via Groq — AI insights generation
- **[Streamlit](https://streamlit.io)** — Interactive web UI
- **Pandas + NumPy + SciPy** — Data processing
- **Matplotlib + Seaborn** — Visualizations

---

## ☁️ Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → Select your repo → `app.py`
4. Under **Secrets**, add:
   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   ```
5. Click **Deploy** 🚀

---

## 🧪 Run Tests

```bash
pip install pytest pytest-cov
pytest tests/ -v
```

---

## 📊 Sample Datasets

The app includes built-in sample datasets (no upload needed):
- **Titanic** — Classic survival prediction dataset
- **Iris** — Flower classification (good for correlations)
- **Tips** — Restaurant tipping patterns
- **Diamonds** — Diamond pricing data

---

## 📸 Screenshots

> Upload your dataset → Click Run → Get instant insights

| Dashboard | Visualizations | AI Insights |
|-----------|---------------|-------------|
| Overview metrics | 6 auto charts | Llama 3 recommendations |

---

## 🤝 Contributing

PRs welcome! Please add tests for new features.

---

## 📄 License

MIT License — see [LICENSE](LICENSE)

---

*Built with ❤️ using LangGraph + Llama 3 + Streamlit*
