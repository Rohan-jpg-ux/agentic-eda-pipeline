#!/usr/bin/env bash
# ============================================================
# deploy.sh — Initialize git, push to GitHub, deploy to Streamlit Cloud
# Usage: bash deploy.sh <github-username> <repo-name>
# ============================================================

set -e

GITHUB_USER="${1:-YOUR_USERNAME}"
REPO_NAME="${2:-agentic-eda-pipeline}"
REPO_URL="https://github.com/${GITHUB_USER}/${REPO_NAME}.git"

echo ""
echo "🚀 Agentic EDA Pipeline — Deploy Script"
echo "======================================="
echo "GitHub: ${REPO_URL}"
echo ""

# ─── Step 1: Git init ─────────────────────────────────────────
if [ ! -d ".git" ]; then
    echo "📁 Initializing git repository..."
    git init
    git branch -M main
fi

# ─── Step 2: Stage all files ──────────────────────────────────
echo "📦 Staging files..."
git add .
git status --short

# ─── Step 3: Commit ───────────────────────────────────────────
echo ""
echo "💾 Creating initial commit..."
git commit -m "feat: Agentic AI Pipeline for Automated EDA

- LangGraph orchestration with 7-node pipeline
- Llama 3 (70B) via Groq for AI insights
- Missing values, outlier detection, statistics, correlations
- 6 auto-generated visualizations
- Streamlit dark-theme UI
- 15 pytest tests, all passing
- GitHub Actions CI/CD
- Streamlit Cloud deployment ready"

# ─── Step 4: Push to GitHub ────────────────────────────────────
echo ""
echo "🌐 Pushing to GitHub..."
echo "   → Make sure you've created the repo at: https://github.com/new"
echo "   → Repo name: ${REPO_NAME}"
echo ""

git remote remove origin 2>/dev/null || true
git remote add origin "${REPO_URL}"
git push -u origin main

echo ""
echo "✅ Code pushed to GitHub!"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🌐 DEPLOY TO STREAMLIT CLOUD"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "1. Go to: https://share.streamlit.io"
echo "2. Click: New app"
echo "3. Repository: ${GITHUB_USER}/${REPO_NAME}"
echo "4. Branch: main"
echo "5. Main file: app.py"
echo "6. Click: Advanced settings → Secrets:"
echo ""
echo "   GROQ_API_KEY = \"gsk_your_key_here\""
echo ""
echo "7. Click: Deploy!"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📝 UPDATE README with your live URL"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "sed -i 's|your-app.streamlit.app|YOUR_APP.streamlit.app|g' README.md"
echo "sed -i 's|YOUR_USERNAME|${GITHUB_USER}|g' README.md"
echo "git add README.md && git commit -m 'docs: add live demo link' && git push"
echo ""
