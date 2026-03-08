# FinancialAI — Intelligent Financial Document Analyzer

**Final Class Project | Python for Everyone**

A Python application that combines three AI/ML technologies to analyze financial data:

1. **OCR** — Read any image (documents, charts, invoices) and extract text using deep learning
2. **Machine Learning** — Predict stock prices using a Random Forest Regressor (scikit-learn)
3. **Local LLM** — Chat with your financial data using a private AI running on your own machine

---

## Project Structure

```
Final-Project/
├── main.py                  # Entry point — menu-driven CLI
├── requirements.txt         # Python dependencies
├── README.md                # This file
│
├── scripts/                 # All source modules
│   ├── __init__.py
│   ├── ocr_engine.py        # Image → text (EasyOCR / deep learning)
│   ├── ml_predictor.py      # Stock price prediction (Random Forest)
│   ├── local_llm.py         # Local LLM integration via Ollama
│   ├── financial_data.py    # Portfolio analysis (extends Class-6 work)
│   └── demo_generator.py    # Generates sample financial images for demo
│
└── data/                    # All data inputs and outputs
    └── demo_images/         # Auto-generated demo images for OCR testing
        ├── financial_report.png
        ├── invoice.png
        └── stock_chart.png
```

---

## Features

### Module 1 — OCR (Image to Text)

- Uses **EasyOCR** (CRNN-based deep learning model) — no external binaries needed
- Preprocessing pipeline: resize, contrast enhancement, sharpening
- Outputs text with per-region confidence scores
- Detects financial values: dollar amounts, percentages, large numbers
- Works on: scanned reports, invoices, chart annotations, screenshots

### Module 2 — Machine Learning (Stock Price Prediction)

- **Model**: `RandomForestRegressor` (200 trees, max depth 10)
- **Features engineered** from raw OHLCV data:
  - Moving Averages: MA-5, MA-10, MA-20, MA-50
  - RSI (Relative Strength Index, 14-period)
  - MACD and Signal Line
  - Bollinger Bands (upper, lower, width)
  - Volume moving average and change
  - Daily returns and 20-day rolling volatility
- **Evaluation**: R² score, RMSE, MAE on held-out test set (last 20%)
- **Output**: Next-day price prediction + feature importance chart

### Module 3 — Local LLM (Private AI Chat)

- Connects to **Ollama** — a local LLM runtime that runs models on your machine
- All data stays on your private network — nothing sent externally
- Supports streaming output (tokens appear as generated)
- **RAG (Retrieval Augmented Generation)**: OCR text + ML predictions are injected into the model's context for grounded, data-specific answers
- Multi-turn conversation history maintained per session

### Module 4 — Portfolio Analysis (Class-6 Extended)

Builds on the `financial_functions.py` and `funds.py` work from Class-6:

- Sharpe Ratio (risk-adjusted returns)
- Max Drawdown (worst peak-to-trough decline)
- Annualized Volatility
- Correlation matrix heatmap across portfolio
- SPY top-holdings auto-portfolio (from `funds.py` pattern)

---

## Setup

### 1. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac / Linux
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

> **Note:** EasyOCR downloads its model (~100 MB) on first run.

### 3. Install Ollama for the Local LLM module

Download from **https://ollama.com** and install.

Then pull a model:

```bash
ollama pull llama3.2           # Recommended — general purpose
ollama pull dolphin-mistral    # Uncensored — no content restrictions
ollama pull mistral            # Fast 7B model
```

Ollama runs as a background service on `http://localhost:11434`.

### 4. Run the application

```bash
python main.py
```

---

## Usage

When you run `main.py` you get an interactive menu:

```
[1]  OCR  - Image to Text
[2]  ML   - Stock Price Prediction (Random Forest)
[3]  LLM  - Local Private AI Chat
[4]  Portfolio Analysis (Class-6 extended)
[5]  Full Pipeline Demo (OCR + ML + LLM)
[6]  About this project
[0]  Exit
```

**Recommended demo flow for presentation:**

1. Run `[1]` → generate demo images → run OCR on `financial_report.png`
2. Run `[2]` → enter `AAPL` → see ML prediction and feature importance chart
3. Run `[3]` → chat with the LLM using the ML results as context
4. Or run `[5]` (Full Pipeline) to do all three steps automatically

---

## Technologies Used

| Library | Purpose |
|---------|---------|
| `yfinance` | Stock & ETF historical data (Yahoo Finance) |
| `pandas` | DataFrame manipulation, time-series |
| `numpy` | Numerical computing |
| `scikit-learn` | Random Forest, StandardScaler, metrics |
| `easyocr` | Deep learning OCR (CRNN architecture) |
| `Pillow` | Image loading and preprocessing |
| `matplotlib` | Charts and visualizations |
| `seaborn` | Correlation heatmaps |
| `requests` | HTTP client for Ollama REST API |
| `Ollama` | Local LLM runtime (llama3.2, dolphin-mistral, etc.) |

---

## Python Concepts Demonstrated

- **Object-Oriented Programming** — 5 classes (`OCREngine`, `StockPredictor`, `LocalLLM`, `FinancialDataFetcher`)
- **Modules & Packages** — clean `scripts/` package with `__init__.py`
- **Exception Handling** — try/except in all I/O and network calls
- **File I/O** — image reading, CSV/Excel export to `data/`
- **Data structures** — DataFrames, Series, dicts, lists
- **Functions** — feature engineering, preprocessing, metric calculation
- **API Communication** — REST HTTP requests to Ollama streaming API
- **Machine Learning pipeline** — fetch → engineer → scale → train → evaluate → predict
- **RAG pattern** — context injection for grounded LLM responses

---

## Ollama Model Guide

| Model | Size | Notes |
|-------|------|-------|
| `llama3.2` | ~2 GB | Best for financial analysis. Recommended. |
| `dolphin-mistral` | ~4 GB | Fine-tuned without RLHF restrictions. Fully local, no guardrails. |
| `mistral` | ~4 GB | Fast and capable 7B model. |
| `qwen2.5` | ~4 GB | Excellent multilingual support. |

**Minimum hardware for local LLM:**
- RAM: 8 GB (for 7B parameter models)
- Disk: ~5 GB per model
- GPU: Optional but significantly speeds up generation

---

## Building on Class-6 Work

This project extends the financial analytics work from Class-6:

| Class-6 File | Extended In |
|---|---|
| `financial_functions.py` | `scripts/financial_data.py` |
| `funds.py` (SPY holdings) | `get_spy_portfolio()` in `financial_data.py` |
| `stock-prices.py` (multi-ticker) | `plot_portfolio_performance()` |
| `plot.py` / `keg.py` | ML prediction charts in `ml_predictor.py` |
