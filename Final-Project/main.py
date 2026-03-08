"""
FinancialAI - Intelligent Financial Document Analyzer
======================================================
Final Class Project | Python for Everyone

Combines three powerful AI/ML technologies:
  1. OCR (Optical Character Recognition) - Read images and convert to text
  2. Machine Learning - Random Forest stock price prediction
  3. Local LLM - Chat with your financial data using a private AI

Architecture:
  ocr_engine.py    -> Image to text using EasyOCR (deep learning based)
  ml_predictor.py  -> Stock prediction using scikit-learn Random Forest
  local_llm.py     -> Local LLM via Ollama (llama3.2, dolphin-mistral, etc.)
  financial_data.py -> Portfolio analysis building on Class-6 yfinance work

Author: Final Project - Python for Everyone
"""

import os
import sys
import time

# Add scripts/ directory to path so modules can be imported directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))


# ─────────────────────────────────────────────
#  BANNER
# ─────────────────────────────────────────────

BANNER = r"""
  _______ _                       _      _____          _
 |  ____(_)                     (_)    |  __ \        | |
 | |__   _ _ __   __ _ _ __   ___  __ _| |__) |__   __| |
 |  __| | | '_ \ / _` | '_ \ / __| |/ _` |  ___/ _ \ / _` |
 | |    | | | | | (_| | | | | (__| | (_| | |  |  __/ (_| |
 |_|    |_|_| |_|\__,_|_| |_|\___|_|\__,_|_|   \___|\__,_|

         LOCAL AI  |  PRIVATE NETWORK  |  UNCENSORED LLM
         Image OCR + ML Prediction + LLM Chat

         Final Project - Python for Everyone
"""


def clear():
    os.system('cls' if os.name == 'nt' else 'clear')


def print_separator(char='─', width=60):
    print(char * width)


def print_header(title):
    print()
    print_separator('═')
    print(f"  {title}")
    print_separator('═')


def pause():
    input("\n  [Press Enter to continue...]")


# ─────────────────────────────────────────────
#  MENU FUNCTIONS
# ─────────────────────────────────────────────

def menu_ocr():
    """OCR: Image to Text demonstration."""
    print_header("MODULE 1: OCR - IMAGE TO TEXT")
    print("""
  This module reads any image (financial documents, charts,
  invoices, screenshots) and extracts all text using deep
  learning based OCR (EasyOCR - CRNN architecture).

  Use cases:
    - Extract data from scanned financial reports
    - Read stock charts with price annotations
    - Parse invoices and receipts automatically
    - Digitize printed financial statements
""")

    print("  Options:")
    print("  [1] Run OCR on a demo-generated image")
    print("  [2] Run OCR on your own image file")
    print("  [3] Generate demo images (financial report, invoice, chart)")
    print("  [0] Back to main menu")

    choice = input("\n  Select: ").strip()

    if choice == '3':
        print("\n  Generating demo images...")
        try:
            from demo_generator import generate_all
            paths = generate_all()
            print(f"\n  Created {len(paths)} images.")
        except Exception as e:
            print(f"  Error generating images: {e}")
        pause()
        return

    if choice in ('1', '2'):
        from ocr_engine import OCREngine

        if choice == '1':
            demo_dir = os.path.join(os.path.dirname(__file__), "data", "demo_images")
            if not os.path.exists(demo_dir) or not os.listdir(demo_dir):
                print("\n  No demo images found. Generating them first...")
                try:
                    from demo_generator import generate_all
                    generate_all()
                except Exception as e:
                    print(f"  Could not generate demo images: {e}")
                    pause()
                    return

            images = [f for f in os.listdir(demo_dir) if f.endswith('.png')]
            if not images:
                print("  No images found in demo_images/")
                pause()
                return

            print("\n  Available demo images:")
            for i, img in enumerate(images, 1):
                print(f"    [{i}] {img}")

            img_choice = input("\n  Select image number: ").strip()
            try:
                image_path = os.path.join(demo_dir, images[int(img_choice) - 1])
            except (ValueError, IndexError):
                print("  Invalid selection.")
                pause()
                return
        else:
            image_path = input("\n  Enter full path to image: ").strip().strip('"')
            if not os.path.exists(image_path):
                print(f"  File not found: {image_path}")
                pause()
                return

        print(f"\n  Processing: {os.path.basename(image_path)}")
        engine = OCREngine()
        extracted = engine.display_results(image_path)

        if extracted:
            financial = engine.extract_financial_data(image_path)
            if financial['financial_items']:
                print("\n  DETECTED FINANCIAL VALUES:")
                for item in financial['financial_items'][:10]:
                    print(f"    {item}")

        pause()


def menu_ml():
    """ML: Stock price prediction with Random Forest."""
    print_header("MODULE 2: ML - STOCK PRICE PREDICTION")
    print("""
  This module uses a Random Forest Regressor (scikit-learn) to
  predict next-day stock closing prices.

  Machine Learning concepts demonstrated:
    - Feature Engineering (MA, RSI, MACD, Bollinger Bands)
    - Train/Test Split (80/20, time-ordered)
    - StandardScaler normalization
    - Random Forest ensemble (200 decision trees)
    - Model evaluation: R², RMSE, MAE
    - Feature importance analysis
""")

    ticker = input("  Enter stock ticker [AAPL]: ").strip().upper() or "AAPL"
    period = input("  Training period [2y] (1y/2y/5y): ").strip() or "2y"

    print(f"\n  Running ML pipeline for {ticker}...\n")

    from ml_predictor import StockPredictor

    predictor = StockPredictor(ticker=ticker, period=period)

    try:
        predictor.fetch_data()
        results = predictor.train()

        print("\n" + "─" * 50)
        print(f"  MODEL RESULTS for {ticker}")
        print("─" * 50)
        print(f"  R² Score:    {results['r2']:.4f}  (closer to 1.0 = better)")
        print(f"  RMSE:        ${results['rmse']:.2f}")
        print(f"  MAE:         ${results['mae']:.2f}")

        pred, current, change_pct = predictor.predict_next_day()
        direction = "↑ UP" if change_pct > 0 else "↓ DOWN"
        color_hint = "(bullish)" if change_pct > 0 else "(bearish)"

        print(f"\n  NEXT DAY PREDICTION:")
        print(f"  Current Price:   ${current:.2f}")
        print(f"  Predicted Price: ${pred:.2f}")
        print(f"  Expected Change: {direction} {abs(change_pct):.2f}% {color_hint}")

        importance = predictor.get_feature_importance()
        print(f"\n  TOP 5 MOST IMPORTANT FEATURES:")
        for feat, score in importance.head(5).items():
            bar = '█' * int(score * 100)
            print(f"  {feat:<20} {bar} {score:.3f}")

        plot_choice = input("\n  Show prediction chart? [y/n]: ").strip().lower()
        if plot_choice == 'y':
            save_path = os.path.join(os.path.dirname(__file__), "data", f"{ticker}_prediction.png")
            predictor.plot_results(save_path=save_path)

        # Store summary for LLM context
        return predictor.summary()

    except Exception as e:
        print(f"\n  Error: {e}")
        pause()
        return None


def menu_llm(ml_context="", ocr_context=""):
    """Local LLM: Chat with financial data."""
    print_header("MODULE 3: LOCAL LLM - PRIVATE AI CHAT")
    print("""
  This module connects to a locally running Ollama LLM server.
  All processing happens on YOUR machine / private network.
  No data is sent to external services.

  Supported models (install with 'ollama pull <model>'):
    llama3.2         - Meta's LLaMA 3.2 (recommended)
    dolphin-mistral  - Uncensored fine-tune, no restrictions
    mistral          - Fast and capable 7B model
    qwen2.5          - Excellent multilingual support
    llama3           - LLaMA 3 8B

  RAG (Retrieval Augmented Generation):
    Your ML predictions and OCR-extracted text are automatically
    injected into the LLM's context for grounded financial analysis.
""")

    from local_llm import LocalLLM

    llm = LocalLLM()

    print("  Checking Ollama server...")
    status = llm.get_status()
    print()
    for line in status.split('\n'):
        print(f"  {line}")
    print()

    if not llm.is_available():
        print("""
  SETUP GUIDE:
  ─────────────────────────────────────────────
  1. Download Ollama:    https://ollama.com
  2. Install and run:   ollama serve
  3. Pull a model:      ollama pull llama3.2
                        ollama pull dolphin-mistral
  4. Re-run this option
  ─────────────────────────────────────────────
""")
        pause()
        return

    models = llm.list_models()
    if models:
        print(f"  Available models: {', '.join(models)}")
        model_choice = input(f"  Select model [{models[0]}]: ").strip()
        llm.model = model_choice if model_choice in models else models[0]
    else:
        print("  No models found. Run: ollama pull llama3.2")
        pause()
        return

    print(f"\n  Using model: {llm.model}")

    # Build context from previous modules
    context_parts = []
    if ml_context:
        context_parts.append(f"ML PREDICTION RESULTS:\n{ml_context}")
    if ocr_context:
        context_parts.append(f"OCR EXTRACTED TEXT:\n{ocr_context}")

    combined_context = "\n\n".join(context_parts) if context_parts else ""

    if combined_context:
        print("\n  [Context from ML + OCR modules loaded into LLM]")
        print("  You can ask questions about your analysis results!")
    else:
        print("\n  [No context loaded - run ML and OCR modules first for grounded analysis]")
        print("  You can still ask general financial questions.")

    print()
    llm.interactive_chat(context=combined_context)
    pause()


def menu_portfolio():
    """Portfolio analysis using financial_data.py."""
    print_header("MODULE 4: PORTFOLIO ANALYSIS")
    print("""
  Advanced financial analytics building on Class-6 work.
  Includes Sharpe ratio, volatility, max drawdown, and
  correlation analysis across a portfolio of stocks.
""")

    print("  Options:")
    print("  [1] Analyze custom portfolio")
    print("  [2] Auto-analyze SPY top holdings (Class-6 pattern)")
    print("  [0] Back")

    choice = input("\n  Select: ").strip()

    if choice not in ('1', '2'):
        return

    from financial_data import FinancialDataFetcher

    if choice == '2':
        fetcher = FinancialDataFetcher()
        tickers = fetcher.get_spy_portfolio()
        fetcher.tickers = tickers[:8]  # top 8 to keep it manageable
    else:
        raw = input("  Enter tickers (comma-separated, e.g. AAPL,MSFT,GOOG): ").strip()
        tickers = [t.strip().upper() for t in raw.split(',') if t.strip()]
        if not tickers:
            tickers = ["AAPL", "MSFT", "GOOG"]
        fetcher = FinancialDataFetcher(tickers=tickers)

    period = input("  Period [1y] (6mo/1y/2y): ").strip() or "1y"

    print(f"\n  Analyzing portfolio: {', '.join(fetcher.tickers)}\n")

    try:
        summary = fetcher.portfolio_summary(period=period)
        print_header("PORTFOLIO SUMMARY")
        print(summary.to_string(index=False))

        plot_choice = input("\n  Show performance charts? [y/n]: ").strip().lower()
        if plot_choice == 'y':
            fetcher.plot_portfolio_performance(period=period)
            if len(fetcher.tickers) > 1:
                fetcher.correlation_heatmap(period=period)
    except Exception as e:
        print(f"\n  Error: {e}")

    pause()


def menu_full_pipeline():
    """Run the complete pipeline: OCR -> ML -> LLM."""
    print_header("FULL PIPELINE DEMO")
    print("""
  This demo runs all three modules in sequence:
  1. OCR: Read a financial document image
  2. ML:  Predict stock prices
  3. LLM: Ask AI about the combined results

  This demonstrates the full power of the FinancialAI system.
""")

    ticker = input("  Enter stock ticker for ML analysis [AAPL]: ").strip().upper() or "AAPL"

    # Step 1: Generate and OCR a demo image
    print("\n" + "─" * 50)
    print("  STEP 1: GENERATING & READING DOCUMENT IMAGE")
    print("─" * 50)

    ocr_text = ""
    try:
        from demo_generator import create_demo_dir, generate_financial_report_image
        create_demo_dir()
        img_path = generate_financial_report_image()

        from ocr_engine import OCREngine
        engine = OCREngine()
        ocr_text = engine.extract_text(img_path)
        print(f"\n  Extracted {len(ocr_text.split())} words from document image.")
        print(f"  Preview: {ocr_text[:200]}...")
    except Exception as e:
        print(f"  OCR step skipped: {e}")

    # Step 2: ML prediction
    print("\n" + "─" * 50)
    print(f"  STEP 2: ML PREDICTION FOR {ticker}")
    print("─" * 50)

    ml_text = ""
    try:
        from ml_predictor import StockPredictor
        predictor = StockPredictor(ticker=ticker, period="1y")
        predictor.fetch_data()
        predictor.train()
        pred, current, change_pct = predictor.predict_next_day()
        ml_text = predictor.summary()

        direction = "UP" if change_pct > 0 else "DOWN"
        print(f"\n  {ticker}: ${current:.2f} → ${pred:.2f} ({direction} {abs(change_pct):.2f}%)")
    except Exception as e:
        print(f"  ML step skipped: {e}")

    # Step 3: LLM analysis
    print("\n" + "─" * 50)
    print("  STEP 3: LOCAL LLM ANALYSIS")
    print("─" * 50)

    from local_llm import LocalLLM
    llm = LocalLLM()

    if not llm.is_available():
        print("\n  Ollama not running. Skipping LLM step.")
        print("  Install Ollama and pull a model to enable AI chat.")
        pause()
        return

    if not llm.setup():
        pause()
        return

    context = f"Stock Analysis for: {ticker}\n\n"
    if ml_text:
        context += f"ML PREDICTION:\n{ml_text}\n\n"
    if ocr_text:
        context += f"DOCUMENT CONTENT (via OCR):\n{ocr_text[:1500]}\n"

    print(f"\n  Asking {llm.model} to analyze the data...\n")
    prompt = (
        f"I have ML predictions and document data for {ticker}. "
        f"Please provide: (1) a summary of what the ML model predicts, "
        f"(2) key insights from the financial document, "
        f"(3) your overall investment perspective, "
        f"(4) key risks to watch."
    )

    llm.chat(prompt, context=context)

    print("\n  Continue chatting? (Ctrl+C or type 'quit' to stop)")
    llm.interactive_chat(context=context)
    pause()


# ─────────────────────────────────────────────
#  ABOUT SCREEN
# ─────────────────────────────────────────────

def show_about():
    print_header("ABOUT THIS PROJECT")
    print("""
  FinancialAI - Final Class Project
  Python for Everyone

  ┌─────────────────────────────────────────────────────┐
  │  TECHNOLOGIES USED                                   │
  ├─────────────────────────────────────────────────────┤
  │  yfinance     Stock/ETF data (Class 6 foundation)   │
  │  pandas       Data manipulation                      │
  │  numpy        Numerical computing                    │
  │  matplotlib   Data visualization                     │
  │  seaborn      Statistical plots                      │
  │  scikit-learn Machine learning (Random Forest)       │
  │  easyocr      OCR / Image-to-Text (deep learning)   │
  │  Pillow       Image preprocessing                    │
  │  requests     HTTP client for Ollama API             │
  │  Ollama       Local LLM server runtime               │
  ├─────────────────────────────────────────────────────┤
  │  ML MODELS                                           │
  ├─────────────────────────────────────────────────────┤
  │  RandomForestRegressor  Stock price prediction       │
  │  EasyOCR (CRNN-based)   Text recognition            │
  │  LLaMA 3.2 / Mistral    Local language model        │
  │  dolphin-mistral        Uncensored local LLM        │
  ├─────────────────────────────────────────────────────┤
  │  CONCEPTS DEMONSTRATED                               │
  ├─────────────────────────────────────────────────────┤
  │  - Object-Oriented Programming (5 classes)          │
  │  - Feature engineering & normalization              │
  │  - Train/test evaluation with metrics               │
  │  - REST API communication (Ollama HTTP API)         │
  │  - RAG (Retrieval Augmented Generation)             │
  │  - Streaming LLM output                             │
  │  - Image preprocessing for ML                       │
  │  - Portfolio risk metrics (Sharpe, Drawdown)        │
  └─────────────────────────────────────────────────────┘

  Built on top of Class-6 financial analytics work:
    - financial_functions.py (yfinance, stock data)
    - funds.py (SPY portfolio analysis)
    - stock-prices.py (multi-ticker visualization)
""")
    pause()


# ─────────────────────────────────────────────
#  MAIN MENU
# ─────────────────────────────────────────────

def main():
    # Shared context between modules (for LLM RAG)
    session_ml_context = ""
    session_ocr_context = ""

    while True:
        clear()
        print(BANNER)
        print_separator()
        print("  MAIN MENU")
        print_separator()
        print("  [1]  OCR - Image to Text")
        print("  [2]  ML  - Stock Price Prediction (Random Forest)")
        print("  [3]  LLM - Local Private AI Chat")
        print("  [4]  Portfolio Analysis (Class-6 extended)")
        print("  [5]  Full Pipeline Demo (OCR + ML + LLM)")
        print("  [6]  About this project")
        print("  [0]  Exit")
        print_separator()

        if session_ml_context:
            print(f"  [Context loaded: ML results available for LLM]")
        if session_ocr_context:
            print(f"  [Context loaded: OCR text available for LLM]")

        choice = input("\n  Select option: ").strip()

        if choice == '1':
            menu_ocr()

        elif choice == '2':
            result = menu_ml()
            if result:
                session_ml_context = result
                print("\n  [ML results saved - available for LLM analysis]")
                time.sleep(1.5)

        elif choice == '3':
            menu_llm(ml_context=session_ml_context, ocr_context=session_ocr_context)

        elif choice == '4':
            menu_portfolio()

        elif choice == '5':
            menu_full_pipeline()

        elif choice == '6':
            show_about()

        elif choice == '0':
            print("\n  Goodbye!\n")
            sys.exit(0)

        else:
            print("  Invalid option. Please select 0-6.")
            time.sleep(1)


if __name__ == "__main__":
    main()
