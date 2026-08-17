# 📈 End-to-End Multi-Stock Price Forecasting Engine

An automated, production-grade Machine Learning pipeline designed to forecast the next-day closing prices of major tech equities (NVDA, AAPL, GOOGL, MSFT, AMZN). 

This system features a fully autonomous Continuous Integration (CI) pipeline that ingests daily post-market data, retrains multiple model architectures, and tracks performance metrics over time without human intervention.

## ✨ Key Features

* **Fully Automated MLOps Pipeline:** Scheduled GitHub Actions trigger daily to ingest new market data, retrain models, and commit updated byte-weights (`.pkl`, `.keras`) back to the repository.
* **Automated Experiment Tracking:** A custom tracking system logs daily validation metrics (RMSE/MAE) to a CSV ledger and auto-generates a dynamic [Performance Dashboard](MODEL_PERFORMANCE.md) to monitor model drift.
* **Resilient Data Ingestion:** Utilizes custom HTTP session handling to bypass institutional data-center IP rate-limiting, ensuring stable daily data extraction from Yahoo Finance via cloud servers.
* **Interactive Local UI:** Built with Gradio, allowing users to select target assets, compare model architectures, and visualize historical trends against predicted vectors.
* **Architecture Scalability:** Implements Object-Oriented Design and the Factory Method Pattern to seamlessly hot-swap between Deep Learning (LSTM) and Statistical (Linear Regression, Random Forest) models.

## 📊 Performance Benchmarks
Benchmarked across a chronological train/test split to prevent look-ahead bias. **Linear Regression** consistently outperforms complex architectures in handling short-term volatility for these specific assets. 

*(Note: These metrics are updated automatically by the CI/CD pipeline. Check [MODEL_PERFORMANCE.md](MODEL_PERFORMANCE.md) for the latest daily run).*

| Asset | Best Model | RMSE | MAE |
| :--- | :--- | :--- | :--- |
| **NVDA** | Linear Regression | 5.19 | 3.96 |
| **AAPL** | Linear Regression | 4.37 | 3.04 |
| **GOOGL**| Linear Regression | 5.34 | 3.67 |
| **MSFT** | Linear Regression | 7.88 | 5.38 |
| **AMZN** | Linear Regression | 4.74 | 3.43 |

## 🛠️ Tech Stack
* **Language:** Python 3.12
* **Machine Learning:** Scikit-Learn (Linear Regression, Random Forest), TensorFlow/Keras (LSTM)
* **Data Engineering:** Pandas, NumPy, YFinance
* **MLOps & CI/CD:** GitHub Actions, Model Artifact Serialization (`joblib`)
* **Frontend:** Gradio


## 💻 Local Setup & Quickstart

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/mariemulemuthoni/multi-stock-predictor.git](https://github.com/mariemulemuthoni/multi-stock-predictor.git)
   cd multi-stock-predictor
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Run the Gradio UI locally:**
   ```bash
   python app.py
   ```
   *The interactive dashboard will launch on a URL.*

4. **Trigger a manual pipeline retraining:**
   ```bash
   python main.py
   ```

## 📁 Repository Structure

```text
├── .github/workflows/   # CI/CD pipelines (daily_prediction.yml)
├── config/              # Configuration-as-code (YAML settings)
├── logs/                # Automated execution logs & metrics_history.csv
├── models/              # Serialized model artifacts (.pkl, .keras)
├── src/                 # Core modular pipeline
│   ├── data_loader.py   # Ingestion, scaling, and IP rate-limit handling
│   ├── feature_eng.py   # Time-series windowing and transformations
│   ├── model_factory.py # OOP Model Registry and Adapters
│   └── trainer.py       # Execution logic and dashboard generation
├── MODEL_PERFORMANCE.md # Auto-generated tracking dashboard
├── app.py               # Gradio interactive UI
├── main.py              # CLI entry point for training
└── requirements.txt     # Python dependencies