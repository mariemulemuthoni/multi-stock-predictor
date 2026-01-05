# End-to-End NVIDIA Stock Forecaster

This project implements an automated machine learning pipeline to forecast NVIDIA (NVDA) stock prices. It compares traditional regression algorithms against deep learning architectures, automatically selecting the optimal model for deployment based on validation performance.

## Project Overview

The system is designed to identify patterns in historical stock market data spanning the last 10 years. It utilizes a Champion/Challenger deployment strategy to ensure that only the most accurate model is utilized for inference.

### Architecture
1.  **Data Ingestion:** Automated retrieval of historical financial data via the Yahoo Finance API.
2.  **Preprocessing:** Chronological train-test splitting to prevent data leakage, utilizing MinMax scaling for feature normalization.
3.  **Model Training:**
    * **Linear Regression:** Established as a baseline.
    * **Random Forest:** Evaluated for its ability to handle non-linear relationships.
    * **Long Short-Term Memory (LSTM):** A Recurrent Neural Network optimized for sequential time-series data.
4.  **Evaluation:** Models are compared using Root Mean Square Error (RMSE).
5.  **Dynamic Deployment**: A full-stack web interface built with Gradio that detects the serialized champion model type (Keras vs. Sklearn) and adjusts input tensor shapes automatically for real-time inference.

## Performance

The pipeline evaluates models on a hold-out test set (latest 20% of data). In the current iteration, the Linear Regression model demonstrated superior performance in capturing the immediate price trend versus complex non-linear models.

| Model | RMSE (USD) | MAE (USD) |
| :--- | :--- | :--- |
| **Linear Regression (Champion)** | **3.99** | **2.95** |
| Random Forest | 85.90 | 76.83 |
| LSTM | 27.84 | 22.81 |

*Note: Results may vary based on market volatility and random seed initialization.*

## Installation

### Prerequisites
* Python 3.8 or higher
* Git

### Setup
1.  Clone the repository:
    ```bash
    git clone [https://github.com/mariemulemuthoni/nvidia-price-predictor.git](https://github.com/mariemulemuthoni/nvidia-price-predictor.git)
    cd nvidia-price-predictor
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## How to Use

### 1. Model Training
Run the Jupyter Notebook to train the models and generate the serialization files.
* Open `notebooks/nvidia_prediction.ipynb`
* Execute all cells.
* The system will automatically save the best-performing model to the `models/` directory.

### 2. Run Inference
Launch the web application to generate predictions.
```bash
python app.py