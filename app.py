import os

# Silence TensorFlow logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import gradio as gr
import joblib
import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# Model and Scaler loading
model = None
scaler = None
model_type = None 

models_dir = "models"
pkl_path = os.path.join(models_dir, "best_model.pkl")
keras_path = os.path.join(models_dir, "best_model.keras")
scaler_path = os.path.join(models_dir, "scaler.pkl")

try:
    # Load Scaler
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    else:
        raise FileNotFoundError("Scaler not found. Run notebook first.")

    # Load Model (Dynamic Selection)
    if os.path.exists(keras_path):
        print(f"Loading Keras/LSTM model from {keras_path}...")
        model = tf.keras.models.load_model(keras_path)
        model_type = 'Long Short-Term Memory (LSTM)'
        
    elif os.path.exists(pkl_path):
        print(f"Loading Scikit-Learn model from {pkl_path}...")
        model = joblib.load(pkl_path)

        # Determine specific model type
        if isinstance(model, RandomForestRegressor):
            model_type = "Random Forest Regressor"
        else:
            model_type = "Linear Regression"
        
    else:
        raise FileNotFoundError("No model file found (checked .pkl and .keras).")

except Exception as e:
    print(f"Critical Error: {e}")

# Prediction Logic
def predict_next_day(ticker):
    if model is None or scaler is None:
        return "Error: System not initialized.", "Error", "Error"

    try:
        # Fetch Data
        df = yf.download(ticker, period="6mo", auto_adjust=True)
        
        # Data Validation for MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        if 'Close' not in df.columns:
             return "Error: 'Close' column missing.", "Error", "Error"

        data = df['Close'].values
        if len(data) < 60:
            return f"Error: Need 60 days of data, found {len(data)}.", "Error", "Error"

        # Use training scaler to avoid data leakage
        last_60_days = data[-60:].reshape(-1, 1)
        last_60_scaled = scaler.transform(last_60_days)
        
        # Dynamic Input Shaping based on Model Type
        if model_type == 'Long Short-Term Memory (LSTM)':
            X_input = last_60_scaled.reshape(1, 60, 1)
        else:
            X_input = last_60_scaled.flatten().reshape(1, -1)
        
        # Inference
        predicted_scaled = model.predict(X_input)
        
        # Inverse Transform results to USD
        predicted_price = scaler.inverse_transform(predicted_scaled.reshape(-1, 1))
        
        # Calculate Percentage Change
        current_price = data[-1]
        next_price = predicted_price[0][0]
        change = ((next_price - current_price) / current_price) * 100
        
        return (
            f"${current_price:.2f}", 
            f"${next_price:.2f}", 
            f"{change:+.2f}%"
        )

    except Exception as e:
        return f"Error: {str(e)}", "Error", "Error"

# Launch Application
iface = gr.Interface(
    fn=predict_next_day,
    inputs=gr.Textbox(value="NVDA", label="Enter Stock Ticker"),
    outputs=[
        gr.Textbox(label="Current Price"),
        gr.Textbox(label="Predicted Next Day"),
        gr.Textbox(label="Predicted Change")
    ],
    title="NVIDIA Stock Price Predictor",
    description=f"""
    This application predicts NVIDIA's next closing price using historical market data.
    <br><br>
    <b>Current Champion Model:</b> {model_type} 
    <br>
    <i>(Selected automatically by the training pipeline based on lowest RMSE).</i>
    """
)

if __name__ == "__main__":
    iface.launch(share=True)