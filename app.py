import os
# Silence TensorFlow Logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.utils import load_config, setup_logger
from src.data_loader import MarketDataManager
from src.feature_eng import FeatureEngineer
from src.model_factory import ModelFactory

# Load Config
config = load_config("config/config.yaml")
logger = setup_logger("App")

def make_prediction(ticker, model_type):
    """
    Main function called by Gradio.
    1. Fetches latest data.
    2. Loads the requested model.
    3. Predicts the next price.
    """
    try:
        # Setup
        data_manager = MarketDataManager(tickers=[ticker], period=config['data']['period'])
        feature_engine = FeatureEngineer(
            lookback_days=config['data']['lookback_days'], 
            test_size=config['data']['test_size']
        )

        # Get Data
        df = data_manager.fetch_data(ticker)
        
        if df is None or len(df) < config['data']['lookback_days']:
            return "Error fetching data", None

        # Prepare Data for Prediction 
        scaled_train, scaled_test, raw_data = feature_engine.preprocess(df)
        
        # Get the last sequence (the most recent 60 days) to predict tomorrow
        last_60_days = df['Close'].values[-config['data']['lookback_days']:]
        last_60_days_scaled = feature_engine.scaler.transform(last_60_days.reshape(-1, 1))
        
        # Reshape for model
        X_input = np.array([last_60_days_scaled]) 
        
        # Load Model 
        logger.info(f"Loading model: {ticker}_{model_type}")
        model = ModelFactory.get_model(model_type, config)
        
        model_path = os.path.join(config['paths']['models_dir'], f"{ticker}_{model_type}")
        
        if not (os.path.exists(f"{model_path}.pkl") or os.path.exists(f"{model_path}.keras")):
            return f"Model not found! Have you run main.py to train {ticker}?", None
            
        model.load(model_path)

        # Predict 
        pred_scaled = model.predict(X_input)
        pred_price = feature_engine.inverse_transform(pred_scaled)[0][0]

        # Visualization 
        fig = plt.figure(figsize=(10, 5))
        history_plot = df['Close'].tail(30)
        plt.plot(history_plot.index, history_plot.values, label="Historical Data", color='blue')
        
        next_date = history_plot.index[-1] + pd.Timedelta(days=1)
        plt.scatter([next_date], [pred_price], color='red', label=f"Prediction: ${pred_price:.2f}", zorder=5)
        plt.plot([history_plot.index[-1], next_date], [history_plot.values[-1], pred_price], color='red', linestyle='--')
        
        plt.title(f"{ticker} Price Forecast ({model_type.upper()})")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        return f"${pred_price:.2f}", fig

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return f"Error: {str(e)}", None

# Gradio UI Layout
with gr.Blocks(title="Institutional Stock Predictor") as demo:
    gr.Markdown("# 📈 Multi-Asset Price Predictor")
    gr.Markdown("Select an asset and a model to forecast the next closing price.")
    
    with gr.Row():
        with gr.Column():
            ticker_dropdown = gr.Dropdown(
                choices=config['data']['tickers'], 
                label="Asset (Ticker)", 
                value="NVDA"
            )
            model_dropdown = gr.Dropdown(
                choices=['lstm', 'random_forest', 'linear_regression'], 
                label="Model Architecture", 
                value="linear_regression"
            )
            predict_btn = gr.Button("Generate Forecast", variant="primary")
            
        with gr.Column():
            output_price = gr.Textbox(label="Predicted Price (Next Close)")
            output_plot = gr.Plot(label="Market Trend Analysis")

    predict_btn.click(
        fn=make_prediction, 
        inputs=[ticker_dropdown, model_dropdown], 
        outputs=[output_price, output_plot]
    )

# Launch
if __name__ == "__main__":
    demo.launch(
        auth=("admin", "pass1234"), 
        share=False,
        theme=gr.themes.Soft()
    )