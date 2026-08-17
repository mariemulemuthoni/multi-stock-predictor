import os
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any
from src.utils import setup_logger
from src.data_loader import MarketDataManager
from src.feature_eng import FeatureEngineer
from src.model_factory import ModelFactory
from sklearn.metrics import mean_squared_error, mean_absolute_error

class Trainer:
    """
    Manages the full lifecycle of the training process:
    Data Ingestion -> Preprocessing -> Model Training -> Evaluation -> Serialization -> Tracking.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = setup_logger(__name__)
        
        self.data_manager = MarketDataManager(
            tickers=config['data']['tickers'],
            period=config['data']['period']
        )
        self.feature_engine = FeatureEngineer(
            lookback_days=config['data']['lookback_days'],
            test_size=config['data']['test_size']
        )
        
        # Ensures model and log directories exist
        os.makedirs(config['paths']['models_dir'], exist_ok=True)
        self.logs_dir = config['paths'].get('logs_dir', 'logs')
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Define the tracking path
        self.metrics_file = os.path.join(self.logs_dir, 'metrics_history.csv')

    def train_all(self):
        """Iterates through configured assets and model architectures."""
        tickers = self.config['data']['tickers']
        model_types = ['lstm', 'random_forest', 'linear_regression']

        self.logger.info(f"Starting batch training for: {tickers}")

        for ticker in tickers:
            self.logger.info(f"--- Processing Asset: {ticker} ---")
            
            df = self.data_manager.fetch_data(ticker)
            if df is None:
                continue 

            # Prepare train/test splits and handle scaling
            scaled_train, scaled_test, raw_test_data = self.feature_engine.preprocess(df)
            
            # Generate sliding window sequences
            X_train, y_train = self.feature_engine.create_sequences(scaled_train)
            X_test, y_test = self.feature_engine.create_sequences(scaled_test)
            
            for model_name in model_types:
                self._train_and_evaluate(
                    ticker, model_name, 
                    X_train, y_train, 
                    X_test, y_test, 
                    raw_test_data
                )
                
        # Generate the visual dashboard after all training is complete
        self._generate_markdown_dashboard()

    def _train_and_evaluate(self, ticker, model_name, X_train, y_train, X_test, y_test, raw_test_data):
        """Executes training, inference, and evaluation for a specific model instance."""
        try:
            model = ModelFactory.get_model(model_name, self.config)
            
            model.train(X_train, y_train)
            
            # Generate predictions and inverse-scale them to original price range
            predictions_scaled = model.predict(X_test)
            predictions_real = self.feature_engine.inverse_transform(predictions_scaled)
            
            # Ground truth for evaluation
            true_values = raw_test_data
            
            # Calculate performance metrics
            mse = mean_squared_error(true_values, predictions_real)
            rmse = np.sqrt(mse) 
            mae = mean_absolute_error(true_values, predictions_real)
            
            self.logger.info(f"[{ticker}] {model_name} Performance -> RMSE: {rmse:.4f}, MAE: {mae:.4f}")
            
            # Serialize model to disk
            save_path = os.path.join(
                self.config['paths']['models_dir'], 
                f"{ticker}_{model_name}"
            )
            model.save(save_path)
            
            # Append metrics to tracking file
            self._log_metrics_to_csv(ticker, model_name, rmse, mae)
            
        except Exception as e:
            self.logger.error(f"Failed to train {model_name} on {ticker}. Details: {e}")

    def _log_metrics_to_csv(self, ticker: str, model_name: str, rmse: float, mae: float):
        """Appends the latest evaluation scores to a permanent historical CSV."""
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file_exists = os.path.isfile(self.metrics_file)
        
        with open(self.metrics_file, 'a') as f:
            if not file_exists:
                f.write("Timestamp,Ticker,Model,RMSE,MAE\n")
            f.write(f"{date_str},{ticker},{model_name},{rmse:.4f},{mae:.4f}\n")

    def _generate_markdown_dashboard(self):
        """Reads the historical CSV and generates a Markdown dashboard of the last 7 runs."""
        try:
            if not os.path.exists(self.metrics_file):
                return
                
            df = pd.read_csv(self.metrics_file)
            
            # Extract up to the last 7 unique training timestamps
            recent_timestamps = sorted(df['Timestamp'].unique())[-7:]
            recent_runs = df[df['Timestamp'].isin(recent_timestamps)]
            
            # Sort by Ticker, then Model, then Timestamp (newest to oldest)
            recent_runs = recent_runs.sort_values(
                by=['Ticker', 'Model', 'Timestamp'], 
                ascending=[True, True, False]
            )
            
            dashboard_path = "MODEL_PERFORMANCE.md"
            
            with open(dashboard_path, "w", encoding="utf-8") as f:
                f.write("# 📊 Automated Model Performance Dashboard\n\n")
                f.write(f"> **Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (UTC)\n\n")
                f.write("This dashboard tracks the last 7 training runs to monitor model degradation or improvement over time. It is automatically generated by the CI/CD pipeline.\n\n")
                
                # Create a separate, clean table for each asset
                tickers = recent_runs['Ticker'].unique()
                for ticker in tickers:
                    f.write(f"## 📈 {ticker} Performance History\n")
                    f.write("| Timestamp | Model Architecture | RMSE (Lower is Better) | MAE (Lower is Better) |\n")
                    f.write("| :--- | :--- | :--- | :--- |\n")
                    
                    ticker_data = recent_runs[recent_runs['Ticker'] == ticker]
                    
                    for _, row in ticker_data.iterrows():
                        f.write(f"| {row['Timestamp']} | {row['Model']} | **{row['RMSE']:.4f}** | {row['MAE']:.4f} |\n")
                    f.write("\n---\n\n")
                    
            self.logger.info("Markdown dashboard successfully generated with historical trends.")
            
        except Exception as e:
            self.logger.error(f"Failed to generate dashboard: {e}")