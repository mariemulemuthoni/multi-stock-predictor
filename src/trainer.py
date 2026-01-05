import os
import numpy as np
from typing import Dict, Any
from src.utils import setup_logger
from src.data_loader import MarketDataManager
from src.feature_eng import FeatureEngineer
from src.model_factory import ModelFactory
from sklearn.metrics import mean_squared_error, mean_absolute_error

class Trainer:
    """
    Manages the full lifecycle of the training process:
    Data Ingestion -> Preprocessing -> Model Training -> Evaluation -> Serialization.
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
        
        # Ensures model artifact directory exists
        os.makedirs(config['paths']['models_dir'], exist_ok=True)

    def train_all(self):
        """Iterates through configured assets and model architectures."""
        tickers = self.config['data']['tickers']
        # Currently supported architectures
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
            
        except Exception as e:
            self.logger.error(f"Failed to train {model_name} on {ticker}. Details: {e}")