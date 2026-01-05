import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Optional
from src.utils import setup_logger

class FeatureEngineer:
    """
    Handles time-series specific transformations including scaling and sliding-window sequence generation.
    """

    def __init__(self, lookback_days: int = 60, test_size: float = 0.2):
        self.lookback_days = lookback_days
        self.test_size = test_size
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.logger = setup_logger(__name__)
        self.train_data_len = 0

    def preprocess(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Splits and scales data. 
        The scaler is fit only on training data to prevent lookahead bias (data leakage).
        """
        data = df.values
        self.train_data_len = int(len(data) * (1 - self.test_size))

        train_raw = data[:self.train_data_len]
        
        # Fit on training set, apply to both train and test
        self.scaler.fit(train_raw)
        scaled_train = self.scaler.transform(train_raw)

        # Include lookback buffer for the test set
        test_raw = data[self.train_data_len - self.lookback_days:]
        scaled_test = self.scaler.transform(test_raw)

        self.logger.info(f"Preprocessing complete. Train size: {len(scaled_train)}, Test inputs: {len(scaled_test)}")
        
        # Return pure future data for accurate evaluation
        raw_test_truth = data[self.train_data_len:]
        return scaled_train, scaled_test, raw_test_truth

    def create_sequences(self, dataset: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Converts array into X (features) and y (target) using a sliding window.
        X = [t-60 ... t-1], y = [t]
        """
        X, y = [], []
        for i in range(self.lookback_days, len(dataset)):
            X.append(dataset[i-self.lookback_days:i, 0])
            y.append(dataset[i, 0])
            
        return np.array(X), np.array(y)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Reverts normalized data back to original dollar values."""
        return self.scaler.inverse_transform(data)