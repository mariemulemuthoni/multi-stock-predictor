import joblib
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from keras.callbacks import EarlyStopping
from abc import ABC, abstractmethod
from typing import Dict, Any
from src.utils import setup_logger

class BaseModel(ABC):
    """
    Abstract Base Class enforcing a unified interface for all predictive models.
    Ensures different architectures (sklearn vs keras) can be swapped seamlessly in the pipeline.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.logger = setup_logger(self.__class__.__name__)

    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Executes model training/fitting logic."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generates predictions. Output must be 2D array (Samples, 1)."""
        pass

    @abstractmethod
    def save(self, file_path: str):
        """Serializes the model to disk using architecture-specific formats."""
        pass
    
    @abstractmethod
    def load(self, file_path: str):
        """Loads model artifacts from disk."""
        pass

class LSTMModel(BaseModel):
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        self.logger.info("Initializing LSTM architecture...")
        
        units = self.config['model']['lstm']['units']
        dropout = self.config['model']['lstm']['dropout']
        
        # 3D Input Shape: (Samples, TimeSteps, Features)
        self.model = Sequential([
            Input(shape=(X_train.shape[1], 1)),
            LSTM(units=units, return_sequences=True),
            Dropout(dropout),
            LSTM(units=units, return_sequences=False),
            Dropout(dropout),
            Dense(units=1)
        ])
        
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        
        self.logger.info("Starting Deep Learning training loop...")
        early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        
        self.model.fit(
            X_train, y_train,
            epochs=self.config['model']['lstm']['epochs'],
            batch_size=self.config['model']['lstm']['batch_size'],
            callbacks=[early_stop],
            verbose=1
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def save(self, file_path: str):
        path = f"{file_path}.keras"
        self.logger.info(f"Saving Keras artifact to {path}")
        self.model.save(path)
        
    def load(self, file_path: str):
        path = f"{file_path}.keras"
        self.model = tf.keras.models.load_model(path)

class SklearnModel(BaseModel):
    """
    Adapter class allowing Scikit-Learn models to function within the Deep Learning pipeline structure.
    """
    
    def __init__(self, config: Dict[str, Any], model_type: str):
        super().__init__(config)
        self.model_type = model_type
        
        if model_type == 'linear_regression':
            self.model = LinearRegression()
        elif model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=config['model']['random_forest']['n_estimators'],
                n_jobs=-1,
                random_state=config['model']['seed']
            )
        else:
            raise ValueError(f"Unsupported model architecture: {model_type}")

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        # Flatten 3D tensors back to 2D matrices for standard ML algorithms
        if X_train.ndim == 3:
            nsamples, nx, ny = X_train.shape
            X_train = X_train.reshape((nsamples, nx*ny))
            
        self.logger.info(f"Fitting {self.model_type}...")
        self.model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 3:
            nsamples, nx, ny = X.shape
            X = X.reshape((nsamples, nx*ny))
        
        preds = self.model.predict(X)
        return preds.reshape(-1, 1) # Enforce consistency with LSTM output shape

    def save(self, file_path: str):
        path = f"{file_path}.pkl"
        self.logger.info(f"Pickling model to {path}")
        joblib.dump(self.model, path)

    def load(self, file_path: str):
        path = f"{file_path}.pkl"
        self.model = joblib.load(path)

class ModelFactory:
    """
    Implements the Factory Design Pattern to dynamically instantiate
    model classes based on configuration strings.
    """
    @staticmethod
    def get_model(model_type: str, config: Dict[str, Any]) -> BaseModel:
        if model_type == 'lstm':
            return LSTMModel(config)
        elif model_type in ['linear_regression', 'random_forest']:
            return SklearnModel(config, model_type)
        else:
            raise ValueError(f"Unknown model type requested: {model_type}")