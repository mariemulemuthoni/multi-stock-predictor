import yfinance as yf
import pandas as pd
from typing import Optional, List
from src.utils import setup_logger

class MarketDataManager:
    """
    Manages data ingestion from external financial APIs.
    Handles connectivity errors and schema normalization.
    """

    def __init__(self, tickers: List[str], period: str = "10y"):
        self.tickers = tickers
        self.period = period
        self.logger = setup_logger(__name__)

    def fetch_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Retrieves historical OHLC data for a specific asset.
        
        Returns:
            pd.DataFrame: A clean DataFrame containing only the 'Close' price.
        """
        if ticker not in self.tickers:
            self.logger.warning(f"Ticker '{ticker}' is unconfigured but will attempt fetch.")
        
        self.logger.info(f"Acquiring market data for: {ticker}")
        
        try:
            df = yf.download(ticker, period=self.period, progress=False, auto_adjust=True)
            
            if df.empty:
                self.logger.error(f"API returned empty dataset for: {ticker}")
                return None

            # Normalize yfinance response structure (handle potential MultiIndex)
            if isinstance(df.columns, pd.MultiIndex):
                df = df['Close']
            
            # Standardize output format
            data = df[[ticker]].rename(columns={ticker: 'Close'}) if isinstance(df, pd.DataFrame) else df.to_frame(name='Close')
            
            self.logger.info(f"Ingestion successful: {len(data)} records loaded.")
            return data

        except Exception as e:
            self.logger.critical(f"Data ingestion failure for {ticker}. Stack trace: {e}")
            return None