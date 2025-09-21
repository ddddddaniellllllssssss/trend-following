import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any
import config

class DataProcessor:
    """Handles loading and preprocessing of OHLC data"""

    def __init__(self, data_path: str = None):
        self.data_path = data_path or config.DATA_PATH
        self.data = None

    def load_data(self) -> pd.DataFrame:
        """Load CSV data and convert to proper format"""
        df = pd.read_csv(self.data_path)

        # Convert timestamp to datetime
        df['datetime'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('datetime', inplace=True)

        # Ensure proper column names and types
        df = df[['open', 'high', 'low', 'close']].astype(float)

        # Filter by date range
        start_date = pd.to_datetime(config.START_DATE)
        end_date = pd.to_datetime(config.END_DATE)
        df = df[(df.index >= start_date) & (df.index <= end_date)]

        self.data = df
        return df

    def get_ohlc(self) -> Dict[str, np.ndarray]:
        """Return OHLC data as numpy arrays"""
        if self.data is None:
            self.load_data()

        return {
            'open': self.data['open'].values,
            'high': self.data['high'].values,
            'low': self.data['low'].values,
            'close': self.data['close'].values,
            'datetime': self.data.index.values
        }

    def validate_data(self) -> bool:
        """Validate data quality"""
        if self.data is None:
            return False

        # Check for missing values
        if self.data.isnull().any().any():
            print("Warning: Missing values found in data")
            return False

        # Check for invalid OHLC relationships
        invalid_bars = (
            (self.data['high'] < self.data['low']) |
            (self.data['high'] < self.data['open']) |
            (self.data['high'] < self.data['close']) |
            (self.data['low'] > self.data['open']) |
            (self.data['low'] > self.data['close'])
        )

        if invalid_bars.any():
            print(f"Warning: {invalid_bars.sum()} invalid OHLC bars found")
            return False

        return True