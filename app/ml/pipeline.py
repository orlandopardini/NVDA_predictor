"""
Data pipeline utilities for time series preparation.

This module provides classes and functions for preparing time series data
for LSTM model training, including windowing, scaling, and train/val splitting.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple


class SeriesWindow:
    """
    Creates sliding windows for time series forecasting.
    
    Transforms a time series into supervised learning format with
    lookback windows (X) and forecast horizons (y).
    
    Attributes:
        lookback (int): Number of past timesteps to use as features
        horizon (int): Number of future timesteps to predict
        
    Example:
        >>> window = SeriesWindow(lookback=60, horizon=1)
        >>> X, y = window.make(prices)
        >>> X.shape  # (n_samples, 60)
        >>> y.shape  # (n_samples,)
    """
    
    def __init__(self, lookback: int = 60, horizon: int = 1):
        """
        Initialize the windowing transformer.
        
        Args:
            lookback: Number of past timesteps (default: 60)
            horizon: Number of future timesteps (default: 1)
        """
        self.lookback = lookback
        self.horizon = horizon

    def make(self, series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create windowed samples from time series.
        
        Args:
            series: 1D array of time series values
            
        Returns:
            Tuple of (X, y) where:
                X: Array of shape (n_samples, lookback)
                y: Array of shape (n_samples,)
                
        Example:
            >>> series = np.array([1, 2, 3, 4, 5, 6])
            >>> X, y = window.make(series)  # lookback=3, horizon=1
            >>> X  # [[1,2,3], [2,3,4], [3,4,5]]
            >>> y  # [4, 5, 6]
        """
        X, y = [], []
        for i in range(self.lookback, len(series) - self.horizon + 1):
            X.append(series[i-self.lookback:i])
            y.append(series[i+self.horizon-1])
        return np.array(X), np.array(y)


def train_val_split(df: pd.DataFrame, ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split time series data into train and validation sets.
    
    Uses temporal split (not random) to preserve time ordering.
    
    Args:
        df: DataFrame to split
        ratio: Fraction of data for training (default: 0.8 = 80/20 split)
        
    Returns:
        Tuple of (train_df, val_df)
        
    Example:
        >>> train, val = train_val_split(prices, ratio=0.8)
        >>> len(train) / len(prices)  # 0.8
    """
    n = len(df)
    cut = int(n * ratio)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def scale_fit_transform(
    train: np.ndarray, 
    val: np.ndarray
) -> Tuple[MinMaxScaler, np.ndarray, np.ndarray]:
    """
    Fit scaler on training data and transform both sets.
    
    Fits MinMaxScaler [0,1] on training data only to prevent data leakage,
    then applies the same transformation to validation data.
    
    Args:
        train: Training data array
        val: Validation data array
        
    Returns:
        Tuple of (scaler, train_scaled, val_scaled)
        
    Example:
        >>> scaler, train_s, val_s = scale_fit_transform(train, val)
        >>> train_s.min(), train_s.max()  # (0.0, 1.0)
    """
    scaler = MinMaxScaler()
    train_s = scaler.fit_transform(train)
    val_s = scaler.transform(val)
    return scaler, train_s, val_s
