"""
Model monitoring utilities for drift detection and retrain triggers.

This module provides functions to monitor model performance degradation
and determine when retraining is necessary based on error metrics.
"""

from datetime import timedelta
import numpy as np
import pandas as pd


def rolling_mae(real: pd.Series, pred: pd.Series, window: int = 20) -> float:
    """
    Calculate rolling Mean Absolute Error over a window.
    
    Args:
        real: Series of actual values
        pred: Series of predicted values
        window: Rolling window size (default: 20)
        
    Returns:
        float: Rolling MAE for the last window
        
    Example:
        >>> rolling_mae(real_prices, predictions, window=30)
        2.45
    """
    diff = (real - pred).abs()
    return diff.rolling(window).mean().iloc[-1]


def needs_retrain(recent_mae: float, baseline_mae: float, threshold: float = 1.25) -> bool:
    """
    Determine if model needs retraining based on performance degradation.
    
    Args:
        recent_mae: Recent Mean Absolute Error
        baseline_mae: Baseline MAE from training/validation
        threshold: Degradation threshold multiplier (default: 1.25 = 25% worse)
        
    Returns:
        bool: True if retraining is recommended
        
    Example:
        >>> needs_retrain(recent_mae=5.0, baseline_mae=3.0, threshold=1.25)
        True  # 5.0 > 1.25 * 3.0
    """
    if baseline_mae is None:
        return False
    return recent_mae > threshold * baseline_mae
