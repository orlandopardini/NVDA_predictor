"""
LSTM Model Training and Management Module

This module provides comprehensive functionality for training, evaluating, and managing
LSTM models for stock price prediction. It handles the complete training pipeline,
including data preparation, model selection, evaluation, and persistence.

Key Features:
    - Fast model training with configurable hyperparameters
    - Automatic model selection based on combined performance metrics
    - Database integration with retry logic for concurrent access
    - Support for multiple LSTM/GRU architectures via model_zoo
    - Time-series validation with proper lookback windows
    - Monitoring integration via Prometheus metrics

Typical Usage:
    # Train all models and select winner
    result = train_all_models_fast(
        ticker="AAPL",
        lookback=60,
        horizon=1,
        epochs=50,
        reuse_if_exists=False
    )
    
    # Load best performing model
    model, scaler, registry = load_best_model("AAPL")
    
    # Make prediction
    prediction = predict_horizon(model, scaler, price_series, lookback=60)

Author: Refactored with Clean Code principles
Date: 2024
"""

import os
import json
import glob
import time
from datetime import datetime
from typing import Tuple, Dict, List, Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy.exc import OperationalError
from tensorflow import keras
from tensorflow.keras import layers

from .model_zoo import build_model, MODEL_NAMES
from .constants import DEFAULT_LOOKBACK, DEFAULT_HORIZON, MODELS_DIR
from .eval import metrics_from_series, rolling_backtest_1step
from ..models import db, PrecoDiario, ModelRegistry
from ..utils.timing import Stopwatch
from ..monitoring import RETRAIN_COUNT, RETRAIN_DURATION


# =============================================================================
# DATABASE UTILITIES WITH RETRY LOGIC
# =============================================================================

def _add_registry_with_retry(
    registry: ModelRegistry,
    max_tries: int = 6
) -> None:
    """
    Add a model registry to the database with retry logic for locked database.
    
    Handles SQLite database locking by implementing exponential backoff retry
    strategy. This is crucial when multiple processes/threads attempt to write
    to the database concurrently.
    
    Args:
        registry: The ModelRegistry object to add to the database.
        max_tries: Maximum number of retry attempts before giving up.
    
    Raises:
        OperationalError: If error is not database lock-related or max retries exceeded.
    
    Note:
        - Uses exponential backoff: 0.5s, 1.0s, 1.5s, 2.0s, etc.
        - Last attempt is made without try-catch to surface actual error
    """
    for attempt in range(max_tries):
        try:
            db.session.add(registry)
            db.session.commit()
            return
        except OperationalError as e:
            # Only retry on database lock errors
            if "database is locked" not in str(e).lower():
                db.session.rollback()
                raise
            
            db.session.rollback()
            
            # Exponential backoff: 0.5s * (attempt + 1)
            if attempt < max_tries - 1:
                time.sleep(0.5 * (attempt + 1))
    
    # Final attempt without catch - let error propagate if it fails
    db.session.add(registry)
    db.session.commit()


def _update_winner_with_retry(
    ticker: str,
    winner_version: str,
    max_tries: int = 6
) -> None:
    """
    Update winner flag in database with retry logic for concurrent access.
    
    Ensures only one model per ticker is marked as winner (is_winner=True).
    Clears previous winner flag and sets new winner atomically with retry
    logic to handle database locking.
    
    Args:
        ticker: Stock ticker symbol (e.g., "AAPL").
        winner_version: Version string of the winning model.
        max_tries: Maximum number of retry attempts.
    
    Raises:
        OperationalError: If error is not database lock-related or max retries exceeded.
    
    Note:
        - Clears TensorFlow/Keras session after successful update
        - Uses same retry strategy as _add_registry_with_retry
    """
    for attempt in range(max_tries):
        try:
            # Clear previous winner
            ModelRegistry.query.filter_by(
                ticker=ticker,
                is_winner=True
            ).update({"is_winner": False})
            
            # Set new winner
            db.session.query(ModelRegistry).filter(
                ModelRegistry.version == winner_version
            ).update({"is_winner": True})
            
            db.session.commit()
            
            # Clear Keras session to free memory
            _clear_keras_session()
            return
            
        except OperationalError as e:
            if "database is locked" not in str(e).lower():
                db.session.rollback()
                raise
            
            db.session.rollback()
            
            if attempt < max_tries - 1:
                time.sleep(0.5 * (attempt + 1))
    
    # Final attempt
    ModelRegistry.query.filter_by(ticker=ticker, is_winner=True).update({"is_winner": False})
    db.session.query(ModelRegistry).filter(
        ModelRegistry.version == winner_version
    ).update({"is_winner": True})
    db.session.commit()
    _clear_keras_session()


def _clear_keras_session() -> None:
    """
    Clear TensorFlow/Keras backend session to free GPU/CPU memory.
    
    Should be called after model training/loading to prevent memory leaks.
    Silently catches exceptions if Keras backend is not available.
    """
    try:
        from tensorflow.keras import backend as K
        K.clear_session()
    except Exception:
        pass  # Keras backend not available or already cleared


# =============================================================================
# DATA PREPARATION UTILITIES
# =============================================================================

def _prepare_series(ticker: str) -> pd.Series:
    """
    Load and prepare price series from database for a given ticker.
    
    Queries PrecoDiario table, extracts closing prices, and returns a clean
    time-indexed pandas Series suitable for model training.
    
    Args:
        ticker: Stock ticker symbol (e.g., "AAPL").
    
    Returns:
        Pandas Series with datetime index and closing prices, NaN values removed.
    
    Raises:
        ValueError: If no data found for the ticker in database.
    
    Example:
        >>> series = _prepare_series("AAPL")
        >>> print(series.head())
        2024-01-01    150.25
        2024-01-02    151.30
        2024-01-03    149.80
        dtype: float64
    """
    query_result = (
        PrecoDiario.query
        .filter_by(ticker=ticker)
        .order_by(PrecoDiario.date.asc())
        .all()
    )
    
    if not query_result:
        raise ValueError(f"No data found for ticker: {ticker}")
    
    close_prices = [record.close for record in query_result]
    dates = pd.to_datetime([record.date for record in query_result])
    
    series = pd.Series(close_prices, index=dates)
    series = series.dropna()
    
    return series


def _train_val_split(
    series: pd.Series,
    val_ratio: float = 0.2
) -> Tuple[pd.Series, pd.Series]:
    """
    Split time series into training and validation sets.
    
    Uses chronological split to maintain temporal order (no random shuffle).
    Ensures minimum 10 samples in training set to avoid degenerate cases.
    
    Args:
        series: Time series data to split.
        val_ratio: Fraction of data to use for validation (default: 0.2 = 20%).
    
    Returns:
        Tuple of (train_series, validation_series), both as copies.
    
    Note:
        - Training set always has at least 10 samples
        - Split is chronological: train comes before validation
        - Returns independent copies to avoid inadvertent modifications
    
    Example:
        >>> series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> train, val = _train_val_split(series, val_ratio=0.2)
        >>> len(train), len(val)
        (8, 2)
    """
    n_total = len(series)
    n_train = max(10, int((1 - val_ratio) * n_total))
    
    train = series.iloc[:n_train].copy()
    validation = series.iloc[n_train:].copy()
    
    return train, validation


def _make_supervised(
    scaled_array: np.ndarray,
    lookback: int,
    horizon: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create supervised learning dataset from time series array.
    
    Transforms a univariate time series into sequences suitable for LSTM training.
    Each input sequence has 'lookback' timesteps, and targets the value 'horizon'
    steps into the future.
    
    Args:
        scaled_array: 2D array of shape (n_samples, 1) with scaled values.
        lookback: Number of past timesteps to use as input features.
        horizon: Number of steps ahead to predict (1 = next day).
    
    Returns:
        Tuple of (X, y) where:
            - X: Array of shape (n_sequences, lookback, 1) - input sequences
            - y: Array of shape (n_sequences, 1) - target values
        Returns empty arrays if insufficient data for creating sequences.
    
    Example:
        >>> data = np.array([[1], [2], [3], [4], [5]])
        >>> X, y = _make_supervised(data, lookback=2, horizon=1)
        >>> X.shape, y.shape
        ((2, 2, 1), (2, 1))
        >>> X[0]  # First sequence: [1, 2]
        array([[1], [2]])
        >>> y[0]  # Target: 3 (one step ahead)
        array([3])
    """
    X_sequences = []
    y_targets = []
    
    # Create sequences: need lookback history + horizon future
    for t in range(lookback, len(scaled_array) - (horizon - 1)):
        # Input: lookback timesteps before t
        X_sequences.append(scaled_array[t - lookback:t])
        
        # Target: value at t + horizon - 1 (0-indexed)
        y_targets.append(scaled_array[t + (horizon - 1)])
    
    # Handle insufficient data case
    if not X_sequences:
        return np.empty((0, lookback, 1)), np.empty((0, 1))
    
    return np.stack(X_sequences), np.stack(y_targets)


def scale_fit_transform_for_train(
    scaler: MinMaxScaler,
    train: pd.Series,
    val: pd.Series,
    lookback: int,
    horizon: int
) -> Tuple[MinMaxScaler, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit scaler on training data and create supervised learning sequences.
    
    CRITICAL: Scaler is fit ONLY on training data to prevent data leakage.
    Validation data is transformed using the training scaler but never influences it.
    
    Args:
        scaler: MinMaxScaler instance (will be fitted in-place).
        train: Training time series.
        val: Validation time series.
        lookback: Number of past timesteps for input sequences.
        horizon: Number of steps ahead to predict.
    
    Returns:
        Tuple of:
            - scaler: Fitted MinMaxScaler
            - X_train: Training input sequences (n_train, lookback, 1)
            - y_train: Training targets (n_train, 1)
            - X_val: Validation input sequences (n_val, lookback, 1)
            - y_val: Validation targets (n_val, 1)
    
    Note:
        - Validation sequences may include trailing training data for lookback window
        - Empty validation set returns empty arrays (valid edge case)
        - Scaler is modified in-place and returned for convenience
    """
    # Convert to 2D arrays for sklearn
    train_array = train.values.reshape(-1, 1)
    val_array = val.values.reshape(-1, 1) if len(val) > 0 else np.empty((0, 1))
    
    # Fit scaler ONLY on training data
    scaler.fit(train_array)
    
    # Transform both sets
    train_scaled = scaler.transform(train_array)
    val_scaled = scaler.transform(val_array) if len(val) > 0 else np.empty((0, 1))
    
    # Create supervised sequences for training
    X_train, y_train = _make_supervised(train_scaled, lookback, horizon)
    
    # Create supervised sequences for validation
    # Concatenate trailing training data to provide lookback context
    if len(val) > 0:
        combined = np.concatenate([train_scaled[-lookback:], val_scaled], axis=0)
        X_val, y_val = _make_supervised(combined, lookback, horizon)
    else:
        X_val, y_val = _make_supervised(train_scaled, lookback, horizon)
    
    return scaler, X_train, y_train, X_val, y_val


# =============================================================================
# METRICS CALCULATION
# =============================================================================

def _metrics_from_arrays(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate regression metrics from true and predicted arrays.
    
    Computes standard error metrics: MAE, RMSE, MAPE, and Pearson correlation.
    All metrics are in the same scale as the input data (not normalized).
    
    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
    
    Returns:
        Dictionary with keys: "mae", "rmse", "mape", "pearson_corr".
    
    Note:
        - MAPE uses 1e-9 epsilon to avoid division by zero
        - Pearson returns 0.0 if insufficient data (< 2 points)
        - All values are Python floats for JSON serialization
    """
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9)))) * 100.0
    
    # Pearson correlation coefficient
    if len(y_true) > 1:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        pearson = 0.0
    
    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "pearson_corr": pearson
    }


# =============================================================================
# MODEL TRAINING ORCHESTRATION
# =============================================================================

def _get_existing_winner(ticker: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve existing winner model from database if available.
    
    Args:
        ticker: Stock ticker symbol.
    
    Returns:
        Dictionary with winner model info or None if no winner exists.
    """
    record = (
        ModelRegistry.query
        .filter_by(ticker=ticker, is_winner=True)
        .order_by(ModelRegistry.registered_at.desc())
        .first()
    )
    
    if not record:
        return None
    
    return {
        "model_id": record.model_id,
        "model_name": record.model_name,
        "metrics": {
            "mae": record.mae,
            "rmse": record.rmse,
            "mape": record.mape
        },
        "version": record.version,
        "path_model": record.path_model,
        "path_scaler": record.path_scaler,
        "registered_at": record.registered_at.isoformat() if record.registered_at else None
    }


def _prepare_training_data(
    ticker: str,
    lookback: int,
    horizon: int,
    val_ratio: float = 0.2
) -> Tuple[MinMaxScaler, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load data from database and prepare for training.
    
    Args:
        ticker: Stock ticker symbol.
        lookback: Number of past timesteps for input.
        horizon: Number of steps ahead to predict.
        val_ratio: Fraction for validation split.
    
    Returns:
        Tuple of (scaler, X_train, y_train, X_val, y_val).
    
    Raises:
        ValueError: If insufficient data for lookback/horizon configuration.
    """
    close_series = _prepare_series(ticker)
    train, val = _train_val_split(close_series, val_ratio=val_ratio)
    
    scaler = MinMaxScaler()
    scaler, X_train, y_train, X_val, y_val = scale_fit_transform_for_train(
        scaler, train, val, lookback, horizon
    )
    
    if X_train.size == 0 or X_val.size == 0:
        raise ValueError(
            f"Insufficient data for ticker {ticker} with "
            f"lookback={lookback} and horizon={horizon}"
        )
    
    return scaler, X_train, y_train, X_val, y_val


def _train_single_model(
    model_id: int,
    ticker: str,
    lookback: int,
    horizon: int,
    epochs: int,
    batch_size: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    scaler: MinMaxScaler
) -> Dict[str, Any]:
    """
    Train a single model and return its metrics and metadata.
    
    Args:
        model_id: Model architecture ID (1-10).
        ticker: Stock ticker symbol.
        lookback: Input sequence length.
        horizon: Prediction horizon.
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        X_train, y_train: Training data.
        X_val, y_val: Validation data.
        scaler: Fitted MinMaxScaler for inverse transform.
    
    Returns:
        Dictionary with model metadata, metrics, and file paths.
    """
    # Build and train model
    model = build_model(model_id, input_shape=(lookback, 1))
    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0
    )
    
    # Predict on validation set
    y_val_pred = model.predict(X_val, verbose=0).reshape(-1)
    
    # Calculate metrics in original scale
    y_val_original = scaler.inverse_transform(y_val.reshape(-1, 1)).reshape(-1)
    y_pred_original = scaler.inverse_transform(y_val_pred.reshape(-1, 1)).reshape(-1)
    metrics = _metrics_from_arrays(y_val_original, y_pred_original)
    
    # Save model and scaler
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    version = f"{ticker}_{model_id}_{timestamp}"
    path_model = os.path.join(MODELS_DIR, f"{version}.keras")
    path_scaler = os.path.join(MODELS_DIR, f"{version}.scaler")
    
    model.save(path_model)
    joblib.dump(scaler, path_scaler)
    
    # Create registry entry
    registry = ModelRegistry(
        ticker=ticker,
        model_id=model_id,
        model_name=MODEL_NAMES[model_id],
        version=version,
        path_model=path_model,
        path_scaler=path_scaler,
        mae=metrics["mae"],
        rmse=metrics["rmse"],
        mape=metrics["mape"],
        r2=None,  # Not used in current version
        accuracy=None,  # Not used in current version
        pearson_corr=metrics.get("pearson_corr"),
        params=json.dumps({
            "lookback": lookback,
            "horizon": horizon,
            "epochs": epochs,
            "batch_size": batch_size
        }),
        is_winner=False
    )
    
    _add_registry_with_retry(registry)
    
    return {
        "model_id": model_id,
        "model_name": MODEL_NAMES[model_id],
        "metrics": metrics,
        "version": version,
        "path_model": path_model,
        "path_scaler": path_scaler,
        "registered_at": datetime.utcnow().isoformat() + "Z"
    }


def _select_winner(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Select best model based on RMSE (primary) and MAE (tiebreaker).
    
    Args:
        results: List of result dictionaries from _train_single_model.
    
    Returns:
        Dictionary with winner model information.
    """
    # Sort by RMSE (ascending), then MAE (ascending)
    sorted_results = sorted(
        results,
        key=lambda r: (r["metrics"]["rmse"], r["metrics"]["mae"])
    )
    
    return sorted_results[0]


def _record_monitoring_metrics(
    ticker: str,
    elapsed: float,
    reused: bool
) -> None:
    """
    Record training metrics to Prometheus monitoring system.
    
    Args:
        ticker: Stock ticker symbol.
        elapsed: Training duration in seconds.
        reused: Whether existing model was reused (True) or retrained (False).
    """
    if not (RETRAIN_COUNT and RETRAIN_DURATION):
        return
    
    try:
        import psutil
        process = psutil.Process()
        
        mode = "fast" if reused else "retrain"
        
        # Record count and duration
        RETRAIN_COUNT.labels(ticker=ticker, mode=mode).inc()
        RETRAIN_DURATION.labels(ticker=ticker, mode=mode).observe(elapsed)
        
        # Record resource usage
        try:
            from ..monitoring import TRAIN_RAM_USAGE, TRAIN_CPU_PERCENT
            
            ram_mb = process.memory_info().rss / 1024 / 1024
            cpu_pct = process.cpu_percent(interval=0.1)
            
            TRAIN_RAM_USAGE.labels(ticker=ticker, mode=mode).set(ram_mb)
            TRAIN_CPU_PERCENT.labels(ticker=ticker, mode=mode).set(cpu_pct)
        except Exception:
            pass  # Resource metrics not critical
            
    except Exception:
        pass  # Monitoring failure should not break training


def train_all_models_fast(
    ticker: str,
    lookback: int = DEFAULT_LOOKBACK,
    horizon: int = DEFAULT_HORIZON,
    epochs: int = 1,
    batch_size: int = 32,
    reuse_if_exists: bool = True
) -> Dict[str, Any]:
    """
    Train all 10 LSTM/GRU models and select the best performing one.
    
    This function orchestrates the complete training pipeline:
    1. Check if winner already exists (if reuse_if_exists=True)
    2. Load and prepare training/validation data
    3. Train all 10 model architectures from model_zoo
    4. Evaluate each on validation set (MAE, RMSE, MAPE, Pearson)
    5. Select winner based on RMSE (lower is better)
    6. Update database with winner flag
    7. Record monitoring metrics
    
    Args:
        ticker: Stock ticker symbol (e.g., "AAPL").
        lookback: Number of past days to use as input (default: 60).
        horizon: Number of days ahead to predict (default: 1).
        epochs: Number of training epochs per model (default: 1 for fast mode).
        batch_size: Training batch size (default: 32).
        reuse_if_exists: If True, return existing winner without retraining.
    
    Returns:
        Dictionary with keys:
            - "results": List of all model results with metrics
            - "winner": Best performing model info
            - "reused": Boolean indicating if existing model was reused
            - "walltime_sec": Total execution time in seconds
    
    Raises:
        ValueError: If no data found for ticker or insufficient data for configuration.
    
    Example:
        >>> result = train_all_models_fast("AAPL", epochs=50, reuse_if_exists=False)
        >>> print(f"Winner: {result['winner']['model_name']}")
        >>> print(f"RMSE: {result['winner']['metrics']['rmse']:.2f}")
    
    Note:
        - Uses MinMaxScaler fitted only on training data to prevent leakage
        - Validation set is 20% of total data (chronological split)
        - All 10 models trained sequentially (no parallelization)
        - Winner selection based on validation RMSE only
        - Clears Keras session after training to free memory
    """
    stopwatch = Stopwatch()
    
    # Check for existing winner
    if reuse_if_exists:
        winner_info = _get_existing_winner(ticker)
        if winner_info:
            elapsed = stopwatch.stop()
            _record_monitoring_metrics(ticker, max(0.1, elapsed), reused=True)
            
            return {
                "results": [winner_info],
                "winner": winner_info,
                "reused": True,
                "walltime_sec": max(0.1, elapsed)
            }
    
    # Prepare training data
    scaler, X_train, y_train, X_val, y_val = _prepare_training_data(
        ticker, lookback, horizon, val_ratio=0.2
    )
    
    # Train all 10 models
    results = []
    for model_id in range(1, 11):
        result = _train_single_model(
            model_id=model_id,
            ticker=ticker,
            lookback=lookback,
            horizon=horizon,
            epochs=epochs,
            batch_size=batch_size,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            scaler=scaler
        )
        results.append(result)
    
    # Select and mark winner
    winner = _select_winner(results)
    _update_winner_with_retry(ticker, winner["version"])
    
    # Clear Keras session and record metrics
    _clear_keras_session()
    elapsed = stopwatch.stop()
    _record_monitoring_metrics(ticker, max(0.1, elapsed), reused=False)
    
    return {
        "results": results,
        "winner": winner,
        "reused": False,
        "walltime_sec": max(0.1, elapsed)
    }


# =============================================================================
# MODEL LOADING AND PREDICTION
# =============================================================================

def load_best_model(ticker: str) -> Tuple[keras.Model, MinMaxScaler, ModelRegistry]:
    """
    Load the best performing model for a ticker based on combined score.
    
    Calculates a combined score using normalized RMSE (60% weight) and Pearson
    correlation (40% weight). Lower RMSE is better, higher Pearson is better.
    
    Combined Score = 0.6 * (RMSE_normalized) + 0.4 * (1 - Pearson_normalized)
    
    Args:
        ticker: Stock ticker symbol (e.g., "AAPL").
    
    Returns:
        Tuple of:
            - model: Loaded Keras model ready for prediction
            - scaler: Fitted MinMaxScaler for data transformation
            - registry: ModelRegistry database record for metadata
    
    Raises:
        ValueError: If no models found for ticker in database.
    
    Example:
        >>> model, scaler, registry = load_best_model("AAPL")
        >>> print(f"Best model: {registry.model_name}")
        >>> print(f"RMSE: {registry.rmse:.2f}")
    
    Note:
        - Only considers models with non-null RMSE values
        - Pearson correlation defaults to 0.0 if null
        - Score normalization prevents bias towards metrics with larger scales
    """
    # Query all models for ticker with valid RMSE
    candidates = (
        ModelRegistry.query
        .filter_by(ticker=ticker)
        .filter(ModelRegistry.rmse.isnot(None))
        .all()
    )
    
    if not candidates:
        raise ValueError(f"No models found for ticker: {ticker}")
    
    # Extract metric values for normalization
    rmse_values = [c.rmse for c in candidates]
    pearson_values = [
        c.pearson_corr if c.pearson_corr is not None else 0.0
        for c in candidates
    ]
    
    min_rmse = min(rmse_values)
    max_rmse = max(rmse_values)
    min_pearson = min(pearson_values)
    max_pearson = max(pearson_values)
    
    # Calculate combined score for each candidate
    best_record = None
    best_score = float('inf')
    
    for candidate in candidates:
        # Normalize RMSE: 0 = best, 1 = worst
        rmse_normalized = (candidate.rmse - min_rmse) / (max_rmse - min_rmse + 1e-10)
        
        # Normalize Pearson: 1 = best, 0 = worst
        pearson_value = candidate.pearson_corr if candidate.pearson_corr is not None else 0.0
        pearson_normalized = (pearson_value - min_pearson) / (max_pearson - min_pearson + 1e-10)
        
        # Combined score: 60% RMSE weight, 40% Pearson weight
        score = (0.6 * rmse_normalized) + (0.4 * (1 - pearson_normalized))
        
        if score < best_score:
            best_score = score
            best_record = candidate
    
    if not best_record:
        raise ValueError(f"Failed to select best model for ticker: {ticker}")
    
    # Load model and scaler from disk
    model = keras.models.load_model(best_record.path_model)
    scaler = joblib.load(best_record.path_scaler)
    
    return model, scaler, best_record


def predict_horizon(
    model: keras.Model,
    scaler: MinMaxScaler,
    close_series: pd.Series,
    lookback: int = DEFAULT_LOOKBACK,
    horizon: int = DEFAULT_HORIZON
) -> float:
    """
    Make a multi-step ahead prediction using trained LSTM model.
    
    Takes the most recent 'lookback' values from the series, scales them,
    and uses the model to predict 'horizon' steps into the future.
    
    Args:
        model: Trained Keras LSTM model.
        scaler: Fitted MinMaxScaler (same one used during training).
        close_series: Time series of closing prices.
        lookback: Number of past timesteps to use as input.
        horizon: Number of steps ahead (must match model's training horizon).
    
    Returns:
        Predicted price in original scale (float).
    
    Example:
        >>> model, scaler, _ = load_best_model("AAPL")
        >>> prices = pd.Series([150, 151, 149, 152, 155, ...])
        >>> prediction = predict_horizon(model, scaler, prices, lookback=60, horizon=1)
        >>> print(f"Next day prediction: ${prediction:.2f}")
    
    Note:
        - Requires at least 'lookback' values in close_series
        - Prediction is returned in original price scale (inverse transform applied)
        - For horizon > 1, this is a direct multi-step prediction, not iterative
    """
    # Convert to scaled array
    series_array = close_series.values.reshape(-1, 1)
    series_scaled = scaler.transform(series_array)
    
    # Extract last lookback values
    last_window = series_scaled[-lookback:]
    
    # Reshape for model input: (1, lookback, 1)
    X_input = last_window.reshape(1, lookback, 1)
    
    # Predict (scaled)
    prediction_scaled = model.predict(X_input, verbose=0)[0][0]
    
    # Inverse transform to original scale
    prediction_original = scaler.inverse_transform([[prediction_scaled]])[0][0]
    
    return float(prediction_original)


# =============================================================================
# LEGACY FUNCTIONS (Deprecated - kept for backward compatibility)
# =============================================================================

def build_lstm(input_shape: Tuple[int, int]) -> keras.Model:
    """
    Build a simple LSTM model (LEGACY - use model_zoo instead).
    
    This function is kept for backward compatibility only.
    New code should use build_model() from model_zoo.py.
    
    Args:
        input_shape: Tuple of (lookback, n_features).
    
    Returns:
        Compiled Keras Sequential model.
    
    Deprecated:
        Use model_zoo.build_model(model_id=1) instead.
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def fit_model(
    close_series: pd.Series,
    lookback: int = DEFAULT_LOOKBACK,
    horizon: int = DEFAULT_HORIZON,
    epochs: int = 15,
    batch_size: int = 32
) -> Tuple[keras.Model, MinMaxScaler, Dict[str, float]]:
    """
    Train a single LSTM model on price series (LEGACY).
    
    This function is kept for backward compatibility only.
    New code should use train_all_models_fast() instead.
    
    Args:
        close_series: Pandas Series of closing prices.
        lookback: Number of past timesteps for input.
        horizon: Number of steps ahead to predict.
        epochs: Number of training epochs.
        batch_size: Training batch size.
    
    Returns:
        Tuple of (model, scaler, metrics_dict).
    
    Deprecated:
        Use train_all_models_fast() for better model selection.
    """
    # Convert series to array
    array = close_series.values.reshape(-1, 1)
    df = pd.DataFrame(array, columns=['close'])
    
    # Split and scale
    train_df, val_df = _train_val_split(pd.Series(array.flatten()), val_ratio=0.2)
    scaler = MinMaxScaler()
    
    scaler, X_train, y_train, X_val, y_val = scale_fit_transform_for_train(
        scaler, train_df, val_df, lookback, horizon
    )
    
    # Build and train model
    model = build_lstm((lookback, 1))
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0
    )
    
    # Predict and calculate metrics
    y_val_pred = model.predict(X_val, verbose=0)
    
    # Metrics in scaled space
    mae = np.mean(np.abs(y_val_pred.flatten() - y_val.flatten()))
    rmse = np.sqrt(np.mean((y_val_pred.flatten() - y_val.flatten()) ** 2))
    mape = float(np.mean(np.abs((y_val.flatten() - y_val_pred.flatten()) / (y_val.flatten() + 1e-9)))) * 100
    
    # R² (coefficient of determination)
    y_true = y_val.flatten()
    y_hat = y_val_pred.flatten()
    ss_res = np.sum((y_true - y_hat) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) + 1e-9
    r2 = float(1.0 - ss_res / ss_tot)
    
    # Directional accuracy
    val_array_scaled = scaler.transform(val_df.values.reshape(-1, 1))
    y_prev = val_array_scaled.flatten()[lookback - 1:-1]
    y_true_dir = y_true
    dir_true = np.sign(y_true_dir - y_prev[:len(y_true_dir)])
    dir_pred = np.sign(y_hat - y_prev[:len(y_true_dir)])
    hits = int(np.sum(dir_true == dir_pred))
    accuracy = float(hits) / len(dir_true) if len(dir_true) > 0 else 0.0
    
    metrics = {
        'mae': float(mae),
        'rmse': float(rmse),
        'mape': float(mape),
        'r2': float(r2),
        'hits': hits,
        'accuracy': accuracy
    }
    
    return model, scaler, metrics
