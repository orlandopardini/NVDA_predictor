"""
API Training Routes - Endpoints para treinamento de modelos LSTM.

Este módulo contém rotas para:
- Treinamento básico com 5 modelos padrão
- Treinamento customizado com arquitetura definida pelo usuário
- Treinamento avançado com 30 arquiteturas (fast/optimized)
- Atualização do modelo vencedor

Rotas:
    POST /api/train - Treino básico (5 modelos)
    POST /api/train-custom - Treino com arquitetura customizada
    POST /api/train-advanced - Treino avançado (30 modelos)
    POST /api/models/update-winner - Atualiza flag de vencedor

Autor: Sistema de Trading LSTM
Data: 2025-01-14
"""

import os
import time
import json
import psutil
import joblib
import numpy as np
from datetime import datetime
from flask import Blueprint, request, jsonify
from flasgger import swag_from

from .. import db
from ..models import ModelRegistry
from ..ml.trainer import train_all_models_fast
from ..ml.trainer_advanced import train_all_models_with_optimization, train_all_models_fast_mode
from ..ml.model_zoo import build_custom_model
from ..ml.data import load_close_series
from ..ml.trainer import _metrics_from_arrays
from ..ml.constants import DEFAULT_LOOKBACK, DEFAULT_HORIZON, MODELS_DIR
from ..utils.data_helpers import update_winner_flag

# Blueprint
api_train_bp = Blueprint('api_train', __name__)


@api_train_bp.post('/train')
def train():
    """
    Treina 5 modelos básicos e retorna o vencedor.
    
    JSON Body / Query Params:
        ticker (str): Símbolo do ticker (default: NVDA)
        lookback (int): Janela de lookback (default: 60)
        horizon (int): Horizonte de previsão (default: 1)
        epochs (int): Número de epochs (default: 1)
        batch_size (int): Tamanho do batch (default: 32)
        force (bool): Forçar retreino mesmo se existir modelo (default: False)
    
    Returns:
        JSON com resultados do treino:
        - ticker: Símbolo treinado
        - version: Timestamp do modelo vencedor
        - winner: Detalhes do melhor modelo
        - all_models: Lista com resultados dos 5 modelos
        - reused: Se reutilizou modelos existentes
        - duration_sec: Tempo total de treino
        - mae, rmse, mape, r2, accuracy: Métricas do vencedor
    
    Notes:
        - Treina 5 arquiteturas: Simple LSTM, Bidirectional, Stacked, etc.
        - Seleciona vencedor baseado em score combinado (RMSE + Pearson)
        - Se force=False e modelos existem, pode reutilizar
    
    Example:
        POST /api/train
        Body: {
            "ticker": "NVDA",
            "epochs": 10,
            "force": true
        }
        
        Response:
        {
            "ticker": "NVDA",
            "version": "20250114_123456",
            "winner": {
                "model_id": 3,
                "model_name": "Stacked LSTM",
                "metrics": {
                    "mae": 1.23,
                    "rmse": 2.45,
                    "mape": 0.015,
                    "r2": 0.95,
                    "accuracy": 0.97
                }
            },
            "all_models": [...],
            "duration_sec": 45.3
        }
    """
    payload = request.get_json(silent=True) or {}
    ticker = (payload.get('ticker') or request.args.get('ticker') or 'NVDA').upper()
    lookback = int(payload.get('lookback', DEFAULT_LOOKBACK))
    horizon = int(payload.get('horizon', DEFAULT_HORIZON))
    epochs = int(payload.get('epochs', 1))
    batch = int(payload.get('batch_size', 32))
    force = bool(payload.get('force', False))

    out = train_all_models_fast(
        ticker, lookback, horizon,
        epochs=epochs, batch_size=batch, reuse_if_exists=not force
    )

    w = out.get("winner", {})
    m = w.get("metrics", {})
    
    return jsonify({
        "ticker": ticker,
        "version": w.get("version"),
        "winner": w,
        "all_models": out.get("results", []),
        "reused": out.get("reused", False),
        "duration_sec": round(out.get("walltime_sec", 0.0), 2),
        "mae": m.get("mae"),
        "rmse": m.get("rmse"),
        "mape": m.get("mape"),
        "r2": m.get("r2"),
        "accuracy": m.get("accuracy"),
    })


@api_train_bp.post('/train-custom')
def train_custom_model():
    """
    Treina modelo LSTM totalmente personalizado.
    
    JSON Body:
        config (dict): Configuração das layers:
            - layers (list): Lista de dicts com type, units, activation, etc.
            - optimizer (str): Nome do otimizador (default: adam)
            - learning_rate (float): Taxa de aprendizado
        lookback (int): Janela de lookback (default: 60)
        horizon (int): Horizonte de previsão (default: 1)
        epochs (int): Número de epochs (default: 50)
        batch_size (int): Tamanho do batch (default: 32)
        validation_split (float): % para validação (default: 0.2)
        patience (int): Early stopping patience (default: 10)
        reduce_lr (bool): Usar ReduceLROnPlateau (default: True)
    
    Returns:
        JSON com resultados:
        - status: "success"
        - model_name: Nome do modelo salvo
        - metrics: {mae, rmse, mape, pearson_corr}
        - history: {loss, val_loss, mae, val_mae}
        - epochs_trained: Epochs executados (pode ser < config se early stop)
        - model_path: Caminho do modelo salvo
        - resources: {duration_sec, ram_used_mb, cpu_percent_avg}
        - predictions: {y_true, y_pred, residuals, split_index}
        - architecture: {summary, text, total_params, trainable_params}
    
    Example:
        POST /api/train-custom
        Body: {
            "config": {
                "layers": [
                    {"type": "LSTM", "units": 128, "return_sequences": true},
                    {"type": "Dropout", "rate": 0.2},
                    {"type": "LSTM", "units": 64},
                    {"type": "Dense", "units": 32, "activation": "relu"},
                    {"type": "Dense", "units": 1}
                ],
                "optimizer": "adam",
                "learning_rate": 0.001
            },
            "epochs": 50,
            "patience": 10
        }
    """
    try:
        # Tracking de recursos
        start_time = time.perf_counter()
        process = psutil.Process()
        ram_start = process.memory_info().rss / 1024 / 1024  # MB
        cpu_start = process.cpu_percent(interval=0.1)
        
        payload = request.get_json() or {}
        ticker = 'NVDA'  # Forçar NVIDIA
        
        # Configuração do modelo
        config = payload.get('config', {})
        if not config.get('layers'):
            return jsonify({"error": "Configuração de layers obrigatória"}), 400
        
        # Hiperparâmetros
        lookback = payload.get('lookback', DEFAULT_LOOKBACK)
        horizon = payload.get('horizon', DEFAULT_HORIZON)
        epochs = payload.get('epochs', 50)
        batch_size = payload.get('batch_size', 32)
        validation_split = payload.get('validation_split', 0.2)
        patience = payload.get('patience', 10)
        
        # Carregar dados
        close_series = load_close_series(ticker)
        if close_series is None or len(close_series) < lookback + 100:
            return jsonify({"error": "Dados insuficientes"}), 400
        
        # Preparar datasets
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(close_series.values.reshape(-1, 1))
        
        X, y = [], []
        for i in range(lookback, len(scaled) - horizon + 1):
            X.append(scaled[i - lookback:i])
            y.append(scaled[i + horizon - 1, 0])
        
        X = np.array(X)
        y = np.array(y)
        
        # Split treino/validação
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Construir modelo
        input_shape = (lookback, 1)
        model = build_custom_model(config, input_shape)
        
        # Callbacks
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
        ]
        
        if payload.get('reduce_lr', True):
            callbacks.append(ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7
            ))
        
        # Treinar
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0
        )
        
        # Avaliar
        y_pred_scaled = model.predict(X_val, verbose=0).flatten()
        y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_true = scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
        
        metrics = _metrics_from_arrays(y_true, y_pred)
        
        # Salvar modelo
        os.makedirs(MODELS_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{ticker}_CUSTOM_{timestamp}"
        model_path = os.path.join(MODELS_DIR, f"{model_name}.keras")
        scaler_path = os.path.join(MODELS_DIR, f"{model_name}.scaler")
        
        model.save(model_path)
        joblib.dump(scaler, scaler_path)
        
        # Registrar no banco
        registry = ModelRegistry(
            ticker=ticker,
            model_id=999,  # ID especial para custom
            model_name=f"Custom Model ({len(config.get('layers', []))} layers)",
            version=timestamp,
            path_model=model_path,
            path_scaler=scaler_path,
            mae=metrics['mae'],
            rmse=metrics['rmse'],
            mape=metrics['mape'],
            pearson_corr=metrics.get('pearson_corr'),
            r2=None,
            accuracy=None,
            is_winner=False,
            registered_at=datetime.now()
        )
        db.session.add(registry)
        db.session.commit()
        
        # Atualiza winner
        update_winner_flag(ticker, db.session)
        
        # Recursos
        duration = time.perf_counter() - start_time
        ram_end = process.memory_info().rss / 1024 / 1024
        cpu_end = process.cpu_percent(interval=0.1)
        
        # Predições completas
        y_pred_all_scaled = model.predict(X, verbose=0).flatten()
        y_pred_all = scaler.inverse_transform(y_pred_all_scaled.reshape(-1, 1)).flatten()
        y_true_all = scaler.inverse_transform(y.reshape(-1, 1)).flatten()
        residuals = y_true_all - y_pred_all
        
        # Arquitetura
        import io
        architecture_summary = []
        stream = io.StringIO()
        model.summary(print_fn=lambda x: stream.write(x + '\n'))
        architecture_text = stream.getvalue()
        
        for layer in model.layers:
            layer_info = {
                'name': layer.name,
                'type': layer.__class__.__name__,
                'output_shape': str(layer.output.shape if hasattr(layer, 'output') else 'N/A'),
                'params': int(layer.count_params())
            }
            if hasattr(layer, 'units'):
                layer_info['units'] = int(layer.units)
            if hasattr(layer, 'activation'):
                act = layer.activation
                layer_info['activation'] = str(act.__name__ if hasattr(act, '__name__') else act)
            architecture_summary.append(layer_info)
        
        return jsonify({
            "status": "success",
            "model_name": model_name,
            "metrics": {
                "mae": float(metrics['mae']),
                "rmse": float(metrics['rmse']),
                "mape": float(metrics['mape']),
                "pearson_corr": float(metrics.get('pearson_corr', 0))
            },
            "history": {
                'loss': [float(x) for x in history.history.get('loss', [])],
                'val_loss': [float(x) for x in history.history.get('val_loss', [])],
                'mae': [float(x) for x in history.history.get('mae', [])],
                'val_mae': [float(x) for x in history.history.get('val_mae', [])]
            },
            "epochs_trained": len(history.history['loss']),
            "model_path": model_path,
            "config": config,
            "resources": {
                "duration_sec": round(duration, 2),
                "ram_used_mb": round(ram_end - ram_start, 2),
                "cpu_percent_avg": round((cpu_start + cpu_end) / 2, 2)
            },
            "predictions": {
                "y_true": [float(x) for x in y_true_all.tolist()],
                "y_pred": [float(x) for x in y_pred_all.tolist()],
                "residuals": [float(x) for x in residuals.tolist()],
                "split_index": split_idx
            },
            "architecture": {
                "summary": architecture_summary,
                "text": architecture_text,
                "total_params": int(model.count_params()),
                "trainable_params": int(sum([l.count_params() for l in model.layers if l.trainable]))
            }
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@api_train_bp.post('/train-advanced')
def train_advanced():
    """
    Modo avançado: Treina com 30 modelos LSTM/GRU + otimização de hiperparâmetros.
    
    JSON Body:
        mode (str): "fast" ou "optimized"
        model_ids (list[int]): IDs dos modelos (1-30) ou null = todos
        lookback (int): Janela de lookback (default: 60)
        horizon (int): Horizonte de previsão (default: 1)
        
        # Para mode="fast":
        epochs (int): Número de epochs (default: 1)
        batch_size (int): Tamanho do batch (default: 32)
        
        # Para mode="optimized":
        optimization_strategy (str): "grid" / "random" / "bayesian"
        n_trials (int): Número de trials por modelo (default: 20)
    
    Returns:
        JSON com resultados:
        
        # mode="fast":
        - status: "success"
        - mode: "fast"
        - winner: Melhor modelo
        - total_models: Total de modelos treinados
        - results: Lista com resultados de cada modelo
        
        # mode="optimized":
        - status: "success"
        - mode: "optimized"
        - winner: Melhor configuração encontrada
        - total_models_tested: Total de trials executados
        - total_time: Tempo total
        - avg_time_per_model: Tempo médio por trial
        - optimization_strategy: Estratégia usada
        - all_results: Resultados detalhados
    
    Example:
        # Modo rápido
        POST /api/train-advanced
        Body: {
            "mode": "fast",
            "model_ids": [1, 2, 3, 4, 5],
            "epochs": 1
        }
        
        # Modo otimizado
        POST /api/train-advanced
        Body: {
            "mode": "optimized",
            "model_ids": null,
            "optimization_strategy": "random",
            "n_trials": 20
        }
    """
    try:
        payload = request.get_json() or {}
        
        mode = payload.get('mode', 'fast')
        ticker = 'NVDA'  # sempre NVIDIA
        model_ids = payload.get('model_ids')  # None = todos os 30
        lookback = payload.get('lookback', DEFAULT_LOOKBACK)
        horizon = payload.get('horizon', DEFAULT_HORIZON)
        
        if mode == 'fast':
            epochs = payload.get('epochs', 1)
            batch_size = payload.get('batch_size', 32)
            
            result = train_all_models_fast_mode(
                ticker=ticker,
                model_ids=model_ids,
                lookback=lookback,
                horizon=horizon,
                epochs=epochs,
                batch_size=batch_size
            )
            
            update_winner_flag(ticker, db.session)
            
            return jsonify({
                "status": "success",
                "mode": "fast",
                "winner": result['winner'],
                "total_models": len(result['results']),
                "results": result['results']
            })
        
        elif mode == 'optimized':
            strategy = payload.get('optimization_strategy', 'random')
            n_trials = payload.get('n_trials', 20)
            
            result = train_all_models_with_optimization(
                ticker=ticker,
                model_ids=model_ids,
                lookback=lookback,
                horizon=horizon,
                optimization_strategy=strategy,
                n_trials_per_model=n_trials,
                save_models=True,
                verbose=True
            )
            
            update_winner_flag(ticker, db.session)
            
            return jsonify({
                "status": "success",
                "mode": "optimized",
                "winner": result['winner'],
                "total_models_tested": result['total_models_tested'],
                "total_time": result['total_elapsed_time'],
                "avg_time_per_model": result['avg_time_per_model'],
                "optimization_strategy": result['optimization_strategy'],
                "all_results": result['results']
            })
        
        else:
            return jsonify({"error": f"Modo inválido: {mode}. Use 'fast' ou 'optimized'"}), 400
    
    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@api_train_bp.post('/models/update-winner')
def update_winner():
    """
    Recalcula e atualiza o flag is_winner baseado em score combinado.
    
    Query Params:
        ticker (str): Símbolo do ticker (default: NVDA)
    
    Returns:
        JSON com:
        - status: "success" ou "error"
        - message: Mensagem descritiva
    
    Notes:
        - Útil para atualizar modelos antigos
        - Útil após importação manual de dados
        - Recalcula score combinado: 60% RMSE + 40% Pearson
    
    Example:
        POST /api/models/update-winner?ticker=NVDA
        
        Response:
        {
            "status": "success",
            "message": "Winner flag atualizado para NVDA"
        }
    """
    ticker = request.args.get('ticker', 'NVDA').upper()
    try:
        update_winner_flag(ticker, db.session)
        return jsonify({
            "status": "success",
            "message": f"Winner flag atualizado para {ticker}"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500
