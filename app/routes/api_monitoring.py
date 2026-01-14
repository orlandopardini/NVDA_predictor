"""
API Monitoring Routes - Endpoints para monitoramento e health checks.

Este módulo contém rotas para:
- Health check da aplicação
- Métricas de treino e retreino
- Histórico de métricas
- Backtest de modelos
- Update diário (cron job)
- Progresso de treino

Rotas:
    GET /api/health - Health check
    GET /api/metrics - Métricas de resultados
    GET /api/retrain/history - Histórico de retreino
    GET /api/metrics/history - Histórico temporal de métricas
    GET /api/backtest - Backtest rolling
    POST /api/tasks/daily_update - Update diário (cron)
    GET /api/train-progress - Progresso do treino

Autor: Sistema de Trading LSTM
Data: 2025-01-14
"""

import pandas as pd
from flask import Blueprint, request, jsonify, current_app
from flasgger import swag_from

from .. import db
from ..models import PrecoDiario, ResultadoMetricas, RetrainHistory, ModelRegistry
from ..ml.trainer import load_best_model
from ..ml.eval import rolling_backtest_1step, metrics_from_series
from ..ml.constants import DEFAULT_LOOKBACK
from ..utils.auth_helpers import _auth_ok

# Blueprint
api_monitoring_bp = Blueprint('api_monitoring', __name__)


@api_monitoring_bp.get('/health')
def health():
    """
    Health check da aplicação.
    
    Returns:
        JSON com:
        - ok: True (sempre)
    
    Notes:
        - Usado por load balancers e monitoring tools
        - Retorna 200 OK se aplicação está responsiva
    
    Example:
        GET /api/health
        
        Response:
        {
            "ok": true
        }
    """
    return jsonify({'ok': True})


@api_monitoring_bp.get('/metrics')
@swag_from({'tags': ['Métricas']})
def metrics():
    """
    Retorna métricas dos últimos treinos.
    
    Query Params:
        ticker (str): Símbolo do ticker (default: NVDA)
    
    Returns:
        JSON com lista de métricas (últimos 50):
        - id, ticker, version, horizon
        - mae, rmse, mape, accuracy
        - trained_at
    
    Example:
        GET /api/metrics?ticker=NVDA
        
        Response:
        [
            {
                "id": 123,
                "ticker": "NVDA",
                "version": "20250114_123456",
                "horizon": 1,
                "mae": 1.23,
                "rmse": 2.45,
                "mape": 0.015,
                "accuracy": 0.97,
                "trained_at": "2025-01-14T12:34:56"
            },
            ...
        ]
    """
    ticker = request.args.get('ticker', 'NVDA').upper()
    
    rows = (ResultadoMetricas.query
            .filter_by(ticker=ticker)
            .order_by(ResultadoMetricas.trained_at.desc())
            .limit(50)
            .all())
    
    return jsonify([
        {
            'id': r.id,
            'ticker': r.ticker,
            'version': r.model_version,
            'horizon': r.horizon,
            'mae': r.mae,
            'rmse': r.rmse,
            'mape': r.mape,
            'accuracy': r.accuracy,
            'trained_at': r.trained_at.isoformat()
        } for r in rows
    ])


@api_monitoring_bp.get('/retrain/history')
@swag_from({'tags': ['Métricas']})
def retrain_history():
    """
    Retorna histórico de retreinos.
    
    Query Params:
        ticker (str): Símbolo do ticker (default: NVDA)
    
    Returns:
        JSON com lista de retreinos (últimos 100):
        - id, ticker, version
        - mae, rmse, mape
        - trigger (motivo do retreino)
        - created_at
    
    Example:
        GET /api/retrain/history?ticker=NVDA
        
        Response:
        [
            {
                "id": 45,
                "ticker": "NVDA",
                "version": "20250114_123456",
                "mae": 1.23,
                "rmse": 2.45,
                "mape": 0.015,
                "trigger": "manual",
                "created_at": "2025-01-14T12:34:56"
            },
            ...
        ]
    """
    ticker = request.args.get('ticker', 'NVDA').upper()
    
    rows = (RetrainHistory.query
            .filter_by(ticker=ticker)
            .order_by(RetrainHistory.created_at.desc())
            .limit(100)
            .all())
    
    return jsonify([
        {
            'id': r.id,
            'ticker': r.ticker,
            'version': r.model_version,
            'mae': r.mae,
            'rmse': r.rmse,
            'mape': r.mape,
            'trigger': r.trigger,
            'created_at': r.created_at.isoformat()
        } for r in rows
    ])


@api_monitoring_bp.get('/metrics/history')
def metrics_history():
    """
    Retorna evolução temporal das métricas.
    
    Query Params:
        ticker (str): Símbolo do ticker (default: NVDA)
    
    Returns:
        JSON com:
        - ticker: Símbolo consultado
        - history: Lista temporal com métricas
            - when: Timestamp
            - mae, rmse, mape, r2, pearson_corr
    
    Notes:
        - Ordenado cronologicamente
        - Útil para gráficos de evolução
    
    Example:
        GET /api/metrics/history?ticker=NVDA
        
        Response:
        {
            "ticker": "NVDA",
            "history": [
                {
                    "when": "2025-01-14T12:34:56",
                    "mae": 1.23,
                    "rmse": 2.45,
                    "mape": 0.015,
                    "r2": 0.95,
                    "pearson_corr": 0.98
                },
                ...
            ]
        }
    """
    ticker = (request.args.get('ticker') or 'NVDA').upper()
    
    rows = (ModelRegistry.query
            .filter_by(ticker=ticker)
            .order_by(ModelRegistry.registered_at.asc())
            .all())
    
    hist = []
    for r in rows:
        if r.mae is None:
            continue
        
        ts = r.registered_at.isoformat() if r.registered_at else None
        hist.append({
            "when": ts,
            "mae": r.mae,
            "rmse": r.rmse,
            "mape": r.mape,
            "r2": r.r2,
            "pearson_corr": r.pearson_corr,
        })
    
    return jsonify({"ticker": ticker, "history": hist})


@api_monitoring_bp.get('/backtest')
def backtest():
    """
    Executa backtest rolling 1-step do modelo vencedor.
    
    Query Params:
        ticker (str): Símbolo do ticker (default: NVDA)
        window (int): Janela de treino (default: 180)
        lookback (int): Lookback do modelo (default: 60)
    
    Returns:
        JSON com:
        - ticker: Símbolo consultado
        - version: Versão do modelo
        - registered_at: Data de registro
        - metrics: {mae, rmse, mape, r2, pearson_corr, accuracy}
        - series: {dates, y_true, y_pred}
    
    Notes:
        - Usa modelo vencedor (is_winner=True)
        - Backtest em janela deslizante
        - Retorna séries para plotagem
    
    Errors:
        - 400: Nenhum modelo vencedor / dados insuficientes
    
    Example:
        GET /api/backtest?ticker=NVDA&window=180
        
        Response:
        {
            "ticker": "NVDA",
            "version": "20250114_123456",
            "registered_at": "2025-01-14T12:34:56",
            "metrics": {
                "mae": 1.23,
                "rmse": 2.45,
                "mape": 0.015,
                "r2": 0.95,
                "pearson_corr": 0.98,
                "accuracy": 0.97
            },
            "series": {
                "dates": ["2024-09-01", ...],
                "y_true": [180.5, ...],
                "y_pred": [181.2, ...]
            }
        }
    """
    ticker = request.args.get('ticker', 'NVDA').upper()
    window = int(request.args.get('window', 180))
    lookback = int(request.args.get('lookback', DEFAULT_LOOKBACK))

    # Carrega vencedor
    model, scaler, rec = load_best_model(ticker)

    # Série temporal
    rows = (PrecoDiario.query
            .filter_by(ticker=ticker)
            .order_by(PrecoDiario.date.asc())
            .all())
    
    if not rows:
        return jsonify({"error": "no data"}), 400

    close = pd.Series(
        [r.close for r in rows],
        index=pd.to_datetime([r.date for r in rows])
    )

    # Backtest
    df_pred = rolling_backtest_1step(model, scaler, close, lookback=lookback, window=window)
    mets = metrics_from_series(df_pred)

    out = {
        "ticker": ticker,
        "version": rec.version,
        "registered_at": rec.registered_at.isoformat() if rec.registered_at else None,
        "metrics": mets,
        "series": {
            "dates": [d.strftime("%Y-%m-%d") for d in df_pred.index],
            "y_true": df_pred["y_true"].tolist(),
            "y_pred": df_pred["y_pred"].tolist(),
        }
    }
    
    return jsonify(out)


@api_monitoring_bp.post('/tasks/daily_update')
@swag_from({
    'tags': ['Tarefas'],
    'description': 'Endpoint para ser chamado pelo Render Cron Job'
})
def daily_update():
    """
    Update diário: atualiza dados + retreina modelo.
    
    JSON Body:
        ticker (str): Símbolo do ticker (default: NVDA)
    
    Returns:
        JSON com:
        - ok: True
        - updated: Resultado do update_data
        - train: Resultado do train
    
    Security:
        Requer autenticação via X-API-KEY header
    
    Notes:
        - Chamado por cron job (ex: Render Cron Job)
        - Atualiza dados históricos
        - Retreina modelo
        - Útil para manter modelos atualizados
    
    Example:
        POST /api/tasks/daily_update
        Headers: X-API-KEY: your-key
        Body: {"ticker": "NVDA"}
        
        Response:
        {
            "ok": true,
            "updated": {
                "ticker": "NVDA",
                "rows_added": 1,
                "range": ["2010-01-01", "2025-01-14"]
            },
            "train": {
                "ticker": "NVDA",
                "winner": {...},
                "duration_sec": 45.3
            }
        }
    """
    if not _auth_ok(request):
        return jsonify({'error': 'unauthorized'}), 401
    
    ticker = (request.json or {}).get('ticker', 'NVDA').upper()

    # Update data
    from .api_data import update_data as update_data_func
    with current_app.test_request_context(f'/api/update_data?ticker={ticker}'):
        resp = update_data_func()

    # Train
    from .api_train import train as train_func
    with current_app.test_request_context('/api/train', json={'ticker': ticker}):
        tr = train_func()

    return jsonify({'ok': True, 'updated': resp, 'train': tr})


@api_monitoring_bp.get('/train-progress')
def get_train_progress():
    """
    Retorna o progresso atual do treino avançado.
    
    Returns:
        JSON com:
        - is_training: bool (se está treinando)
        - mode: "fast" | "optimized"
        - current_model: Modelo atual sendo treinado
        - total_models: Total de modelos
        - current_trial: Trial atual (modo optimized)
        - total_trials: Total de trials (modo optimized)
        - percent: Progresso (0-100)
        - message: Mensagem descritiva
        - model_name: Nome do modelo atual
        - error: Mensagem de erro se houver
    
    Notes:
        - Usado para polling do frontend
        - Atualiza barra de progresso em tempo real
    
    Example:
        GET /api/train-progress
        
        Response:
        {
            "is_training": true,
            "mode": "fast",
            "current_model": 15,
            "total_models": 30,
            "current_trial": 0,
            "total_trials": 0,
            "percent": 50.0,
            "message": "Treinando modelo 15/30: Bidirectional GRU Deep",
            "model_name": "Bidirectional GRU Deep",
            "error": null
        }
    """
    from ..ml.training_progress import get_training_progress
    
    progress = get_training_progress()
    return jsonify(progress.get_progress())
