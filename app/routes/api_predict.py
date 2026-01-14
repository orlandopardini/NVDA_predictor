"""
API Prediction Routes - Endpoints para predições com modelos treinados.

Este módulo contém rotas para:
- Predição de 1 passo à frente (próximo dia útil)
- Simulação multi-passo (até 252 dias)
- Predição com modelos carregados pelo usuário

Rotas:
    GET /api/predict - Prediz próximo dia útil
    GET /api/simulate - Simula N dias à frente
    POST /api/predict-loaded-model - Prediz com modelo uploaded

Autor: Sistema de Trading LSTM
Data: 2025-01-14
"""

import time
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from pandas.tseries.offsets import BDay
from flask import Blueprint, request, jsonify
from flasgger import swag_from
from tensorflow import keras
import joblib

from ..models import PrecoDiario, ModelRegistry
from ..ml.trainer import load_best_model
from ..ml.data import load_close_series
from ..ml.constants import DEFAULT_LOOKBACK, DEFAULT_HORIZON
from ..monitoring import INFERENCE_LATENCY

# Blueprint
api_predict_bp = Blueprint('api_predict', __name__)


@api_predict_bp.get('/predict')
@swag_from({
    'tags': ['Modelo'],
    'parameters': [
        {'name': 'ticker', 'in': 'query', 'schema': {'type': 'string'}, 'required': True},
        {'name': 'date', 'in': 'query', 'schema': {'type': 'string'}, 'required': False},
        {'name': 'horizon', 'in': 'query', 'schema': {'type': 'integer'}, 'required': False},
    ],
    'responses': {200: {'description': 'Previsão gerada com sucesso'}}
})
def predict():
    """
    Prediz preço do próximo dia útil usando modelo vencedor.
    
    Query Params:
        ticker (str): Símbolo do ticker (default: NVDA)
        lookback (int): Janela de lookback (default: 60)
        horizon (int): Horizonte de previsão (default: 1)
    
    Returns:
        JSON com:
        - ticker: Símbolo consultado
        - version: Versão do modelo usado
        - pred: Preço predito
        - date_cutoff: Última data disponível na base
        - date_next: Próximo dia ÚTIL (previsão)
    
    Notes:
        - Usa modelo marcado como is_winner=True
        - Predição baseada nos últimos lookback dias
        - date_next considera apenas dias úteis (BDay)
        - Latência registrada no Prometheus
    
    Errors:
        - 404: Nenhum modelo vencedor encontrado (treinar primeiro)
        - 400: Dados insuficientes (< lookback dias)
    
    Example:
        GET /api/predict?ticker=NVDA
        
        Response:
        {
            "ticker": "NVDA",
            "version": "20250114_123456",
            "pred": 285.32,
            "date_cutoff": "2025-01-13",
            "date_next": "2025-01-14"
        }
    """
    ticker = request.args.get('ticker', 'NVDA').upper()
    lookback = int(request.args.get('lookback', DEFAULT_LOOKBACK))

    # Carrega melhor modelo
    try:
        model, scaler, rec = load_best_model(ticker)
    except ValueError:
        return jsonify({
            "error": "No winner model for this ticker; treine primeiro via POST /api/train"
        }), 404

    # Lê série temporal (prioriza adj_close)
    rows = (PrecoDiario.query
            .filter_by(ticker=ticker)
            .order_by(PrecoDiario.date.asc())
            .all())
    
    if not rows:
        return jsonify({"error": "no data"}), 400

    idx = pd.to_datetime([r.date for r in rows])
    vals = [(r.adj_close if r.adj_close is not None else r.close) for r in rows]
    close = pd.Series(vals, index=idx, name="close").dropna()

    if len(close) < lookback:
        return jsonify({
            "error": f"Insufficient series length ({len(close)}) for lookback={lookback}"
        }), 400

    # Última janela
    s = close.values.reshape(-1, 1)
    s_scaled = scaler.transform(s)
    last = s_scaled[-lookback:].reshape(1, lookback, 1)

    # Predição com timing
    last_dt = pd.to_datetime(close.index[-1])
    next_dt = (last_dt + BDay(1)).date()
    
    t0 = time.perf_counter()
    yhat_scaled = model.predict(last, verbose=0)[0][0]
    dur = time.perf_counter() - t0
    
    yhat = float(scaler.inverse_transform([[yhat_scaled]])[0][0])
    
    # Registra latência no Prometheus
    INFERENCE_LATENCY.labels(ticker=ticker, version=rec.version).observe(dur)

    return jsonify({
        "ticker": ticker,
        "version": rec.version,
        "pred": yhat,
        "date_cutoff": last_dt.date().isoformat(),
        "date_next": next_dt.isoformat()
    })


@api_predict_bp.get('/simulate')
def simulate():
    """
    Prevê N dias à frente (1 dia por vez, recursivo).
    
    Query Params:
        ticker (str): Símbolo do ticker (default: NVDA)
        date (str): Data alvo YYYY-MM-DD (ou usar steps)
        steps (int): Número de dias úteis à frente (max: 252)
    
    Returns:
        JSON com:
        - ticker: Símbolo consultado
        - from: Última data disponível
        - to: Data alvo final
        - steps: Número de passos executados
        - version: Versão do modelo usado
        - series: Lista de {date, pred}
    
    Notes:
        - Se date fornecido, calcula steps automaticamente
        - Se steps fornecido, calcula data alvo
        - Limite de 252 dias (1 ano útil) para segurança
        - Predição recursiva: usa predições anteriores como input
    
    Errors:
        - 400: Nenhum modelo vencedor / dados insuficientes
        - 400: Data alvo deve ser futura
    
    Example:
        GET /api/simulate?ticker=NVDA&steps=30
        
        Response:
        {
            "ticker": "NVDA",
            "from": "2025-01-13",
            "to": "2025-02-20",
            "steps": 30,
            "version": "20250114_123456",
            "series": [
                {"date": "2025-01-14", "pred": 285.32},
                {"date": "2025-01-15", "pred": 287.15},
                ...
            ]
        }
    """
    ticker = (request.args.get('ticker') or 'NVDA').upper()
    date_str = request.args.get('date')
    steps_in = request.args.get('steps', type=int)

    # Vencedor mais recente
    rec = (ModelRegistry.query
           .filter_by(ticker=ticker, is_winner=True)
           .order_by(ModelRegistry.registered_at.desc())
           .first())
    
    if not rec:
        return jsonify({"error": "No winner model for this ticker"}), 400

    close = load_close_series(ticker)
    if close is None or close.empty:
        return jsonify({"error": "No price series"}), 400

    import json
    params = json.loads(rec.params or '{}')
    lookback = int(params.get('lookback', 60))
    last_dt = pd.to_datetime(close.index.max()).date()

    # Resolve steps
    if steps_in and steps_in > 0:
        steps = int(min(steps_in, 252))
        target_dt = (pd.Timestamp(last_dt) + BDay(steps)).date()
    else:
        if not date_str:
            return jsonify({"error": "missing date or steps"}), 400
        
        target_dt = pd.to_datetime(date_str).date()
        if target_dt <= last_dt:
            return jsonify({"error": "target must be after last available date"}), 400
        
        steps = len(pd.bdate_range(last_dt, target_dt)) - 1
        steps = int(min(steps, 252))

    # Carrega modelo e scaler
    model = keras.models.load_model(rec.path_model, compile=False)
    scaler = joblib.load(rec.path_scaler)

    s = close.values.astype(float).reshape(-1, 1)
    s_sc = scaler.transform(s)
    
    if len(s_sc) < lookback:
        return jsonify({"error": "series shorter than lookback"}), 400
    
    window = s_sc[-lookback:].reshape(1, lookback, 1)

    preds, cur = [], last_dt
    for _ in range(steps):
        yhat_sc = float(model.predict(window, verbose=0).reshape(-1)[0])
        yhat = float(scaler.inverse_transform([[yhat_sc]])[0, 0])
        cur = (pd.bdate_range(cur, periods=2)[-1]).date()
        preds.append({"date": cur.isoformat(), "pred": yhat})
        window = np.concatenate([window[:, 1:, :], [[[yhat_sc]]]], axis=1)

    return jsonify({
        "ticker": ticker,
        "from": last_dt.isoformat(),
        "to": target_dt.isoformat(),
        "steps": steps,
        "version": rec.version,
        "series": preds
    })


@api_predict_bp.post('/predict-loaded-model')
def predict_loaded_model():
    """
    Faz predição usando modelo carregado pelo usuário (uploaded).
    
    JSON Body:
        model_name (str): Nome do modelo (ex: AAPL_1_20250114_123456)
        ticker (str): Símbolo do ticker (default: AAPL)
        lookback (int): Janela de lookback (default: 60)
        horizon (int): Horizonte de previsão (default: 1)
    
    Returns:
        JSON com:
        - status: "success"
        - ticker: Símbolo usado
        - current_price: Preço atual
        - predicted_price: Preço predito
        - change_percent: % de mudança
        - lower_bound: Limite inferior (95% do predito)
        - upper_bound: Limite superior (105% do predito)
        - lookback: Janela usada
        - horizon: Horizonte usado
    
    Notes:
        - Busca modelo em models/uploaded/ primeiro
        - Se não encontrar, busca em models/
        - Baixa dados recentes do Yahoo Finance
        - Calcula intervalo de confiança estimado (±5%)
    
    Errors:
        - 400: model_name obrigatório
        - 404: Modelo não encontrado
        - 400: Dados insuficientes (<lookback dias)
    
    Example:
        POST /api/predict-loaded-model
        Body: {
            "model_name": "AAPL_1_20250114_123456",
            "ticker": "AAPL"
        }
        
        Response:
        {
            "status": "success",
            "ticker": "AAPL",
            "current_price": 185.50,
            "predicted_price": 187.25,
            "change_percent": 0.94,
            "lower_bound": 177.89,
            "upper_bound": 196.61,
            "lookback": 60,
            "horizon": 1
        }
    """
    try:
        data = request.get_json()
        model_name = data.get('model_name')
        ticker = data.get('ticker', 'AAPL')
        lookback = data.get('lookback', 60)
        horizon = data.get('horizon', 1)
        
        if not model_name:
            return jsonify({"error": "model_name é obrigatório"}), 400
        
        # Buscar caminhos do modelo
        import os
        models_dir = os.path.join(os.getcwd(), 'models')
        
        # Tenta uploaded/ primeiro
        model_path = os.path.join(models_dir, 'uploaded', f"{model_name}.keras")
        scaler_path = os.path.join(models_dir, 'uploaded', f"{model_name}.scaler")
        
        # Se não encontrar, tenta raiz
        if not os.path.exists(model_path):
            model_path = os.path.join(models_dir, f"{model_name}.keras")
            scaler_path = os.path.join(models_dir, f"{model_name}.scaler")
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            return jsonify({"error": f"Modelo não encontrado: {model_name}"}), 404
        
        # Carrega modelo e scaler
        model = keras.models.load_model(model_path)
        scaler = joblib.load(scaler_path)
        
        # Baixa dados históricos
        end_date = datetime.now()
        start_date = end_date - pd.Timedelta(days=lookback + 100)
        
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if df.empty or len(df) < lookback:
            return jsonify({
                "error": f"Dados insuficientes para {ticker}. Mínimo: {lookback} dias"
            }), 400
        
        # Prepara dados
        prices = df['Close'].values[-lookback:]
        prices_scaled = scaler.transform(prices.reshape(-1, 1))
        X = prices_scaled.reshape(1, lookback, 1)
        
        # Predição
        pred_scaled = model.predict(X, verbose=0)[0][0]
        pred_price = scaler.inverse_transform([[pred_scaled]])[0][0]
        
        current_price = float(df['Close'].iloc[-1])
        change_percent = ((pred_price - current_price) / current_price) * 100
        
        # Intervalo de confiança (±5% estimativa)
        lower_bound = pred_price * 0.95
        upper_bound = pred_price * 1.05
        
        return jsonify({
            "status": "success",
            "ticker": ticker,
            "current_price": float(current_price),
            "predicted_price": float(pred_price),
            "change_percent": float(change_percent),
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "lookback": lookback,
            "horizon": horizon
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500
