"""
API Models Routes - Endpoints para gerenciamento de modelos treinados.

Este módulo contém rotas para:
- Consulta do melhor modelo (winner)
- Listagem e resumo de modelos
- Informações dos 30 modelos avançados
- Download de modelos treinados
- Upload de modelos externos
- Predições com modelos avançados

Rotas:
    GET /api/models/best - Melhor modelo do ticker
    GET /api/models/summary - Resumo de todos os modelos
    GET /api/models-info - Info dos 30 modelos avançados
    GET /api/download-model - Download de modelo (.keras ou .scaler)
    POST /api/load-model - Upload de modelo externo
    GET /api/advanced-model-predictions - Predições para visualização

Autor: Sistema de Trading LSTM
Data: 2025-01-14
"""

import os
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from flask import Blueprint, request, jsonify, send_file
from flasgger import swag_from
from tensorflow import keras

from .. import db
from ..models import PrecoDiario, ModelRegistry
from ..ml.trainer import load_best_model
from ..ml.model_zoo_advanced import ADVANCED_MODEL_NAMES, ADVANCED_MODEL_DESCRIPTIONS
from ..utils.data_helpers import update_winner_flag

# Blueprint
api_models_bp = Blueprint('api_models', __name__)


@api_models_bp.get('/models/best')
def best_model():
    """
    Retorna detalhes do modelo vencedor (is_winner=True).
    
    Query Params:
        ticker (str): Símbolo do ticker (default: NVDA)
    
    Returns:
        JSON com:
        - ticker: Símbolo consultado
        - model_id: ID do modelo (1-30 ou 999 para custom)
        - model_name: Nome descritivo
        - version: Timestamp do modelo
        - registered_at: Data de registro
        - metrics: {mae, rmse, mape, r2, accuracy, pearson_corr}
    
    Errors:
        - 404: Nenhum modelo vencedor encontrado
    
    Example:
        GET /api/models/best?ticker=NVDA
        
        Response:
        {
            "ticker": "NVDA",
            "model_id": 15,
            "model_name": "Bidirectional GRU Deep",
            "version": "20250114_123456",
            "registered_at": "2025-01-14T12:34:56",
            "metrics": {
                "mae": 1.23,
                "rmse": 2.45,
                "mape": 0.015,
                "r2": 0.95,
                "accuracy": 0.97,
                "pearson_corr": 0.98
            }
        }
    """
    ticker = request.args.get('ticker', 'NVDA').upper()
    
    try:
        _, _, rec = load_best_model(ticker)
    except Exception:
        return jsonify({"error": "no winner"}), 404
    
    return jsonify({
        "ticker": ticker,
        "model_id": rec.model_id,
        "model_name": rec.model_name,
        "version": rec.version,
        "registered_at": rec.registered_at.isoformat() if rec.registered_at else None,
        "metrics": {
            "mae": rec.mae,
            "rmse": rec.rmse,
            "mape": rec.mape,
            "r2": rec.r2,
            "accuracy": rec.accuracy,
            "pearson_corr": rec.pearson_corr
        }
    })


@api_models_bp.get('/models/summary')
def models_summary():
    """
    Retorna resumo de todos os modelos com score combinado.
    
    Query Params:
        ticker (str): Símbolo do ticker (default: NVDA)
    
    Returns:
        JSON com:
        - ticker: Símbolo consultado
        - models: Lista ordenada por score (melhor primeiro)
            - model_id, model_name, version
            - mae, rmse, mape, r2, accuracy, pearson_corr
            - is_winner: True para o melhor
            - combined_score: Score calculado (menor = melhor)
            - registered_at: Data de registro
    
    Notes:
        - Score = 60% RMSE + 40% (1 - Pearson)
        - Ordena por score (menor primeiro)
        - Atualiza is_winner no banco automaticamente
    
    Example:
        GET /api/models/summary?ticker=NVDA
        
        Response:
        {
            "ticker": "NVDA",
            "models": [
                {
                    "model_id": 15,
                    "model_name": "Bidirectional GRU Deep",
                    "version": "20250114_123456",
                    "mae": 1.23,
                    "rmse": 2.45,
                    "mape": 0.015,
                    "r2": 0.95,
                    "accuracy": 0.97,
                    "pearson_corr": 0.98,
                    "is_winner": true,
                    "combined_score": 0.0523,
                    "registered_at": "2025-01-14T12:34:56"
                },
                ...
            ]
        }
    """
    ticker = request.args.get('ticker', 'NVDA').upper()
    
    try:
        rows = (ModelRegistry.query
                .filter_by(ticker=ticker)
                .filter(ModelRegistry.rmse.isnot(None))
                .all())
    except Exception as e:
        return jsonify({
            "ticker": ticker,
            "models": [],
            "note": f"{type(e).__name__}: {e}"
        }), 200
    
    if not rows:
        return jsonify({"ticker": ticker, "models": []})
    
    # Calcula scores
    rmse_values = [r.rmse for r in rows]
    pearson_values = [r.pearson_corr if r.pearson_corr is not None else 0.0 for r in rows]
    
    min_rmse = min(rmse_values)
    max_rmse = max(rmse_values)
    min_pearson = min(pearson_values)
    max_pearson = max(pearson_values)
    
    scored_models = []
    for r in rows:
        rmse_norm = (r.rmse - min_rmse) / (max_rmse - min_rmse + 1e-10)
        pearson_val = r.pearson_corr if r.pearson_corr is not None else 0.0
        pearson_norm = (pearson_val - min_pearson) / (max_pearson - min_pearson + 1e-10)
        score = (0.6 * rmse_norm) + (0.4 * (1 - pearson_norm))
        scored_models.append((score, r))
    
    scored_models.sort(key=lambda x: x[0])
    
    # Atualiza winner
    update_winner_flag(ticker, db.session)
    
    out = []
    for idx, (score, r) in enumerate(scored_models):
        out.append({
            "model_id": r.model_id,
            "model_name": r.model_name,
            "version": r.version,
            "mae": r.mae,
            "rmse": r.rmse,
            "mape": r.mape,
            "r2": r.r2,
            "accuracy": r.accuracy,
            "pearson_corr": r.pearson_corr,
            "is_winner": (idx == 0),
            "combined_score": round(score, 4),
            "registered_at": r.registered_at.isoformat() if r.registered_at else None
        })
    
    return jsonify({"ticker": ticker, "models": out})


@api_models_bp.get('/models-info')
def get_models_info():
    """
    Retorna informações sobre os 30 modelos avançados disponíveis.
    
    Returns:
        JSON com:
        - total_models: Total de modelos (30)
        - models: Lista com [id, name, description]
        - categories: Mapa de categorias com IDs
    
    Example:
        GET /api/models-info
        
        Response:
        {
            "total_models": 30,
            "models": [
                {
                    "id": 1,
                    "name": "LSTM Classic",
                    "description": "LSTM básico com 50 unidades..."
                },
                ...
            ],
            "categories": {
                "LSTM Base & Variants": [1, 2, 3, 4, 5],
                "GRU Base & Variants": [6, 7, 8, 9, 10],
                ...
            }
        }
    """
    models_info = [
        {
            "id": mid,
            "name": name,
            "description": ADVANCED_MODEL_DESCRIPTIONS.get(mid, "Sem descrição disponível")
        }
        for mid, name in ADVANCED_MODEL_NAMES.items()
    ]
    
    return jsonify({
        "total_models": len(models_info),
        "models": models_info,
        "categories": {
            "LSTM Base & Variants": list(range(1, 6)),
            "GRU Base & Variants": list(range(6, 11)),
            "Bidirectional": list(range(11, 16)),
            "Stacked Deep": list(range(16, 21)),
            "Residual & Skip": list(range(21, 26)),
            "Attention & Hybrid": list(range(26, 31))
        }
    })


@api_models_bp.route('/download-model', methods=['GET'])
def download_model():
    """
    Faz download de modelo ou scaler treinado.
    
    Query Params:
        model_name (str): Nome do modelo (ex: NVDA_1_20250114_123456)
        file_type (str): "model" ou "scaler" (default: model)
    
    Returns:
        Arquivo binário (.keras ou .scaler)
    
    Errors:
        - 400: model_name obrigatório
        - 400: file_type inválido
        - 404: Arquivo não encontrado
    
    Example:
        GET /api/download-model?model_name=NVDA_1_20250114_123456&file_type=model
        
        Response:
        File: NVDA_1_20250114_123456.keras (binary)
    """
    try:
        model_name = request.args.get('model_name')
        file_type = request.args.get('file_type', 'model')
        
        if not model_name:
            return jsonify({"error": "model_name é obrigatório"}), 400
        
        models_dir = os.path.join(os.getcwd(), 'models')
        
        if file_type == 'model':
            file_path = os.path.join(models_dir, f"{model_name}.keras")
        elif file_type == 'scaler':
            file_path = os.path.join(models_dir, f"{model_name}.scaler")
        else:
            return jsonify({"error": "file_type inválido (use 'model' ou 'scaler')"}), 400
        
        if not os.path.exists(file_path):
            return jsonify({"error": f"Arquivo não encontrado: {file_path}"}), 404
        
        return send_file(
            file_path,
            as_attachment=True,
            download_name=os.path.basename(file_path),
            mimetype='application/octet-stream'
        )
    
    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@api_models_bp.route('/load-model', methods=['POST'])
def load_model_endpoint():
    """
    Faz upload de modelo externo (.keras + .scaler).
    
    Form Data:
        model_file (file): Arquivo .keras
        scaler_file (file): Arquivo .scaler
    
    Returns:
        JSON com:
        - status: "success"
        - message: Mensagem descritiva
        - model_name: Nome do modelo salvo
        - params: Total de parâmetros
        - layers: Número de camadas
    
    Notes:
        - Salva em models/uploaded/
        - Valida modelo carregando-o
        - Registra no banco com model_id=999 (uploaded)
        - Atualiza winner flag automaticamente
    
    Errors:
        - 400: Ambos arquivos obrigatórios
        - 400: Arquivos vazios
        - 500: Erro ao carregar/validar
    
    Example:
        POST /api/load-model
        Form: model_file=model.keras, scaler_file=model.scaler
        
        Response:
        {
            "status": "success",
            "message": "Modelo carregado com sucesso",
            "model_name": "UPLOADED_20250114_123456",
            "params": 12800,
            "layers": 5
        }
    """
    try:
        if 'model_file' not in request.files or 'scaler_file' not in request.files:
            return jsonify({"error": "Envie ambos os arquivos: model_file e scaler_file"}), 400
        
        model_file = request.files['model_file']
        scaler_file = request.files['scaler_file']
        
        if model_file.filename == '' or scaler_file.filename == '':
            return jsonify({"error": "Arquivos vazios"}), 400
        
        # Diretório de upload
        upload_dir = os.path.join(os.getcwd(), 'models', 'uploaded')
        os.makedirs(upload_dir, exist_ok=True)
        
        # Salva arquivos
        model_path = os.path.join(upload_dir, model_file.filename)
        scaler_path = os.path.join(upload_dir, scaler_file.filename)
        
        model_file.save(model_path)
        scaler_file.save(scaler_path)
        
        # Valida carregando
        model = keras.models.load_model(model_path)
        scaler = joblib.load(scaler_path)
        
        total_params = int(model.count_params())
        num_layers = len(model.layers)
        
        # Registra no banco
        model_name = os.path.splitext(model_file.filename)[0]
        ticker = model_name.split('_')[0] if '_' in model_name else 'UPLOADED'
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        registry = ModelRegistry(
            ticker=ticker,
            model_id=999,
            model_name=f"Uploaded Model ({num_layers} layers)",
            version=timestamp,
            path_model=model_path,
            path_scaler=scaler_path,
            mae=0.0,
            rmse=0.0,
            mape=0.0,
            r2=None,
            accuracy=None,
            is_winner=False,
            registered_at=datetime.now()
        )
        db.session.add(registry)
        db.session.commit()
        
        update_winner_flag(ticker, db.session)
        
        return jsonify({
            "status": "success",
            "message": "Modelo carregado com sucesso",
            "model_name": model_name,
            "params": total_params,
            "layers": num_layers
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@api_models_bp.get('/advanced-model-predictions')
def get_advanced_model_predictions():
    """
    Retorna predições do modelo avançado para gráficos.
    
    Query Params:
        model_id (int): ID do modelo (1-30)
        version (str): Versão ou "latest" (default: latest)
    
    Returns:
        JSON com:
        - dates: Lista de datas
        - actual: Valores reais
        - predicted: Valores preditos
        - metrics: {rmse, mae, mape}
    
    Notes:
        - Usa conjunto de validação (20% dos dados)
        - Se version="latest", busca is_winner=True
        - Retorna apenas período de validação para visualização
    
    Errors:
        - 400: model_id obrigatório
        - 404: Modelo não encontrado
    
    Example:
        GET /api/advanced-model-predictions?model_id=15&version=latest
        
        Response:
        {
            "dates": ["2024-09-01", "2024-09-02", ...],
            "actual": [180.5, 182.3, ...],
            "predicted": [181.2, 183.1, ...],
            "metrics": {
                "rmse": 2.45,
                "mae": 1.23,
                "mape": 0.015
            }
        }
    """
    try:
        model_id = request.args.get('model_id', type=int)
        version = request.args.get('version', 'latest')
        ticker = 'NVDA'
        
        if not model_id:
            return jsonify({"error": "model_id é obrigatório"}), 400
        
        # Busca modelo
        if version == 'latest':
            model_reg = ModelRegistry.query.filter_by(
                ticker=ticker,
                model_id=model_id,
                is_winner=True
            ).order_by(ModelRegistry.registered_at.desc()).first()
        else:
            model_reg = ModelRegistry.query.filter_by(
                ticker=ticker,
                model_id=model_id,
                version=version
            ).first()
        
        if not model_reg:
            return jsonify({"error": "Modelo não encontrado"}), 404
        
        # Carrega modelo e scaler
        model = keras.models.load_model(model_reg.path_model)
        scaler = joblib.load(model_reg.path_scaler)
        
        # Carrega dados
        rows = db.session.query(PrecoDiario).filter_by(ticker=ticker).order_by(PrecoDiario.date.asc()).all()
        dates = [r.date.strftime('%Y-%m-%d') for r in rows]
        close = np.array([r.close for r in rows], dtype=np.float64)
        
        # Split 80/20
        split_idx = int(len(close) * 0.8)
        val_close = close[split_idx:]
        val_dates = dates[split_idx:]
        
        # Normaliza
        val_scaled = scaler.transform(val_close.reshape(-1, 1)).flatten()
        
        # Cria janelas (lookback=60, horizon=1)
        lookback = 60
        horizon = 1
        
        X_val, y_val = [], []
        for i in range(len(val_scaled) - lookback - horizon + 1):
            X_val.append(val_scaled[i:i+lookback])
            y_val.append(val_scaled[i+lookback:i+lookback+horizon])
        
        X_val = np.array(X_val).reshape(-1, lookback, 1)
        y_val = np.array(y_val)
        
        # Predições
        y_pred = model.predict(X_val, verbose=0)
        
        # Desnormaliza
        y_val_real = scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
        y_pred_real = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        
        result_dates = val_dates[lookback+horizon-1:]
        
        return jsonify({
            "dates": result_dates[:len(y_val_real)],
            "actual": y_val_real.tolist(),
            "predicted": y_pred_real.tolist(),
            "metrics": {
                "rmse": float(model_reg.rmse),
                "mae": float(model_reg.mae),
                "mape": float(model_reg.mape)
            }
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500
