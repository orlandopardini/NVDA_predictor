# app/ml/trainer_advanced.py
"""
🎯 TRAINER AVANÇADO - Sistema de Treino com Otimização de Hiperparâmetros
Suporta 30 modelos + Grid/Random/Bayesian Search
"""
import os
import time
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sqlalchemy.exc import OperationalError
import logging

from .model_zoo_advanced import build_advanced_model, ADVANCED_MODEL_NAMES
from .hyperparameter_optimizer import optimize_hyperparameters, HyperparameterOptimizer
from .constants import DEFAULT_LOOKBACK, DEFAULT_HORIZON, MODELS_DIR
from .eval import metrics_from_series
from ..models import db, PrecoDiario, ModelRegistry, ResultadoMetricas
from ..monitoring import RETRAIN_COUNT, RETRAIN_DURATION, TRAIN_RAM_USAGE, TRAIN_CPU_PERCENT
from .training_progress import get_training_progress

logger = logging.getLogger(__name__)

# Helpers do trainer original
def _prepare_series(ticker: str) -> np.ndarray:
    """Carrega séries temporais do banco de dados"""
    rows = db.session.query(PrecoDiario).filter_by(ticker=ticker).order_by(PrecoDiario.date.asc()).all()
    if not rows:
        raise ValueError(f"Nenhum dado encontrado para ticker={ticker}")
    close = np.array([r.close for r in rows], dtype=np.float64)
    return close

def _train_val_split(series: np.ndarray, val_ratio: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    """Split train/val"""
    n = len(series)
    split_idx = int(n * (1 - val_ratio))
    return series[:split_idx], series[split_idx:]

def _make_supervised(arr: np.ndarray, lookback: int, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    """Cria janelas deslizantes (X, y)"""
    X_list, y_list = [], []
    for i in range(len(arr) - lookback - horizon + 1):
        X_list.append(arr[i:i+lookback])
        y_list.append(arr[i+lookback:i+lookback+horizon])
    return np.array(X_list), np.array(y_list)

def _add_registry_with_retry(reg, tries=6):
    """Adiciona registro ao banco com retry"""
    for i in range(tries):
        try:
            db.session.add(reg)
            db.session.commit()
            return
        except OperationalError as e:
            if "database is locked" not in str(e).lower():
                db.session.rollback()
                raise
            db.session.rollback()
            time.sleep(0.5 * (i + 1))
    db.session.add(reg)
    db.session.commit()

def _update_winner_with_retry(ticker, winner_version, tries=6):
    """Atualiza winner flag com retry"""
    for i in range(tries):
        try:
            ModelRegistry.query.filter_by(ticker=ticker, is_winner=True).update({"is_winner": False})
            db.session.query(ModelRegistry).filter(ModelRegistry.version == winner_version).update({"is_winner": True})
            db.session.commit()
            return
        except OperationalError as e:
            if "database is locked" not in str(e).lower():
                db.session.rollback()
                raise
            db.session.rollback()
            time.sleep(0.5 * (i + 1))
    ModelRegistry.query.filter_by(ticker=ticker, is_winner=True).update({"is_winner": False})
    db.session.query(ModelRegistry).filter(ModelRegistry.version == winner_version).update({"is_winner": True})
    db.session.commit()


def train_single_model_optimized(
    model_id: int,
    ticker: str,
    lookback: int = DEFAULT_LOOKBACK,
    horizon: int = DEFAULT_HORIZON,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    epochs: int = 30,
    dropout_rate: float = 0.2,
    activation: str = 'relu',
    use_early_stopping: bool = True,
    verbose: int = 0
) -> float:
    """
    Treina UM ÚNICO modelo com hiperparâmetros específicos.
    Retorna RMSE de validação (para otimização).
    """
    # Carrega dados
    close = _prepare_series(ticker)
    train, val = _train_val_split(close, val_ratio=0.2)
    
    # Normaliza
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train.reshape(-1, 1)).flatten()
    val_scaled = scaler.transform(val.reshape(-1, 1)).flatten()
    
    # Cria janelas
    Xtr, ytr = _make_supervised(train_scaled, lookback, horizon)
    Xva, yva = _make_supervised(val_scaled, lookback, horizon)
    
    # Reshape para LSTM
    Xtr = Xtr.reshape(Xtr.shape[0], Xtr.shape[1], 1)
    Xva = Xva.reshape(Xva.shape[0], Xva.shape[1], 1)
    
    # Build model
    model = build_advanced_model(
        model_id=model_id,
        input_shape=(lookback, 1),
        activation=activation,
        dropout_rate=dropout_rate
    )
    
    # Compila com learning rate customizado
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    # Callbacks
    callbacks = []
    if use_early_stopping:
        callbacks.append(EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=0
        ))
        callbacks.append(ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=0
        ))
    
    # Treina
    model.fit(
        Xtr, ytr,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(Xva, yva),
        callbacks=callbacks,
        verbose=verbose
    )
    
    # Avalia
    yva_pred = model.predict(Xva, verbose=0)
    
    # Desnormaliza
    inv = lambda x: scaler.inverse_transform(x.reshape(-1, 1)).flatten()
    yva_real = inv(yva.flatten())
    yva_pred_real = inv(yva_pred.flatten())
    
    # Calcula RMSE
    rmse = np.sqrt(np.mean((yva_real - yva_pred_real) ** 2))
    
    # Limpa memória
    K.clear_session()
    
    return rmse


def train_all_models_with_optimization(
    ticker: str = 'NVDA',
    model_ids: List[int] = None,
    lookback: int = DEFAULT_LOOKBACK,
    horizon: int = DEFAULT_HORIZON,
    optimization_strategy: str = 'random',
    n_trials_per_model: int = 20,
    save_models: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    MODO AVANÇADO: Treina múltiplos modelos com otimização de hiperparâmetros.
    
    Args:
        ticker: Sempre 'NVDA'
        model_ids: Lista de IDs para testar (ex: [1, 2, 3] ou None = todos 30)
        lookback: Janela temporal
        horizon: Horizonte de predição
        optimization_strategy: 'grid', 'random', 'bayesian'
        n_trials_per_model: Quantas combinações testar por modelo
        save_models: Salvar melhores modelos no disco
        verbose: Exibir progresso
    
    Returns:
        {
            'results': [...],  # todos os resultados
            'winner': {...},   # melhor modelo global
            'optimization_summary': {...}
        }
    """
    logger.info(f"🎯 Iniciando treino avançado: strategy={optimization_strategy}, trials={n_trials_per_model}")
    
    if model_ids is None:
        model_ids = list(range(1, 31))  # 1..30
    
    # Inicializa progresso
    progress = get_training_progress()
    progress.start_training(mode='optimized', total_models=len(model_ids), total_trials=n_trials_per_model)
    
    global_start = time.time()
    all_results = []
    
    try:
        for idx, model_id in enumerate(model_ids, 1):
            model_name = ADVANCED_MODEL_NAMES[model_id]
            logger.info(f"Testando modelo {model_id}/30: {model_name}")
            
            # Atualiza progresso para o início deste modelo
            progress.update_progress(
                current_model=idx,
                model_name=model_name,
                current_trial=0
            )
            
            # Função wrapper para otimização com callback de progresso
            def train_fn(mid, tick, **params):
                return train_single_model_optimized(
                    model_id=mid,
                    ticker=tick,
                    lookback=lookback,
                    horizon=horizon,
                    verbose=0,
                    **params
                )
            
            # Otimiza hiperparâmetros para este modelo
            try:
                opt_result = optimize_hyperparameters(
                    model_id=model_id,
                    ticker=ticker,
                    train_fn=train_fn,
                    strategy=optimization_strategy,
                    n_trials=n_trials_per_model,
                    verbose=False,
                    progress_callback=lambda trial_num: progress.update_progress(
                        current_model=idx,
                        model_name=model_name,
                        current_trial=trial_num
                    )
                )
                
                best_params = opt_result['best_params']
                best_score = opt_result['best_score']
                
                logger.info(f"  ✅ Melhor RMSE: {best_score:.4f} | Params: {best_params}")
                
                # Salva resultado
                result = {
                    'model_id': model_id,
                    'model_name': model_name,
                    'best_params': best_params,
                    'best_rmse': best_score,
                    'n_trials': opt_result['n_trials'],
                    'elapsed_time': opt_result['elapsed_time']
                }
                all_results.append(result)
            
            except Exception as e:
                logger.error(f"  ❌ Falhou: {e}")
                continue
        
        if not all_results:
            progress.finish_training(success=False, error="Nenhum modelo foi treinado com sucesso")
            raise RuntimeError("Nenhum modelo foi treinado com sucesso!")
    
    except Exception as e:
        progress.finish_training(success=False, error=str(e))
        raise
    
    # Encontra campeão global
    winner = min(all_results, key=lambda r: r['best_rmse'])
    logger.info(f"CAMPEÃO: Modelo {winner['model_id']} ({winner['model_name']}) - RMSE: {winner['best_rmse']:.4f}")
    
    # Re-treina campeão com melhores parâmetros e SALVA
    if save_models:
        logger.info("Salvando campeão...")
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Treina modelo final
        close = _prepare_series(ticker)
        train, val = _train_val_split(close, val_ratio=0.2)
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train.reshape(-1, 1)).flatten()
        val_scaled = scaler.transform(val.reshape(-1, 1)).flatten()
        
        Xtr, ytr = _make_supervised(train_scaled, lookback, horizon)
        Xva, yva = _make_supervised(val_scaled, lookback, horizon)
        Xtr = Xtr.reshape(Xtr.shape[0], Xtr.shape[1], 1)
        Xva = Xva.reshape(Xva.shape[0], Xva.shape[1], 1)
        
        best_params = winner['best_params']
        model = build_advanced_model(
            model_id=winner['model_id'],
            input_shape=(lookback, 1),
            activation=best_params.get('activation', 'relu'),
            dropout_rate=best_params.get('dropout_rate', 0.2)
        )
        
        optimizer = keras.optimizers.Adam(learning_rate=best_params.get('learning_rate', 0.001))
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        model.fit(
            Xtr, ytr,
            epochs=best_params.get('epochs', 30),
            batch_size=best_params.get('batch_size', 32),
            validation_data=(Xva, yva),
            callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
            verbose=1 if verbose else 0
        )
        
        # Salva arquivos
        os.makedirs(MODELS_DIR, exist_ok=True)
        model_path = os.path.join(MODELS_DIR, f"{ticker}_{winner['model_id']}_{version}.keras")
        scaler_path = os.path.join(MODELS_DIR, f"{ticker}_{winner['model_id']}_{version}.scaler")
        
        model.save(model_path)
        joblib.dump(scaler, scaler_path)
        
        # Salva metadata
        metadata = {
            'ticker': ticker,
            'model_id': winner['model_id'],
            'model_name': winner['model_name'],
            'version': version,
            'lookback': lookback,
            'horizon': horizon,
            'best_params': best_params,
            'rmse': winner['best_rmse'],
            'optimization_strategy': optimization_strategy,
            'n_trials': winner['n_trials']
        }
        
        metadata_path = os.path.join(MODELS_DIR, f"{ticker}_{winner['model_id']}_{version}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Registra no banco
        yva_pred = model.predict(Xva, verbose=0)
        inv = lambda x: scaler.inverse_transform(x.reshape(-1, 1)).flatten()
        yva_real = inv(yva.flatten())
        yva_pred_real = inv(yva_pred.flatten())
        
        mae = np.mean(np.abs(yva_real - yva_pred_real))
        mape = np.mean(np.abs((yva_real - yva_pred_real) / yva_real)) * 100
        pearson = float(np.corrcoef(yva_real, yva_pred_real)[0, 1]) if len(yva_real) > 1 else 0.0
        
        # Registra no ModelRegistry
        reg = ModelRegistry(
            ticker=ticker,
            model_id=winner['model_id'],
            model_name=winner['model_name'],
            version=version,
            path_model=model_path,
            path_scaler=scaler_path,
            mae=float(mae),
            rmse=float(winner['best_rmse']),
            mape=float(mape),
            pearson_corr=pearson,
            is_winner=True,
            model_metadata=json.dumps(metadata)
        )
        
        _update_winner_with_retry(ticker, version)
        _add_registry_with_retry(reg)
        
        # IMPORTANTE: Também salva em ResultadoMetricas para aparecer na tela principal
        resultado = ResultadoMetricas(
            ticker=ticker,
            model_version=version,
            horizon=horizon,
            mae=float(mae),
            rmse=float(winner['best_rmse']),
            mape=float(mape),
            trained_at=datetime.now()
        )
        _add_registry_with_retry(resultado)
        
        logger.info(f"Salvo: {model_path}")
    
    elapsed_total = time.time() - global_start
    
    # Métricas Prometheus
    RETRAIN_COUNT.labels(ticker=ticker, mode='advanced').inc()
    RETRAIN_DURATION.labels(ticker=ticker, mode='advanced').observe(elapsed_total)
    
    # Finaliza progresso
    progress.finish_training(success=True)
    
    # Adiciona versão e métricas ao winner se foi salvo
    if save_models:
        winner['version'] = version
        winner['mae'] = float(mae)
        winner['mape'] = float(mape)
        winner['pearson_corr'] = pearson
    
    summary = {
        'results': all_results,
        'winner': winner,
        'total_models_tested': len(all_results),
        'total_elapsed_time': elapsed_total,
        'optimization_strategy': optimization_strategy,
        'avg_time_per_model': elapsed_total / len(all_results) if all_results else 0
    }
    
    return summary


def _save_model_to_disk_and_registry(
    model: keras.Model,
    scaler: MinMaxScaler,
    ticker: str,
    model_id: int,
    model_name: str,
    lookback: int,
    horizon: int,
    rmse: float,
    mae: float,
    mape: float,
    pearson_corr: float,
    is_winner: bool = False,
    epochs: int = 1,
    batch_size: int = 32
) -> str:
    """
    🎯 Salva modelo no disco + registra no banco (ModelRegistry + ResultadoMetricas).
    
    Aplica Single Responsibility Principle: função dedicada para persistência.
    
    Args:
        model: Modelo Keras treinado
        scaler: MinMaxScaler ajustado
        ticker: Ticker do ativo
        model_id: ID do modelo (1-30)
        model_name: Nome descritivo do modelo
        lookback: Janela temporal usada
        horizon: Horizonte de predição
        rmse: Root Mean Squared Error
        mae: Mean Absolute Error
        mape: Mean Absolute Percentage Error
        pearson_corr: Correlação de Pearson
        is_winner: Se é o modelo campeão
        epochs: Número de epochs usadas no treinamento
        batch_size: Tamanho do batch
        
    Returns:
        version: String de versão do modelo salvo (formato: YYYYMMDD_HHMMSS)
    
    Raises:
        OSError: Se falhar ao salvar arquivos no disco
        OperationalError: Se falhar ao registrar no banco após retries
    """
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Garante que diretório existe
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Define caminhos dos arquivos
    model_path = os.path.join(MODELS_DIR, f"{ticker}_{model_id}_{version}.keras")
    scaler_path = os.path.join(MODELS_DIR, f"{ticker}_{model_id}_{version}.scaler")
    metadata_path = os.path.join(MODELS_DIR, f"{ticker}_{model_id}_{version}.json")
    
    # Salva modelo e scaler
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    logger.info(f"💾 Modelo salvo: {model_path}")
    
    # Salva metadata
    metadata = {
        'ticker': ticker,
        'model_id': model_id,
        'model_name': model_name,
        'version': version,
        'lookback': lookback,
        'horizon': horizon,
        'epochs': epochs,
        'batch_size': batch_size,
        'rmse': float(rmse),
        'mae': float(mae),
        'mape': float(mape),
        'pearson_corr': float(pearson_corr),
        'is_winner': is_winner
    }
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Registra no ModelRegistry
    reg = ModelRegistry(
        ticker=ticker,
        model_id=model_id,
        model_name=model_name,
        version=version,
        path_model=model_path,
        path_scaler=scaler_path,
        mae=float(mae),
        rmse=float(rmse),
        mape=float(mape),
        pearson_corr=float(pearson_corr),
        is_winner=is_winner,
        model_metadata=json.dumps(metadata)
    )
    _add_registry_with_retry(reg)
    logger.info(f"📝 Registrado no banco: ModelRegistry (version={version}, is_winner={is_winner})")
    
    # Registra em ResultadoMetricas (para tela principal)
    resultado = ResultadoMetricas(
        ticker=ticker,
        model_version=version,
        horizon=horizon,
        mae=float(mae),
        rmse=float(rmse),
        mape=float(mape),
        trained_at=datetime.now()
    )
    _add_registry_with_retry(resultado)
    logger.info(f"📝 Registrado no banco: ResultadoMetricas")
    
    return version


def train_all_models_fast_mode(
    ticker: str = 'NVDA',
    model_ids: List[int] = None,
    lookback: int = DEFAULT_LOOKBACK,
    horizon: int = DEFAULT_HORIZON,
    epochs: int = 1,
    batch_size: int = 32,
    save_all_models: bool = False
) -> Dict[str, Any]:
    """
    🚀 MODO RÁPIDO: Treina múltiplos modelos com parâmetros fixos (rápido).
    
    Diferença vs modo otimizado:
    - Usa hiperparâmetros padrão (sem busca)
    - Treina com 1 epoch por padrão (vs 30+)
    - Ideal para exploração rápida da arquitetura
    
    ⚠️ ATUALIZAÇÃO: Agora salva o modelo vencedor no banco para habilitar gráficos!
    
    Args:
        ticker: Ticker do ativo (sempre 'NVDA' no sistema)
        model_ids: Lista de IDs dos modelos a testar (1-30), None = todos
        lookback: Janela temporal de entrada
        horizon: Horizonte de predição
        epochs: Número de epochs de treinamento (padrão: 1 para velocidade)
        batch_size: Tamanho do batch
        save_all_models: Se True, salva TODOS os modelos; se False, só o vencedor
        
    Returns:
        {
            'results': List[Dict] - Resultados de todos os modelos testados
            'winner': Dict - Melhor modelo com campos:
                - model_id: int
                - model_name: str
                - version: str  ⭐ NOVO: versão do modelo salvo
                - rmse: float
                - mae: float
                - mape: float
                - pearson_corr: float
            'mode': str - Sempre 'fast'
            'total_models': int - Quantidade de modelos testados
        }
    
    Raises:
        ValueError: Se ticker não tiver dados
        RuntimeError: Se nenhum modelo treinar com sucesso
    """
    logger.info(f"🚀 MODO RÁPIDO: testando {len(model_ids) if model_ids else 30} modelos com {epochs} epoch(s)")
    
    if model_ids is None:
        model_ids = list(range(1, 31))
    
    # Inicializa progresso
    progress = get_training_progress()
    progress.start_training(mode='fast', total_models=len(model_ids), total_trials=1)
    
    global_start = time.time()
    
    try:
        results = []
        
        # Carrega dados uma vez (reutiliza para todos os modelos)
        close = _prepare_series(ticker)
        train, val = _train_val_split(close, val_ratio=0.2)
        
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train.reshape(-1, 1)).flatten()
        val_scaled = scaler.transform(val.reshape(-1, 1)).flatten()
        
        Xtr, ytr = _make_supervised(train_scaled, lookback, horizon)
        Xva, yva = _make_supervised(val_scaled, lookback, horizon)
        Xtr = Xtr.reshape(Xtr.shape[0], Xtr.shape[1], 1)
        Xva = Xva.reshape(Xva.shape[0], Xva.shape[1], 1)
        
        # Treina todos os modelos
        trained_models = []  # Guarda (model, metrics) para salvar depois
        
        for idx, model_id in enumerate(model_ids, 1):
            try:
                model_name = ADVANCED_MODEL_NAMES[model_id]
                
                # Atualiza progresso
                progress.update_progress(
                    current_model=idx,
                    model_name=model_name,
                    current_trial=1
                )
                
                # Treina modelo
                model = build_advanced_model(model_id, input_shape=(lookback, 1))
                model.fit(Xtr, ytr, epochs=epochs, batch_size=batch_size, verbose=0)
                
                # Avalia no validation set
                yva_pred = model.predict(Xva, verbose=0)
                inv = lambda x: scaler.inverse_transform(x.reshape(-1, 1)).flatten()
                yva_real = inv(yva.flatten())
                yva_pred_real = inv(yva_pred.flatten())
                
                # Calcula métricas
                rmse = float(np.sqrt(np.mean((yva_real - yva_pred_real) ** 2)))
                mae = float(np.mean(np.abs(yva_real - yva_pred_real)))
                mape = float(np.mean(np.abs((yva_real - yva_pred_real) / yva_real)) * 100)
                pearson = float(np.corrcoef(yva_real, yva_pred_real)[0, 1]) if len(yva_real) > 1 else 0.0
                
                result = {
                    'model_id': model_id,
                    'model_name': model_name,
                    'rmse': rmse,
                    'mae': mae,
                    'mape': mape,
                    'pearson_corr': pearson
                }
                results.append(result)
                
                # Guarda modelo + métricas para salvar depois
                trained_models.append((model, result))
                
                logger.info(f"✅ Modelo {model_id} ({model_name}): RMSE={rmse:.4f}, MAE={mae:.4f}")
                
                # Limpa memória após cada modelo
                K.clear_session()
            
            except Exception as e:
                logger.error(f"❌ Modelo {model_id} falhou: {e}")
                continue
        
        if not results:
            progress.finish_training(success=False, error="Nenhum modelo foi treinado com sucesso")
            raise RuntimeError("Nenhum modelo foi treinado com sucesso!")
        
        # Encontra campeão
        winner = min(results, key=lambda r: r['rmse'])
        winner_idx = results.index(winner)
        winner_model, _ = trained_models[winner_idx]
        
        logger.info(f"🏆 CAMPEÃO: Modelo {winner['model_id']} ({winner['model_name']}) - RMSE: {winner['rmse']:.4f}")
        
        # ⭐ SALVA O CAMPEÃO NO BANCO (para habilitar gráficos!)
        # Limpa flags de is_winner antigas
        _update_winner_with_retry(ticker, None)  # Limpa todos os is_winner primeiro
        
        # Salva campeão com is_winner=True
        winner_version = _save_model_to_disk_and_registry(
            model=winner_model,
            scaler=scaler,
            ticker=ticker,
            model_id=winner['model_id'],
            model_name=winner['model_name'],
            lookback=lookback,
            horizon=horizon,
            rmse=winner['rmse'],
            mae=winner['mae'],
            mape=winner['mape'],
            pearson_corr=winner['pearson_corr'],
            is_winner=True,
            epochs=epochs,
            batch_size=batch_size
        )
        
        # Adiciona version ao winner (NECESSÁRIO para os gráficos!)
        winner['version'] = winner_version
        
        # Opcionalmente salva todos os outros modelos
        if save_all_models:
            logger.info(f"💾 Salvando todos os {len(trained_models)-1} modelos restantes...")
            for model, result in trained_models:
                if result['model_id'] == winner['model_id']:
                    continue  # Já salvamos o campeão
                
                try:
                    version = _save_model_to_disk_and_registry(
                        model=model,
                        scaler=scaler,
                        ticker=ticker,
                        model_id=result['model_id'],
                        model_name=result['model_name'],
                        lookback=lookback,
                        horizon=horizon,
                        rmse=result['rmse'],
                        mae=result['mae'],
                        mape=result['mape'],
                        pearson_corr=result['pearson_corr'],
                        is_winner=False,
                        epochs=epochs,
                        batch_size=batch_size
                    )
                    result['version'] = version  # Adiciona version aos results
                except Exception as e:
                    logger.warning(f"⚠️ Falha ao salvar modelo {result['model_id']}: {e}")
        
        elapsed_total = time.time() - global_start
        
        # Métricas Prometheus
        RETRAIN_COUNT.labels(ticker=ticker, mode='fast').inc()
        RETRAIN_DURATION.labels(ticker=ticker, mode='fast').observe(elapsed_total)
        
        progress.finish_training(success=True)
        logger.info(f"✅ Treino rápido concluído em {elapsed_total:.2f}s")
        
    except Exception as e:
        progress.finish_training(success=False, error=str(e))
        logger.error(f"❌ Erro no treino rápido: {e}")
        raise
    
    return {
        'results': results,
        'winner': winner,
        'mode': 'fast',
        'total_models': len(results),
        'elapsed_time': elapsed_total
    }
