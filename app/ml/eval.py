"""
Módulo de avaliação e métricas para modelos LSTM.

Este módulo fornece funções para:
- Realizar backtesting de modelos em janelas deslizantes
- Calcular métricas de performance (MAE, RMSE, MAPE, R², Accuracy, Pearson)
- Avaliar qualidade de previsões em séries temporais

Princípios aplicados:
- Type hints para segurança de tipos
- Docstrings detalhadas seguindo Google Style
- Funções pequenas e coesas (Single Responsibility)
- Tratamento robusto de erros
"""

from typing import Dict, Optional, Tuple, Any
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .constants import DEFAULT_LOOKBACK, DEFAULT_HORIZON


def _prepare_backtest_data(
    close: pd.Series,
    lookback: int,
    window: int
) -> Tuple[pd.Series, int]:
    """
    Prepara dados para backtesting ajustando janela e índice inicial.
    
    Args:
        close: Série temporal de preços de fechamento
        lookback: Número de períodos para contexto histórico
        window: Tamanho da janela de backtesting
        
    Returns:
        Tupla contendo (segmento de dados, índice inicial)
        
    Raises:
        ValueError: Se dados insuficientes para o lookback
    """
    if len(close) < lookback:
        raise ValueError(
            f"Dados insuficientes: {len(close)} < lookback {lookback}"
        )
    
    adjusted_window = min(window, len(close))
    start_idx = max(0, len(close) - adjusted_window - lookback)
    segment = close.iloc[start_idx:].copy()
    
    return segment, start_idx


def _scale_and_reshape_data(
    segment: pd.Series,
    scaler: Any
) -> NDArray[np.float64]:
    """
    Normaliza e prepara dados para entrada no modelo LSTM.
    
    Args:
        segment: Segmento de dados a ser normalizado
        scaler: Scaler pré-treinado (MinMaxScaler ou similar)
        
    Returns:
        Array normalizado pronto para LSTM
    """
    values = segment.values.reshape(-1, 1)
    scaled_values = scaler.transform(values)
    return scaled_values


def _generate_predictions(
    model: Any,
    scaled_data: NDArray[np.float64],
    scaler: Any,
    segment: pd.Series,
    lookback: int
) -> Tuple[list, list]:
    """
    Gera previsões iterativas usando janela deslizante.
    
    Args:
        model: Modelo LSTM treinado
        scaled_data: Dados normalizados
        scaler: Scaler para inverter transformação
        segment: Segmento original com índices
        lookback: Tamanho da janela de lookback
        
    Returns:
        Tupla (lista de previsões, lista de índices temporais)
    """
    predictions = []
    indices = []
    
    for t in range(lookback, len(segment)):
        # Prepara input: [batch=1, timesteps=lookback, features=1]
        X = scaled_data[t - lookback:t].reshape(1, lookback, 1)
        
        # Predição normalizada
        y_pred_scaled = model.predict(X, verbose=0)[0][0]
        
        # Inverter normalização
        y_pred = scaler.inverse_transform([[y_pred_scaled]])[0][0]
        
        predictions.append(y_pred)
        indices.append(segment.index[t])
    
    return predictions, indices


def rolling_backtest_1step(
    model: Any,
    scaler: Any,
    close: pd.Series,
    lookback: int = 60,
    window: int = 180
) -> pd.DataFrame:
    """
    Realiza backtesting com janela deslizante de 1 passo à frente.
    
    Utiliza SEMPRE o scaler fornecido (pré-treinado) para garantir
    consistência nas transformações. Ideal para avaliar modelos em
    produção com dados históricos.
    
    Args:
        model: Modelo LSTM/GRU treinado com método predict()
        scaler: Scaler pré-treinado (ex: MinMaxScaler)
        close: Série temporal de preços de fechamento com DatetimeIndex
        lookback: Número de períodos históricos para input (padrão: 60)
        window: Tamanho da janela de backtesting em dias (padrão: 180)
        
    Returns:
        DataFrame com colunas:
            - y_true: Valores reais (escala original)
            - y_pred: Valores previstos (escala original)
            - index: DatetimeIndex correspondente
            
    Raises:
        ValueError: Se dados insuficientes ou scaler/model inválidos
        
    Example:
        >>> model = load_model('model.keras')
        >>> scaler = joblib.load('scaler.pkl')
        >>> results = rolling_backtest_1step(model, scaler, prices_df['close'])
        >>> metrics = metrics_from_series(results)
    """
    # Validações
    if not hasattr(model, 'predict'):
        raise ValueError("Model deve ter método predict()")
    if not hasattr(scaler, 'transform'):
        raise ValueError("Scaler deve ter método transform()")
    
    # 1. Preparar dados
    segment, _ = _prepare_backtest_data(close, lookback, window)
    
    # 2. Normalizar dados
    scaled_data = _scale_and_reshape_data(segment, scaler)
    
    # 3. Gerar previsões
    predictions, indices = _generate_predictions(
        model, scaled_data, scaler, segment, lookback
    )
    
    # 4. Construir DataFrame de resultados
    y_true = segment.iloc[lookback:].values.astype(float)
    
    result_df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": np.array(predictions)
    }, index=pd.to_datetime(indices))
    
    return result_df


def _find_column(
    columns: pd.Index,
    possible_names: Tuple[str, ...]
) -> Optional[str]:
    """
    Encontra coluna no DataFrame usando nomes alternativos.
    
    Args:
        columns: Índice de colunas do DataFrame
        possible_names: Tupla de nomes possíveis (case-insensitive)
        
    Returns:
        Nome da coluna encontrada ou None
    """
    cols_lower = {c.lower(): c for c in columns}
    
    for name in possible_names:
        if name in cols_lower:
            return cols_lower[name]
    
    return None


def _calculate_regression_metrics(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64]
) -> Dict[str, float]:
    """
    Calcula métricas padrão de regressão.
    
    Args:
        y_true: Valores reais
        y_pred: Valores previstos
        
    Returns:
        Dicionário com MAE, RMSE, MAPE, R²
    """
    # Mean Absolute Error
    mae = float(np.mean(np.abs(y_true - y_pred)))
    
    # Root Mean Squared Error
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    
    # Mean Absolute Percentage Error
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9)))) * 100.0
    
    # R² Score (Coefficient of Determination)
    ss_residual = float(np.sum((y_true - y_pred) ** 2))
    ss_total = float(np.sum((y_true - np.mean(y_true)) ** 2)) + 1e-9
    r2 = float(1.0 - ss_residual / ss_total)
    
    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "r2": r2
    }


def _calculate_directional_accuracy(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64]
) -> float:
    """
    Calcula acurácia direcional (previsão de subida/descida).
    
    Args:
        y_true: Valores reais
        y_pred: Valores previstos
        
    Returns:
        Acurácia entre 0 e 1
    """
    # Valores anteriores (t-1)
    y_prev = np.r_[y_true[0], y_true[:-1]]
    
    # Direção real: sinal de (y_t - y_{t-1})
    direction_true = np.sign(y_true - y_prev)
    
    # Direção prevista: sinal de (y_pred_t - y_{t-1})
    direction_pred = np.sign(y_pred - y_prev)
    
    # Proporção de acertos na direção
    accuracy = float(np.mean(direction_true == direction_pred))
    
    return accuracy


def _calculate_pearson_correlation(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64]
) -> float:
    """
    Calcula coeficiente de correlação de Pearson.
    
    Args:
        y_true: Valores reais
        y_pred: Valores previstos
        
    Returns:
        Coeficiente entre -1 e 1
    """
    if len(y_true) <= 1:
        return 0.0
    
    correlation_matrix = np.corrcoef(y_true, y_pred)
    pearson_corr = float(correlation_matrix[0, 1])
    
    return pearson_corr


def metrics_from_series(df_pred: pd.DataFrame) -> Dict[str, float]:
    """
    Calcula métricas completas a partir de DataFrame de previsões.
    
    Aceita nomes de colunas flexíveis:
        - Valores reais: y_true | real | true | target
        - Valores previstos: y_pred | pred | previsto | forecast
    
    Args:
        df_pred: DataFrame com colunas de valores reais e previstos
        
    Returns:
        Dicionário contendo:
            - mae: Mean Absolute Error
            - rmse: Root Mean Squared Error
            - mape: Mean Absolute Percentage Error (%)
            - r2: R² Score (Coefficient of Determination)
            - accuracy: Acurácia direcional (0-1)
            - pearson_corr: Coeficiente de correlação de Pearson (-1 a 1)
            
    Raises:
        ValueError: Se colunas necessárias não forem encontradas
        
    Example:
        >>> df = pd.DataFrame({'y_true': [1, 2, 3], 'y_pred': [1.1, 2.1, 2.9]})
        >>> metrics = metrics_from_series(df)
        >>> print(f"RMSE: {metrics['rmse']:.4f}")
    """
    # Encontrar colunas de valores reais
    true_col = _find_column(
        df_pred.columns,
        ("y_true", "real", "true", "target")
    )
    
    # Encontrar colunas de valores previstos
    pred_col = _find_column(
        df_pred.columns,
        ("y_pred", "pred", "previsto", "forecast")
    )
    
    # Validar presença de colunas
    if true_col is None or pred_col is None:
        raise ValueError(
            f"Colunas não encontradas no DataFrame. "
            f"Colunas disponíveis: {list(df_pred.columns)}. "
            f"Esperado: (y_true/real/true/target) e (y_pred/pred/previsto/forecast)"
        )
    
    # Extrair arrays
    y_true = df_pred[true_col].astype(float).to_numpy()
    y_pred = df_pred[pred_col].astype(float).to_numpy()
    
    # Calcular métricas
    regression_metrics = _calculate_regression_metrics(y_true, y_pred)
    directional_acc = _calculate_directional_accuracy(y_true, y_pred)
    pearson = _calculate_pearson_correlation(y_true, y_pred)
    
    # Combinar todas as métricas
    all_metrics = {
        **regression_metrics,
        "accuracy": directional_acc,
        "pearson_corr": pearson
    }
    
    return all_metrics
