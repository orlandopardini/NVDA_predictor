"""
📊 MODELS - Modelos de Dados SQLAlchemy
========================================

Define a estrutura do banco de dados SQLite para o sistema de
previsão de preços de ações usando LSTM.

Modelos:
    - PrecoDiario: Dados históricos OHLCV (preços diários)
    - ResultadoMetricas: Métricas de avaliação dos modelos
    - RetrainHistory: Histórico de retreinamentos
    - ModelRegistry: Registro de modelos treinados

Padrões aplicados:
    - Properties para cálculos derivados
    - Métodos de negócio para validações
    - Representação clara (__repr__)
    - Docstrings completas
    - Constraints de integridade
"""

from typing import Dict, Any, Optional
from datetime import datetime, date
from . import db


class PrecoDiario(db.Model):
    """
    Modelo para armazenar dados históricos de preços (OHLCV).
    
    Representa um dia de negociação com preços de abertura, máxima,
    mínima, fechamento, fechamento ajustado e volume.
    
    Attributes:
        id (int): Chave primária auto-incrementada
        ticker (str): Símbolo do ativo (ex: 'NVDA', 'AAPL')
        date (date): Data da negociação
        open (float): Preço de abertura
        high (float): Preço máximo do dia
        low (float): Preço mínimo do dia
        close (float): Preço de fechamento
        adj_close (float): Preço de fechamento ajustado (splits, dividendos)
        volume (int): Volume de negociações
        created_at (datetime): Data de criação do registro
        
    Constraints:
        - Unique: (ticker, date) - Não permite duplicatas
        - Index: ticker, date - Para queries rápidas
        
    Example:
        >>> preco = PrecoDiario(
        ...     ticker='NVDA',
        ...     date=date(2023, 1, 15),
        ...     open=150.0,
        ...     high=155.0,
        ...     low=148.0,
        ...     close=153.0,
        ...     adj_close=153.0,
        ...     volume=50000000
        ... )
        >>> db.session.add(preco)
        >>> db.session.commit()
    """
    __tablename__ = 'preco_diario'
    
    # Colunas
    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String(16), index=True, nullable=False)
    date = db.Column(db.Date, index=True, nullable=False)
    open = db.Column(db.Float)
    high = db.Column(db.Float)
    low = db.Column(db.Float)
    close = db.Column(db.Float)
    adj_close = db.Column(db.Float)
    volume = db.Column(db.BigInteger)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Constraints
    __table_args__ = (
        db.UniqueConstraint('ticker', 'date', name='uq_ticker_date'),
    )
    
    @property
    def daily_return(self) -> Optional[float]:
        """
        Calcula o retorno percentual do dia (close vs open).
        
        Returns:
            Retorno percentual ou None se open for inválido
            
        Example:
            >>> preco.daily_return
            2.0  # 2% de alta no dia
        """
        if self.open and self.open > 0:
            return ((self.close - self.open) / self.open) * 100
        return None
    
    @property
    def price_range(self) -> Optional[float]:
        """
        Calcula a amplitude de preço do dia (high - low).
        
        Returns:
            Diferença entre máxima e mínima ou None
            
        Example:
            >>> preco.price_range
            7.0  # $7 de variação
        """
        if self.high is not None and self.low is not None:
            return self.high - self.low
        return None
    
    @property
    def is_up_day(self) -> bool:
        """
        Verifica se foi um dia de alta (close > open).
        
        Returns:
            True se fechamento > abertura, False caso contrário
            
        Example:
            >>> preco.is_up_day
            True  # Dia de alta
        """
        return self.close > self.open if (self.close and self.open) else False
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converte o registro para dicionário (útil para APIs JSON).
        
        Returns:
            Dicionário com todos os campos
            
        Example:
            >>> preco.to_dict()
            {'ticker': 'NVDA', 'date': '2023-01-15', ...}
        """
        return {
            'id': self.id,
            'ticker': self.ticker,
            'date': self.date.isoformat() if self.date else None,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'adj_close': self.adj_close,
            'volume': self.volume,
            'daily_return': self.daily_return,
            'price_range': self.price_range,
            'is_up_day': self.is_up_day,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    def __repr__(self) -> str:
        """Representação legível do objeto."""
        return (
            f"<PrecoDiario(ticker='{self.ticker}', "
            f"date={self.date}, close={self.close})>"
        )

class ResultadoMetricas(db.Model):
    """
    Modelo para armazenar métricas de avaliação de modelos treinados.
    
    Armazena resultados de avaliação como RMSE, MAE, MAPE, além de
    informações sobre acurácia direcional e drift detection.
    
    Attributes:
        id (int): Chave primária
        ticker (str): Símbolo do ativo
        model_version (str): Versão do modelo (timestamp)
        horizon (int): Horizonte de predição (dias à frente)
        split_start (date): Data de início da avaliação
        split_end (date): Data de fim da avaliação
        mae (float): Mean Absolute Error
        rmse (float): Root Mean Squared Error
        mape (float): Mean Absolute Percentage Error
        r2 (float): R² Score (coeficiente de determinação)
        hits (int): Número de acertos direcionais
        accuracy (float): Acurácia direcional (%)
        drift_mae (float): MAE do drift detector
        trained_at (datetime): Data/hora do treinamento
        
    Example:
        >>> resultado = ResultadoMetricas(
        ...     ticker='NVDA',
        ...     model_version='20230115_143000',
        ...     horizon=5,
        ...     mae=2.5,
        ...     rmse=3.2,
        ...     mape=1.8,
        ...     accuracy=65.5
        ... )
        >>> db.session.add(resultado)
    """
    __tablename__ = 'resultado_metricas'
    
    # Colunas
    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String(16), index=True, nullable=False)
    model_version = db.Column(db.String(64), index=True, nullable=False)
    horizon = db.Column(db.Integer, default=1)
    split_start = db.Column(db.Date)
    split_end = db.Column(db.Date)
    mae = db.Column(db.Float)
    rmse = db.Column(db.Float)
    mape = db.Column(db.Float)
    r2 = db.Column(db.Float)
    hits = db.Column(db.Integer)
    accuracy = db.Column(db.Float)
    drift_mae = db.Column(db.Float)
    trained_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    @property
    def performance_grade(self) -> str:
        """
        Classifica a performance do modelo baseado no MAPE.
        
        Escala:
            - Excelente: MAPE < 2%
            - Bom: 2% ≤ MAPE < 5%
            - Razoável: 5% ≤ MAPE < 10%
            - Ruim: MAPE ≥ 10%
            
        Returns:
            String com classificação
            
        Example:
            >>> resultado.performance_grade
            'Excelente'
        """
        if self.mape is None:
            return 'N/A'
        if self.mape < 2.0:
            return 'Excelente'
        elif self.mape < 5.0:
            return 'Bom'
        elif self.mape < 10.0:
            return 'Razoável'
        else:
            return 'Ruim'
    
    @property
    def is_accurate(self) -> bool:
        """
        Verifica se o modelo tem boa acurácia direcional (>50%).
        
        Returns:
            True se acurácia > 50%, False caso contrário
        """
        return self.accuracy > 50.0 if self.accuracy is not None else False
    
    @property
    def days_since_training(self) -> Optional[int]:
        """
        Calcula quantos dias se passaram desde o treinamento.
        
        Returns:
            Número de dias ou None se trained_at for None
            
        Example:
            >>> resultado.days_since_training
            3  # Treinado há 3 dias
        """
        if self.trained_at:
            delta = datetime.utcnow() - self.trained_at
            return delta.days
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário (útil para APIs)."""
        return {
            'id': self.id,
            'ticker': self.ticker,
            'model_version': self.model_version,
            'horizon': self.horizon,
            'mae': self.mae,
            'rmse': self.rmse,
            'mape': self.mape,
            'r2': self.r2,
            'accuracy': self.accuracy,
            'performance_grade': self.performance_grade,
            'is_accurate': self.is_accurate,
            'days_since_training': self.days_since_training,
            'trained_at': self.trained_at.isoformat() if self.trained_at else None
        }
    
    def __repr__(self) -> str:
        """Representação legível do objeto."""
        return (
            f"<ResultadoMetricas(ticker='{self.ticker}', "
            f"version='{self.model_version}', "
            f"rmse={self.rmse:.2f if self.rmse else 'N/A'})>"
        )

class RetrainHistory(db.Model):
    """
    Modelo para rastrear histórico de retreinamentos.
    
    Registra cada retreinamento automático ou manual, incluindo
    o motivo do retreinamento (trigger) e estatísticas de drift.
    
    Attributes:
        id (int): Chave primária
        ticker (str): Símbolo do ativo
        model_version (str): Versão do modelo retreinado
        train_start (date): Início do período de treino
        train_end (date): Fim do período de treino
        eval_start (date): Início do período de avaliação
        eval_end (date): Fim do período de avaliação
        mae (float): MAE após retreinamento
        rmse (float): RMSE após retreinamento
        mape (float): MAPE após retreinamento
        r2 (float): R² após retreinamento
        trigger (str): Motivo do retreinamento ('manual', 'drift', 'scheduled')
        drift_stat (float): Estatística de drift detection
        created_at (datetime): Data/hora do retreinamento
        
    Example:
        >>> retrain = RetrainHistory(
        ...     ticker='NVDA',
        ...     model_version='20230115_143000',
        ...     trigger='drift',
        ...     drift_stat=0.85,
        ...     mae=2.3,
        ...     rmse=3.0
        ... )
    """
    __tablename__ = 'retrain_history'
    
    # Colunas
    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String(16), index=True, nullable=False)
    model_version = db.Column(db.String(64), index=True, nullable=False)
    train_start = db.Column(db.Date)
    train_end = db.Column(db.Date)
    eval_start = db.Column(db.Date)
    eval_end = db.Column(db.Date)
    mae = db.Column(db.Float)
    rmse = db.Column(db.Float)
    mape = db.Column(db.Float)
    r2 = db.Column(db.Float)
    trigger = db.Column(db.String(16))
    drift_stat = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    @property
    def train_duration_days(self) -> Optional[int]:
        """
        Calcula duração do período de treinamento em dias.
        
        Returns:
            Número de dias ou None se datas não disponíveis
        """
        if self.train_start and self.train_end:
            return (self.train_end - self.train_start).days
        return None
    
    @property
    def eval_duration_days(self) -> Optional[int]:
        """
        Calcula duração do período de avaliação em dias.
        
        Returns:
            Número de dias ou None se datas não disponíveis
        """
        if self.eval_start and self.eval_end:
            return (self.eval_end - self.eval_start).days
        return None
    
    @property
    def has_drift(self) -> bool:
        """
        Verifica se o retreinamento foi causado por drift.
        
        Returns:
            True se trigger == 'drift'
        """
        return self.trigger == 'drift' if self.trigger else False
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return {
            'id': self.id,
            'ticker': self.ticker,
            'model_version': self.model_version,
            'trigger': self.trigger,
            'drift_stat': self.drift_stat,
            'mae': self.mae,
            'rmse': self.rmse,
            'mape': self.mape,
            'r2': self.r2,
            'train_duration_days': self.train_duration_days,
            'eval_duration_days': self.eval_duration_days,
            'has_drift': self.has_drift,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    def __repr__(self) -> str:
        """Representação legível do objeto."""
        return (
            f"<RetrainHistory(ticker='{self.ticker}', "
            f"trigger='{self.trigger}', "
            f"version='{self.model_version}')>"
        )

class ModelRegistry(db.Model):
    """
    Modelo para registro central de todos os modelos treinados.
    
    Mantém catálogo de modelos com metadados, métricas de performance,
    caminhos dos arquivos salvos e flag para indicar o modelo campeão.
    
    Attributes:
        id (int): Chave primária
        ticker (str): Símbolo do ativo
        model_id (int): ID do tipo de modelo (1-30 para modelos avançados)
        model_name (str): Nome descritivo (ex: 'LSTM_BiDirectional')
        version (str): Versão timestamp (formato: YYYYMMDD_HHMMSS)
        path_model (str): Caminho completo do arquivo .keras
        path_scaler (str): Caminho completo do arquivo .scaler
        mae (float): Mean Absolute Error
        rmse (float): Root Mean Squared Error
        mape (float): Mean Absolute Percentage Error
        r2 (float): R² Score
        accuracy (float): Acurácia direcional
        pearson_corr (float): Correlação de Pearson
        params (str): JSON com hiperparâmetros usados
        metadata (str): JSON com metadados adicionais
        is_winner (bool): Flag indicando se é o melhor modelo atual
        registered_at (datetime): Data/hora de registro
        
    Indexes:
        - ticker: Para queries por ativo
        - is_winner: Para buscar modelo campeão rapidamente
        - registered_at: Para ordenar por data
        
    Example:
        >>> model = ModelRegistry(
        ...     ticker='NVDA',
        ...     model_id=1,
        ...     model_name='LSTM_Simple',
        ...     version='20230115_143000',
        ...     path_model='models/NVDA_1_20230115_143000.keras',
        ...     path_scaler='models/NVDA_1_20230115_143000.scaler',
        ...     rmse=3.2,
        ...     is_winner=True
        ... )
    """
    __tablename__ = "model_registry"
    
    # Colunas
    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String, index=True, nullable=False)
    model_id = db.Column(db.Integer, nullable=False)
    model_name = db.Column(db.String, nullable=False)
    version = db.Column(db.String, nullable=False)
    path_model = db.Column(db.String, nullable=False)
    path_scaler = db.Column(db.String, nullable=False)
    mae = db.Column(db.Float)
    rmse = db.Column(db.Float)
    mape = db.Column(db.Float)
    r2 = db.Column(db.Float)
    accuracy = db.Column(db.Float)
    pearson_corr = db.Column(db.Float)
    params = db.Column(db.Text)
    model_metadata = db.Column(db.Text)  # JSON com metadados adicionais
    is_winner = db.Column(db.Boolean, default=False, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)  # Adicionado
    registered_at = db.Column(db.DateTime, server_default=db.func.now(), index=True)
    
    @property
    def is_recent(self) -> bool:
        """
        Verifica se o modelo foi treinado recentemente (últimas 24h).
        
        Returns:
            True se registrado nas últimas 24 horas
            
        Example:
            >>> model.is_recent
            True  # Treinado hoje
        """
        if self.registered_at:
            delta = datetime.utcnow() - self.registered_at
            return delta.total_seconds() < 86400  # 24 horas
        return False
    
    @property
    def performance_score(self) -> Optional[float]:
        """
        Calcula score normalizado de performance (0-100).
        
        Combina múltiplas métricas em um score único:
            - MAPE (peso 40%)
            - Pearson correlation (peso 30%)
            - Accuracy (peso 30%)
            
        Returns:
            Score de 0 a 100 ou None se métricas indisponíveis
            
        Example:
            >>> model.performance_score
            78.5  # Bom modelo
        """
        if self.mape is None:
            return None
        
        # Componente MAPE (invertido: menor é melhor)
        # MAPE < 2% = 100 pontos, MAPE > 10% = 0 pontos
        mape_score = max(0, min(100, 100 - (self.mape * 10)))
        
        # Componente Pearson (0.0 a 1.0 → 0 a 100)
        pearson_score = (self.pearson_corr * 100) if self.pearson_corr else 50
        
        # Componente Accuracy (já está em %)
        accuracy_score = self.accuracy if self.accuracy else 50
        
        # Média ponderada
        total_score = (
            mape_score * 0.4 +
            pearson_score * 0.3 +
            accuracy_score * 0.3
        )
        
        return round(total_score, 2)
    
    @property
    def quality_grade(self) -> str:
        """
        Classifica qualidade do modelo baseado no performance_score.
        
        Escala:
            - A (Excelente): score ≥ 80
            - B (Bom): 60 ≤ score < 80
            - C (Regular): 40 ≤ score < 60
            - D (Ruim): score < 40
            
        Returns:
            Grade de A a D
            
        Example:
            >>> model.quality_grade
            'A'  # Excelente
        """
        score = self.performance_score
        if score is None:
            return 'N/A'
        
        if score >= 80:
            return 'A'
        elif score >= 60:
            return 'B'
        elif score >= 40:
            return 'C'
        else:
            return 'D'
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário (útil para APIs)."""
        return {
            'id': self.id,
            'ticker': self.ticker,
            'model_id': self.model_id,
            'model_name': self.model_name,
            'version': self.version,
            'mae': self.mae,
            'rmse': self.rmse,
            'mape': self.mape,
            'r2': self.r2,
            'accuracy': self.accuracy,
            'pearson_corr': self.pearson_corr,
            'is_winner': self.is_winner,
            'is_recent': self.is_recent,
            'performance_score': self.performance_score,
            'quality_grade': self.quality_grade,
            'registered_at': self.registered_at.isoformat() if self.registered_at else None
        }
    
    def __repr__(self) -> str:
        """Representação legível do objeto."""
        winner_flag = " [WINNER]" if self.is_winner else ""
        return (
            f"<ModelRegistry(ticker='{self.ticker}', "
            f"model='{self.model_name}', "
            f"version='{self.version}', "
            f"rmse={self.rmse:.2f if self.rmse else 'N/A'}){winner_flag}>"
        )
