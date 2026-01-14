"""
🌐 WEB ROUTES - Rotas de Páginas HTML
=====================================

Blueprint Flask para servir páginas HTML da aplicação.
Sistema dedicado ao ticker NVDA (NVIDIA).

Responsabilidades:
    - Renderizar páginas principais (home, simulação, logs)
    - Renderizar páginas de treino (padrão, avançado, customizado)
    - Renderizar página de monitoramento (métricas Prometheus)
    - Buscar dados do banco para exibição

Padrões aplicados:
    - Type hints completos
    - Docstrings estilo Google
    - Single Responsibility Principle (SRP)
    - Tratamento de erros com fallbacks
"""

from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, date
from flask import Blueprint, render_template, Response
import logging

from ..models import ResultadoMetricas, PrecoDiario
from .. import db

# Configuração
logger = logging.getLogger(__name__)

# Constantes
DEFAULT_TICKER = 'NVDA'  # Sistema dedicado ao ticker NVDA
MAX_RECENT_PRICES = 30   # Número de preços recentes para exibir na home

# Blueprint
web_bp = Blueprint('web', __name__)


# =============================================================================
# FUNÇÕES AUXILIARES (Lógica de Negócio Extraída - SRP)
# =============================================================================

def _get_latest_model_result(ticker: str) -> Optional[ResultadoMetricas]:
    """
    Busca o resultado do modelo mais recente para um ticker.
    
    Args:
        ticker: Símbolo do ativo (ex: 'NVDA')
        
    Returns:
        ResultadoMetricas mais recente ou None se não existir
        
    Example:
        >>> latest = _get_latest_model_result('NVDA')
        >>> if latest:
        ...     print(f"RMSE: {latest.rmse}")
    """
    try:
        return (
            ResultadoMetricas.query
            .filter_by(ticker=ticker)
            .order_by(ResultadoMetricas.trained_at.desc())
            .first()
        )
    except Exception as e:
        logger.error(f"Erro ao buscar último modelo para {ticker}: {e}")
        return None


def _get_recent_prices(ticker: str, limit: int = MAX_RECENT_PRICES) -> List[PrecoDiario]:
    """
    Busca os preços mais recentes para um ticker.
    
    Args:
        ticker: Símbolo do ativo (ex: 'NVDA')
        limit: Número máximo de registros a retornar
        
    Returns:
        Lista de PrecoDiario ordenada do mais antigo para o mais recente
        (ordem cronológica invertida para facilitar plotagem)
        
    Example:
        >>> prices = _get_recent_prices('NVDA', limit=30)
        >>> print(f"Preços de {prices[0].date} até {prices[-1].date}")
    """
    try:
        prices = (
            PrecoDiario.query
            .filter_by(ticker=ticker)
            .order_by(PrecoDiario.date.desc())
            .limit(limit)
            .all()
        )
        # Inverte para ordem cronológica (mais antigo → mais recente)
        return list(reversed(prices))
    except Exception as e:
        logger.error(f"Erro ao buscar preços para {ticker}: {e}")
        return []


def _get_last_available_date(ticker: str) -> Optional[date]:
    """
    Busca a data mais recente com dados disponíveis no banco.
    
    Args:
        ticker: Símbolo do ativo (ex: 'NVDA')
        
    Returns:
        Data mais recente ou None se não houver dados
        
    Example:
        >>> last_date = _get_last_available_date('NVDA')
        >>> print(f"Dados disponíveis até: {last_date}")
    """
    try:
        last_record = (
            PrecoDiario.query
            .filter_by(ticker=ticker)
            .order_by(PrecoDiario.date.desc())
            .first()
        )
        return last_record.date if last_record else None
    except Exception as e:
        logger.error(f"Erro ao buscar última data para {ticker}: {e}")
        return None


def _prepare_home_context(ticker: str) -> Dict[str, Any]:
    """
    Prepara contexto completo para renderização da página home.
    
    Aplica SRP: função dedicada a preparar dados para a view.
    
    Args:
        ticker: Símbolo do ativo (ex: 'NVDA')
        
    Returns:
        Dicionário com dados necessários para o template:
            - ticker: str
            - latest: ResultadoMetricas | None
            - prices: List[PrecoDiario]
            - last_date: date | None
            
    Example:
        >>> context = _prepare_home_context('NVDA')
        >>> print(f"Ticker: {context['ticker']}")
        >>> print(f"Preços disponíveis: {len(context['prices'])}")
    """
    return {
        'ticker': ticker,
        'latest': _get_latest_model_result(ticker),
        'prices': _get_recent_prices(ticker),
        'last_date': _get_last_available_date(ticker)
    }


# =============================================================================
# ROTAS - Página Principal
# =============================================================================

@web_bp.get('/')
def home() -> str:
    """
    Renderiza página principal (dashboard) do sistema.
    
    Exibe:
        - Gráfico de preços históricos recentes (30 dias)
        - Estatísticas do último modelo treinado (RMSE, MAE, MAPE)
        - Data da última atualização dos dados
        - Ticker fixo: NVDA (NVIDIA)
        
    Returns:
        HTML renderizado com template 'index.html'
        
    Template Context:
        ticker (str): Símbolo do ativo ('NVDA')
        latest (ResultadoMetricas | None): Último modelo treinado
        prices (List[PrecoDiario]): 30 preços mais recentes
        last_date (date | None): Data mais recente com dados
        
    Example:
        GET http://127.0.0.1:5000/
        → Renderiza dashboard com dados NVDA
    """
    logger.info(f"Renderizando página home para ticker {DEFAULT_TICKER}")
    
    try:
        context = _prepare_home_context(DEFAULT_TICKER)
        return render_template('index.html', **context)
    except Exception as e:
        logger.error(f"Erro ao renderizar home: {e}", exc_info=True)
        # Fallback: renderiza com dados vazios
        return render_template(
            'index.html',
            ticker=DEFAULT_TICKER,
            latest=None,
            prices=[],
            last_date=None
        )


# =============================================================================
# ROTAS - Páginas de Funcionalidades
# =============================================================================

@web_bp.get('/simulate')
def simulate() -> str:
    """
    Renderiza página de simulação de investimento.
    
    Permite ao usuário:
        - Simular estratégias de compra/venda
        - Testar diferentes cenários de investimento
        - Visualizar retornos hipotéticos
        
    Returns:
        HTML renderizado com template 'simulate.html'
        
    Example:
        GET http://127.0.0.1:5000/simulate
        → Renderiza página de simulação
    """
    logger.info("Renderizando página de simulação")
    return render_template('simulate.html')


@web_bp.get('/logs')
def logs() -> str:
    """
    Renderiza página de visualização de logs do sistema.
    
    Exibe:
        - Logs de treinos
        - Logs de erros
        - Histórico de operações
        
    Returns:
        HTML renderizado com template 'logs.html'
        
    Example:
        GET http://127.0.0.1:5000/logs
        → Renderiza página de logs
    """
    logger.info("Renderizando página de logs")
    return render_template('logs.html')


@web_bp.get('/monitoring')
def monitoring() -> str:
    """
    Renderiza página de monitoramento com métricas Prometheus.
    
    Exibe:
        - Métricas de performance (CPU, RAM)
        - Contadores de requisições
        - Duração de operações
        - Gráficos Grafana integrados
        
    Returns:
        HTML renderizado com template 'monitoring.html'
        
    Example:
        GET http://127.0.0.1:5000/monitoring
        → Renderiza dashboard de monitoramento
    """
    logger.info("Renderizando página de monitoramento")
    return render_template('monitoring.html')


# =============================================================================
# ROTAS - Páginas de Treino de Modelos
# =============================================================================

@web_bp.get('/custom-model')
def custom_model() -> str:
    """
    Renderiza página de criação de modelo customizado.
    
    Permite ao usuário:
        - Configurar arquitetura personalizada (camadas, neurônios)
        - Ajustar hiperparâmetros (learning rate, dropout, epochs)
        - Testar configurações experimentais
        - Salvar modelos customizados
        
    Returns:
        HTML renderizado com template 'custom_model.html'
        
    Example:
        GET http://127.0.0.1:5000/custom-model
        → Renderiza editor de modelos customizados
    """
    logger.info("Renderizando página de modelo customizado")
    return render_template('custom_model.html')


@web_bp.get('/advanced-training')
def advanced_training() -> str:
    """
    Renderiza página de treino avançado com múltiplos modelos.
    
    Funcionalidades:
        - Testa 30 arquiteturas LSTM/GRU diferentes
        - Dois modos disponíveis:
            * Modo Rápido: Treina com hiperparâmetros fixos (1 epoch)
            * Modo Otimizado: Busca hiperparâmetros ótimos (Grid/Random/Bayesian)
        - Compara resultados automaticamente
        - Seleciona melhor modelo (menor RMSE)
        - Exibe 4 gráficos de análise:
            1. Previsões vs Real
            2. Scatter Plot
            3. Análise de Resíduos
            4. Histograma de Erros
        
    Returns:
        HTML renderizado com template 'advanced_training.html'
        
    Example:
        GET http://127.0.0.1:5000/advanced-training
        → Renderiza página de treino avançado
        
    Note:
        Esta página usa trainer_advanced.py que foi recentemente
        corrigido para salvar modelos no banco (bug fix de gráficos).
    """
    logger.info("Renderizando página de treino avançado")
    return render_template('advanced_training.html')

