"""
Custom Exceptions System
Define exceções específicas do domínio da aplicação para melhor tratamento de erros.
Hierarquia de exceções organizada por contexto (API, Database, ML, etc).
"""
from typing import Optional, Dict, Any
from http import HTTPStatus


class StockLSTMException(Exception):
    """
    Exceção base para todas as exceções customizadas da aplicação.
    
    Attributes:
        message: Mensagem de erro descritiva
        details: Detalhes adicionais do erro
        status_code: Código HTTP associado (para exceções de API)
    """
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        status_code: int = HTTPStatus.INTERNAL_SERVER_ERROR
    ):
        """
        Args:
            message: Mensagem de erro
            details: Dicionário com detalhes adicionais
            status_code: Código de status HTTP
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.status_code = status_code
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converte a exceção para dicionário (útil para JSON responses).
        
        Returns:
            Dicionário com informações do erro
        """
        return {
            'error': self.__class__.__name__,
            'message': self.message,
            'details': self.details,
            'status_code': self.status_code
        }


# ==================== API Exceptions ====================

class APIException(StockLSTMException):
    """Exceção base para erros relacionados à API."""
    pass


class ValidationError(APIException):
    """Erro de validação de dados de entrada."""
    
    def __init__(self, message: str, field: Optional[str] = None, **kwargs):
        """
        Args:
            message: Mensagem de erro
            field: Campo que falhou na validação
            **kwargs: Argumentos adicionais
        """
        details = kwargs.pop('details', {})
        if field:
            details['field'] = field
        
        super().__init__(
            message=message,
            details=details,
            status_code=HTTPStatus.BAD_REQUEST
        )


class ResourceNotFoundError(APIException):
    """Recurso solicitado não foi encontrado."""
    
    def __init__(self, resource_type: str, resource_id: Any, **kwargs):
        """
        Args:
            resource_type: Tipo do recurso (ex: 'Ticker', 'Model', 'User')
            resource_id: Identificador do recurso
            **kwargs: Argumentos adicionais
        """
        message = f"{resource_type} não encontrado: {resource_id}"
        details = {
            'resource_type': resource_type,
            'resource_id': str(resource_id)
        }
        details.update(kwargs.pop('details', {}))
        
        super().__init__(
            message=message,
            details=details,
            status_code=HTTPStatus.NOT_FOUND
        )


class RateLimitExceededError(APIException):
    """Limite de taxa de requisições excedido."""
    
    def __init__(self, limit: str, retry_after: Optional[int] = None):
        """
        Args:
            limit: Descrição do limite (ex: "100 per hour")
            retry_after: Segundos até poder tentar novamente
        """
        message = f"Limite de requisições excedido: {limit}"
        details = {'limit': limit}
        if retry_after:
            details['retry_after'] = retry_after
        
        super().__init__(
            message=message,
            details=details,
            status_code=HTTPStatus.TOO_MANY_REQUESTS
        )


class AuthenticationError(APIException):
    """Erro de autenticação."""
    
    def __init__(self, message: str = "Autenticação necessária", **kwargs):
        super().__init__(
            message=message,
            status_code=HTTPStatus.UNAUTHORIZED,
            **kwargs
        )


class AuthorizationError(APIException):
    """Erro de autorização (permissão negada)."""
    
    def __init__(self, message: str = "Permissão negada", resource: Optional[str] = None):
        details = {}
        if resource:
            details['resource'] = resource
        
        super().__init__(
            message=message,
            details=details,
            status_code=HTTPStatus.FORBIDDEN
        )


# ==================== Database Exceptions ====================

class DatabaseException(StockLSTMException):
    """Exceção base para erros de banco de dados."""
    pass


class DatabaseConnectionError(DatabaseException):
    """Erro ao conectar ao banco de dados."""
    
    def __init__(self, message: str = "Erro ao conectar ao banco de dados", **kwargs):
        super().__init__(
            message=message,
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            **kwargs
        )


class DatabaseLockError(DatabaseException):
    """Banco de dados está bloqueado (comum em SQLite)."""
    
    def __init__(self, operation: Optional[str] = None, retries: int = 0):
        message = "Banco de dados temporariamente bloqueado"
        if operation:
            message += f" durante operação: {operation}"
        
        details = {'retries': retries}
        
        super().__init__(
            message=message,
            details=details,
            status_code=HTTPStatus.SERVICE_UNAVAILABLE
        )


class IntegrityError(DatabaseException):
    """Violação de integridade (constraint violation)."""
    
    def __init__(self, message: str, constraint: Optional[str] = None):
        details = {}
        if constraint:
            details['constraint'] = constraint
        
        super().__init__(
            message=message,
            details=details,
            status_code=HTTPStatus.CONFLICT
        )


class RecordNotFoundError(DatabaseException):
    """Registro não encontrado no banco de dados."""
    
    def __init__(self, table: str, conditions: Dict[str, Any]):
        message = f"Registro não encontrado em {table}"
        details = {
            'table': table,
            'conditions': conditions
        }
        
        super().__init__(
            message=message,
            details=details,
            status_code=HTTPStatus.NOT_FOUND
        )


# ==================== ML/Model Exceptions ====================

class MLException(StockLSTMException):
    """Exceção base para erros relacionados a Machine Learning."""
    pass


class ModelNotFoundError(MLException):
    """Modelo não encontrado."""
    
    def __init__(self, model_name: str, ticker: Optional[str] = None):
        message = f"Modelo não encontrado: {model_name}"
        details = {'model_name': model_name}
        if ticker:
            message += f" para ticker {ticker}"
            details['ticker'] = ticker
        
        super().__init__(
            message=message,
            details=details,
            status_code=HTTPStatus.NOT_FOUND
        )


class ModelLoadError(MLException):
    """Erro ao carregar modelo."""
    
    def __init__(self, model_path: str, reason: Optional[str] = None):
        message = f"Erro ao carregar modelo: {model_path}"
        if reason:
            message += f" - {reason}"
        
        details = {'model_path': model_path}
        if reason:
            details['reason'] = reason
        
        super().__init__(
            message=message,
            details=details,
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR
        )


class ModelTrainingError(MLException):
    """Erro durante treinamento do modelo."""
    
    def __init__(self, message: str, epoch: Optional[int] = None, **kwargs):
        details = kwargs.pop('details', {})
        if epoch is not None:
            details['epoch'] = epoch
        
        super().__init__(
            message=message,
            details=details,
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR
        )


class InvalidModelConfigError(MLException):
    """Configuração de modelo inválida."""
    
    def __init__(self, message: str, config_key: Optional[str] = None):
        details = {}
        if config_key:
            details['config_key'] = config_key
        
        super().__init__(
            message=message,
            details=details,
            status_code=HTTPStatus.BAD_REQUEST
        )


class InsufficientDataError(MLException):
    """Dados insuficientes para treinar ou fazer predição."""
    
    def __init__(
        self,
        required: int,
        available: int,
        data_type: str = "samples"
    ):
        message = (
            f"Dados insuficientes: requeridos {required} {data_type}, "
            f"disponíveis {available}"
        )
        details = {
            'required': required,
            'available': available,
            'data_type': data_type
        }
        
        super().__init__(
            message=message,
            details=details,
            status_code=HTTPStatus.BAD_REQUEST
        )


class PredictionError(MLException):
    """Erro durante predição."""
    
    def __init__(self, message: str, model_name: Optional[str] = None):
        details = {}
        if model_name:
            details['model_name'] = model_name
        
        super().__init__(
            message=message,
            details=details,
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR
        )


# ==================== Data Exceptions ====================

class DataException(StockLSTMException):
    """Exceção base para erros relacionados a dados."""
    pass


class DataFetchError(DataException):
    """Erro ao buscar dados de fonte externa (ex: Yahoo Finance)."""
    
    def __init__(self, ticker: str, source: str = "Yahoo Finance", reason: Optional[str] = None):
        message = f"Erro ao buscar dados de {ticker} do {source}"
        if reason:
            message += f": {reason}"
        
        details = {
            'ticker': ticker,
            'source': source
        }
        if reason:
            details['reason'] = reason
        
        super().__init__(
            message=message,
            details=details,
            status_code=HTTPStatus.BAD_GATEWAY
        )


class InvalidTickerError(DataException):
    """Ticker inválido ou não suportado."""
    
    def __init__(self, ticker: str, reason: Optional[str] = None):
        message = f"Ticker inválido: {ticker}"
        if reason:
            message += f" - {reason}"
        
        details = {'ticker': ticker}
        if reason:
            details['reason'] = reason
        
        super().__init__(
            message=message,
            details=details,
            status_code=HTTPStatus.BAD_REQUEST
        )


class DataQualityError(DataException):
    """Problemas de qualidade nos dados."""
    
    def __init__(self, message: str, issues: Optional[Dict[str, Any]] = None):
        details = issues or {}
        
        super().__init__(
            message=message,
            details=details,
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY
        )


# ==================== Configuration Exceptions ====================

class ConfigurationError(StockLSTMException):
    """Erro de configuração."""
    
    def __init__(self, message: str, config_key: Optional[str] = None):
        details = {}
        if config_key:
            details['config_key'] = config_key
        
        super().__init__(
            message=message,
            details=details,
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR
        )


# ==================== Timeout Exceptions ====================

class TimeoutError(StockLSTMException):
    """Operação excedeu tempo limite."""
    
    def __init__(self, operation: str, timeout: float):
        message = f"Timeout na operação '{operation}' após {timeout}s"
        details = {
            'operation': operation,
            'timeout': timeout
        }
        
        super().__init__(
            message=message,
            details=details,
            status_code=HTTPStatus.REQUEST_TIMEOUT
        )


# Exportar todas as exceções
__all__ = [
    # Base
    'StockLSTMException',
    
    # API
    'APIException',
    'ValidationError',
    'ResourceNotFoundError',
    'RateLimitExceededError',
    'AuthenticationError',
    'AuthorizationError',
    
    # Database
    'DatabaseException',
    'DatabaseConnectionError',
    'DatabaseLockError',
    'IntegrityError',
    'RecordNotFoundError',
    
    # ML
    'MLException',
    'ModelNotFoundError',
    'ModelLoadError',
    'ModelTrainingError',
    'InvalidModelConfigError',
    'InsufficientDataError',
    'PredictionError',
    
    # Data
    'DataException',
    'DataFetchError',
    'InvalidTickerError',
    'DataQualityError',
    
    # Configuration
    'ConfigurationError',
    
    # Timeout
    'TimeoutError',
]
