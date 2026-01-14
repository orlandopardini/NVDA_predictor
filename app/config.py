"""
Configuration Management System
Centraliza todas as configurações do sistema usando dataclasses e variáveis de ambiente.
Suporta múltiplos ambientes: Development, Testing, Production.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class BaseConfig:
    """Configuração base compartilhada por todos os ambientes."""
    
    # Flask Core
    SECRET_KEY: str = field(default_factory=lambda: os.getenv('SECRET_KEY', 'dev-secret-change-in-production'))
    FLASK_APP: str = 'wsgi.py'
    
    # Database
    SQLALCHEMY_TRACK_MODIFICATIONS: bool = False
    SQLALCHEMY_ENGINE_OPTIONS: dict = field(default_factory=lambda: {
        "connect_args": {"timeout": 30},
        "pool_pre_ping": True,
        "pool_recycle": 3600,
    })
    
    # Swagger/API Documentation
    SWAGGER: dict = field(default_factory=lambda: {
        'title': 'Stock LSTM API',
        'uiversion': 3,
        'description': 'API para previsão de preços de ações usando LSTM',
        'version': '2.0',
        'termsOfService': '',
    })
    
    # ML Model Settings
    MODELS_DIR: Path = field(default_factory=lambda: Path('models'))
    DEFAULT_LOOKBACK: int = 60
    DEFAULT_HORIZON: int = 1
    MAX_EPOCHS: int = 100
    BATCH_SIZE: int = 32
    VALIDATION_SPLIT: float = 0.2
    
    # Training Settings
    ENABLE_EARLY_STOPPING: bool = True
    EARLY_STOPPING_PATIENCE: int = 10
    REDUCE_LR_PATIENCE: int = 5
    MIN_LEARNING_RATE: float = 1e-7
    
    # Data Settings
    MAX_TICKER_LENGTH: int = 16
    DEFAULT_TICKER: str = 'AAPL'
    YFINANCE_TIMEOUT: int = 30
    
    # Monitoring & Metrics
    ENABLE_PROMETHEUS: bool = True
    PROMETHEUS_PORT: Optional[int] = None
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_DEFAULT: str = "100 per hour"
    RATE_LIMIT_STORAGE_URL: Optional[str] = None
    
    # Logging
    LOG_LEVEL: str = 'INFO'
    LOG_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE: Optional[str] = None
    
    # Cache
    CACHE_TYPE: str = 'simple'
    CACHE_DEFAULT_TIMEOUT: int = 300
    
    # Security
    MAX_CONTENT_LENGTH: int = 16 * 1024 * 1024  # 16MB
    CORS_ENABLED: bool = False
    CORS_ORIGINS: str = "*"
    
    def __post_init__(self):
        """Validações e inicializações pós-criação."""
        # Garantir que MODELS_DIR existe
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Validar valores numéricos
        if self.DEFAULT_LOOKBACK < 1:
            raise ValueError("DEFAULT_LOOKBACK deve ser >= 1")
        if self.DEFAULT_HORIZON < 1:
            raise ValueError("DEFAULT_HORIZON deve ser >= 1")
        if not (0 < self.VALIDATION_SPLIT < 1):
            raise ValueError("VALIDATION_SPLIT deve estar entre 0 e 1")
    
    @property
    def DATABASE_URI(self) -> str:
        """Retorna a URI do banco de dados."""
        raise NotImplementedError("Deve ser implementado nas subclasses")


@dataclass
class DevelopmentConfig(BaseConfig):
    """Configuração para ambiente de desenvolvimento."""
    
    DEBUG: bool = True
    TESTING: bool = False
    LOG_LEVEL: str = 'DEBUG'
    EXPLAIN_TEMPLATE_LOADING: bool = False
    
    # Database - SQLite local
    SQLALCHEMY_ECHO: bool = False  # Set True para debug SQL
    
    # Cache desabilitado em dev
    CACHE_TYPE: str = 'null'
    
    # Rate limiting mais permissivo
    RATE_LIMIT_DEFAULT: str = "200 per hour"
    
    @property
    def DATABASE_URI(self) -> str:
        """Database URI para desenvolvimento."""
        instance_path = Path('instance')
        instance_path.mkdir(parents=True, exist_ok=True)
        db_path = instance_path / 'app.db'
        # Converter para caminho absoluto e usar forward slashes (POSIX)
        abs_path = db_path.absolute().as_posix()
        return os.getenv('DATABASE_URL', f'sqlite:///{abs_path}')


@dataclass
class TestingConfig(BaseConfig):
    """Configuração para ambiente de testes."""
    
    DEBUG: bool = False
    TESTING: bool = True
    LOG_LEVEL: str = 'WARNING'
    
    # Database in-memory para testes
    SQLALCHEMY_ECHO: bool = False
    
    # Desabilitar features externas em testes
    ENABLE_PROMETHEUS: bool = False
    RATE_LIMIT_ENABLED: bool = False
    YFINANCE_TIMEOUT: int = 5
    
    # Cache desabilitado
    CACHE_TYPE: str = 'null'
    
    # Modelos menores para testes rápidos
    MAX_EPOCHS: int = 5
    EARLY_STOPPING_PATIENCE: int = 2
    
    @property
    def DATABASE_URI(self) -> str:
        """Database URI para testes (in-memory)."""
        return os.getenv('TEST_DATABASE_URL', 'sqlite:///:memory:')


@dataclass
class ProductionConfig(BaseConfig):
    """Configuração para ambiente de produção."""
    
    DEBUG: bool = False
    TESTING: bool = False
    LOG_LEVEL: str = field(default_factory=lambda: os.getenv('LOG_LEVEL', 'INFO'))
    
    # Security
    SECRET_KEY: str = field(default_factory=lambda: os.getenv('SECRET_KEY'))
    SESSION_COOKIE_SECURE: bool = True
    SESSION_COOKIE_HTTPONLY: bool = True
    SESSION_COOKIE_SAMESITE: str = 'Lax'
    
    # Database - PostgreSQL ou SQLite otimizado
    SQLALCHEMY_ECHO: bool = False
    SQLALCHEMY_ENGINE_OPTIONS: dict = field(default_factory=lambda: {
        "connect_args": {"timeout": 30},
        "pool_size": 10,
        "max_overflow": 20,
        "pool_pre_ping": True,
        "pool_recycle": 3600,
    })
    
    # Cache ativo
    CACHE_TYPE: str = 'simple'  # Ou 'redis' se disponível
    
    # Rate limiting mais restritivo
    RATE_LIMIT_DEFAULT: str = "60 per hour"
    
    # Logging para arquivo
    LOG_FILE: str = field(default_factory=lambda: os.getenv('LOG_FILE', 'logs/app.log'))
    
    # CORS configurável
    CORS_ENABLED: bool = field(default_factory=lambda: os.getenv('CORS_ENABLED', 'False').lower() == 'true')
    CORS_ORIGINS: str = field(default_factory=lambda: os.getenv('CORS_ORIGINS', ''))
    
    def __post_init__(self):
        """Validações específicas de produção."""
        super().__post_init__()
        
        # Validar SECRET_KEY em produção
        if not self.SECRET_KEY or self.SECRET_KEY == 'dev-secret-change-in-production':
            raise ValueError(
                "SECRET_KEY deve ser definida em produção via variável de ambiente!"
            )
        
        # Criar diretório de logs
        if self.LOG_FILE:
            log_dir = Path(self.LOG_FILE).parent
            log_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def DATABASE_URI(self) -> str:
        """Database URI para produção."""
        db_url = os.getenv('DATABASE_URL')
        if not db_url:
            # Fallback para SQLite se DATABASE_URL não estiver definida
            instance_path = Path('instance')
            instance_path.mkdir(parents=True, exist_ok=True)
            db_path = instance_path / 'app_production.db'
            return f'sqlite:///{db_path}'
        
        # Fix para Heroku: postgres:// -> postgresql://
        if db_url.startswith('postgres://'):
            db_url = db_url.replace('postgres://', 'postgresql://', 1)
        
        return db_url


# Mapeamento de ambientes
_config_map = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'dev': DevelopmentConfig,
    'test': TestingConfig,
    'prod': ProductionConfig,
}


def get_config(env: Optional[str] = None) -> BaseConfig:
    """
    Retorna a configuração apropriada baseada no ambiente.
    
    Args:
        env: Nome do ambiente ('development', 'testing', 'production')
             Se None, usa FLASK_ENV ou default 'development'
    
    Returns:
        Instância da configuração apropriada
    
    Raises:
        ValueError: Se o ambiente especificado não for válido
    
    Examples:
        >>> config = get_config('production')
        >>> print(config.DEBUG)
        False
        
        >>> config = get_config()  # Usa FLASK_ENV
        >>> print(config.DATABASE_URI)
    """
    if env is None:
        env = os.getenv('FLASK_ENV', 'development').lower()
    
    env = env.lower()
    
    if env not in _config_map:
        raise ValueError(
            f"Ambiente inválido: '{env}'. "
            f"Opções válidas: {', '.join(_config_map.keys())}"
        )
    
    config_class = _config_map[env]
    return config_class()


# Instância global da configuração atual
config = get_config()


# Exportar para fácil importação
__all__ = [
    'BaseConfig',
    'DevelopmentConfig',
    'TestingConfig',
    'ProductionConfig',
    'get_config',
    'config',
]
