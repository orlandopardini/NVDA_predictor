"""
Logging Configuration System
Sistema centralizado de logging com suporte a múltiplos handlers, formatters e níveis.
Configuração estruturada para diferentes ambientes e componentes do sistema.
"""
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Dict, Any, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    from .schemas import LoggingConfig


# Cores para terminal (opcional)
class LogColors:
    """Códigos ANSI para colorir logs no terminal."""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    # Níveis
    DEBUG = '\033[36m'      # Cyan
    INFO = '\033[32m'       # Green
    WARNING = '\033[33m'    # Yellow
    ERROR = '\033[31m'      # Red
    CRITICAL = '\033[35m'   # Magenta
    
    # Componentes
    HTTP = '\033[94m'       # Blue
    DB = '\033[96m'         # Cyan
    ML = '\033[92m'         # Bright Green


class ColoredFormatter(logging.Formatter):
    """Formatter que adiciona cores aos logs no terminal."""
    
    LEVEL_COLORS = {
        logging.DEBUG: LogColors.DEBUG,
        logging.INFO: LogColors.INFO,
        logging.WARNING: LogColors.WARNING,
        logging.ERROR: LogColors.ERROR,
        logging.CRITICAL: LogColors.CRITICAL,
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Formata o log com cores."""
        # Adicionar cor ao nível
        levelname = record.levelname
        if record.levelno in self.LEVEL_COLORS:
            levelname_color = (
                f"{self.LEVEL_COLORS[record.levelno]}"
                f"{levelname:<8}"
                f"{LogColors.RESET}"
            )
            record.levelname = levelname_color
        
        # Formatar
        result = super().format(record)
        
        # Resetar levelname original
        record.levelname = levelname
        
        return result


class StructuredFormatter(logging.Formatter):
    """Formatter que adiciona campos estruturados aos logs."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Adiciona campos extras ao log."""
        # Adicionar informações extras
        record.module_path = f"{record.module}.{record.funcName}"
        
        # Adicionar context se disponível
        if hasattr(record, 'user_id'):
            record.user_context = f"user={record.user_id}"
        else:
            record.user_context = ""
        
        if hasattr(record, 'request_id'):
            record.request_context = f"req={record.request_id}"
        else:
            record.request_context = ""
        
        return super().format(record)


def setup_logging(config: 'LoggingConfig') -> logging.Logger:
    """
    Configura o sistema de logging da aplicação usando Parameter Object Pattern.
    
    Args:
        config: Objeto LoggingConfig contendo todas as configurações de logging
    
    Returns:
        Logger raiz configurado
    
    Examples:
        >>> from app.schemas import LoggingConfig
        >>> config = LoggingConfig(app_name='my-app', log_level='DEBUG', log_file='logs/app.log')
        >>> logger = setup_logging(config)
        >>> logger.info("Application started")
    """
    # Obter logger raiz
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.log_level.upper()))
    
    # Limpar handlers existentes
    root_logger.handlers.clear()
    
    # Format string padrão
    log_format = config.log_format
    if log_format is None:
        log_format = (
            '%(asctime)s - '
            '%(name)-20s - '
            '%(levelname)-8s - '
            '%(module)s:%(lineno)d - '
            '%(message)s'
        )
    
    # Console Handler
    if config.enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        
        if config.enable_colors and sys.stdout.isatty():
            console_formatter = ColoredFormatter(
                log_format,
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        else:
            console_formatter = StructuredFormatter(
                log_format,
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # File Handler com rotação
    if config.log_file:
        log_path = Path(config.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            config.log_file,
            maxBytes=config.max_bytes,
            backupCount=config.backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        
        # Formato sem cores para arquivo
        file_formatter = StructuredFormatter(
            log_format,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Configurar loggers de bibliotecas externas
    _configure_third_party_loggers(config.log_level)
    
    # Log inicial
    app_logger = logging.getLogger(config.app_name)
    app_logger.info(f"Logging configurado - Nível: {config.log_level}")
    if config.log_file:
        app_logger.info(f"Log file: {config.log_file}")
    
    return root_logger


def _configure_third_party_loggers(log_level: str):
    """Configura níveis de log para bibliotecas externas."""
    # Reduzir verbosidade de bibliotecas externas
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('h5py').setLevel(logging.WARNING)
    logging.getLogger('h5py._conv').setLevel(logging.WARNING)
    
    # SQLAlchemy - sempre no modo WARNING para reduzir verbosidade
    logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
    logging.getLogger('sqlalchemy.pool').setLevel(logging.WARNING)
    logging.getLogger('sqlalchemy.dialects').setLevel(logging.WARNING)


def get_logger(name: str, **kwargs) -> logging.Logger:
    """
    Retorna um logger configurado para um módulo específico.
    
    Args:
        name: Nome do logger (geralmente __name__)
        **kwargs: Campos extras para adicionar a todos os logs
    
    Returns:
        Logger configurado
    
    Examples:
        >>> logger = get_logger(__name__, component='api')
        >>> logger.info("Request received", extra={'request_id': '123'})
    """
    logger = logging.getLogger(name)
    
    # Adicionar adapter para campos extras permanentes
    if kwargs:
        logger = logging.LoggerAdapter(logger, kwargs)
    
    return logger


class LogContext:
    """
    Context manager para adicionar campos extras temporariamente.
    
    Examples:
        >>> logger = get_logger(__name__)
        >>> with LogContext(logger, request_id='abc123'):
        ...     logger.info("Processing request")
    """
    
    def __init__(self, logger: logging.Logger, **context):
        """
        Args:
            logger: Logger a ser usado
            **context: Campos de contexto a adicionar
        """
        self.logger = logger
        self.context = context
        self.old_factory = None
    
    def __enter__(self):
        """Ativa o contexto."""
        # Salvar factory antiga
        self.old_factory = logging.getLogRecordFactory()
        
        # Criar nova factory que adiciona contexto
        context = self.context
        
        def record_factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            for key, value in context.items():
                setattr(record, key, value)
            return record
        
        logging.setLogRecordFactory(record_factory)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Desativa o contexto."""
        # Restaurar factory antiga
        if self.old_factory:
            logging.setLogRecordFactory(self.old_factory)


def log_execution_time(logger: logging.Logger, level: int = logging.INFO):
    """
    Decorator para logar tempo de execução de funções.
    
    Args:
        logger: Logger a ser usado
        level: Nível de log
    
    Examples:
        >>> logger = get_logger(__name__)
        >>> @log_execution_time(logger)
        ... def my_function():
        ...     pass
    """
    def decorator(func):
        from functools import wraps
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            func_name = f"{func.__module__}.{func.__name__}"
            
            try:
                logger.log(level, f"Iniciando {func_name}")
                result = func(*args, **kwargs)
                elapsed = (datetime.now() - start_time).total_seconds()
                logger.log(
                    level,
                    f"Concluído {func_name} em {elapsed:.2f}s"
                )
                return result
            except Exception as e:
                elapsed = (datetime.now() - start_time).total_seconds()
                logger.error(
                    f"Erro em {func_name} após {elapsed:.2f}s: {e}",
                    exc_info=True
                )
                raise
        
        return wrapper
    return decorator


def log_function_call(logger: logging.Logger, level: int = logging.DEBUG):
    """
    Decorator para logar chamadas de função com argumentos.
    
    Args:
        logger: Logger a ser usado
        level: Nível de log
    
    Examples:
        >>> logger = get_logger(__name__)
        >>> @log_function_call(logger)
        ... def add(a, b):
        ...     return a + b
    """
    def decorator(func):
        from functools import wraps
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__name__}"
            
            # Logar argumentos (cuidado com dados sensíveis!)
            args_repr = [repr(a) for a in args[:3]]  # Limitar a 3 args
            if len(args) > 3:
                args_repr.append(f"... +{len(args)-3} more")
            
            kwargs_repr = [f"{k}={v!r}" for k, v in list(kwargs.items())[:3]]
            if len(kwargs) > 3:
                kwargs_repr.append(f"... +{len(kwargs)-3} more")
            
            signature = ", ".join(args_repr + kwargs_repr)
            logger.log(level, f"Chamando {func_name}({signature})")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


# Exportar componentes principais
__all__ = [
    'setup_logging',
    'get_logger',
    'LogContext',
    'log_execution_time',
    'log_function_call',
    'LogColors',
    'ColoredFormatter',
    'StructuredFormatter',
]
