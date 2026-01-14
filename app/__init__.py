"""
Stock LSTM Flask Application Factory.

This module provides the main Flask application factory with complete
configuration for database, blueprints, monitoring, and API documentation.

The application follows the factory pattern for better testability and
supports multiple configurations (dev, test, production).

Example:
    >>> from app import create_app
    >>> app = create_app()
    >>> app.run(debug=True)
"""

from sqlalchemy import event
from sqlalchemy.engine import Engine
from flask import Flask
from flasgger import Swagger
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from pathlib import Path
import os

# Global database and migration instances
# Initialized here before importing models to avoid circular dependencies
db = SQLAlchemy()
migrate = Migrate()


def create_app() -> Flask:
    """
    Create and configure the Flask application.
    
    This factory function creates a Flask app with:
    - SQLite database with WAL mode for better concurrency
    - 6 blueprints (5 API + 1 web interface)
    - Swagger/OpenAPI documentation
    - Prometheus monitoring metrics
    - Database migrations support
    
    Returns:
        Flask: Configured Flask application instance
        
    Example:
        >>> app = create_app()
        >>> with app.app_context():
        ...     # Run database operations
        ...     pass
    """
    app = Flask(__name__, instance_relative_config=True,
                static_folder="static", static_url_path="/static")
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret')

    # Create instance folder for database
    Path(app.instance_path).mkdir(parents=True, exist_ok=True)
    
    # Database configuration (SQLite with production-ready settings)
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv(
        'DATABASE_URL',
        f"sqlite:///{os.path.join(app.instance_path, 'app.db')}"
    )
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {"connect_args": {"timeout": 30}}
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    # Swagger/OpenAPI documentation
    app.config['SWAGGER'] = {'title': 'Stock LSTM API', 'uiversion': 3}
    Swagger(app)

    # Initialize extensions
    db.init_app(app)
    migrate.init_app(app, db)

    @event.listens_for(Engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        """
        Configure SQLite for better concurrency and performance.
        
        Sets:
        - WAL mode: Write-Ahead Logging for concurrent reads/writes
        - NORMAL synchronous: Balance between safety and speed
        - 5s busy timeout: Retry on database locks
        """
        try:
            if dbapi_connection.__class__.__module__.startswith("sqlite3"):
                cur = dbapi_connection.cursor()
                cur.execute("PRAGMA journal_mode=WAL;")
                cur.execute("PRAGMA synchronous=NORMAL;")
                cur.execute("PRAGMA busy_timeout=5000;")
                cur.close()
        except Exception:
            pass

    # Import models after db is initialized (avoid circular imports)
    from . import models  # noqa

    # Register all blueprints
    # API blueprints (refactored from single api.py into 5 modules)
    from .routes.api_data import api_data_bp
    from .routes.api_train import api_train_bp
    from .routes.api_predict import api_predict_bp
    from .routes.api_models import api_models_bp
    from .routes.api_monitoring import api_monitoring_bp
    from .routes.web import web_bp
    
    # All API routes use /api prefix
    app.register_blueprint(api_data_bp, url_prefix='/api')
    app.register_blueprint(api_train_bp, url_prefix='/api')
    app.register_blueprint(api_predict_bp, url_prefix='/api')
    app.register_blueprint(api_models_bp, url_prefix='/api')
    app.register_blueprint(api_monitoring_bp, url_prefix='/api')
    
    # Web interface (no prefix)
    app.register_blueprint(web_bp)

    # Setup Prometheus monitoring (metrics endpoint at /metrics)
    from .monitoring import setup_monitoring
    setup_monitoring(app)

    # Create all database tables
    with app.app_context():
        db.create_all()
    
    return app
