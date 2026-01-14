"""
Prometheus monitoring integration for Flask application.

This module provides comprehensive Prometheus metrics for:
- HTTP request metrics (count, latency, in-progress)
- ML model inference timing
- Training/retraining duration and resource usage
- System resource consumption (RAM, CPU)

All metrics are exposed at /metrics endpoint in Prometheus format.

Example:
    >>> from app.monitoring import HTTP_REQUESTS, HTTP_LATENCY
    >>> HTTP_REQUESTS.labels('GET', '/api/predict', '200').inc()
    >>> HTTP_LATENCY.labels('GET', '/api/predict').observe(0.245)
"""

import time
from flask import Response, Flask
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST, REGISTRY
from typing import Any


def _get_or_create(
    metric_cls: type, 
    name: str, 
    documentation: str, 
    labelnames: tuple = (), 
    **kwargs: Any
) -> Any:
    """
    Get existing metric or create new one (idempotent).
    
    Prevents "Duplicated timeseries" errors when reloading modules
    by reusing existing metrics from the registry.
    
    Args:
        metric_cls: Prometheus metric class (Counter, Gauge, Histogram)
        name: Metric name (snake_case)
        documentation: Human-readable description
        labelnames: Tuple of label names for dimensions
        **kwargs: Additional metric-specific arguments (e.g., buckets)
        
    Returns:
        Prometheus metric instance
        
    Example:
        >>> counter = _get_or_create(Counter, "requests", "Total requests", ["method"])
        >>> counter.labels(method="GET").inc()
    """
    # Check if metric already exists in registry
    existing = REGISTRY._names_to_collectors.get(name)
    if existing is not None:
        return existing
    return metric_cls(name, documentation, labelnames, **kwargs)


# HTTP Request Metrics
HTTP_REQUESTS = _get_or_create(
    Counter, "http_requests", "Total de requisições HTTP",
    ["method", "endpoint", "http_status"]
)

HTTP_LATENCY = _get_or_create(
    Histogram, "http_request_duration_seconds", "Duração das requisições HTTP (s)",
    ["method", "endpoint"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10)
)

INPROGRESS = _get_or_create(
    Gauge, "http_requests_in_progress", "Requisições em andamento",
    ["endpoint"]
)

# ML Model Metrics
INFERENCE_LATENCY = _get_or_create(
    Histogram, "inference_seconds", "Tempo de inferência do modelo (s)",
    ["ticker", "version"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5)
)

# Training Metrics
RETRAIN_COUNT = _get_or_create(
    Counter, "retrain_total", "Qtd de retreinagens", ["ticker", "mode"]
)

RETRAIN_DURATION = _get_or_create(
    Histogram, "retrain_duration_seconds", "Tempo de treino/retreino (s)",
    ["ticker", "mode"],
    buckets=(5, 10, 20, 30, 45, 60, 90, 120, 180, 300, 600, 1200)
)

# Resource Usage During Training
TRAIN_RAM_USAGE = _get_or_create(
    Gauge, "train_ram_usage_mb", "Uso de RAM durante treino (MB)",
    ["ticker", "mode"]
)

TRAIN_CPU_PERCENT = _get_or_create(
    Gauge, "train_cpu_percent", "Uso de CPU durante treino (%)",
    ["ticker", "mode"]
)


def setup_monitoring(app: Flask) -> None:
    """
    Configure comprehensive Prometheus monitoring for Flask app.
    
    Sets up:
    - /metrics endpoint for Prometheus scraping
    - Automatic request timing middleware
    - Request counting with labels (method, path, status)
    - In-progress request tracking
    
    Args:
        app: Flask application instance
        
    Example:
        >>> from flask import Flask
        >>> app = Flask(__name__)
        >>> setup_monitoring(app)
        >>> # Metrics now available at http://localhost:5000/metrics
    """
    @app.get("/metrics")
    def _metrics() -> Response:
        """Prometheus metrics endpoint."""
        return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)
    
    @app.before_request
    def _start() -> None:
        """Start timing and increment in-progress counter."""
        try:
            from flask import request
            request._start_time = time.perf_counter()
            INPROGRESS.labels(endpoint=request.path).inc()
        except Exception:
            pass

    @app.after_request
    def _end(response: Response) -> Response:
        """
        Record request metrics after completion.
        
        Measures:
        - Request duration (histogram)
        - Request count (counter with status code)
        - Decrements in-progress gauge
        """
        try:
            from flask import request
            dt = time.perf_counter() - getattr(request, "_start_time", time.perf_counter())
            HTTP_LATENCY.labels(request.method, request.path).observe(dt)
            HTTP_REQUESTS.labels(request.method, request.path, str(response.status_code)).inc()
            INPROGRESS.labels(endpoint=request.path).dec()
        except Exception:
            pass
        return response
