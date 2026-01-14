"""
Simple Prometheus metrics endpoint for application monitoring.

Exposes system and process-level metrics (CPU, Memory) in Prometheus format
for monitoring and alerting. Lightweight alternative to full APM solutions.
"""

import os
import psutil
from flask import Response
from prometheus_client import Gauge, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST

# Separate registry to avoid conflicts with other prometheus instrumentation
REGISTRY = CollectorRegistry()

# Metrics definitions
CPU_PERCENT = Gauge("app_cpu_percent", "CPU total (%) do sistema", registry=REGISTRY)
PROC_CPU_PERCENT = Gauge("app_process_cpu_percent", "CPU (%) do processo Flask", registry=REGISTRY)
MEM_RSS_BYTES = Gauge("app_memory_rss_bytes", "RAM (RSS) do processo em bytes", registry=REGISTRY)
MEM_VMS_BYTES = Gauge("app_memory_vms_bytes", "Memória virtual (VMS) do processo em bytes", registry=REGISTRY)

# Initialize process handle and baseline CPU measurement
_ps = psutil.Process(os.getpid())
_ps.cpu_percent(None)  # First call establishes baseline for delta measurements


def _collect_now() -> None:
    """
    Collect current system and process metrics.
    
    Updates all Prometheus gauges with current values for:
    - System CPU usage (%)
    - Process CPU usage (%)
    - Process RSS memory (bytes)
    - Process VMS memory (bytes)
    """
    CPU_PERCENT.set(psutil.cpu_percent(interval=0.0))
    PROC_CPU_PERCENT.set(_ps.cpu_percent(interval=None))
    mi = _ps.memory_info()
    MEM_RSS_BYTES.set(mi.rss)
    MEM_VMS_BYTES.set(mi.vms)


def metrics_endpoint() -> Response:
    """
    Flask endpoint that returns Prometheus metrics.
    
    Returns:
        Response: Prometheus-formatted metrics with appropriate content type
        
    Example:
        >>> from flask import Flask
        >>> app = Flask(__name__)
        >>> app.route('/metrics')(metrics_endpoint)
    """
    _collect_now()
    data = generate_latest(REGISTRY)
    return Response(data, mimetype=CONTENT_TYPE_LATEST)
