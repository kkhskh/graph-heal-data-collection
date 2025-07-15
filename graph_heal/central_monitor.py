import time
import threading
import requests
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import os
from prometheus_client import Counter, Gauge, Histogram, CollectorRegistry, start_http_server

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('central_monitor')

# Create a custom registry
REGISTRY = CollectorRegistry()

# Define metrics
REQUEST_COUNT = Counter(
    'service_request_total',
    'Total number of requests',
    ['service', 'endpoint', 'method', 'status'],
    registry=REGISTRY
)

REQUEST_LATENCY = Histogram(
    'service_request_duration_seconds',
    'Request latency in seconds',
    ['service', 'endpoint'],
    registry=REGISTRY
)

SERVICE_HEALTH = Gauge(
    'service_health',
    'Service health status (1 for healthy, 0 for unhealthy)',
    ['service'],
    registry=REGISTRY
)

SERVICE_AVAILABILITY = Gauge(
    'service_availability_percentage',
    'Service availability percentage',
    ['service'],
    registry=REGISTRY
)

def parse_prometheus_metrics(metrics_text: str) -> Dict[str, float]:
    """Parse Prometheus-formatted metrics text into a dictionary of metric values."""
    metrics = {}
    for line in metrics_text.split('\n'):
        if line.startswith('#') or not line.strip():
            continue
        try:
            # Split the line into metric name and value
            parts = line.split(' ')
            if len(parts) >= 2:
                metric_name = parts[0]
                value = float(parts[-1])
                metrics[metric_name] = value
        except (ValueError, IndexError):
            continue
    return metrics

class CentralMonitor:
    """
    Centralized monitoring service using Prometheus.
    """
    def __init__(self, services_config: Optional[List[Dict[str, str]]] = None, poll_interval: int = 5):
        """
        Initialize the central monitor.
        
        Args:
            services_config: List of service configurations
            poll_interval: Polling interval in seconds
        """
        self.services = services_config or []
        self.poll_interval = poll_interval
        self.stop_event = threading.Event()
        self.monitor_thread = None
        self.service_status = {}
        self.service_metrics = {}
        self.availability_history = {s["name"]: [] for s in self.services}
        
        # Start Prometheus metrics server
        start_http_server(8000, registry=REGISTRY)
    
    def start_monitoring(self):
        """Start the monitoring thread."""
        if self.monitor_thread and self.monitor_thread.is_alive():
            logger.warning("Monitoring thread already running")
            return
        
        self.stop_event.clear()
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info("Started central monitoring")
    
    def stop_monitoring(self):
        """Stop the monitoring thread."""
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.stop_event.set()
            self.monitor_thread.join()
            logger.info("Stopped central monitoring")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while not self.stop_event.is_set():
            for service in self.services:
                try:
                    # Check metrics endpoint (which includes health status)
                    metrics_url = f"{service['url'].rstrip('/')}/{service['metrics_endpoint'].lstrip('/')}"
                    start_time = time.time()
                    metrics_response = requests.get(metrics_url, timeout=1)
                    latency = time.time() - start_time
                    
                    # Record request metrics
                    REQUEST_COUNT.labels(
                        service=service["id"],
                        endpoint="/metrics",
                        method="GET",
                        status=metrics_response.status_code
                    ).inc()
                    
                    REQUEST_LATENCY.labels(
                        service=service["id"],
                        endpoint="/metrics"
                    ).observe(latency)
                    
                    if metrics_response.status_code == 200:
                        # Service is healthy if metrics endpoint responds
                        is_healthy = True
                        SERVICE_HEALTH.labels(service=service["id"]).set(1)
                        
                        # Parse Prometheus metrics
                        metrics = parse_prometheus_metrics(metrics_response.text)
                        current_time = datetime.now()
                        metrics["timestamp"] = current_time.isoformat()
                        self.service_metrics[service["id"]] = metrics
                        
                        # Update service status
                        self.service_status[service["id"]] = {
                            "name": service["name"],
                            "health": "healthy",
                            "last_check": current_time.isoformat()
                        }
                    else:
                        # Service is unhealthy
                        is_healthy = False
                        SERVICE_HEALTH.labels(service=service["id"]).set(0)
                        
                        self.service_status[service["id"]] = {
                            "name": service["name"],
                            "health": "unhealthy",
                            "last_check": datetime.now().isoformat()
                        }
                    
                    # Update availability history
                    self.availability_history[service["name"]].append(is_healthy)
                    if len(self.availability_history[service["name"]]) > 720:  # Keep last hour
                        self.availability_history[service["name"]] = self.availability_history[service["name"]][-720:]
                    
                    # Calculate and update availability percentage
                    history = self.availability_history[service["name"]]
                    availability = (sum(history) / len(history)) * 100 if history else 0
                    SERVICE_AVAILABILITY.labels(service=service["id"]).set(availability)
                    self.service_status[service["id"]]["availability"] = availability
                    
                except requests.RequestException as e:
                    logger.warning(f"Failed to check {service['name']}: {e}")
                    SERVICE_HEALTH.labels(service=service["id"]).set(0)
                    
                    current_time = datetime.now()
                    self.service_status[service["id"]] = {
                        "name": service["name"],
                        "health": "unhealthy",
                        "last_check": current_time.isoformat(),
                        "error": str(e)
                    }
                    
                    # Update availability history
                    self.availability_history[service["name"]].append(False)
                    if len(self.availability_history[service["name"]]) > 720:
                        self.availability_history[service["name"]] = self.availability_history[service["name"]][-720:]
                    
                    # Calculate availability percentage
                    history = self.availability_history[service["name"]]
                    availability = (sum(history) / len(history)) * 100 if history else 0
                    SERVICE_AVAILABILITY.labels(service=service["id"]).set(availability)
                    self.service_status[service["id"]]["availability"] = availability
            
            # Sleep until next check
            time.sleep(self.poll_interval)
    
    def get_service_status(self, service_id: str) -> Optional[Dict[str, Any]]:
        """Get the current status of a service."""
        return self.service_status.get(service_id)
    
    def get_service_metrics(self, service_id: str) -> Optional[Dict[str, Any]]:
        """Get the current metrics of a service."""
        return self.service_metrics.get(service_id)
    
    def get_all_services_status(self) -> Dict[str, Dict[str, Any]]:
        """Get the current status of all services."""
        return self.service_status.copy()
    
    def get_all_services_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get the current metrics of all services."""
        return self.service_metrics.copy() 