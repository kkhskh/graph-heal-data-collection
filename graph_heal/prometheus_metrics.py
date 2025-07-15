from prometheus_client import Counter, Gauge, Histogram, generate_latest, REGISTRY
from typing import Dict, Any

def _counter(name: str, doc: str, **kwargs):
    if name in REGISTRY._names_to_collectors:
        return REGISTRY._names_to_collectors[name]
    return Counter(name, doc, **kwargs)


def _gauge(name: str, doc: str, **kwargs):
    if name in REGISTRY._names_to_collectors:
        return REGISTRY._names_to_collectors[name]
    return Gauge(name, doc, **kwargs)


def _histogram(name: str, doc: str, **kwargs):
    if name in REGISTRY._names_to_collectors:
        return REGISTRY._names_to_collectors[name]
    return Histogram(name, doc, **kwargs)


# Idempotent metric objects
REQUEST_COUNT = _counter(
    'service_request_total',
    'Total number of requests',
    labelnames=['service', 'endpoint', 'method', 'status']
)

REQUEST_LATENCY = _histogram(
    'service_request_duration_seconds',
    'Request latency in seconds',
    labelnames=['service', 'endpoint']
)

SERVICE_HEALTH = _gauge(
    'service_health',
    'Service health status (1 for healthy, 0 for unhealthy)',
    labelnames=['service']
)

SERVICE_AVAILABILITY = _gauge(
    'service_availability_percentage',
    'Service availability percentage',
    labelnames=['service']
)

def format_metrics(metrics_data: Dict[str, Any], service_name: str) -> bytes:
    """
    Format metrics data in Prometheus format.
    
    Args:
        metrics_data: Dictionary containing metrics data
        service_name: Name of the service
    
    Returns:
        Prometheus-formatted metrics as bytes
    """
    # Update metrics with the latest data
    if 'requests' in metrics_data:
        for endpoint, data in metrics_data['requests'].items():
            REQUEST_COUNT.labels(
                service=service_name,
                endpoint=endpoint,
                method=data.get('method', 'GET'),
                status=data.get('status', '200')
            ).inc(data.get('count', 0))
            
            if 'latency' in data:
                REQUEST_LATENCY.labels(
                    service=service_name,
                    endpoint=endpoint
                ).observe(data['latency'])
    
    if 'health' in metrics_data:
        SERVICE_HEALTH.labels(service=service_name).set(
            1 if metrics_data['health'].get('status') == 'healthy' else 0
        )
    
    if 'availability' in metrics_data:
        SERVICE_AVAILABILITY.labels(service=service_name).set(
            metrics_data['availability']
        )
    
    return generate_latest() 