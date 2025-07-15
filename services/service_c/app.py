from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, Gauge, Counter, Histogram
from flask import Flask, Response, request, jsonify
import os
import time
import threading
import psutil
from functools import wraps
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
import random

app = Flask(__name__)

# Define metrics
cpu_usage = Gauge('service_cpu_usage', 'CPU usage in percentage')
memory_usage = Gauge('service_memory_usage', 'Memory usage in MB')
request_count = Counter('service_request_count', 'Total number of requests')
request_errors = Counter('request_errors_total', 'Total 5xx errors')
circuit_breaker_state = Gauge('service_circuit_breaker_state', 'Circuit breaker state (0=closed, 1=open)')
service_health = Gauge('service_health', 'Service health status (1.0=healthy, 0.7=warning, 0.4=degraded, 0.1=critical)')
service_latency_seconds = Gauge('service_latency_seconds', '95th percentile request latency in seconds')
dependency_health = Gauge('service_dependency_health', 'Health of dependent services', ['dependency'])
request_latency = Histogram('request_latency_seconds', 'Request latency in seconds', buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0])

# Initialize dependency health metric
dependency_health.labels(dependency='service_b').set(1.0)  # Initialize with healthy state

# Circuit breaker settings
CIRCUIT_BREAKER_THRESHOLD = 3  # Number of failures before opening
CIRCUIT_BREAKER_TIMEOUT = 10  # Seconds to wait before half-open
circuit_breaker_failures = 0
circuit_breaker_last_failure = 0
circuit_breaker_state_value = 0  # 0=closed, 1=open, 2=half-open

# Global variable to control CPU fault injection
cpu_fault_active = False
cpu_fault_thread = None

def simulate_cpu_load(duration):
    global cpu_fault_active
    cpu_fault_active = True
    end_time = time.time() + duration
    while time.time() < end_time and cpu_fault_active:
        # Simulate CPU load
        _ = [i * i for i in range(1000)]
    cpu_fault_active = False

def check_dependency_health():
    """Check health of Service B"""
    try:
        response = requests.get('http://service_b:5000/health', timeout=0.5)
        if response.status_code == 200:
            data = response.json()
            health_score = data.get('health_score', 0.0)
            dependency_health.labels(dependency='service_b').set(health_score)
            return health_score
    except:
        dependency_health.labels(dependency='service_b').set(0.0)
    return 0.0

def get_histogram_p95(histogram, metric_name):
    # Use prometheus_client public API to get bucket samples
    try:
        for metric in histogram.collect():
            for sample in metric.samples:
                # sample: (name, labels, value, timestamp)
                if sample.name.endswith('_bucket'):
                    # Collect all buckets
                    buckets = []
                    for s in metric.samples:
                        if s.name.endswith('_bucket'):
                            buckets.append((float(s.labels['le']), s.value))
                    buckets.sort()
                    total = buckets[-1][1] if buckets else 0
                    if total == 0:
                        return 0.0
                    target = total * 0.95
                    current = 0
                    for upper_bound, count in buckets:
                        current = count
                        if current >= target:
                            return upper_bound
        return 0.0
    except Exception as e:
        app.logger.error(f"Error extracting histogram p95: {str(e)}")
        return 0.0

def calculate_health_status():
    """Calculate overall health status based on real metrics"""
    try:
        health = 1.0
        
        # Get real metrics
        cpu = psutil.cpu_percent(interval=None)
        memory = psutil.Process().memory_info().rss / 1024**2   # MB
        latency = get_histogram_p95(request_latency, 'request_latency')  # seconds

        # Local penalties - ONLY if values are truly bad
        if cpu > 80:          health *= 0.8
        if memory > 400:      health *= 0.8
        if latency > 0.2:     health *= 0.8     # 200 ms

        # Dependency penalty
        upstream = dependency_health.labels(dependency="service_b")._value.get()
        if upstream < 0.7:
            health *= 0.6
        
        # Round to 2 decimal places
        health = round(health, 2)
        service_health.set(health)
        return health
    except Exception as e:
        app.logger.error(f"Error calculating health status: {str(e)}")
        return 0.1  # Return critical health on error

@app.errorhandler(Exception)
def handle_exception(e):
    """Handle all exceptions and increment error counter"""
    request_errors.inc()
    return "Internal Server Error", 500

@app.route('/health')
def health():
    """Health check endpoint"""
    start_time = time.time()
    health_score = calculate_health_status()
    response = jsonify({
        'status': 'healthy' if health_score > 0.7 else 'unhealthy',
        'health_score': health_score,
        'cpu_usage': psutil.cpu_percent(),
        'memory_usage': psutil.Process().memory_info().rss / 1_048_576,  # Convert to MB
        'error_rate': request_errors._value.get() / max(request_count._value.get(), 1),
        'circuit_breaker_state': 'closed' if circuit_breaker_state_value == 0 else 'open' if circuit_breaker_state_value == 1 else 'half-open',
        'cpu_fault_active': cpu_fault_active
    })
    request_latency.observe(time.time() - start_time)
    return response

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_upstream_health():
    """Get health metric from upstream service"""
    try:
        response = requests.get('http://service_b:5000/health', timeout=0.5)
        if response.status_code == 200:
            data = response.json()
            return data.get('health_score', 1.0)
        return 1.0
    except Exception as e:
        app.logger.error(f"Error getting upstream health: {str(e)}")
        return 1.0

@app.route('/metrics')
def metrics():
    # Simulate some metrics
    cpu = random.uniform(0.1, 0.9)
    memory = random.uniform(0.2, 0.8)
    latency = random.uniform(0.01, 0.1)
    
    # Calculate health score based on metrics
    health_score = 1.0
    if cpu > 0.8:
        health_score *= 0.8
    if memory > 0.8:
        health_score *= 0.8
    if latency > 0.05:
        health_score *= 0.8
        
    # Check upstream service health
    up_health = get_upstream_health()
    if up_health < 0.7:
        health_score *= 0.6
    
    # Set metrics
    service_health.set(health_score)  # Set as float
    cpu_usage.set(cpu)
    memory_usage.set(memory)
    service_latency_seconds.set(latency)
    
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

@app.route('/fault/cpu', methods=['POST'])
def inject_cpu_fault():
    """Inject CPU fault for testing"""
    start_time = time.time()
    global cpu_fault_thread
    duration = request.json.get('duration', 30)
    
    if cpu_fault_thread and cpu_fault_thread.is_alive():
        response = jsonify({'status': 'fault already active'})
    else:
        cpu_fault_thread = threading.Thread(target=simulate_cpu_load, args=(duration,))
        cpu_fault_thread.daemon = True
        cpu_fault_thread.start()
        response = jsonify({'status': 'cpu fault started', 'duration': duration})
    
    request_latency.observe(time.time() - start_time)
    return response

@app.route('/')
def root():
    """Root endpoint"""
    return "Service C is running", 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 