from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, Gauge, Counter, Histogram
from flask import Flask, Response, request, jsonify
import os
import time
import threading
import psutil
from functools import wraps
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
import traceback

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

# Circuit breaker settings
CIRCUIT_BREAKER_THRESHOLD = 5  # Number of failures before opening
CIRCUIT_BREAKER_TIMEOUT = 30  # Seconds to wait before half-open
circuit_breaker_failures = 0
circuit_breaker_last_failure = 0
circuit_breaker_state_value = 0  # 0=closed, 1=open, 2=half-open

# Global variable to control CPU fault injection
cpu_fault_active = False
cpu_fault_thread = None

memory_fault_active = False
memory_fault_thread = None

def simulate_cpu_load(duration):
    global cpu_fault_active
    cpu_fault_active = True
    end_time = time.time() + duration
    while time.time() < end_time and cpu_fault_active:
        # Simulate CPU load
        _ = [i * i for i in range(1000)]
    cpu_fault_active = False

def simulate_memory_load(duration):
    global memory_fault_active
    memory_fault_active = True
    end_time = time.time() + duration
    mem_list = []
    try:
        while time.time() < end_time and memory_fault_active:
            mem_list.append(bytearray(10 * 1024 * 1024))  # Allocate 10MB chunks
            time.sleep(0.1)
    except Exception as e:
        app.logger.error(f"Memory fault error: {e}")
    finally:
        mem_list.clear()
        memory_fault_active = False

def check_dependency_health():
    """Check health of dependent services"""
    try:
        # For service_a, we don't have any dependencies
        return 1.0
    except Exception as e:
        app.logger.error(f"Error checking dependency health: {str(e)}")
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
    global cpu_fault_active
    try:
        health_score = 1.0
        
        # If CPU fault is active, immediately reduce health
        if cpu_fault_active:
            health_score = 0.4  # Set to degraded state
            return health_score  # Return immediately when fault is active
        
        # Get real CPU usage
        cpu_percent = psutil.cpu_percent()
        if cpu_percent > 80:
            health_score *= 0.7
        
        # Get real latency (95th percentile) only if we have data
        latency_p95 = get_histogram_p95(request_latency, 'request_latency')
        if latency_p95 > 0.3:  # >300ms
            health_score *= 0.7
        
        # Check dependency health
        dependency_score = check_dependency_health()
        if dependency_score < 0.7:
            health_score *= 0.6
        
        # If circuit breaker is open, reduce health
        if circuit_breaker_state_value == 1:
            health_score *= 0.5
        
        # Round to 2 decimal places
        health_score = round(health_score, 2)
        return health_score
    except Exception as e:
        app.logger.error(f"Error calculating health status: {str(e)}")
        return 0.1  # Return critical health on error

@app.errorhandler(Exception)
def handle_exception(e):
    request_errors.inc()
    app.logger.error("Exception occurred", exc_info=True)
    return "Internal Server Error", 500

@app.route('/health')
def health():
    """Health check endpoint"""
    start_time = time.time()
    try:
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
    except Exception as e:
        app.logger.error(f"Error in health endpoint: {str(e)}")
        request_errors.inc()
        return "Internal Server Error", 500

@app.route('/metrics')
def metrics():
    start_time = time.time()
    
    try:
        # Set real system metrics
        cpu_usage.set(psutil.cpu_percent())
        memory_usage.set(psutil.Process().memory_info().rss / 1_048_576)  # Convert to MB
        
        request_count.inc()
        
        # Set latency metric from histogram using public API
        latency_p95 = get_histogram_p95(request_latency, 'request_latency')
        service_latency_seconds.set(latency_p95)
        
        # Check dependency health and update metric
        dependency_score = check_dependency_health()
        dependency_health.labels(dependency='service_a').set(dependency_score)
        
        # Update health status - always recalculate
        health_score = calculate_health_status()
        service_health.set(health_score)  # Set as float
        
        # Update circuit breaker state
        circuit_breaker_state.set(circuit_breaker_state_value)
        
        # Force update of all metrics
        generate_latest()
        
        response = Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)
        request_latency.observe(time.time() - start_time)
        return response
    except Exception as e:
        app.logger.error(f"Error in metrics endpoint: {str(e)}")
        request_errors.inc()
        return "Internal Server Error", 500

@app.route('/fault/cpu', methods=['POST'])
def inject_cpu_fault():
    """Inject CPU fault for testing"""
    start_time = time.time()
    global cpu_fault_thread, cpu_fault_active
    try:
        duration = request.json.get('duration', 30)
        app.logger.info(f"Received CPU fault request with duration {duration}")
        
        if cpu_fault_thread and cpu_fault_thread.is_alive():
            app.logger.info("Fault already active")
            response = jsonify({'status': 'fault already active'})
        else:
            app.logger.info("Starting new CPU fault thread")
            cpu_fault_thread = threading.Thread(target=simulate_cpu_load, args=(duration,))
            cpu_fault_thread.daemon = True
            cpu_fault_thread.start()
            response = jsonify({'status': 'cpu fault started', 'duration': duration})
        
        request_latency.observe(time.time() - start_time)
        return response
    except Exception as e:
        app.logger.error(f"Error in CPU fault injection: {str(e)}")
        request_errors.inc()
        return "Internal Server Error", 500

@app.route('/fault/memory', methods=['POST'])
def inject_memory_fault():
    app.logger.info("/fault/memory endpoint called")
    try:
        app.logger.info(f"Request data: {request.data}")
        data = request.get_json(force=True)
        app.logger.info(f"Parsed JSON: {data}")
        duration = data.get('duration', 30)
        app.logger.info(f"Parsed duration: {duration}")
    except Exception as e:
        app.logger.error(f"Error parsing request payload: {e}", exc_info=True)
        return jsonify({'error': 'Invalid request payload'}), 400
    global memory_fault_thread, memory_fault_active
    try:
        if memory_fault_thread and memory_fault_thread.is_alive():
            app.logger.info("Memory fault already active")
            response = jsonify({'status': 'fault already active'})
        else:
            app.logger.info("Starting new memory fault thread")
            def safe_simulate_memory_load(duration):
                try:
                    simulate_memory_load(duration)
                except Exception as e:
                    app.logger.error(f"Exception in memory fault thread: {e}", exc_info=True)
            
            memory_fault_thread = threading.Thread(target=safe_simulate_memory_load, args=(duration,))
            memory_fault_thread.daemon = True
            memory_fault_thread.start()
            response = jsonify({'status': 'memory fault started', 'duration': duration})
        
        request_latency.observe(time.time() - start_time)
        return response
    except Exception as e:
        app.logger.error(f"Error in memory fault injection: {e}", exc_info=True)
        return "Internal Server Error", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 
