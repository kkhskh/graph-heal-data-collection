#!/bin/bash

# Configuration
EXPERIMENT_NAME="cpu_fault_injection"
DURATION=60  # Duration in seconds
METRICS_INTERVAL=5  # Metrics collection interval in seconds
FAULT_DURATION=30  # CPU fault duration in seconds

# Function to check if Prometheus is ready
check_prometheus() {
    echo "Waiting for Prometheus to be ready..."
    until curl -sf http://localhost:9090/-/ready >/dev/null; do
        echo "Prometheus not ready yet, waiting..."
        sleep 2
    done
    echo "Prometheus is ready!"
}

# Function to check if a service is healthy
check_service() {
    local service=$1
    local port=$2
    echo "Checking health of $service..."
    until curl -s "http://localhost:$port/health" | grep -q "healthy"; do
        echo "$service not ready yet, waiting..."
        sleep 2
    done
    echo "$service is healthy!"
}

# Create experiment directory
EXPERIMENT_DIR="experiments/${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$EXPERIMENT_DIR"

# Wait for all services to be ready
echo "Waiting for all services to be ready..."
check_prometheus
check_service "service_a" "5001"
check_service "service_b" "5002"
check_service "service_c" "5003"
check_service "service_d" "5004"

# Start metrics collection
echo "Starting metrics collection..."
./scripts/capture_metrics.sh "$EXPERIMENT_DIR" "$DURATION" "$METRICS_INTERVAL" &
METRICS_PID=$!

# Wait for initial metrics to be collected
sleep 5

# Inject CPU fault into Service A
echo "Injecting CPU fault into Service A..."
curl -X POST http://localhost:5001/fault/cpu -H "Content-Type: application/json" -d "{\"duration\": $FAULT_DURATION}"

# Wait for experiment duration
echo "Waiting for experiment duration..."
sleep $DURATION

# Stop metrics collection
echo "Stopping metrics collection..."
kill $METRICS_PID

# Process metrics
echo "Processing metrics..."
./scripts/process_metrics.sh "$EXPERIMENT_DIR"

# Generate visualization
echo "Generating visualization..."
./scripts/visualize_metrics.sh "$EXPERIMENT_DIR"

echo "Experiment completed. Results saved in $EXPERIMENT_DIR" 