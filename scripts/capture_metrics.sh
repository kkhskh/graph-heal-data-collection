#!/bin/bash

# Check if required arguments are provided
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
    echo "Usage: $0 <experiment_directory> <duration> <interval>"
    exit 1
fi

EXPERIMENT_DIR="$1"
DURATION="$2"
INTERVAL="$3"
END_TIME=$((SECONDS + DURATION))

# Create metrics files for each service
for service in a b c d; do
    touch "${EXPERIMENT_DIR}/metrics_${service}.txt"
done

# Capture metrics until duration is reached
while [ $SECONDS -lt $END_TIME ]; do
    timestamp=$(date +%s)
    
    # Capture metrics from each service
    for service in a b c d; do
        case $service in
            a) port=5001 ;;
            b) port=5002 ;;
            c) port=5003 ;;
            d) port=5004 ;;
        esac
        curl -s "http://localhost:${port}/metrics" | grep -E "service_(cpu_usage|memory_bytes|latency_seconds|health|circuit_breaker_state|error_count)" | \
            awk -v ts="$timestamp" '{print ts, $0}' >> "${EXPERIMENT_DIR}/metrics_${service}.txt"
    done
    
    sleep $INTERVAL
done 