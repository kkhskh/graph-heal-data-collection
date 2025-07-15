#!/bin/bash

# Check if experiment directory is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <experiment_directory>"
    exit 1
fi

EXPERIMENT_DIR="$1"
METRICS_FILE="${EXPERIMENT_DIR}/metrics.csv"
PROCESSED_FILE="${EXPERIMENT_DIR}/processed_metrics.csv"

# Create processed metrics file with headers
echo "timestamp,service,metric,value,labels" > "$PROCESSED_FILE"

# Process each metrics file
for service in a b c d; do
    metrics_file="${EXPERIMENT_DIR}/metrics_${service}.txt"
    if [ -f "$metrics_file" ]; then
        echo "Processing metrics for service ${service}..."
        
        # Extract and process metrics
        while IFS= read -r line; do
            # Skip help and type lines
            if [[ "$line" == *"# HELP"* ]] || [[ "$line" == *"# TYPE"* ]]; then
                continue
            fi
            
            # Extract timestamp and metric name
            timestamp=$(echo "$line" | awk '{print $1}')
            metric_name=$(echo "$line" | awk '{print $2}')
            
            # Extract labels if present
            labels=""
            if [[ "$line" == *"{"* ]]; then
                labels=$(echo "$line" | sed -n 's/.*{\(.*\)}.*/\1/p')
                # Extract value after labels
                value=$(echo "$line" | sed -n 's/.*} \(.*\)/\1/p')
            else
                # Extract value as last field
                value=$(echo "$line" | awk '{print $NF}')
            fi
            
            # Skip if any field is empty
            if [ -z "$timestamp" ] || [ -z "$metric_name" ] || [ -z "$value" ]; then
                continue
            fi
            
            # Write processed metrics
            echo "${timestamp},${service},${metric_name},${value},${labels}" >> "$PROCESSED_FILE"
        done < "$metrics_file"
    fi
done

echo "Metrics processing completed. Results saved in ${PROCESSED_FILE}" 