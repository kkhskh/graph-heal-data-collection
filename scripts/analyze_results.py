import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import re

def parse_metrics(metrics_text):
    """Parse Prometheus metrics text into a dictionary."""
    metrics = {}
    for line in metrics_text.split('\n'):
        if line.startswith('#') or not line.strip():
            continue
        match = re.match(r'(\w+)\s+([\d.]+)', line)
        if match:
            name, value = match.groups()
            metrics[name] = float(value)
    return metrics

def load_experiment_data(filename):
    """Load experiment data from a JSON file."""
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # Parse timestamps and metrics
    timestamps = []
    cpu_usage = []
    memory_usage = []
    response_time = []
    
    for entry in data['metrics']:
        timestamp = datetime.fromisoformat(entry['timestamp'])
        metrics = parse_metrics(entry['metrics'])
        
        timestamps.append(timestamp)
        cpu_usage.append(metrics.get('service_cpu_usage', 0))
        memory_usage.append(metrics.get('service_memory_usage', 0))
        response_time.append(metrics.get('service_response_time', 0))
    
    return {
        'experiment_name': data['experiment_name'],
        'service': data['service'],
        'fault_type': data['fault_type'],
        'duration': data['duration'],
        'timestamps': timestamps,
        'cpu_usage': cpu_usage,
        'memory_usage': memory_usage,
        'response_time': response_time
    }

def plot_metrics(data, output_dir='results/plots'):
    """Plot metrics for an experiment."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
    fig.suptitle(f"{data['experiment_name']} - {data['fault_type']} fault on service_{data['service']}")
    
    # Plot CPU usage
    ax1.plot(data['timestamps'], data['cpu_usage'], 'b-', label='CPU Usage')
    ax1.set_ylabel('CPU Usage (%)')
    ax1.set_title('CPU Usage Over Time')
    ax1.grid(True)
    
    # Plot memory usage
    ax2.plot(data['timestamps'], data['memory_usage'], 'r-', label='Memory Usage')
    ax2.set_ylabel('Memory Usage (MB)')
    ax2.set_title('Memory Usage Over Time')
    ax2.grid(True)
    
    # Plot response time
    ax3.plot(data['timestamps'], data['response_time'], 'g-', label='Response Time')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Response Time (ms)')
    ax3.set_title('Response Time Over Time')
    ax3.grid(True)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{data['experiment_name']}.png")
    plt.close()

def main():
    # Load and plot results for each experiment
    experiments = [
        'results/cpu_experiment.json',
        'results/memory_experiment.json',
        'results/network_experiment.json'
    ]
    
    for experiment_file in experiments:
        print(f"Analyzing {experiment_file}...")
        data = load_experiment_data(experiment_file)
        plot_metrics(data)
        print(f"Plots saved to results/plots/{data['experiment_name']}.png")

if __name__ == "__main__":
    main() 