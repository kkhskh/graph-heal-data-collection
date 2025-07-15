import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import matplotlib.dates as mdates

# Load results from the unified script
with open('results/detection_accuracy_results_unified.json', 'r') as f:
    results = json.load(f)

# Set a serious, Harvard-style aesthetic
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18
})

# Function to plot bar charts for accuracy metrics
def plot_accuracy_metrics():
    experiments = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    methods = ['threshold_based', 'graph_heal']
    versions = ['old_version', 'new_version']

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        x = np.arange(len(experiments))
        width = 0.2

        for j, method in enumerate(methods):
            for k, version in enumerate(versions):
                values = [results[exp][version][method][metric] for exp in experiments]
                ax.bar(x + j * width + k * width / 2, values, width / 2, label=f'{method} ({version})')

        ax.set_xlabel('Experiment')
        ax.set_ylabel(metric.capitalize() + ' (%)')
        ax.set_title(f'{metric.capitalize()} Comparison')
        ax.set_xticks(x + width)
        ax.set_xticklabels(experiments)
        ax.legend()

    plt.tight_layout()
    plt.savefig('results/accuracy_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def extract_prometheus_metric(metrics_text, metric_name):
    for line in metrics_text.split('\n'):
        if line.startswith(metric_name + ' '):
            try:
                return float(line.split()[1])
            except Exception:
                return 0.0
    return 0.0

# Function to plot time series of metrics during fault periods
def plot_time_series():
    for experiment in results.keys():
        # Load experiment data
        with open(f'results/{experiment}.json', 'r') as f:
            data = json.load(f)

        timestamps = [datetime.fromisoformat(entry['timestamp']) for entry in data['metrics']]
        metrics = {
            'service_cpu_usage': [extract_prometheus_metric(entry['metrics'], 'service_cpu_usage') for entry in data['metrics']],
            'service_memory_usage': [extract_prometheus_metric(entry['metrics'], 'service_memory_usage') for entry in data['metrics']],
            'service_response_time': [extract_prometheus_metric(entry['metrics'], 'service_response_time') for entry in data['metrics']]
        }

        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        for i, (metric_name, values) in enumerate(metrics.items()):
            ax = axes[i]
            ax.plot(timestamps, values, label=metric_name)
            ax.set_xlabel('Time')
            ax.set_ylabel(metric_name)
            ax.set_title(f'{metric_name} Over Time')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            ax.legend()

        plt.tight_layout()
        plt.savefig(f'results/{experiment}_time_series.png', dpi=300, bbox_inches='tight')
        plt.close()

def plot_metric_separately(metric):
    experiments = list(results.keys())
    methods = ['threshold_based', 'graph_heal']
    versions = ['old_version', 'new_version']
    colors = {
        ('threshold_based', 'old_version'): '#1f77b4',  # blue
        ('threshold_based', 'new_version'): '#aec7e8',  # light blue
        ('graph_heal', 'old_version'): '#d62728',       # red
        ('graph_heal', 'new_version'): '#ff9896',       # light red
    }
    labels = {
        ('threshold_based', 'old_version'): 'Threshold-Based (Old)',
        ('threshold_based', 'new_version'): 'Threshold-Based (New)',
        ('graph_heal', 'old_version'): 'GRAPH-HEAL (Old)',
        ('graph_heal', 'new_version'): 'GRAPH-HEAL (New)',
    }
    x = np.arange(len(experiments))
    width = 0.18
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, method in enumerate(methods):
        for j, version in enumerate(versions):
            values = [results[exp][version][method][metric] for exp in experiments]
            offset = (i * 2 + j - 1.5) * width
            ax.bar(x + offset, values, width, color=colors[(method, version)], label=labels[(method, version)])
    ax.set_xlabel('Experiment')
    ax.set_ylabel(f'{metric.capitalize()} (%)')
    ax.set_title(f'{metric.capitalize()} Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([exp.replace('_experiment', '').capitalize() for exp in experiments])
    # Place legend outside
    legend = ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0., frameon=False)
    # Add explanation below the plot
    explanation = (
        "Legend: 'Old' = Original detection logic/thresholds. 'New' = Improved detection logic/thresholds.\n"
        "'Threshold-Based' = Simple threshold method. 'GRAPH-HEAL' = Z-score-based anomaly detection."
    )
    plt.gcf().text(0.01, -0.12, explanation, fontsize=11, ha='left', va='top', wrap=True)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(f'results/{metric}_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

# Generate each plot separately
for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
    plot_metric_separately(metric)

print("Separate comparison graphs generated and saved to the 'results' directory.")

# Run the plotting functions
plot_accuracy_metrics()
plot_time_series() 