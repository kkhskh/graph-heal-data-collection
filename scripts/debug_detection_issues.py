#!/usr/bin/env python3
"""
Debug script to analyze why detection results are poor.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

def analyze_experiment_data():
    """Analyze the experiment data to understand the detection issues."""
    
    # Load a sample experiment
    exp_file = 'results/processed/cpu_train_cpu_experiment_1.json'
    with open(exp_file, 'r') as f:
        experiment = json.load(f)
    
    print("=== EXPERIMENT DATA ANALYSIS ===")
    print(f"Experiment: {experiment.get('experiment_name', 'unknown')}")
    print(f"Fault type: {experiment.get('fault_type', 'unknown')}")
    print(f"Pattern: {experiment.get('pattern', 'unknown')}")
    print(f"Duration: {experiment.get('duration', 'unknown')}")
    print(f"Intensity: {experiment.get('intensity', 'unknown')}")
    
    # Analyze metrics
    metrics = experiment['metrics']
    cpu_values = [m['service_cpu_usage'] for m in metrics]
    
    print(f"\n=== METRIC ANALYSIS ===")
    print(f"Total samples: {len(cpu_values)}")
    print(f"CPU usage range: {min(cpu_values):.2f} - {max(cpu_values):.2f}")
    print(f"CPU usage mean: {np.mean(cpu_values):.2f}")
    print(f"CPU usage std: {np.std(cpu_values):.2f}")
    print(f"CPU usage change: {cpu_values[-1] - cpu_values[0]:.2f}")
    print(f"CPU usage change rate: {(cpu_values[-1] - cpu_values[0]) / len(cpu_values):.4f} per sample")
    
    # Analyze fault periods
    fault_periods = experiment['fault_periods']
    print(f"\n=== FAULT PERIODS ===")
    for i, period in enumerate(fault_periods):
        print(f"Period {i+1}: {period['start']} to {period['end']}")
        print(f"  Type: {period['type']}")
        print(f"  Pattern: {period['pattern']}")
    
    # Convert to ground truth
    timestamps = [datetime.fromisoformat(ts) for ts in experiment['timestamps']]
    ground_truth = [False] * len(timestamps)
    
    for period in fault_periods:
        start_time = datetime.fromisoformat(period['start'])
        end_time = datetime.fromisoformat(period['end'])
        
        for i, ts in enumerate(timestamps):
            if start_time <= ts <= end_time:
                ground_truth[i] = True
    
    print(f"\n=== GROUND TRUTH ANALYSIS ===")
    print(f"Total ground truth positives: {sum(ground_truth)}")
    print(f"Ground truth positive rate: {sum(ground_truth) / len(ground_truth) * 100:.1f}%")
    print(f"First 20 ground truth: {ground_truth[:20]}")
    
    # Test detection with different parameters
    print(f"\n=== DETECTION PARAMETER TESTING ===")
    
    # Current parameters
    current_params = {
        'window_size': 30,
        'z_score_threshold': 2.0,
        'min_anomaly_duration': 5
    }
    
    # Test with current parameters
    predictions = detect_anomalies(cpu_values, current_params)
    metrics = calculate_metrics(predictions, ground_truth)
    
    print(f"Current parameters: {current_params}")
    print(f"Predictions: {sum(predictions)} positives out of {len(predictions)}")
    print(f"Metrics: {metrics}")
    
    # Test with more sensitive parameters
    sensitive_params = {
        'window_size': 10,
        'z_score_threshold': 0.5,
        'min_anomaly_duration': 1
    }
    
    predictions_sensitive = detect_anomalies(cpu_values, sensitive_params)
    metrics_sensitive = calculate_metrics(predictions_sensitive, ground_truth)
    
    print(f"\nSensitive parameters: {sensitive_params}")
    print(f"Predictions: {sum(predictions_sensitive)} positives out of {len(predictions_sensitive)}")
    print(f"Metrics: {metrics_sensitive}")
    
    # Test with very sensitive parameters
    very_sensitive_params = {
        'window_size': 5,
        'z_score_threshold': 0.1,
        'min_anomaly_duration': 1
    }
    
    predictions_very = detect_anomalies(cpu_values, very_sensitive_params)
    metrics_very = calculate_metrics(predictions_very, ground_truth)
    
    print(f"\nVery sensitive parameters: {very_sensitive_params}")
    print(f"Predictions: {sum(predictions_very)} positives out of {len(predictions_very)}")
    print(f"Metrics: {metrics_very}")
    
    # Plot the data
    plot_analysis(cpu_values, ground_truth, predictions, predictions_sensitive, predictions_very)

def detect_anomalies(metric_values, params):
    """Detect anomalies using specified parameters."""
    window_size = params['window_size']
    z_threshold = params['z_score_threshold']
    min_duration = params['min_anomaly_duration']
    
    metric_values = np.array(metric_values)
    
    # Calculate rolling statistics
    rolling_mean = np.convolve(metric_values, np.ones(window_size)/window_size, mode='same')
    rolling_std = np.array([
        np.std(metric_values[max(0, i-window_size//2):min(len(metric_values), i+window_size//2+1)])
        for i in range(len(metric_values))
    ])
    
    # Calculate z-scores
    z_scores = np.zeros_like(metric_values)
    for i in range(len(metric_values)):
        if rolling_std[i] > 0:
            z_scores[i] = (metric_values[i] - rolling_mean[i]) / rolling_std[i]
    
    # Detect anomalies
    anomalies = np.abs(z_scores) > z_threshold
    
    # Apply minimum duration filter
    for i in range(len(anomalies)):
        if anomalies[i]:
            if i + min_duration <= len(anomalies):
                if not all(anomalies[i:i+min_duration]):
                    anomalies[i] = False
    
    return anomalies.tolist()

def calculate_metrics(predictions, ground_truth):
    """Calculate detection metrics."""
    tp = sum(1 for p, a in zip(predictions, ground_truth) if p and a)
    fp = sum(1 for p, a in zip(predictions, ground_truth) if p and not a)
    fn = sum(1 for p, a in zip(predictions, ground_truth) if not p and a)
    tn = sum(1 for p, a in zip(predictions, ground_truth) if not p and not a)
    
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def plot_analysis(cpu_values, ground_truth, predictions, predictions_sensitive, predictions_very):
    """Plot the analysis results."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: CPU values over time
    axes[0, 0].plot(cpu_values, label='CPU Usage', color='blue')
    axes[0, 0].set_title('CPU Usage Over Time')
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('CPU Usage (%)')
    axes[0, 0].legend()
    
    # Plot 2: Ground truth
    axes[0, 1].plot(ground_truth, label='Ground Truth', color='red', linewidth=2)
    axes[0, 1].set_title('Ground Truth Anomalies')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Anomaly (True/False)')
    axes[0, 1].legend()
    
    # Plot 3: Current predictions
    axes[1, 0].plot(predictions, label='Current Predictions', color='orange')
    axes[1, 0].set_title('Current Parameter Predictions')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Anomaly (True/False)')
    axes[1, 0].legend()
    
    # Plot 4: Sensitive predictions
    axes[1, 1].plot(predictions_sensitive, label='Sensitive Predictions', color='green')
    axes[1, 1].set_title('Sensitive Parameter Predictions')
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Anomaly (True/False)')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('results/debug_detection_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nAnalysis plot saved to: results/debug_detection_analysis.png")

if __name__ == "__main__":
    analyze_experiment_data() 