#!/usr/bin/env python3
"""
Parameter optimization for the StatisticalAnomalyDetector.

This script systematically tests different parameter combinations for the
trend-based anomaly detector to find the optimal set that maximizes the F1-score.
"""

import json
import numpy as np
import os
import itertools
from datetime import datetime
from sklearn.linear_model import LinearRegression

# --- Re-usable Evaluation Logic (from fixed_detection_evaluation.py) ---

class StandaloneDetector:
    """A standalone version of the detector for optimization testing."""
    def __init__(self, params):
        self.params = params
        self.window_size = 20  # Keep window size fixed for now

    def detect_trend_anomalies(self, metrics):
        """Detect anomalies using trend analysis with the given parameters."""
        # This logic is simplified to focus on the core detection algorithm
        # It assumes a single service's metric stream for evaluation.
        metric_keys = ['service_cpu_usage', 'service_memory_usage', 'service_response_time']
        
        all_anomalies = []
        for key in metric_keys:
            if key in metrics[0]:
                values = np.array([m[key] for m in metrics])
                anomalies = self._detect_trends(values)
                all_anomalies.append(anomalies)
        
        # Combine anomalies from different metrics (OR logic)
        if not all_anomalies:
            return [False] * len(metrics)
            
        final_anomalies = np.any(np.array(all_anomalies), axis=0)
        return final_anomalies.tolist()

    def _detect_trends(self, values):
        """Detect trends using linear regression with tunable parameters."""
        trends = np.zeros(len(values), dtype=bool)
        
        for i in range(self.window_size, len(values)):
            window = values[i-self.window_size:i]
            x = np.arange(len(window)).reshape(-1, 1)
            y = window

            try:
                reg = LinearRegression()
                reg.fit(x, y)
                slope = reg.coef_[0]
                r_squared = reg.score(x, y)
                start_val = window[0]
                end_val = window[-1]
                pct_change = abs(end_val - start_val) / start_val if start_val > 0 else 0

                is_anomaly = (
                    abs(slope) > self.params['sensitivity'] and
                    r_squared > self.params['min_r_squared'] and
                    pct_change > self.params['trend_threshold']
                )
                if is_anomaly:
                    trends[i] = True
            except Exception:
                continue
                
        return trends

def calculate_metrics(predictions, ground_truth):
    """Calculate detection metrics."""
    tp = sum(1 for p, a in zip(predictions, ground_truth) if p and a)
    fp = sum(1 for p, a in zip(predictions, ground_truth) if p and not a)
    fn = sum(1 for p, a in zip(predictions, ground_truth) if not p and a)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {'precision': precision, 'recall': recall, 'f1_score': f1}

def convert_fault_periods_to_timesteps(experiment):
    """Convert fault periods to a ground truth vector."""
    timestamps = [datetime.fromisoformat(ts) for ts in experiment['timestamps']]
    ground_truth = [False] * len(timestamps)
    for period in experiment['fault_periods']:
        start_time = datetime.fromisoformat(period['start'])
        end_time = datetime.fromisoformat(period['end'])
        for i, ts in enumerate(timestamps):
            if start_time <= ts <= end_time:
                ground_truth[i] = True
    return ground_truth

# --- Main Optimization Logic ---

def run_optimization():
    """Main function to run the parameter optimization process."""
    print("=== Starting Parameter Optimization ===")

    # Load experiment data
    experiments = []
    data_dir = 'results/processed'
    for filename in os.listdir(data_dir):
        if filename.endswith('.json') and ('_4.json' in filename or '_5.json' in filename):
            filepath = os.path.join(data_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    exp = json.load(f)
                if 'metrics' in exp and 'fault_periods' in exp and exp['metrics']:
                    experiments.append(exp)
            except Exception as e:
                print(f"Skipping {filename}: {e}")
    
    print(f"Loaded {len(experiments)} experiments for optimization.")

    # Define the parameter grid to search
    param_grid = {
        'trend_threshold': [0.01, 0.05, 0.1, 0.15],
        'sensitivity': [0.01, 0.05, 0.1],
        'min_r_squared': [0.3, 0.4, 0.5, 0.6]
    }

    # Generate all parameter combinations
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"Testing {len(param_combinations)} parameter combinations...")

    best_f1_score = -1
    best_params = None
    all_results = []

    # Run evaluation for each combination
    for i, params in enumerate(param_combinations):
        detector = StandaloneDetector(params)
        
        f1_scores = []
        for exp in experiments:
            predictions = detector.detect_trend_anomalies(exp['metrics'])
            ground_truth = convert_fault_periods_to_timesteps(exp)
            metrics = calculate_metrics(predictions, ground_truth)
            f1_scores.append(metrics['f1_score'])
        
        avg_f1_score = np.mean(f1_scores) if f1_scores else 0
        all_results.append({'params': params, 'f1_score': avg_f1_score})
        
        print(f"[{i+1}/{len(param_combinations)}] Params: {params} -> Avg F1-Score: {avg_f1_score:.4f}")

        if avg_f1_score > best_f1_score:
            best_f1_score = avg_f1_score
            best_params = params

    # --- Results ---
    print("\n=== Optimization Complete ===")
    print(f"Best F1-Score: {best_f1_score:.4f}")
    print(f"Optimal Parameters: {best_params}")

    # Save results to a file
    output = {
        'best_f1_score': best_f1_score,
        'best_parameters': best_params,
        'all_results': sorted(all_results, key=lambda x: x['f1_score'], reverse=True),
        'timestamp': datetime.now().isoformat()
    }
    
    os.makedirs('results/optimization', exist_ok=True)
    with open('results/optimization/parameter_optimization_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("\nResults saved to: results/optimization/parameter_optimization_results.json")
    print("You can now update the `StatisticalAnomalyDetector` with these optimal parameters.")

if __name__ == "__main__":
    run_optimization() 