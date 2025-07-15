#!/usr/bin/env python3
"""
Fixed detection evaluation using trend-based detection for gradual changes.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from scipy import stats
from sklearn.linear_model import LinearRegression

class FixedDetector:
    def __init__(self):
        self.params = {
            'trend_threshold': 0.05,  # 5% change threshold
            'window_size': 20,        # Smaller window for trend detection
            'min_duration': 3,        # Minimum trend duration
            'sensitivity': 0.1        # Sensitivity for gradual changes
        }
    
    def detect_trend_anomalies(self, metrics, debug=False):
        """Detect anomalies using trend analysis instead of z-score."""
        cpu_values = np.array([m['service_cpu_usage'] for m in metrics])
        memory_values = np.array([m['service_memory_usage'] for m in metrics])
        response_values = np.array([m['service_response_time'] for m in metrics])
        
        anomalies = np.zeros(len(cpu_values), dtype=bool)
        
        # Detect trends in CPU usage
        cpu_trends = self._detect_trends(cpu_values, 'CPU', debug)
        memory_trends = self._detect_trends(memory_values, 'Memory', debug)
        response_trends = self._detect_trends(response_values, 'Response', debug)
        
        # Combine detections
        anomalies = cpu_trends | memory_trends | response_trends
        
        # Apply minimum duration filter
        anomalies = self._apply_duration_filter(anomalies)
        
        return anomalies.tolist()
    
    def _detect_trends(self, values, metric_name, debug=False):
        """Detect trends using linear regression."""
        window_size = self.params['window_size']
        threshold = self.params['trend_threshold']
        sensitivity = self.params['sensitivity']
        
        trends = np.zeros(len(values), dtype=bool)
        
        for i in range(window_size, len(values)):
            window = values[i-window_size:i]
            x = np.arange(len(window)).reshape(-1, 1)
            y = window
            
            # Fit linear regression
            reg = LinearRegression()
            reg.fit(x, y)
            
            # Calculate slope and R-squared
            slope = reg.coef_[0]
            r_squared = reg.score(x, y)
            
            # Calculate percentage change
            start_val = window[0]
            end_val = window[-1]
            pct_change = abs(end_val - start_val) / start_val if start_val > 0 else 0
            
            # Detect anomaly based on trend
            is_anomaly = (
                abs(slope) > sensitivity and  # Significant slope
                r_squared > 0.3 and          # Good fit
                pct_change > threshold       # Significant change
            )
            
            if is_anomaly:
                trends[i] = True
                
            if debug and i < 50:
                print(f"{metric_name} {i}: slope={slope:.4f}, R²={r_squared:.3f}, change={pct_change:.3f}, anomaly={is_anomaly}")
        
        return trends
    
    def _apply_duration_filter(self, anomalies):
        """Apply minimum duration filter."""
        min_duration = self.params['min_duration']
        filtered = anomalies.copy()
        
        for i in range(len(anomalies)):
            if anomalies[i]:
                if i + min_duration <= len(anomalies):
                    if not all(anomalies[i:i+min_duration]):
                        filtered[i] = False
        
        return filtered
    
    def calculate_metrics(self, predictions, ground_truth):
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
            'f1_score': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn
        }

def convert_fault_periods_to_timesteps(experiment):
    """Convert fault periods to per-timestep ground truth vector."""
    timestamps = [datetime.fromisoformat(ts) for ts in experiment['timestamps']]
    ground_truth = [False] * len(timestamps)
    
    for period in experiment['fault_periods']:
        start_time = datetime.fromisoformat(period['start'])
        end_time = datetime.fromisoformat(period['end'])
        
        for i, ts in enumerate(timestamps):
            if start_time <= ts <= end_time:
                ground_truth[i] = True
                
    return ground_truth

def main():
    print("=== FIXED DETECTION EVALUATION ===")
    
    # Load experiments
    experiments = []
    data_dir = 'results/processed'
    
    for filename in os.listdir(data_dir):
        if filename.endswith('.json') and ('_4.json' in filename or '_5.json' in filename):
            filepath = os.path.join(data_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    experiment = json.load(f)
                
                if 'metrics' in experiment and 'fault_periods' in experiment:
                    experiment['filename'] = filename
                    experiments.append(experiment)
                    
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    print(f"Loaded {len(experiments)} experiments")
    
    # Initialize detector
    detector = FixedDetector()
    
    # Evaluate each experiment
    all_results = []
    
    for i, experiment in enumerate(experiments):
        print(f"\n--- Experiment {i+1}: {experiment['filename']} ---")
        print(f"Type: {experiment.get('fault_type', 'unknown')}")
        print(f"Pattern: {experiment.get('pattern', 'unknown')}")
        
        # Get predictions and ground truth
        predictions = detector.detect_trend_anomalies(experiment['metrics'], debug=(i==0))
        ground_truth = convert_fault_periods_to_timesteps(experiment)
        
        # Calculate metrics
        metrics = detector.calculate_metrics(predictions, ground_truth)
        
        print(f"Predictions: {sum(predictions)}/{len(predictions)} ({sum(predictions)/len(predictions)*100:.1f}%)")
        print(f"Ground truth: {sum(ground_truth)}/{len(ground_truth)} ({sum(ground_truth)/len(ground_truth)*100:.1f}%)")
        print(f"Metrics: Accuracy={metrics['accuracy']:.3f}, Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, F1={metrics['f1_score']:.3f}")
        
        all_results.append(metrics)
    
    # Calculate overall statistics
    print(f"\n=== OVERALL RESULTS ===")
    
    avg_accuracy = np.mean([r['accuracy'] for r in all_results])
    avg_precision = np.mean([r['precision'] for r in all_results])
    avg_recall = np.mean([r['recall'] for r in all_results])
    avg_f1 = np.mean([r['f1_score'] for r in all_results])
    
    print(f"Average Accuracy: {avg_accuracy:.3f}")
    print(f"Average Precision: {avg_precision:.3f}")
    print(f"Average Recall: {avg_recall:.3f}")
    print(f"Average F1-Score: {avg_f1:.3f}")
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'parameters': detector.params,
        'overall_metrics': {
            'accuracy': avg_accuracy,
            'precision': avg_precision,
            'recall': avg_recall,
            'f1_score': avg_f1
        },
        'experiment_results': all_results
    }
    
    os.makedirs('results/fixed_evaluation', exist_ok=True)
    with open('results/fixed_evaluation/fixed_detection_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: results/fixed_evaluation/fixed_detection_results.json")

if __name__ == "__main__":
    main() 