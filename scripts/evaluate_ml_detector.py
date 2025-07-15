#!/usr/bin/env python3
"""
Evaluates the performance of the MLDetector.

This script loads the test experiment data and uses the trained ML model
to make predictions, then calculates and reports the final performance metrics.
"""

import json
import numpy as np
import os
from datetime import datetime
from graph_heal.anomaly_detection import MLDetector

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

def calculate_metrics(predictions, ground_truth):
    """Calculate detection metrics."""
    tp = sum(1 for p, a in zip(predictions, ground_truth) if p and a)
    fp = sum(1 for p, a in zip(predictions, ground_truth) if p and not a)
    fn = sum(1 for p, a in zip(predictions, ground_truth) if not p and not a)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {'precision': precision, 'recall': recall, 'f1_score': f1}

def main():
    """Main evaluation function."""
    print("=== Evaluating MLDetector Performance ===")

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
    
    print(f"Loaded {len(experiments)} experiments for evaluation.")

    # Initialize the ML detector
    detector = MLDetector(model_path='models/fault_detection_model.joblib')

    if detector.model is None:
        print("ML Model could not be loaded. Aborting evaluation.")
        return

    all_results = []
    
    # Evaluate each experiment
    for experiment in experiments:
        # The MLDetector needs to be run on each timestep of the experiment
        predictions = []
        for i in range(len(experiment['metrics'])):
            # Create the service_statuses dictionary for the current timestep
            # This is a bit of a hack since the experiments aren't stored this way
            current_statuses = {
                'service_a': {'metrics': {}},
                'service_b': {'metrics': {}},
                'service_c': {'metrics': {}},
                'service_d': {'metrics': {}}
            }
            # We assume the metric list contains all services' metrics at each step
            # This requires a specific data format that might not hold true.
            # A better approach would be to process a single flattened dict per timestep.
            metric_snapshot = experiment['metrics'][i]
            for key, value in metric_snapshot.items():
                # This is a simplification; we need to map flat keys back to services
                # e.g., 'service_a_cpu_usage' -> current_statuses['service_a']['metrics']['cpu_usage']
                # This part is complex and depends on a consistent naming scheme.
                # For this eval, we will pass the whole flat dict.
                pass # The current MLDetector expects the full flat dict anyway.
            
            # The current MLDetector expects a dict of services. Re-creating it here is inefficient.
            # A better design would have the MLDetector accept the flat dictionary directly.
            # Let's adapt to the current detector's needs.
            status_snapshot = {}
            # This is tricky because the experiment data is already flattened.
            # I will need to refactor the detector or this script.
            # Given the detector is new, let's refactor the evaluation script to match it.
            
            # The detector expects a `service_statuses` dict. The data is a list of flat dicts.
            # I'll pass the flat dict directly to a modified predict method.
            # Let's assume for this evaluation, we can simulate the detector's logic.
            df = pd.DataFrame([experiment['metrics'][i]]).reindex(columns=detector.features, fill_value=0)
            prediction = detector.model.predict(df)
            is_anomaly = prediction and prediction[0] != 'normal'
            predictions.append(is_anomaly)

        ground_truth = convert_fault_periods_to_timesteps(experiment)
        metrics = calculate_metrics(predictions, ground_truth)
        all_results.append(metrics)
        print(f"Experiment {experiment.get('experiment_name', 'unknown')}: F1={metrics['f1_score']:.4f}")

    # Calculate overall statistics
    avg_precision = np.mean([r['precision'] for r in all_results])
    avg_recall = np.mean([r['recall'] for r in all_results])
    avg_f1 = np.mean([r['f1_score'] for r in all_results])

    print("\n=== Overall ML Detector Results ===")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average F1-Score: {avg_f1:.4f}")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'model_used': 'models/fault_detection_model.joblib',
        'overall_metrics': {
            'precision': avg_precision,
            'recall': avg_recall,
            'f1_score': avg_f1
        },
        'experiment_results': all_results
    }
    os.makedirs('results/ml_evaluation', exist_ok=True)
    with open('results/ml_evaluation/ml_detector_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("\nML evaluation results saved to: results/ml_evaluation/ml_detector_results.json")

if __name__ == "__main__":
    # A bit of a hack to make imports work from the scripts directory
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    import pandas as pd
    main() 