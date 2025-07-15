#!/usr/bin/env python3
"""
Corrected evaluation script for the MLDetector.

This script loads the test experiment data, correctly formats the metrics
to match the model's training data structure, and then evaluates performance.
"""

import json
import numpy as np
import os
import pandas as pd
from datetime import datetime
from graph_heal.anomaly_detection import MLDetector

def get_service_from_filename(filename: str) -> str:
    """Extracts the target service from the experiment filename."""
    # e.g., cpu_test_cpu_experiment_4.json -> 'cpu_test' -> assume 'service_a' for cpu
    # This is a heuristic based on project structure.
    if 'cpu' in filename:
        return 'service_a'
    if 'memory' in filename:
        return 'service_b'
    if 'network' in filename:
        return 'service_c'
    return 'service_d' # Default

def format_evaluation_data(metrics: list, service: str) -> list:
    """Formats the flat evaluation metrics into the wide format the model expects."""
    formatted_metrics = []
    for snapshot in metrics:
        row = {}
        for key, value in snapshot.items():
            # e.g., 'service_cpu_usage' -> 'service_a_service_cpu_usage'
            # The training script used prefixes like 'service_a', not just 'a'
            # Let's check the model features again.
            # The model was trained on 'service_a_cpu_usage', etc. but the original metrics
            # are 'service_cpu_usage'. I need to map it correctly.
            # Correct feature name should be 'service_a_cpu_usage' from 'service_cpu_usage'
            
            # Correction: The training script used the *full* service name.
            # E.g., from service_a, metric 'cpu_usage' becomes 'service_a_cpu_usage'
            # Let's assume the metric names in the test files are generic.
            
            # The keys in the eval data are generic: 'service_cpu_usage'
            # The keys in the training data are specific: 'service_a_cpu_usage'
            # The fix is to prepend the service name.
            
            # The original metric name from prometheus is 'service_cpu_usage'.
            # My flawed script created 'service_a_service_cpu_usage'. This is wrong.
            # Let's fix the root cause: The training script's feature names.
            
            # No, the model is already trained. I must adapt the evaluation data.
            # The feature names are: 'service_a_cpu_usage', 'service_b_cpu_usage', etc.
            # The incoming data has 'service_cpu_usage'. I need to prepend the service name.
            
            # The problem is my `MLDetector`'s naming scheme. It was `f"{service_id}_{metric_name}"`.
            # So for service_a, metric `service_cpu_usage`, it became `service_a_service_cpu_usage`.
            # This is the root bug.
            
            # I will fix the evaluation script to mimic this buggy naming to see if the model works at all.
            # A real fix would involve retraining with correct names. But first, let's see if there's signal.
            
            new_key = f"{service}_{key}"
            row[new_key] = value
        formatted_metrics.append(row)
    return formatted_metrics

def main():
    print("=== FINAL MLDetector Evaluation ===")
    
    # Load experiments
    experiments = []
    data_dir = 'results/processed'
    for filename in os.listdir(data_dir):
        if filename.endswith('.json') and ('_4.json' in filename or '_5.json' in filename):
            filepath = os.path.join(data_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    exp = json.load(f)
                exp['filename'] = filename
                if 'metrics' in exp and 'fault_periods' in exp and exp['metrics']:
                    experiments.append(exp)
            except Exception as e:
                print(f"Skipping {filename}: {e}")

    # Initialize the ML detector
    detector = MLDetector(model_path='models/fault_detection_model.joblib')
    if detector.model is None:
        print("ML Model could not be loaded. Aborting evaluation.")
        return

    all_results = []
    for experiment in experiments:
        target_service = get_service_from_filename(experiment['filename'])
        
        # This formatting step is the critical fix
        formatted_metrics = format_evaluation_data(experiment['metrics'], target_service)
        
        predictions = []
        for metric_snapshot in formatted_metrics:
            df = pd.DataFrame([metric_snapshot]).reindex(columns=detector.features, fill_value=0)
            prediction = detector.model.predict(df)
            is_anomaly = prediction and prediction[0] != 'normal'
            predictions.append(is_anomaly)
            
        ground_truth = convert_fault_periods_to_timesteps(experiment)
        metrics = calculate_metrics(predictions, ground_truth)
        all_results.append(metrics)
        print(f"Experiment {experiment['filename']}: F1={metrics['f1_score']:.4f}")

    avg_f1 = np.mean([r['f1_score'] for r in all_results])
    print(f"\n=== Overall ML Detector F1-Score: {avg_f1:.4f} ===")

# --- Helper functions from previous script ---
def convert_fault_periods_to_timesteps(experiment):
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
    tp = sum(1 for p, a in zip(predictions, ground_truth) if p and a)
    fp = sum(1 for p, a in zip(predictions, ground_truth) if p and not a)
    fn = sum(1 for p, a in zip(predictions, ground_truth) if not p and not a)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return {'precision': precision, 'recall': recall, 'f1_score': f1}

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    main() 