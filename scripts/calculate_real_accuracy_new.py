import json
import re
from datetime import datetime
import numpy as np
from typing import Dict, List, Tuple

def parse_metrics(metrics_text: str) -> Dict[str, float]:
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

def load_experiment_data(filename: str) -> Tuple[List[datetime], List[Dict[str, float]]]:
    """Load experiment data from JSON file and parse timestamps and metrics."""
    with open(filename, 'r') as f:
        data = json.load(f)
    
    timestamps = []
    metrics_list = []
    
    for entry in data['metrics']:
        timestamp = datetime.fromisoformat(entry['timestamp'])
        metrics = parse_metrics(entry['metrics'])
        timestamps.append(timestamp)
        metrics_list.append(metrics)
    
    return timestamps, metrics_list

def define_ground_truth_faults() -> Dict[str, Tuple[datetime, datetime]]:
    """Define when faults occurred in the experiments."""
    return {
        'cpu_experiment': (
            datetime.fromisoformat('2025-05-30T19:25:54.419259'),
            datetime.fromisoformat('2025-05-30T19:30:54.419259')
        ),
        'memory_experiment': (
            datetime.fromisoformat('2025-05-30T19:25:54.419259'),
            datetime.fromisoformat('2025-05-30T19:28:54.419259')
        ),
        'network_experiment': (
            datetime.fromisoformat('2025-05-30T19:25:54.419259'),
            datetime.fromisoformat('2025-05-30T19:29:54.419259')
        )
    }

def threshold_based_detection(timestamps: List[datetime], metrics_list: List[Dict[str, float]], fault_type: str) -> List[bool]:
    """Implement threshold-based detection method."""
    detections = []
    
    # Define thresholds based on fault type
    thresholds = {
        'cpu': {'service_cpu_usage': 70.0},
        'memory': {'service_memory_usage': 400.0},
        'network': {'service_response_time': 150.0}
    }
    
    for metrics in metrics_list:
        detected = False
        for metric_name, threshold in thresholds[fault_type].items():
            if metric_name in metrics and metrics[metric_name] > threshold:
                detected = True
                break
        detections.append(detected)
    
    return detections

def graph_heal_detection(timestamps: List[datetime], metrics_list: List[Dict[str, float]], fault_type: str) -> List[bool]:
    """Implement GRAPH-HEAL detection method using z-scores."""
    detections = []
    
    # Define metrics to monitor based on fault type
    metrics_to_monitor = {
        'cpu': ['service_cpu_usage'],
        'memory': ['service_memory_usage'],
        'network': ['service_response_time']
    }
    
    # Calculate z-scores for each metric
    for metric_name in metrics_to_monitor[fault_type]:
        values = [m.get(metric_name, 0) for m in metrics_list]
        mean = np.mean(values)
        std = np.std(values)
        if std == 0:
            continue
        
        z_scores = [(v - mean) / std for v in values]
        detections = [z > 2.0 for z in z_scores]  # Detect anomalies beyond 2 standard deviations
    
    return detections

def calculate_accuracy_metrics(detections: List[bool], ground_truth: Tuple[datetime, datetime], timestamps: List[datetime]) -> Dict[str, float]:
    """Calculate accuracy metrics including precision, recall, and F1-score."""
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for i, (timestamp, detected) in enumerate(zip(timestamps, detections)):
        is_fault_period = ground_truth[0] <= timestamp <= ground_truth[1]
        
        if detected and is_fault_period:
            true_positives += 1
        elif detected and not is_fault_period:
            false_positives += 1
        elif not detected and is_fault_period:
            false_negatives += 1
    
    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (true_positives + (len(timestamps) - true_positives - false_positives - false_negatives)) / len(timestamps)
    
    return {
        'precision': precision * 100,
        'recall': recall * 100,
        'f1_score': f1_score * 100,
        'accuracy': accuracy * 100
    }

def main():
    """Main function to process experiment data and calculate accuracy metrics."""
    experiment_files = [
        'results/cpu_experiment.json',
        'results/memory_experiment.json',
        'results/network_experiment.json'
    ]
    
    ground_truth = define_ground_truth_faults()
    results = {}
    
    for filename in experiment_files:
        experiment_name = filename.split('/')[-1].replace('.json', '')
        fault_type = experiment_name.split('_')[0]
        
        print(f"\nProcessing {experiment_name}...")
        
        # Load and process data
        timestamps, metrics_list = load_experiment_data(filename)
        
        # Run detection methods
        threshold_detections = threshold_based_detection(timestamps, metrics_list, fault_type)
        graph_heal_detections = graph_heal_detection(timestamps, metrics_list, fault_type)
        
        # Calculate metrics
        threshold_metrics = calculate_accuracy_metrics(threshold_detections, ground_truth[experiment_name], timestamps)
        graph_heal_metrics = calculate_accuracy_metrics(graph_heal_detections, ground_truth[experiment_name], timestamps)
        
        # Store results
        results[experiment_name] = {
            'threshold_based': threshold_metrics,
            'graph_heal': graph_heal_metrics
        }
        
        # Print results
        print(f"\nThreshold-Based Detection:")
        print(f"Accuracy: {threshold_metrics['accuracy']:.1f}%")
        print(f"Precision: {threshold_metrics['precision']:.1f}%")
        print(f"Recall: {threshold_metrics['recall']:.1f}%")
        print(f"F1-Score: {threshold_metrics['f1_score']:.1f}%")
        
        print(f"\nGRAPH-HEAL Detection:")
        print(f"Accuracy: {graph_heal_metrics['accuracy']:.1f}%")
        print(f"Precision: {graph_heal_metrics['precision']:.1f}%")
        print(f"Recall: {graph_heal_metrics['recall']:.1f}%")
        print(f"F1-Score: {graph_heal_metrics['f1_score']:.1f}%")
    
    # Save results to file
    with open('results/detection_accuracy_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to results/detection_accuracy_results.json")

if __name__ == '__main__':
    main() 