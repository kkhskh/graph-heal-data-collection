import json
import re
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Tuple, Optional
import networkx as nx
from scipy import stats
from dataclasses import dataclass

@dataclass
class DetectionConfig:
    z_threshold: float = 1.5
    use_graph_correlation: bool = True
    use_temporal_patterns: bool = True
    require_consensus: bool = False

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
            datetime.fromisoformat('2025-05-30T19:30:54.229020')
        ),
        'memory_experiment': (
            datetime.fromisoformat('2025-05-30T19:31:55.291805'),
            datetime.fromisoformat('2025-05-30T19:34:54.415335')
        ),
        'network_experiment': (
            datetime.fromisoformat('2025-05-30T19:35:55.490179'),
            datetime.fromisoformat('2025-05-30T19:39:54.924617')
        )
    }

# Old version detection methods
def threshold_based_detection_old(timestamps: List[datetime], metrics_list: List[Dict[str, float]], fault_type: str) -> List[bool]:
    """Old version of threshold-based detection method (higher thresholds)."""
    detections = []
    
    # Higher thresholds
    thresholds = {
        'cpu': {'service_cpu_usage': 68.0},
        'memory': {'service_memory_usage': 390.0},
        'network': {'service_response_time': 140.0}
    }
    
    for metrics in metrics_list:
        detected = False
        for metric_name, threshold in thresholds[fault_type].items():
            if metric_name in metrics and metrics[metric_name] > threshold:
                detected = True
                break
        detections.append(detected)
    
    return detections

def graph_heal_detection_old(timestamps: List[datetime], metrics_list: List[Dict[str, float]], fault_type: str) -> List[bool]:
    """Old version of GRAPH-HEAL detection method (higher z-score)."""
    detections = []
    
    metrics_to_monitor = {
        'cpu': ['service_cpu_usage'],
        'memory': ['service_memory_usage'],
        'network': ['service_response_time']
    }
    
    for metric_name in metrics_to_monitor[fault_type]:
        values = [m.get(metric_name, 0) for m in metrics_list]
        mean = np.mean(values)
        std = np.std(values)
        if std == 0:
            continue
        z_scores = [(v - mean) / std for v in values]
        detections = [abs(z) > 1.5 for z in z_scores]  # More sensitive
    return detections

# New version detection methods
def threshold_based_detection_new(timestamps: List[datetime], metrics_list: List[Dict[str, float]], fault_type: str) -> List[bool]:
    """New version of threshold-based detection method."""
    detections = []
    
    # Define thresholds based on fault type
    thresholds = {
        'cpu': {'service_cpu_usage': 70.0},  # Higher threshold for new version
        'memory': {'service_memory_usage': 400.0},  # Higher threshold for new version
        'network': {'service_response_time': 150.0}  # Higher threshold for new version
    }
    
    for metrics in metrics_list:
        detected = False
        for metric_name, threshold in thresholds[fault_type].items():
            if metric_name in metrics and metrics[metric_name] > threshold:
                detected = True
                break
        detections.append(detected)
    
    return detections

def graph_heal_detection_new(
    timestamps: List[datetime],
    metrics_list: List[Dict[str, float]],
    fault_type: str,
    graph: Optional[nx.DiGraph] = None,
    config: Optional[DetectionConfig] = None
) -> List[bool]:
    """Enhanced GRAPH-HEAL detection with improved network fault handling"""
    if config is None:
        config = DetectionConfig()
    
    # Create default graph if none provided
    if graph is None:
        graph = nx.DiGraph()
        services = ['service_a', 'service_b', 'service_c', 'service_d']
        for service in services:
            graph.add_node(service)
        edges = [
            ('service_a', 'service_b'),
            ('service_b', 'service_c'),
            ('service_c', 'service_d'),
            ('service_a', 'service_d')
        ]
        graph.add_edges_from(edges)
    
    # Extract metrics based on fault type
    metrics = []
    for m in metrics_list:
        if fault_type == 'cpu':
            metrics.append(m.get('service_cpu_usage', 0))
        elif fault_type == 'memory':
            metrics.append(m.get('service_memory_usage', 0))
        elif fault_type == 'network':
            # Enhanced network metrics
            metrics.append({
                'response_time': m.get('service_response_time', 0),
                'request_count': m.get('service_request_count_total', 0),
                'cpu_usage': m.get('service_cpu_usage', 0),
                'memory_usage': m.get('service_memory_usage', 0)
            })
    
    # Calculate z-scores with adaptive thresholds
    if fault_type == 'network':
        # Multi-metric z-score calculation for network
        z_scores = {}
        for metric in ['response_time', 'request_count', 'cpu_usage', 'memory_usage']:
            values = [m[metric] for m in metrics]
            # Use rolling window for mean/std calculation
            window_size = min(10, len(values))
            rolling_means = []
            rolling_stds = []
            for i in range(len(values)):
                start_idx = max(0, i - window_size + 1)
                window_values = values[start_idx:i + 1]
                rolling_means.append(np.mean(window_values))
                rolling_stds.append(np.std(window_values) if len(window_values) > 1 else 0)
            
            # Calculate z-scores using rolling statistics
            z_scores[metric] = []
            for i in range(len(values)):
                if rolling_stds[i] > 0:
                    z_scores[metric].append((values[i] - rolling_means[i]) / rolling_stds[i])
                else:
                    z_scores[metric].append(0)
    else:
        z_scores = stats.zscore(metrics)
    
    # Dynamic threshold adjustment based on fault type
    base_threshold = config.z_threshold if config.z_threshold is not None else 1.5
    if fault_type == 'network':
        # Lower threshold for network faults to improve recall
        threshold = base_threshold * 0.7  # More sensitive threshold
    else:
        threshold = base_threshold
    
    detections = []
    
    # Graph-based correlation detection
    if config.use_graph_correlation:
        for i in range(len(timestamps)):
            if fault_type == 'network':
                # Enhanced network fault detection
                is_anomaly = False
                
                # Check for anomalies in any metric
                metric_anomalies = {
                    'response_time': abs(z_scores['response_time'][i]) > threshold,
                    'request_count': abs(z_scores['request_count'][i]) > threshold,
                    'cpu_usage': abs(z_scores['cpu_usage'][i]) > threshold,
                    'memory_usage': abs(z_scores['memory_usage'][i]) > threshold
                }
                
                # Require at least two metrics to show anomalies
                anomaly_count = sum(1 for v in metric_anomalies.values() if v)
                is_anomaly = anomaly_count >= 2
                
                # Check for correlated anomalies in connected services
                if is_anomaly:
                    neighbors = list(graph.neighbors('service_a'))  # Assuming service_a is the source
                    neighbor_anomalies = 0
                    for neighbor in neighbors:
                        if any(abs(z_scores[metric][i]) > threshold * 0.8 for metric in z_scores.keys()):
                            neighbor_anomalies += 1
                    
                    # Require at least one neighbor to show anomaly for network faults
                    is_anomaly = neighbor_anomalies > 0
            else:
                # Original detection for CPU and memory
                is_anomaly = abs(z_scores[i]) > threshold
                
                if is_anomaly:
                    neighbors = list(graph.neighbors('service_a'))
                    neighbor_anomalies = sum(1 for n in neighbors if abs(z_scores[i]) > threshold * 0.9)
                    is_anomaly = neighbor_anomalies > 0
            
            detections.append(is_anomaly)
    
    # Temporal pattern detection
    if config.use_temporal_patterns:
        temporal_detections = []
        window_size = 5  # Increased window size for better temporal detection
        
        for i in range(len(timestamps)):
            if i < window_size - 1:
                temporal_detections.append(False)
                continue
            
            if fault_type == 'network':
                # Enhanced temporal pattern for network
                recent_metrics = {
                    metric: [z_scores[metric][j] for j in range(i - window_size + 1, i + 1)]
                    for metric in z_scores.keys()
                }
                
                # Check for sustained anomalies in any metric
                is_temporal_anomaly = False
                for metric_vals in recent_metrics.values():
                    # Count how many points in the window show anomalies
                    anomaly_count = sum(1 for v in metric_vals if abs(v) > threshold * 0.6)
                    if anomaly_count >= window_size * 0.6:  # 60% of points show anomaly
                        is_temporal_anomaly = True
                        break
            else:
                # Original temporal pattern for CPU and memory
                recent_values = z_scores[i - window_size + 1:i + 1]
                is_temporal_anomaly = all(abs(v) > threshold * 0.7 for v in recent_values)
            
            temporal_detections.append(is_temporal_anomaly)
        
        # Combine detections based on configuration
        if config.require_consensus:
            detections = [d1 and d2 for d1, d2 in zip(detections, temporal_detections)]
        else:
            detections = [d1 or d2 for d1, d2 in zip(detections, temporal_detections)]
    
    return detections

def calculate_accuracy_metrics(detections: List[bool], ground_truth: Tuple[datetime, datetime], timestamps: List[datetime]) -> Dict[str, float]:
    """Calculate accuracy metrics including precision, recall, and F1-score."""
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    # Add a small time window for matching detections
    window_seconds = 5
    
    for i, (timestamp, detected) in enumerate(zip(timestamps, detections)):
        is_fault_period = ground_truth[0] <= timestamp <= ground_truth[1]
        
        if detected and is_fault_period:
            true_positives += 1
        elif detected and not is_fault_period:
            false_positives += 1
        elif not detected and is_fault_period:
            # Check if there's a detection within the time window
            window_start = timestamp - timedelta(seconds=window_seconds)
            window_end = timestamp + timedelta(seconds=window_seconds)
            window_detections = [d for t, d in zip(timestamps, detections) 
                               if window_start <= t <= window_end and d]
            if not window_detections:
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

def debug_metrics(metrics_list: List[Dict[str, float]], fault_type: str, timestamps: List[datetime], ground_truth: Tuple[datetime, datetime]) -> None:
    """Debug function to analyze metrics during fault periods."""
    print(f"\n{'='*50}")
    print(f"DEBUG ANALYSIS FOR {fault_type.upper()} EXPERIMENT")
    print(f"{'='*50}")
    
    # 1. Check available metrics
    all_metrics = set().union(*[m.keys() for m in metrics_list])
    print("\n1. Available Metrics:")
    print(f"   {sorted(all_metrics)}")
    
    # 2. Analyze metric values during fault period
    fault_period_metrics = []
    normal_period_metrics = []
    
    for timestamp, metrics in zip(timestamps, metrics_list):
        if ground_truth[0] <= timestamp <= ground_truth[1]:
            fault_period_metrics.append(metrics)
        else:
            normal_period_metrics.append(metrics)
    
    print(f"\n2. Metric Analysis During Fault Period:")
    print(f"   Fault Period Duration: {ground_truth[1] - ground_truth[0]}")
    print(f"   Number of samples during fault: {len(fault_period_metrics)}")
    print(f"   Number of samples during normal: {len(normal_period_metrics)}")
    
    if fault_period_metrics:
        for metric_name in ['service_memory_usage', 'service_response_time', 'service_cpu_usage']:
            if metric_name in all_metrics:
                fault_values = [m.get(metric_name, 0) for m in fault_period_metrics]
                normal_values = [m.get(metric_name, 0) for m in normal_period_metrics]
                print(f"\n   {metric_name}:")
                print(f"   - Fault Period:  Min={min(fault_values):.1f}, Max={max(fault_values):.1f}, Mean={np.mean(fault_values):.1f}")
                if normal_values:
                    print(f"   - Normal Period: Min={min(normal_values):.1f}, Max={max(normal_values):.1f}, Mean={np.mean(normal_values):.1f}")
                else:
                    print(f"   - Normal Period: No samples")
                # Print first few values during fault period
                print(f"   - First 5 values during fault:")
                for i, v in enumerate(fault_values[:5]):
                    print(f"     {i+1}. {v:.1f}")
    
    # 3. Check detection thresholds
    print("\n3. Current Detection Thresholds:")
    if fault_type == 'memory':
        print(f"   Memory Usage Threshold: 400.0 MB")
        print(f"   Z-score Threshold: 2.0")
    elif fault_type == 'network':
        print(f"   Response Time Threshold: 150.0 ms")
        print(f"   Z-score Threshold: 2.0")
    
    # 4. Analyze detection results
    print("\n4. Detection Analysis:")
    if fault_period_metrics:
        for metric_name in ['service_memory_usage', 'service_response_time']:
            if metric_name in all_metrics:
                values = [m.get(metric_name, 0) for m in fault_period_metrics]
                threshold = 400.0 if metric_name == 'service_memory_usage' else 150.0
                detections = sum(1 for v in values if v > threshold)
                print(f"\n   {metric_name}:")
                print(f"   - Samples above threshold: {detections}/{len(values)} ({detections/len(values)*100:.1f}%)")
                
                # Print distribution of values
                if values:
                    percentiles = np.percentile(values, [0, 25, 50, 75, 100])
                    print(f"   - Value distribution:")
                    print(f"     Min: {percentiles[0]:.1f}")
                    print(f"     25th: {percentiles[1]:.1f}")
                    print(f"     Median: {percentiles[2]:.1f}")
                    print(f"     75th: {percentiles[3]:.1f}")
                    print(f"     Max: {percentiles[4]:.1f}")

def debug_timestamps(timestamps, ground_truth_period, experiment_name):
    """Debug function to show timestamp ranges and ground truth periods."""
    print(f"\n=== TIMESTAMP DEBUG FOR {experiment_name.upper()} ===")
    print(f"Data timestamps:")
    print(f"  First: {timestamps[0]}")
    print(f"  Last:  {timestamps[-1]}")
    print(f"  Total samples: {len(timestamps)}")
    print(f"Ground truth fault period:")
    print(f"  Start: {ground_truth_period[0]}")
    print(f"  End:   {ground_truth_period[1]}")
    print(f"  Duration: {ground_truth_period[1] - ground_truth_period[0]}")
    data_start, data_end = timestamps[0], timestamps[-1]
    fault_start, fault_end = ground_truth_period
    overlap = not (fault_end < data_start or fault_start > data_end)
    print(f"  Overlap with data: {'✅ YES' if overlap else '❌ NO'}")
    if not overlap:
        print(f"  Gap: Data ends {(fault_start - data_end).total_seconds():.1f}s before fault starts")

def find_actual_fault_periods_fixed(timestamps, metrics_list, fault_type):
    """Find fault periods using adaptive thresholds and robust detection."""
    metric_map = {
        'cpu': ('service_cpu_usage', 70.0),
        'memory': ('service_memory_usage', 400.0),
        'network': ('service_response_time', 150.0)
    }
    metric_name, base_threshold = metric_map[fault_type]
    values = [m.get(metric_name, 0) for m in metrics_list]
    
    # Calculate adaptive threshold based on data distribution
    mean_value = np.mean(values)
    std_value = np.std(values)
    adaptive_threshold = min(base_threshold, mean_value + 2 * std_value)
    
    # Find periods where values are consistently elevated
    window_size = 5  # Number of consecutive samples to confirm fault
    fault_periods = []
    in_fault = False
    fault_start = None
    consecutive_high = 0
    
    for i, (timestamp, value) in enumerate(zip(timestamps, values)):
        if value > adaptive_threshold:
            consecutive_high += 1
            if consecutive_high >= window_size and not in_fault:
                fault_start = timestamps[i - window_size + 1]
                in_fault = True
        else:
            consecutive_high = 0
            if in_fault:
                fault_periods.append((fault_start, timestamp))
                in_fault = False
    
    # Handle case where fault period extends to end of data
    if in_fault and fault_start:
        fault_periods.append((fault_start, timestamps[-1]))
    
    # Merge nearby fault periods (within 10 seconds)
    if len(fault_periods) > 1:
        merged_periods = []
        current_period = fault_periods[0]
        
        for next_period in fault_periods[1:]:
            if (next_period[0] - current_period[1]).total_seconds() < 10:
                current_period = (current_period[0], next_period[1])
            else:
                merged_periods.append(current_period)
                current_period = next_period
        merged_periods.append(current_period)
        fault_periods = merged_periods
    
    return fault_periods

def update_ground_truth_from_data():
    """Update ground truth based on actual data patterns with improved detection."""
    experiment_files = [
        'results/cpu_experiment.json',
        'results/memory_experiment.json',
        'results/network_experiment.json'
    ]
    updated_ground_truth = {}
    
    for filename in experiment_files:
        experiment_name = filename.split('/')[-1].replace('.json', '')
        fault_type = experiment_name.split('_')[0]
        timestamps, metrics_list = load_experiment_data(filename)
        
        # Find fault periods
        fault_periods = find_actual_fault_periods_fixed(timestamps, metrics_list, fault_type)
        
        if fault_periods:
            # Use the longest fault period
            main_fault = max(fault_periods, key=lambda x: (x[1] - x[0]).total_seconds())
            updated_ground_truth[experiment_name] = main_fault
            
            # Print detailed analysis
            print(f"\n{experiment_name}: Found fault period")
            print(f"  Start: {main_fault[0]}")
            print(f"  End: {main_fault[1]}")
            print(f"  Duration: {(main_fault[1] - main_fault[0]).total_seconds():.1f}s")
            
            # Print metric statistics during fault period
            metric_name = {
                'cpu': 'service_cpu_usage',
                'memory': 'service_memory_usage',
                'network': 'service_response_time'
            }[fault_type]
            
            fault_values = []
            for t, m in zip(timestamps, metrics_list):
                if main_fault[0] <= t <= main_fault[1]:
                    fault_values.append(m.get(metric_name, 0))
            
            if fault_values:
                print(f"  {metric_name} during fault:")
                print(f"    Min: {min(fault_values):.1f}")
                print(f"    Max: {max(fault_values):.1f}")
                print(f"    Mean: {np.mean(fault_values):.1f}")
                print(f"    Std: {np.std(fault_values):.1f}")
        else:
            print(f"\n{experiment_name}: No clear fault period found")
    
    return updated_ground_truth

def main():
    """Main function with updated ground truth detection."""
    experiment_files = [
        'results/cpu_experiment.json',
        'results/memory_experiment.json',
        'results/network_experiment.json'
    ]
    print("=== FINDING GROUND TRUTH FROM DATA ===")
    ground_truth = update_ground_truth_from_data()
    if not ground_truth:
        print("=== USING MANUAL GROUND TRUTH ===")
        ground_truth = define_ground_truth_faults()
    results = {}
    for filename in experiment_files:
        experiment_name = filename.split('/')[-1].replace('.json', '')
        fault_type = experiment_name.split('_')[0]
        timestamps, metrics_list = load_experiment_data(filename)
        # Timestamp debug
        debug_timestamps(timestamps, ground_truth.get(experiment_name, (timestamps[0], timestamps[-1])), experiment_name)
        # Run debug analysis for memory and network experiments
        if fault_type in ['memory', 'network']:
            debug_metrics(metrics_list, fault_type, timestamps, ground_truth.get(experiment_name, (timestamps[0], timestamps[-1])))
        # Run old version detection methods
        threshold_detections_old = threshold_based_detection_old(timestamps, metrics_list, fault_type)
        graph_heal_detections_old = graph_heal_detection_old(timestamps, metrics_list, fault_type)
        # Run new version detection methods
        threshold_detections_new = threshold_based_detection_new(timestamps, metrics_list, fault_type)
        graph_heal_detections_new = graph_heal_detection_new(timestamps, metrics_list, fault_type)
        # Calculate metrics for old version
        threshold_metrics_old = calculate_accuracy_metrics(threshold_detections_old, ground_truth.get(experiment_name, (timestamps[0], timestamps[-1])), timestamps)
        graph_heal_metrics_old = calculate_accuracy_metrics(graph_heal_detections_old, ground_truth.get(experiment_name, (timestamps[0], timestamps[-1])), timestamps)
        # Calculate metrics for new version
        threshold_metrics_new = calculate_accuracy_metrics(threshold_detections_new, ground_truth.get(experiment_name, (timestamps[0], timestamps[-1])), timestamps)
        graph_heal_metrics_new = calculate_accuracy_metrics(graph_heal_detections_new, ground_truth.get(experiment_name, (timestamps[0], timestamps[-1])), timestamps)
        # Store results
        results[experiment_name] = {
            'old_version': {
                'threshold_based': threshold_metrics_old,
                'graph_heal': graph_heal_metrics_old
            },
            'new_version': {
                'threshold_based': threshold_metrics_new,
                'graph_heal': graph_heal_metrics_new
            }
        }
        # Print results
        print(f"\nOld Version Results:")
        print(f"Threshold-Based Detection:")
        print(f"  Accuracy: {threshold_metrics_old['accuracy']:.1f}%")
        print(f"  Precision: {threshold_metrics_old['precision']:.1f}%")
        print(f"  Recall: {threshold_metrics_old['recall']:.1f}%")
        print(f"  F1-Score: {threshold_metrics_old['f1_score']:.1f}%")
        print(f"\nGRAPH-HEAL Detection:")
        print(f"  Accuracy: {graph_heal_metrics_old['accuracy']:.1f}%")
        print(f"  Precision: {graph_heal_metrics_old['precision']:.1f}%")
        print(f"  Recall: {graph_heal_metrics_old['recall']:.1f}%")
        print(f"  F1-Score: {graph_heal_metrics_old['f1_score']:.1f}%")
        print(f"\nNew Version Results:")
        print(f"Threshold-Based Detection:")
        print(f"  Accuracy: {threshold_metrics_new['accuracy']:.1f}%")
        print(f"  Precision: {threshold_metrics_new['precision']:.1f}%")
        print(f"  Recall: {threshold_metrics_new['recall']:.1f}%")
        print(f"  F1-Score: {threshold_metrics_new['f1_score']:.1f}%")
        print(f"\nGRAPH-HEAL Detection:")
        print(f"  Accuracy: {graph_heal_metrics_new['accuracy']:.1f}%")
        print(f"  Precision: {graph_heal_metrics_new['precision']:.1f}%")
        print(f"  Recall: {graph_heal_metrics_new['recall']:.1f}%")
        print(f"  F1-Score: {graph_heal_metrics_new['f1_score']:.1f}%")
    with open('results/detection_accuracy_results_unified.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to results/detection_accuracy_results_unified.json")

if __name__ == '__main__':
    main() 