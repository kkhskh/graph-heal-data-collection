import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../graph-heal')))
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple
from graph_heal.graph_analysis import ServiceGraph
from graph_heal.anomaly_detection import StatisticalAnomalyDetector

class MetricsEvaluator:
    def __init__(self):
        self.graph_metrics = {
            'propagation': {
                'detection_accuracy': [],
                'delay_estimation_error': [],
                'cross_layer_detection': [],
                'cascading_failure_prediction': []
            },
            'localization': {
                'root_cause_accuracy': [],
                'localization_time': [],
                'false_root_cause_rate': [],
                'multi_hop_localization': []
            }
        }
        
        self.statistical_metrics = {
            'propagation': {
                'detection_accuracy': [],
                'delay_estimation_error': [],
                'cross_layer_detection': [],
                'cascading_failure_prediction': []
            },
            'localization': {
                'root_cause_accuracy': [],
                'localization_time': [],
                'false_root_cause_rate': [],
                'multi_hop_localization': []
            }
        }

    def calculate_propagation_metrics(self, 
                                   ground_truth: Dict,
                                   graph_predictions: Dict,
                                   statistical_predictions: Dict) -> Tuple[Dict, Dict]:
        """Calculate propagation-related metrics for both approaches"""
        
        # Propagation Detection Accuracy
        graph_accuracy = self._calculate_detection_accuracy(
            ground_truth['propagated_services'],
            graph_predictions['propagated_services']
        )
        stat_accuracy = self._calculate_detection_accuracy(
            ground_truth['propagated_services'],
            statistical_predictions['propagated_services']
        )
        
        # Propagation Delay Estimation Error
        graph_delay_error = self._calculate_delay_error(
            ground_truth['propagation_delays'],
            graph_predictions['propagation_delays']
        )
        stat_delay_error = self._calculate_delay_error(
            ground_truth['propagation_delays'],
            statistical_predictions['propagation_delays']
        )
        
        # Cross-layer Fault Detection
        graph_cross_layer = self._calculate_cross_layer_detection(
            ground_truth['cross_layer_faults'],
            graph_predictions['cross_layer_faults']
        )
        stat_cross_layer = self._calculate_cross_layer_detection(
            ground_truth['cross_layer_faults'],
            statistical_predictions['cross_layer_faults']
        )
        
        # Cascading Failure Prediction
        graph_cascade = self._calculate_cascade_prediction(
            ground_truth['cascading_failures'],
            graph_predictions['cascading_failures']
        )
        stat_cascade = self._calculate_cascade_prediction(
            ground_truth['cascading_failures'],
            statistical_predictions['cascading_failures']
        )
        
        return (
            {
                'detection_accuracy': graph_accuracy,
                'delay_estimation_error': graph_delay_error,
                'cross_layer_detection': graph_cross_layer,
                'cascading_failure_prediction': graph_cascade
            },
            {
                'detection_accuracy': stat_accuracy,
                'delay_estimation_error': stat_delay_error,
                'cross_layer_detection': stat_cross_layer,
                'cascading_failure_prediction': stat_cascade
            }
        )

    def calculate_localization_metrics(self,
                                    ground_truth: Dict,
                                    graph_predictions: Dict,
                                    statistical_predictions: Dict) -> Tuple[Dict, Dict]:
        """Calculate localization-related metrics for both approaches"""
        
        # Root Cause Accuracy
        graph_accuracy = self._calculate_root_cause_accuracy(
            ground_truth['root_causes'],
            graph_predictions['root_causes']
        )
        stat_accuracy = self._calculate_root_cause_accuracy(
            ground_truth['root_causes'],
            statistical_predictions['root_causes']
        )
        
        # Localization Time
        graph_time = self._calculate_localization_time(
            ground_truth['detection_times'],
            graph_predictions['localization_times']
        )
        stat_time = self._calculate_localization_time(
            ground_truth['detection_times'],
            statistical_predictions['localization_times']
        )
        
        # False Root Cause Rate
        graph_false_rate = self._calculate_false_root_cause_rate(
            ground_truth['root_causes'],
            graph_predictions['root_causes']
        )
        stat_false_rate = self._calculate_false_root_cause_rate(
            ground_truth['root_causes'],
            statistical_predictions['root_causes']
        )
        
        # Multi-hop Localization
        graph_multi_hop = self._calculate_multi_hop_localization(
            ground_truth['multi_hop_paths'],
            graph_predictions['multi_hop_paths']
        )
        stat_multi_hop = self._calculate_multi_hop_localization(
            ground_truth['multi_hop_paths'],
            statistical_predictions['multi_hop_paths']
        )
        
        return (
            {
                'root_cause_accuracy': graph_accuracy,
                'localization_time': graph_time,
                'false_root_cause_rate': graph_false_rate,
                'multi_hop_localization': graph_multi_hop
            },
            {
                'root_cause_accuracy': stat_accuracy,
                'localization_time': stat_time,
                'false_root_cause_rate': stat_false_rate,
                'multi_hop_localization': stat_multi_hop
            }
        )

    def _calculate_detection_accuracy(self, ground_truth: List[str], predictions: List[str]) -> float:
        """Calculate accuracy of propagation detection"""
        true_positives = len(set(ground_truth) & set(predictions))
        false_negatives = len(set(ground_truth) - set(predictions))
        return true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0

    def _calculate_delay_error(self, ground_truth: Dict[str, float], predictions: Dict[str, float]) -> float:
        """Calculate mean absolute error of delay estimation"""
        errors = []
        for service in ground_truth:
            if service in predictions:
                errors.append(abs(ground_truth[service] - predictions[service]))
        return np.mean(errors) if errors else float('inf')

    def _calculate_cross_layer_detection(self, ground_truth: List[str], predictions: List[str]) -> float:
        """Calculate cross-layer fault detection rate"""
        true_positives = len(set(ground_truth) & set(predictions))
        return true_positives / len(ground_truth) if ground_truth else 0.0

    def _calculate_cascade_prediction(self, ground_truth: List[str], predictions: List[str]) -> float:
        """Calculate cascading failure prediction rate"""
        true_positives = len(set(ground_truth) & set(predictions))
        return true_positives / len(ground_truth) if ground_truth else 0.0

    def _calculate_root_cause_accuracy(self, ground_truth: List[str], predictions: List[str]) -> float:
        """Calculate root cause localization accuracy"""
        correct = sum(1 for gt, pred in zip(ground_truth, predictions) if gt == pred)
        return correct / len(ground_truth) if ground_truth else 0.0

    def _calculate_localization_time(self, detection_times: List[float], localization_times: List[float]) -> float:
        """Calculate average localization time"""
        times = [loc - det for det, loc in zip(detection_times, localization_times)]
        return np.mean(times) if times else float('inf')

    def _calculate_false_root_cause_rate(self, ground_truth: List[str], predictions: List[str]) -> float:
        """Calculate false root cause rate"""
        false_positives = sum(1 for gt, pred in zip(ground_truth, predictions) if gt != pred)
        return false_positives / len(ground_truth) if ground_truth else 0.0

    def _calculate_multi_hop_localization(self, ground_truth: List[List[str]], predictions: List[List[str]]) -> float:
        """Calculate multi-hop localization accuracy"""
        correct_paths = sum(1 for gt, pred in zip(ground_truth, predictions) if gt == pred)
        return correct_paths / len(ground_truth) if ground_truth else 0.0

    def plot_metrics_comparison(self, save_path: str = 'results/metrics_comparison.png'):
        """Plot comparison of metrics between graph-based and statistical approaches"""
        metrics = ['detection_accuracy', 'delay_estimation_error', 'cross_layer_detection', 
                  'cascading_failure_prediction', 'root_cause_accuracy', 'localization_time',
                  'false_root_cause_rate', 'multi_hop_localization']
        
        graph_values = []
        stat_values = []
        
        for metric in metrics:
            if metric in self.graph_metrics['propagation']:
                graph_values.append(np.mean(self.graph_metrics['propagation'][metric]))
                stat_values.append(np.mean(self.statistical_metrics['propagation'][metric]))
            else:
                graph_values.append(np.mean(self.graph_metrics['localization'][metric]))
                stat_values.append(np.mean(self.statistical_metrics['localization'][metric]))
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'Metric': metrics * 2,
            'Value': graph_values + stat_values,
            'Approach': ['Graph-based'] * len(metrics) + ['Statistical'] * len(metrics)
        })
        
        # Plot
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x='Metric', y='Value', hue='Approach')
        plt.xticks(rotation=45)
        plt.title('Comparison of Graph-based vs Statistical Approaches')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

def main():
    # Initialize evaluator
    evaluator = MetricsEvaluator()
    
    # Load experiment results - use available files
    experiment_files = [
        'results/processed/cpu_test_cpu_experiment_3.json',
        'results/processed/cpu_test_cpu_experiment_4.json',
        'results/processed/cpu_test_cpu_experiment_5.json',
        'results/processed/memory_test_memory_experiment_4.json'
    ]
    
    for exp_file in experiment_files:
        try:
            with open(exp_file, 'r') as f:
                data = json.load(f)
                
            # Check if this experiment has the required structure
            if 'ground_truth' not in data or 'graph_predictions' not in data or 'statistical_predictions' not in data:
                print(f"Skipping {exp_file}: missing required prediction data")
                continue
                
            # Calculate metrics for this experiment
            graph_prop_metrics, stat_prop_metrics = evaluator.calculate_propagation_metrics(
                data['ground_truth'],
                data['graph_predictions'],
                data['statistical_predictions']
            )
            
            graph_loc_metrics, stat_loc_metrics = evaluator.calculate_localization_metrics(
                data['ground_truth'],
                data['graph_predictions'],
                data['statistical_predictions']
            )
            
            # Store metrics
            for metric, value in graph_prop_metrics.items():
                evaluator.graph_metrics['propagation'][metric].append(value)
            for metric, value in stat_prop_metrics.items():
                evaluator.statistical_metrics['propagation'][metric].append(value)
            for metric, value in graph_loc_metrics.items():
                evaluator.graph_metrics['localization'][metric].append(value)
            for metric, value in stat_loc_metrics.items():
                evaluator.statistical_metrics['localization'][metric].append(value)
                
        except Exception as e:
            print(f"Error processing {exp_file}: {e}")
            continue
    
    # Plot comparison
    evaluator.plot_metrics_comparison()
    
    # Print summary
    print("\nMetrics Summary:")
    print("===============")
    print("\nPropagation Metrics:")
    for metric in evaluator.graph_metrics['propagation']:
        if evaluator.graph_metrics['propagation'][metric]:  # Check if we have data
            graph_mean = np.mean(evaluator.graph_metrics['propagation'][metric])
            stat_mean = np.mean(evaluator.statistical_metrics['propagation'][metric])
            print(f"{metric}:")
            print(f"  Graph-based: {graph_mean:.3f}")
            print(f"  Statistical: {stat_mean:.3f}")
            if stat_mean != 0:
                print(f"  Improvement: {(graph_mean - stat_mean) / stat_mean * 100:.1f}%")
            else:
                print(f"  Improvement: N/A (statistical mean is 0)")
    
    print("\nLocalization Metrics:")
    for metric in evaluator.graph_metrics['localization']:
        if evaluator.graph_metrics['localization'][metric]:  # Check if we have data
            graph_mean = np.mean(evaluator.graph_metrics['localization'][metric])
            stat_mean = np.mean(evaluator.statistical_metrics['localization'][metric])
            print(f"{metric}:")
            print(f"  Graph-based: {graph_mean:.3f}")
            print(f"  Statistical: {stat_mean:.3f}")
            if stat_mean != 0:
                print(f"  Improvement: {(graph_mean - stat_mean) / stat_mean * 100:.1f}%")
            else:
                print(f"  Improvement: N/A (statistical mean is 0)")

if __name__ == "__main__":
    main() 