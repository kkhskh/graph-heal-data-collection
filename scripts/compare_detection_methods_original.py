#!/usr/bin/env python3

import argparse
import sys
import os
import json
from datetime import datetime
import logging
from typing import Dict, List

# Add parent directory to path to import graph_heal modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.statistical_detector import StatisticalDetector
from src.health_monitor import HealthMonitor
from graph_heal.service_graph import ServiceGraph
from graph_heal.health_manager import HealthManager
from graph_heal.anomaly_detection import BayesianFaultLocalizer
from graph_heal.fault_localization import GraphBasedFaultLocalizer
from graph_heal.recovery_system import EnhancedRecoverySystem
from scripts.run_improved_experiments import ImprovedExperimentRunner

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

def run_side_by_side_comparison(experiment_data: List[Dict]) -> Dict:
    """Compare all three methods on same data"""
    
    # Initialize all three detectors
    statistical = StatisticalDetector()
    health_monitor = HealthMonitor()
    service_graph = ServiceGraph()
    graph_heal = HealthManager()
    
    # Initialize advanced GRAPH-HEAL components
    bayesian_localizer = BayesianFaultLocalizer()
    graph_localizer = GraphBasedFaultLocalizer()
    recovery_system = EnhancedRecoverySystem(service_graph, None)  # No Docker client needed for comparison
    
    # Set up service graph with dependencies
    service_graph.add_service('service_a', ['service_b', 'service_c'])
    service_graph.add_service('service_b', ['service_d'])
    service_graph.add_service('service_c', ['service_d'])
    service_graph.add_service('service_d', [])
    
    results = {
        'statistical': {'detections': [], 'tp': 0, 'fp': 0, 'fn': 0},
        'health_based': {'detections': [], 'tp': 0, 'fp': 0, 'fn': 0},
        'graph_heal': {
            'detections': [], 
            'tp': 0, 'fp': 0, 'fn': 0,
            'localization_correct': 0,
            'recovery_success': 0,
            'recovery_time': [],
            'cascade_prevented': 0,
            'bayesian_accuracy': 0,
            'graph_localization_accuracy': 0
        }
    }
    
    fault_start_time = None
    recovery_start_time = None
    service_statuses = []
    
    for snapshot in experiment_data:
        metrics = snapshot
        current_phase = snapshot.get('phase', 'normal')
        is_fault = current_phase in ['onset', 'fault']
        
        # Track fault and recovery timing
        if current_phase == 'onset' and fault_start_time is None:
            fault_start_time = metrics['timestamp']
        elif current_phase == 'recovery' and recovery_start_time is None:
            recovery_start_time = metrics['timestamp']
        
        # Update service graph with current metrics
        service_graph.add_metrics('service_a', metrics, datetime.fromtimestamp(metrics['timestamp']))
        
        # Test all three methods
        stat_detected = bool(statistical.detect_anomaly(metrics))
        
        health_score = health_monitor.calculate_health_score(metrics)
        health_state = health_monitor.get_health_state(health_score)
        health_detected = bool(health_state in ['degraded', 'warning', 'critical'])
        
        # GRAPH-HEAL detection (combines graph analysis with health scoring)
        graph_health_score, graph_health_state = graph_heal.calculate_health_score(metrics)
        affected_services = service_graph.get_affected_services('service_a')
        graph_heal_detected = bool(graph_health_state in ['degraded', 'warning', 'critical'] or 
                                 len(affected_services) > 0)
        
        # Advanced GRAPH-HEAL features
        if graph_heal_detected and is_fault:
            # Bayesian localization
            service_status = {'service_a': metrics}
            bayesian_results = bayesian_localizer.localize([{'service_id': 'service_a', 'severity': 'high'}], service_status)
            if bayesian_results and bayesian_results[0][0] == 'service_a':
                results['graph_heal']['bayesian_accuracy'] += 1
            
            # Graph-based localization
            graph_results = graph_localizer.localize_faults(service_status, [{'service_id': 'service_a', 'severity': 'high'}])
            if graph_results and graph_results[0].get('root_cause') == 'service_a':
                results['graph_heal']['graph_localization_accuracy'] += 1
            
            # Recovery planning
            recovery_plan = recovery_system.get_recovery_plan('service_a', 'cpu', metrics)
            if recovery_plan:
                results['graph_heal']['recovery_success'] += 1
                if recovery_start_time:
                    recovery_time = metrics['timestamp'] - recovery_start_time
                    results['graph_heal']['recovery_time'].append(recovery_time)
            
            # Cascade prevention
            if len(affected_services) > 0:
                # Check if cascade was prevented
                cascade_prevented = False
                for action in recovery_plan:
                    if action.action_type == 'composite' and 'cascading_failure' in action.parameters:
                        cascade_prevented = True
                        break
                if cascade_prevented:
                    results['graph_heal']['cascade_prevented'] += 1
        
        # Debug logging
        logger.debug(f"Phase: {current_phase}, "
                    f"CPU: {metrics.get('service_cpu_usage', 0):.1f}, "
                    f"Memory: {metrics.get('service_memory_usage', 0):.1f}, "
                    f"Health Score: {health_score:.1f}, "
                    f"Health State: {health_state}, "
                    f"Graph Health State: {graph_health_state}, "
                    f"Affected Services: {len(affected_services)}, "
                    f"Bayesian Root Cause: {bayesian_results[0][0] if bayesian_results else 'None'}, "
                    f"Graph Root Cause: {graph_results[0].get('root_cause') if graph_results else 'None'}, "
                    f"Stat Detected: {stat_detected}, "
                    f"Health Detected: {health_detected}, "
                    f"Graph-HEAL Detected: {graph_heal_detected}")
        
        results['statistical']['detections'].append(stat_detected)
        results['health_based']['detections'].append(health_detected)
        results['graph_heal']['detections'].append(graph_heal_detected)
        
        # Calculate confusion matrix
        for method, detected in [
            ('statistical', stat_detected),
            ('health_based', health_detected),
            ('graph_heal', graph_heal_detected)
        ]:
            if detected and is_fault:
                results[method]['tp'] += 1
            elif detected and not is_fault:
                results[method]['fp'] += 1
            elif not detected and is_fault:
                results[method]['fn'] += 1
    
    # Calculate metrics
    for method in ['statistical', 'health_based', 'graph_heal']:
        tp = results[method]['tp']
        fp = results[method]['fp']
        fn = results[method]['fn']
        
        precision = float(tp / (tp + fp) if (tp + fp) > 0 else 0)
        recall = float(tp / (tp + fn) if (tp + fn) > 0 else 0)
        accuracy = float(tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0)
        
        results[method].update({
            'accuracy': accuracy,
            'precision': precision, 
            'recall': recall,
            'f1_score': float(2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0)
        })
    
    # Calculate GRAPH-HEAL specific metrics
    total_faults = sum(1 for s in experiment_data if s.get('phase') in ['onset', 'fault'])
    total_recovery_attempts = sum(1 for s in experiment_data if s.get('phase') == 'recovery')
    
    if total_faults > 0:
        results['graph_heal']['localization_accuracy'] = float(results['graph_heal']['localization_correct'] / total_faults)
        results['graph_heal']['bayesian_accuracy'] = float(results['graph_heal']['bayesian_accuracy'] / total_faults)
        results['graph_heal']['graph_localization_accuracy'] = float(results['graph_heal']['graph_localization_accuracy'] / total_faults)
        results['graph_heal']['cascade_prevention_rate'] = float(results['graph_heal']['cascade_prevented'] / total_faults)
    
    if total_recovery_attempts > 0:
        results['graph_heal']['recovery_effectiveness'] = float(results['graph_heal']['recovery_success'] / total_recovery_attempts)
    
    if results['graph_heal']['recovery_time']:
        results['graph_heal']['avg_recovery_time'] = float(sum(results['graph_heal']['recovery_time']) / len(results['graph_heal']['recovery_time']))
    
    return results

def print_comparison(results: Dict):
    """Print side-by-side comparison"""
    
    print(f"\n{'='*100}")
    print(f" STATISTICAL vs HEALTH-BASED vs GRAPH-HEAL COMPARISON")
    print(f"{'='*100}")
    
    # Basic detection metrics
    print(f"\nDetection Metrics:")
    print(f"{'Metric':<20} {'Statistical':<15} {'Health-Based':<15} {'GRAPH-HEAL':<15} {'Best':<15}")
    print(f"{'-'*100}")
    
    for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
        stat_val = results['statistical'][metric]
        health_val = results['health_based'][metric]
        graph_val = results['graph_heal'][metric]
        
        best_val = max(stat_val, health_val, graph_val)
        best_method = 'Statistical' if best_val == stat_val else 'Health-Based' if best_val == health_val else 'GRAPH-HEAL'
        
        print(f"{metric:<20} {stat_val:<15.3f} {health_val:<15.3f} {graph_val:<15.3f} {best_method:<15}")
    
    # GRAPH-HEAL advanced metrics
    print(f"\nGRAPH-HEAL Advanced Metrics:")
    print(f"{'Metric':<30} {'Value':<15}")
    print(f"{'-'*100}")
    
    advanced_metrics = [
        ('localization_accuracy', 'Root Cause Accuracy'),
        ('bayesian_accuracy', 'Bayesian Localization'),
        ('graph_localization_accuracy', 'Graph Localization'),
        ('cascade_prevention_rate', 'Cascade Prevention'),
        ('recovery_effectiveness', 'Recovery Effectiveness'),
        ('avg_recovery_time', 'Avg Recovery Time (s)')
    ]
    
    for metric, label in advanced_metrics:
        if metric in results['graph_heal']:
            value = results['graph_heal'][metric]
            if metric == 'avg_recovery_time':
                print(f"{label:<30} {value:<15.1f}")
            else:
                print(f"{label:<30} {value:<15.3f}")

def main():
    parser = argparse.ArgumentParser(description='Compare detection methods')
    parser.add_argument('--service', required=True, help='Service to test (A, B, C, or D)')
    parser.add_argument('--type', required=True, choices=['cpu', 'memory', 'latency', 'network'],
                      help='Type of fault to inject')
    args = parser.parse_args()
    
    # Map service letter to service name
    service_map = {
        'A': 'service_a',
        'B': 'service_b',
        'C': 'service_c',
        'D': 'service_d'
    }
    
    if args.service not in service_map:
        logger.error(f"Invalid service: {args.service}. Must be one of: A, B, C, D")
        sys.exit(1)
    
    service_name = service_map[args.service]
    runner = ImprovedExperimentRunner()
    
    # Run the experiment
    metrics = runner.run_fault_injection(service_name, args.type)
    
    # Run comparison
    comparison_results = run_side_by_side_comparison(metrics)
    print_comparison(comparison_results)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    filename = f'comparison_results_{args.service}_{args.type}_{timestamp}.json'
    filepath = os.path.join(results_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    logger.info(f"Saved comparison results to {filepath}")

if __name__ == '__main__':
    main() 