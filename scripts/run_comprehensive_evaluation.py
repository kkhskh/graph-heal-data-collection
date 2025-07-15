#!/usr/bin/env python3
"""
Comprehensive Evaluation Script
Runs multiple experiments and generates detailed performance reports.
"""

import time
import json
import logging
import requests
import subprocess
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ComprehensiveEvaluator:
    """Comprehensive evaluation of Graph-Heal performance."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.services = config.get('services', [])
        self.experiment_configs = config.get('experiments', [])
        self.results_dir = Path(config.get('results_dir', 'data/comprehensive_evaluation'))
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.all_results = []
        self.baseline_results = []
        self.graphheal_results = []
        
    def run_single_fault_experiment(self, target_service: str, fault_type: str = 'cpu') -> Dict:
        """Run a single fault injection experiment."""
        logger.info(f"Running single fault experiment: {target_service} - {fault_type}")
        
        experiment_start = time.time()
        
        # Inject fault
        fault_success = self.inject_fault(target_service, fault_type)
        if not fault_success:
            return {'status': 'fault_injection_failed'}
            
        fault_time = time.time()
        
        # Monitor for detection
        detection_time = None
        detection_found = False
        
        while time.time() - fault_time < 30:  # 30 second timeout
            if self.detect_anomaly(target_service):
                detection_time = time.time()
                detection_found = True
                break
            time.sleep(1)
            
        if not detection_found:
            return {'status': 'detection_timeout'}
            
        # Localize fault
        localization_start = time.time()
        localized_service = self.localize_fault(target_service)
        localization_time = time.time()
        
        # Execute recovery
        recovery_start = time.time()
        recovery_success = self.execute_recovery(localized_service)
        recovery_time = time.time()
        
        # Verify recovery
        time.sleep(5)
        recovery_verified = self.verify_recovery(target_service)
        
        experiment_end = time.time()
        
        return {
            'status': 'success',
            'target_service': target_service,
            'fault_type': fault_type,
            'timing': {
                'experiment_start': experiment_start,
                'fault_injection': fault_time,
                'detection': detection_time,
                'localization_start': localization_start,
                'localization_end': localization_time,
                'recovery_start': recovery_start,
                'recovery_end': recovery_time,
                'experiment_end': experiment_end
            },
            'latencies': {
                'detection_latency': detection_time - fault_time,
                'localization_latency': localization_time - localization_start,
                'recovery_latency': recovery_time - recovery_start,
                'total_recovery_time': recovery_time - fault_time,
                'total_experiment_time': experiment_end - experiment_start
            },
            'accuracy': {
                'localization_correct': localized_service == target_service,
                'recovery_successful': recovery_success,
                'recovery_verified': recovery_verified
            }
        }
        
    def run_baseline_experiment(self, target_service: str) -> Dict:
        """Run baseline experiment with simple threshold detection."""
        logger.info(f"Running baseline experiment: {target_service}")
        
        experiment_start = time.time()
        
        # Inject fault
        fault_success = self.inject_fault(target_service, 'cpu')
        if not fault_success:
            return {'status': 'fault_injection_failed'}
            
        fault_time = time.time()
        
        # Simple threshold-based detection (no dependency analysis)
        detection_time = None
        detection_found = False
        
        while time.time() - fault_time < 30:
            if self.detect_anomaly_simple(target_service):
                detection_time = time.time()
                detection_found = True
                break
            time.sleep(1)
            
        if not detection_found:
            return {'status': 'detection_timeout'}
            
        # Simple recovery (restart target service)
        recovery_start = time.time()
        recovery_success = self.execute_recovery(target_service)
        recovery_time = time.time()
        
        # Verify recovery
        time.sleep(5)
        recovery_verified = self.verify_recovery(target_service)
        
        experiment_end = time.time()
        
        return {
            'status': 'success',
            'target_service': target_service,
            'method': 'baseline',
            'timing': {
                'experiment_start': experiment_start,
                'fault_injection': fault_time,
                'detection': detection_time,
                'recovery_start': recovery_start,
                'recovery_end': recovery_time,
                'experiment_end': experiment_end
            },
            'latencies': {
                'detection_latency': detection_time - fault_time,
                'recovery_latency': recovery_time - recovery_start,
                'total_recovery_time': recovery_time - fault_time,
                'total_experiment_time': experiment_end - experiment_start
            },
            'accuracy': {
                'localization_correct': True,  # Baseline always targets the detected service
                'recovery_successful': recovery_success,
                'recovery_verified': recovery_verified
            }
        }
        
    def run_graphheal_experiment(self, target_service: str) -> Dict:
        """Run Graph-Heal experiment with dependency-aware detection."""
        logger.info(f"Running Graph-Heal experiment: {target_service}")
        
        experiment_start = time.time()
        
        # Inject fault
        fault_success = self.inject_fault(target_service, 'cpu')
        if not fault_success:
            return {'status': 'fault_injection_failed'}
            
        fault_time = time.time()
        
        # Dependency-aware detection
        detection_time = None
        detection_found = False
        
        while time.time() - fault_time < 30:
            if self.detect_anomaly_dependency_aware(target_service):
                detection_time = time.time()
                detection_found = True
                break
            time.sleep(1)
            
        if not detection_found:
            return {'status': 'detection_timeout'}
            
        # Dependency-based localization
        localization_start = time.time()
        localized_service = self.localize_fault_dependency_aware(target_service)
        localization_time = time.time()
        
        # Execute recovery
        recovery_start = time.time()
        recovery_success = self.execute_recovery(localized_service)
        recovery_time = time.time()
        
        # Verify recovery
        time.sleep(5)
        recovery_verified = self.verify_recovery(target_service)
        
        experiment_end = time.time()
        
        return {
            'status': 'success',
            'target_service': target_service,
            'method': 'graphheal',
            'timing': {
                'experiment_start': experiment_start,
                'fault_injection': fault_time,
                'detection': detection_time,
                'localization_start': localization_start,
                'localization_end': localization_time,
                'recovery_start': recovery_start,
                'recovery_end': recovery_time,
                'experiment_end': experiment_end
            },
            'latencies': {
                'detection_latency': detection_time - fault_time,
                'localization_latency': localization_time - localization_start,
                'recovery_latency': recovery_time - recovery_start,
                'total_recovery_time': recovery_time - fault_time,
                'total_experiment_time': experiment_end - experiment_start
            },
            'accuracy': {
                'localization_correct': localized_service == target_service,
                'recovery_successful': recovery_success,
                'recovery_verified': recovery_verified
            }
        }
        
    def inject_fault(self, service: str, fault_type: str) -> bool:
        """Inject a fault into a service."""
        try:
            cmd = [
                'python', 'scripts/inject_cpu_fault.py',
                '--service', service,
                '--duration', '20'
            ]
            subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        except Exception as e:
            logger.error(f"Fault injection failed: {e}")
            return False
            
    def detect_anomaly(self, service: str) -> bool:
        """Detect anomaly in a service."""
        try:
            port = self.get_service_port(service)
            response = requests.get(f"http://localhost:{port}/metrics", timeout=5)
            
            if response.status_code == 200:
                metrics = self.parse_metrics(response.text)
                return self.is_anomalous(metrics)
            return False
        except Exception as e:
            logger.warning(f"Anomaly detection failed for {service}: {e}")
            return False
            
    def detect_anomaly_simple(self, service: str) -> bool:
        """Simple threshold-based anomaly detection."""
        return self.detect_anomaly(service)
        
    def detect_anomaly_dependency_aware(self, service: str) -> bool:
        """Dependency-aware anomaly detection."""
        # For now, use the same detection as simple
        # In a real implementation, this would consider service dependencies
        return self.detect_anomaly(service)
        
    def get_service_port(self, service: str) -> int:
        """Get the port for a service."""
        port_map = {
            'service_a': 5001,
            'service_b': 5002,
            'service_c': 5003,
            'service_d': 5004
        }
        return port_map.get(service, 5000)
        
    def parse_metrics(self, metrics_text: str) -> Dict[str, float]:
        """Parse Prometheus metrics format."""
        metrics = {}
        for line in metrics_text.split('\n'):
            if line.startswith('#') or not line.strip():
                continue
            try:
                if 'service_cpu_usage' in line:
                    value = float(line.split()[-1])
                    metrics['cpu_usage'] = value
                elif 'service_memory_usage' in line:
                    value = float(line.split()[-1])
                    metrics['memory_usage'] = value
                elif 'service_response_time' in line:
                    value = float(line.split()[-1])
                    metrics['response_time'] = value
            except (ValueError, IndexError):
                continue
        return metrics
        
    def is_anomalous(self, metrics: Dict[str, float]) -> bool:
        """Check if metrics indicate an anomaly."""
        if 'cpu_usage' in metrics and metrics['cpu_usage'] > 80:
            return True
        if 'memory_usage' in metrics and metrics['memory_usage'] > 85:
            return True
        if 'response_time' in metrics and metrics['response_time'] > 1.0:
            return True
        return False
        
    def localize_fault(self, target_service: str) -> str:
        """Basic fault localization."""
        return self.localize_fault_dependency_aware(target_service)
        
    def localize_fault_dependency_aware(self, target_service: str) -> str:
        """Dependency-aware fault localization."""
        # Service dependency graph
        service_dependencies = {
            'service_a': ['service_b', 'service_c'],
            'service_b': ['service_d'],
            'service_c': ['service_d'],
            'service_d': []
        }
        
        # Check dependencies first
        dependencies = service_dependencies.get(target_service, [])
        for dep in dependencies:
            if not self.check_service_health(dep):
                return dep
                
        # If no unhealthy dependencies, return target service
        return target_service
        
    def check_service_health(self, service: str) -> bool:
        """Check if a service is healthy."""
        try:
            port = self.get_service_port(service)
            response = requests.get(f"http://localhost:{port}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
            
    def execute_recovery(self, service: str) -> bool:
        """Execute recovery action for a service."""
        try:
            cmd = ['docker', 'restart', service]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Recovery execution failed: {e}")
            return False
            
    def verify_recovery(self, service: str) -> bool:
        """Verify that recovery was successful."""
        try:
            time.sleep(5)
            
            if not self.check_service_health(service):
                return False
                
            port = self.get_service_port(service)
            response = requests.get(f"http://localhost:{port}/metrics", timeout=5)
            
            if response.status_code == 200:
                metrics = self.parse_metrics(response.text)
                return not self.is_anomalous(metrics)
                
            return False
        except Exception as e:
            logger.warning(f"Recovery verification failed: {e}")
            return False
            
    def run_comparison_experiment(self, target_service: str) -> Dict:
        """Run comparison between baseline and Graph-Heal."""
        logger.info(f"Running comparison experiment for {target_service}")
        
        # Run baseline experiment
        baseline_result = self.run_baseline_experiment(target_service)
        
        # Wait for system to stabilize
        time.sleep(30)
        
        # Run Graph-Heal experiment
        graphheal_result = self.run_graphheal_experiment(target_service)
        
        # Compare results
        comparison = self.compare_results(baseline_result, graphheal_result)
        
        return {
            'baseline': baseline_result,
            'graphheal': graphheal_result,
            'comparison': comparison
        }
        
    def compare_results(self, baseline: Dict, graphheal: Dict) -> Dict:
        """Compare baseline and Graph-Heal results."""
        comparison = {}
        
        if baseline.get('status') == 'success' and graphheal.get('status') == 'success':
            baseline_latencies = baseline.get('latencies', {})
            graphheal_latencies = graphheal.get('latencies', {})
            
            # Compare detection latency
            if 'detection_latency' in baseline_latencies and 'detection_latency' in graphheal_latencies:
                bl_detection = baseline_latencies['detection_latency']
                gh_detection = graphheal_latencies['detection_latency']
                comparison['detection_latency_improvement'] = bl_detection - gh_detection
                comparison['detection_latency_ratio'] = gh_detection / bl_detection
                
            # Compare total recovery time
            if 'total_recovery_time' in baseline_latencies and 'total_recovery_time' in graphheal_latencies:
                bl_total = baseline_latencies['total_recovery_time']
                gh_total = graphheal_latencies['total_recovery_time']
                comparison['total_recovery_improvement'] = bl_total - gh_total
                comparison['total_recovery_ratio'] = gh_total / bl_total
                
            # Compare accuracy
            baseline_accuracy = baseline.get('accuracy', {})
            graphheal_accuracy = graphheal.get('accuracy', {})
            
            comparison['localization_accuracy'] = {
                'baseline': baseline_accuracy.get('localization_correct', False),
                'graphheal': graphheal_accuracy.get('localization_correct', False)
            }
            
            comparison['recovery_success'] = {
                'baseline': baseline_accuracy.get('recovery_successful', False),
                'graphheal': graphheal_accuracy.get('recovery_successful', False)
            }
            
        return comparison
        
    def run_comprehensive_evaluation(self) -> Dict:
        """Run comprehensive evaluation across multiple scenarios."""
        logger.info("Starting comprehensive evaluation")
        
        evaluation_results = {
            'experiments': [],
            'summary': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Run experiments for each service
        for service in self.services:
            logger.info(f"Running experiments for {service}")
            
            # Single fault experiment
            single_result = self.run_single_fault_experiment(service)
            evaluation_results['experiments'].append({
                'type': 'single_fault',
                'service': service,
                'result': single_result
            })
            
            # Wait between experiments
            time.sleep(30)
            
            # Comparison experiment
            comparison_result = self.run_comparison_experiment(service)
            evaluation_results['experiments'].append({
                'type': 'comparison',
                'service': service,
                'result': comparison_result
            })
            
            # Wait between experiments
            time.sleep(30)
            
        # Calculate summary statistics
        evaluation_results['summary'] = self.calculate_evaluation_summary(evaluation_results['experiments'])
        
        return evaluation_results
        
    def calculate_evaluation_summary(self, experiments: List[Dict]) -> Dict:
        """Calculate summary statistics for all experiments."""
        summary = {
            'total_experiments': len(experiments),
            'successful_experiments': 0,
            'detection_latencies': [],
            'localization_accuracies': [],
            'recovery_success_rates': [],
            'comparison_metrics': []
        }
        
        for experiment in experiments:
            result = experiment['result']
            
            if experiment['type'] == 'single_fault':
                if result.get('status') == 'success':
                    summary['successful_experiments'] += 1
                    
                    latencies = result.get('latencies', {})
                    if 'detection_latency' in latencies:
                        summary['detection_latencies'].append(latencies['detection_latency'])
                        
                    accuracy = result.get('accuracy', {})
                    if 'localization_correct' in accuracy:
                        summary['localization_accuracies'].append(accuracy['localization_correct'])
                        
                    if 'recovery_successful' in accuracy:
                        summary['recovery_success_rates'].append(accuracy['recovery_successful'])
                        
            elif experiment['type'] == 'comparison':
                comparison = result.get('comparison', {})
                if comparison:
                    summary['comparison_metrics'].append(comparison)
                    
        # Calculate statistics
        if summary['detection_latencies']:
            summary['detection_latency_stats'] = {
                'mean': statistics.mean(summary['detection_latencies']),
                'median': statistics.median(summary['detection_latencies']),
                'std': statistics.stdev(summary['detection_latencies']) if len(summary['detection_latencies']) > 1 else 0
            }
            
        if summary['localization_accuracies']:
            summary['localization_accuracy_rate'] = sum(summary['localization_accuracies']) / len(summary['localization_accuracies'])
            
        if summary['recovery_success_rates']:
            summary['recovery_success_rate'] = sum(summary['recovery_success_rates']) / len(summary['recovery_success_rates'])
            
        return summary
        
    def save_results(self, results: Dict, filename: str = None):
        """Save evaluation results to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"comprehensive_evaluation_{timestamp}.json"
            
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Results saved to {filepath}")
        return filepath
        
    def print_summary(self, results: Dict):
        """Print a summary of the evaluation results."""
        summary = results.get('summary', {})
        
        logger.info("Comprehensive Evaluation Summary")
        logger.info("=" * 50)
        logger.info(f"Total Experiments: {summary.get('total_experiments', 0)}")
        logger.info(f"Successful Experiments: {summary.get('successful_experiments', 0)}")
        
        # Detection latency
        if 'detection_latency_stats' in summary:
            dl = summary['detection_latency_stats']
            logger.info(f"Detection Latency:")
            logger.info(f"  Mean: {dl['mean']:.3f}s")
            logger.info(f"  Median: {dl['median']:.3f}s")
            logger.info(f"  Std Dev: {dl['std']:.3f}s")
            
        # Localization accuracy
        if 'localization_accuracy_rate' in summary:
            logger.info(f"Localization Accuracy: {summary['localization_accuracy_rate']:.1%}")
            
        # Recovery success rate
        if 'recovery_success_rate' in summary:
            logger.info(f"Recovery Success Rate: {summary['recovery_success_rate']:.1%}")
            
        # Comparison metrics
        if 'comparison_metrics' in summary:
            logger.info(f"Comparison Experiments: {len(summary['comparison_metrics'])}")


def main():
    """Main function to run comprehensive evaluation."""
    # Configuration
    config = {
        'services': ['service_a', 'service_b', 'service_c', 'service_d'],
        'experiments': [
            {'type': 'single_fault', 'fault_type': 'cpu'},
            {'type': 'comparison', 'baseline': True, 'graphheal': True}
        ],
        'results_dir': 'data/comprehensive_evaluation'
    }
    
    # Create evaluator
    evaluator = ComprehensiveEvaluator(config)
    
    # Run comprehensive evaluation
    logger.info("Starting comprehensive evaluation")
    results = evaluator.run_comprehensive_evaluation()
    
    # Save results
    evaluator.save_results(results)
    
    # Print summary
    evaluator.print_summary(results)
    
    logger.info("Comprehensive evaluation completed")


if __name__ == "__main__":
    main() 