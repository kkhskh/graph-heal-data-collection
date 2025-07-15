#!/usr/bin/env python3

import argparse
import sys
import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, List
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph_heal.anomaly_detection import StatisticalAnomalyDetector
from graph_heal.config.evaluation_config import EvaluationConfig

class StatisticalExperimentRunner:
    def __init__(self):
        """Initialize the statistical experiment runner"""
        self.config = EvaluationConfig()
        self.detector = StatisticalAnomalyDetector()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Create results directory if it doesn't exist
        self.results_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'results')
        os.makedirs(self.results_dir, exist_ok=True)
        
    def collect_baseline(self, service: str, num_samples: int) -> List[Dict[str, float]]:
        """Collect baseline metrics for a service
        
        Args:
            service: Service name
            num_samples: Number of samples to collect
            
        Returns:
            List of metric samples
        """
        self.logger.info(f"Starting baseline collection for {service}")
        self.logger.info(f"Will collect {num_samples} baseline samples")
        
        samples = []
        for i in range(num_samples):
            # Generate normal metrics with some noise
            metrics = {
                'cpu': 15 + np.random.normal(0, 2),  # 15% ± 2%
                'memory': 40 + np.random.normal(0, 3),  # 40% ± 3%
                'latency': 50 + np.random.normal(0, 5),  # 50ms ± 5ms
                'error_rate': 0.1 + np.random.normal(0, 0.05)  # 0.1% ± 0.05%
            }
            
            # The new detector manages its own history.
            # We just need to pass the metrics in a structured way.
            service_status = {service: {"metrics": metrics}}
            self.detector.detect_anomalies(service_status)
            samples.append(metrics)
            
            if (i + 1) % 10 == 0:
                self.logger.info(f"Collecting baseline sample {i + 1}/{num_samples}")
                
        self.logger.info(f"Completed baseline collection with {len(samples)} samples")
        return samples
        
    def run_fault_injection(self, service: str, fault_type: str, duration: int) -> List[Dict[str, float]]:
        """Run fault injection experiment
        
        Args:
            service: Service name
            fault_type: Type of fault to inject
            duration: Duration of fault injection in seconds
            
        Returns:
            List of metric samples during fault injection
        """
        self.logger.info(f"Starting fault injection experiment for {service} with {fault_type} fault")
        self.logger.info(f"Fault injection duration: {duration} seconds")
        
        samples = []
        for t in range(duration):
            # Generate metrics based on fault type
            metrics = self._generate_fault_metrics(fault_type, t, duration)
            
            # Update detector history and check for anomalies
            service_status = {service: {"metrics": metrics}}
            anomalies = self.detector.detect_anomalies(service_status)
            
            # Add detection results to metrics
            metrics['anomalies'] = anomalies
            samples.append(metrics)
            
            if (t + 1) % 10 == 0:
                self.logger.info(f"Fault injection progress: {t + 1}/{duration} seconds")
                
        self.logger.info(f"Completed fault injection with {len(samples)} samples")
        return samples
        
    def _generate_fault_metrics(self, fault_type: str, t: int, duration: int) -> Dict[str, float]:
        """Generate metrics for a specific fault type
        
        Args:
            fault_type: Type of fault
            t: Current time step
            duration: Total duration
            
        Returns:
            Dictionary of metric values
        """
        progress = t / duration
        metrics = {
            'cpu': 15 + np.random.normal(0, 2),
            'memory': 40 + np.random.normal(0, 3),
            'latency': 50 + np.random.normal(0, 5),
            'error_rate': 0.1 + np.random.normal(0, 0.05)
        }
        
        if fault_type == 'cpu':
            # Gradual CPU increase with noise
            target_cpu = 15 + (50 * progress)  # 15% -> 65%
            noise = np.random.normal(0, 5)
            metrics['cpu'] = target_cpu + noise
            
            # Add periodic spikes
            if t % 30 == 0:
                metrics['cpu'] += np.random.uniform(10, 20)
                
        elif fault_type == 'memory':
            # Gradual memory increase
            target_memory = 40 + (40 * progress)  # 40% -> 80%
            noise = np.random.normal(0, 3)
            metrics['memory'] = target_memory + noise
            
        elif fault_type == 'latency':
            # Gradual latency increase
            target_latency = 50 + (200 * progress)  # 50ms -> 250ms
            noise = np.random.normal(0, 10)
            metrics['latency'] = target_latency + noise
            
        elif fault_type == 'network':
            # Network issues affect multiple metrics
            metrics['latency'] = 50 + (300 * progress)  # 50ms -> 350ms
            metrics['error_rate'] = 0.1 + (5 * progress)  # 0.1% -> 5.1%
            
        return metrics
        
    def analyze_results(self, baseline_samples: List[Dict[str, float]], 
                       fault_samples: List[Dict[str, float]]) -> Dict:
        """Analyze experiment results
        
        Args:
            baseline_samples: Baseline metric samples
            fault_samples: Fault injection metric samples
            
        Returns:
            Dictionary containing analysis results
        """
        # Calculate detection accuracy
        total_anomalies = sum(1 for s in fault_samples if s.get('anomalies'))
        detection_accuracy = total_anomalies / len(fault_samples) if fault_samples else 0
        
        # Calculate false positive rate
        false_positives = sum(1 for s in baseline_samples if s.get('anomalies'))
        false_positive_rate = false_positives / len(baseline_samples) if baseline_samples else 0
        
        # Calculate detection time
        detection_time = 0
        for i, sample in enumerate(fault_samples):
            if sample.get('anomalies'):
                detection_time = i
                break
                
        # Calculate recovery time
        recovery_time = 0
        for i, sample in enumerate(reversed(fault_samples)):
            if sample.get('anomalies'):
                recovery_time = i
                break
                
        # Calculate health state distribution
        health_states = {
            'healthy': 0,
            'degraded': 0,
            'warning': 0,
            'critical': 0
        }
        
        for sample in fault_samples:
            cpu = sample['cpu']
            memory = sample['memory']
            latency = sample['latency']
            error_rate = sample['error_rate']
            
            # Simple health calculation
            health_score = 100
            if cpu > 80: health_score -= 30
            elif cpu > 60: health_score -= 15
            elif cpu > 40: health_score -= 5
            
            if memory > 80: health_score -= 25
            elif memory > 60: health_score -= 10
            
            if latency > 200: health_score -= 20
            elif latency > 100: health_score -= 10
            
            if error_rate > 1: health_score -= 25
            elif error_rate > 0.5: health_score -= 10
            
            # Determine health state
            if health_score > 85: health_states['healthy'] += 1
            elif health_score > 60: health_states['degraded'] += 1
            elif health_score > 30: health_states['warning'] += 1
            else: health_states['critical'] += 1
            
        # Convert counts to percentages
        total_samples = len(fault_samples)
        for state in health_states:
            health_states[state] = (health_states[state] / total_samples) * 100 if total_samples > 0 else 0
            
        return {
            'detection_accuracy': detection_accuracy,
            'false_positive_rate': false_positive_rate,
            'detection_time': detection_time,
            'recovery_time': recovery_time,
            'health_states': health_states
        }
        
    def save_results(self, results: Dict, service: str, fault_type: str) -> None:
        """Save experiment results to file
        
        Args:
            results: Results dictionary
            service: Service name
            fault_type: Type of fault
        """
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        filename = f"statistical_results_{service}_{fault_type}_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
            
        self.logger.info(f"Saved results to {filepath}")
        
def main():
    parser = argparse.ArgumentParser(description='Run statistical baseline experiments')
    parser.add_argument('--service', required=True, help='Service to inject fault into')
    parser.add_argument('--type', required=True, choices=['cpu', 'memory', 'latency', 'network'],
                      help='Type of fault to inject')
    args = parser.parse_args()
    
    runner = StatisticalExperimentRunner()
    
    # Collect baseline
    baseline_samples = runner.collect_baseline(args.service, 60)  # 1 minute baseline
    
    # Run fault injection
    fault_samples = runner.run_fault_injection(args.service, args.type, 300)  # 5 minutes fault
    
    # Analyze results
    results = runner.analyze_results(baseline_samples, fault_samples)
    
    # Save results
    runner.save_results(results, args.service, args.type)
    
if __name__ == '__main__':
    main() 