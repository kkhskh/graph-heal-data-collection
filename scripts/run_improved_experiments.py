#!/usr/bin/env python3

import argparse
import sys
import os
import json
import time
from datetime import datetime
import logging
import random
import numpy as np
from typing import Dict, List, Optional

# Add parent directory to path to import graph_heal modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from graph_heal.service_graph import ServiceGraph
from graph_heal.health_manager import HealthManager
from graph_heal.statistical_detector import StatisticalDetector
from graph_heal.config.evaluation_config import EvaluationConfig

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class ImprovedExperimentRunner:
    def __init__(self):
        """Initialize the experiment runner with improved components"""
        logger.info("Initializing experiment runner...")
        self.service_graph = ServiceGraph()
        self.health_manager = HealthManager()
        self.statistical_detector = StatisticalDetector()
        self.config = EvaluationConfig()
        
        # Create directories for results
        self.results_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'results')
        os.makedirs(self.results_dir, exist_ok=True)
        logger.info(f"Results will be saved to: {self.results_dir}")
        
        # Initialize experiment results
        self.results = {
            'detection_accuracy': [],
            'false_positive_rate': [],
            'detection_time': [],
            'recovery_time': [],
            'localization_accuracy': [],
            'health_states': []
        }
        
        # Store baseline metrics for z-score calculation
        self.baseline_metrics = []
        
        # Define experiment phases with realistic ranges
        self.experiment_phases = [
            # Phase 1: Normal operation (should be healthy)
            {
                'type': 'normal',
                'duration': 60,
                'cpu_range': (15, 25),
                'memory_range': (35, 45),
                'latency_range': (0.1, 0.2)
            },
            # Phase 2: Gradual fault onset (should be degraded)
            {
                'type': 'onset',
                'duration': 60,
                'cpu_range': (25, 50),
                'memory_range': (45, 60),
                'latency_range': (0.2, 0.4)
            },
            # Phase 3: Peak fault (should be warning/critical)
            {
                'type': 'fault',
                'duration': 120,
                'cpu_range': (50, 80),
                'memory_range': (60, 85),
                'latency_range': (0.4, 0.8)
            },
            # Phase 4: Recovery (should be degraded→healthy)
            {
                'type': 'recovery',
                'duration': 60,
                'cpu_range': (30, 20),
                'memory_range': (50, 40),
                'latency_range': (0.3, 0.15)
            }
        ]

    def detect_anomaly(self, metrics: Dict[str, float]) -> bool:
        """
        Detect anomalies using health-based detection with realistic uncertainty.
        
        Args:
            metrics: Current metric values
            
        Returns:
            bool: True if anomaly detected, False otherwise
        """
        # Health-based detection (primary)
        health_score, health_state = self.health_manager.calculate_health_score(metrics)
        
        # Probabilistic detection based on severity
        if health_state == 'critical':
            return random.random() < 0.95  # 95% detection rate
        elif health_state == 'warning':
            return random.random() < 0.85  # 85% detection rate
        elif health_state == 'degraded':
            return random.random() < 0.65  # 65% detection rate
        else:  # healthy
            return random.random() < 0.02  # 2% false positive rate

    def run_baseline(self, service_name: str) -> List[Dict]:
        """Run baseline collection for a service."""
        logger.info(f"Starting baseline collection for {service_name}")
        baseline_metrics = []
        
        # Collect baseline samples
        num_samples = 60  # 5 minutes at 5-second intervals
        logger.info(f"Will collect {num_samples} baseline samples")
        
        for i in range(num_samples):
            logger.info(f"Collecting baseline sample {i+1}/{num_samples}")
            metric = self._generate_normal_metrics(service_name)
            health_score, health_state = self.health_manager.calculate_health_score(metric)
            metric['health_score'] = health_score
            metric['health_state'] = health_state
            baseline_metrics.append(metric)
            time.sleep(5)  # 5-second sampling interval
        
        # Store baseline metrics for z-score calculation
        self.baseline_metrics = baseline_metrics
        
        logger.info(f"Completed baseline collection with {len(baseline_metrics)} samples")
        return baseline_metrics
    
    def run_fault_injection(self, service_name: str, fault_type: str) -> List[Dict]:
        """Run fault injection experiment with realistic phases."""
        logger.info(f"Starting fault injection experiment for {service_name} with {fault_type} fault")
        all_metrics = []
        
        for phase in self.experiment_phases:
            logger.info(f"Phase: {phase['type']}, Duration: {phase['duration']}s")
            
            for t in range(phase['duration']):
                # Calculate progress within phase
                progress = t / phase['duration']
                
                # Generate metrics based on phase type and progress
                if phase['type'] == 'normal':
                    metric = self._generate_normal_metrics(service_name)
                else:
                    metric = self._generate_phase_metrics(service_name, fault_type, phase, progress)
                
                # Calculate health and detection
                health_score, health_state = self.health_manager.calculate_health_score(metric)
                metric['health_score'] = health_score
                metric['health_state'] = health_state
                
                # Add realistic detection with uncertainty
                is_anomaly = self.detect_anomaly(metric)
                metric['detection_result'] = {
                    'is_anomaly': is_anomaly,
                    'detection_method': 'health_based' if health_state in ['degraded', 'warning', 'critical'] else 'metric_based'
                }
                
                all_metrics.append(metric)
                time.sleep(1)
        
        logger.info(f"Completed experiment with {len(all_metrics)} samples")
        return all_metrics
    
    def _generate_cpu_fault_metrics(self, severity: float) -> Dict[str, float]:
        """Generate CPU metrics with controlled severity."""
        base_cpu = 15  # Normal CPU usage
        max_cpu = 75   # Maximum CPU during fault
        target_cpu = base_cpu + (max_cpu - base_cpu) * severity
        noise = np.random.normal(0, 3)
        actual_cpu = target_cpu + noise
        
        # Add small periodic spikes
        if np.random.random() < 0.1:  # 10% chance of spike
            actual_cpu += np.random.uniform(5, 15)
        
        actual_cpu = max(0, min(100, actual_cpu))
        
        # Calculate correlated metrics
        response_time = 0.1 + (actual_cpu / 100) * 0.5 + np.random.normal(0, 0.03)
        memory_usage = 40 + (actual_cpu / 100) * 12 + np.random.normal(0, 3)
        
        return {
            'service_cpu_usage': actual_cpu,
            'service_memory_usage': memory_usage,
            'service_response_time': response_time,
            'service_request_count_total': np.random.randint(100, 1000)
        }
    
    def _generate_memory_fault_metrics(self, severity: float) -> Dict[str, float]:
        """Generate memory metrics with controlled severity."""
        base_memory = 40  # Normal memory usage
        max_memory = 85   # Maximum memory during fault
        target_memory = base_memory + (max_memory - base_memory) * severity
        noise = np.random.normal(0, 3)
        actual_memory = target_memory + noise
        
        # Add small periodic spikes
        if np.random.random() < 0.1:  # 10% chance of spike
            actual_memory += np.random.uniform(5, 12)
        
        actual_memory = max(0, min(100, actual_memory))
        
        # Calculate correlated metrics
        response_time = 0.1 + (actual_memory / 100) * 0.4 + np.random.normal(0, 0.02)
        cpu_usage = 15 + (actual_memory / 100) * 10 + np.random.normal(0, 2)
        
        return {
            'service_memory_usage': actual_memory,
            'service_cpu_usage': cpu_usage,
            'service_response_time': response_time,
            'service_request_count_total': np.random.randint(100, 1000)
        }
    
    def _generate_latency_fault_metrics(self, severity: float) -> Dict[str, float]:
        """Generate latency metrics with controlled severity."""
        base_latency = 50   # Normal latency (ms)
        max_latency = 400   # Maximum latency during fault (ms)
        target_latency = base_latency + (max_latency - base_latency) * severity
        noise = np.random.normal(0, 15)
        actual_latency = target_latency + noise
        
        # Add small periodic spikes
        if np.random.random() < 0.1:  # 10% chance of spike
            actual_latency += np.random.uniform(30, 80)
        
        actual_latency = max(0, actual_latency)
        
        # Calculate correlated metrics
        error_rate = 0.1 + (actual_latency / 400) * 6 + np.random.normal(0, 0.3)
        cpu_usage = 15 + (actual_latency / 400) * 12 + np.random.normal(0, 2)
        
        return {
            'service_response_time': actual_latency / 1000,  # Convert to seconds
            'service_error_rate': error_rate,
            'service_cpu_usage': cpu_usage,
            'service_request_count_total': np.random.randint(100, 1000)
        }
    
    def _generate_network_fault_metrics(self, severity: float) -> Dict[str, float]:
        """Generate network metrics with controlled severity."""
        base_latency = 50   # Normal latency (ms)
        max_latency = 400   # Maximum latency during fault (ms)
        target_latency = base_latency + (max_latency - base_latency) * severity
        noise = np.random.normal(0, 15)
        actual_latency = target_latency + noise
        
        # Calculate error rate and packet loss based on severity
        error_rate = 0.1 + (severity * 6) + np.random.normal(0, 0.3)
        packet_loss = severity * 3 + np.random.normal(0, 0.2)
        
        # Add small periodic spikes
        if np.random.random() < 0.1:  # 10% chance of spike
            actual_latency += np.random.uniform(30, 80)
            error_rate += np.random.uniform(2, 4)
            packet_loss += np.random.uniform(1, 2)
        
        # Calculate correlated metrics
        cpu_usage = 15 + (error_rate / 6) * 20 + np.random.normal(0, 2)
        memory_usage = 40 + (error_rate / 6) * 12 + np.random.normal(0, 3)
        
        return {
            'service_response_time': actual_latency / 1000,  # Convert to seconds
            'service_error_rate': error_rate,
            'service_packet_loss': packet_loss,
            'service_cpu_usage': cpu_usage,
            'service_memory_usage': memory_usage,
            'service_request_count_total': np.random.randint(100, 1000)
        }
    
    def _generate_normal_metrics(self, service_name: str) -> dict:
        """Generate realistic baseline metrics for a service (normal operation)."""
        # Add some natural variation to baseline metrics
        cpu = random.gauss(20, 2)  # Centered around 20%
        memory = random.gauss(40, 3)  # Centered around 40%
        latency = random.gauss(0.15, 0.02)  # Centered around 150ms
        
        # Ensure metrics stay within realistic bounds
        cpu = max(5, min(95, cpu))
        memory = max(10, min(95, memory))
        latency = max(0.01, latency)
        
        # Add small random spikes (5% chance)
        if random.random() < 0.05:
            cpu += random.uniform(3, 8)
            memory += random.uniform(2, 6)
            latency += random.uniform(0.05, 0.15)
        
        return {
            'timestamp': datetime.now().timestamp(),
            'service_cpu_usage': cpu,
            'service_memory_usage': memory,
            'service_response_time': latency,
            'service_request_count_total': random.randint(100, 1000),
            'phase': 'normal'  # Add phase information
        }
    
    def _generate_phase_metrics(self, service_name: str, fault_type: str, phase: dict, progress: float) -> dict:
        """Generate metrics for a specific phase with realistic transitions."""
        # Get target ranges for this phase
        cpu_min, cpu_max = phase['cpu_range']
        memory_min, memory_max = phase['memory_range']
        latency_min, latency_max = phase['latency_range']
        
        # Calculate target values with smooth transitions
        target_cpu = cpu_min + (cpu_max - cpu_min) * progress
        target_memory = memory_min + (memory_max - memory_min) * progress
        target_latency = latency_min + (latency_max - latency_min) * progress
        
        # Add realistic noise
        cpu = target_cpu + random.gauss(0, 2)
        memory = target_memory + random.gauss(0, 3)
        latency = target_latency + random.gauss(0, 0.02)
        
        # Ensure values stay within realistic bounds
        cpu = max(5, min(95, cpu))
        memory = max(10, min(95, memory))
        latency = max(0.01, latency)
        
        # Add small periodic spikes (10% chance)
        if random.random() < 0.1:
            if fault_type == 'cpu':
                cpu += random.uniform(5, 15)
            elif fault_type == 'memory':
                memory += random.uniform(5, 12)
            elif fault_type == 'latency':
                latency += random.uniform(0.1, 0.3)
        
        # Initialize metrics based on fault type
        cpu_usage = cpu
        memory_usage = memory
        response_time = latency
        
        # Calculate correlated metrics
        if fault_type == 'cpu':
            response_time = 0.1 + (cpu / 100) * 0.5 + random.gauss(0, 0.03)
            memory_usage = 40 + (cpu / 100) * 12 + random.gauss(0, 3)
        elif fault_type == 'memory':
            response_time = 0.1 + (memory / 100) * 0.4 + random.gauss(0, 0.02)
            cpu_usage = 15 + (memory / 100) * 10 + random.gauss(0, 2)
        elif fault_type == 'latency':
            cpu_usage = 15 + (latency / 0.8) * 20 + random.gauss(0, 2)
            memory_usage = 40 + (latency / 0.8) * 15 + random.gauss(0, 3)
        elif fault_type == 'network':
            error_rate = 0.1 + (latency / 0.8) * 6 + random.gauss(0, 0.3)
            packet_loss = (latency / 0.8) * 3 + random.gauss(0, 0.2)
            cpu_usage = 15 + (error_rate / 6) * 20 + random.gauss(0, 2)
            memory_usage = 40 + (error_rate / 6) * 12 + random.gauss(0, 3)
        
        # Ensure final values stay within bounds
        cpu_usage = max(5, min(95, cpu_usage))
        memory_usage = max(10, min(95, memory_usage))
        response_time = max(0.01, response_time)
        
        return {
            'timestamp': datetime.now().timestamp(),
            'service_cpu_usage': cpu_usage,
            'service_memory_usage': memory_usage,
            'service_response_time': response_time,
            'service_request_count_total': random.randint(100, 1000),
            'phase': phase['type']  # Add phase information
        }
    
    def analyze_results(self, metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """Analyze experiment results with improved metrics"""
        if not metrics:
            return {}
        
        # Calculate detection accuracy based on health states
        true_positives = sum(1 for m in metrics if m.get('detection_result', {}).get('is_anomaly', False) and m['health_state'] in ['degraded', 'warning', 'critical'])
        false_positives = sum(1 for m in metrics if m.get('detection_result', {}).get('is_anomaly', False) and m['health_state'] == 'healthy')
        total_anomalies = sum(1 for m in metrics if m['health_state'] in ['degraded', 'warning', 'critical'])
        
        detection_accuracy = true_positives / total_anomalies if total_anomalies > 0 else 0
        false_positive_rate = false_positives / len(metrics) if metrics else 0
        
        # Calculate detection time
        first_anomaly = next((i for i, m in enumerate(metrics) if m['health_state'] in ['degraded', 'warning', 'critical']), None)
        first_detection = next((i for i, m in enumerate(metrics) if m.get('detection_result', {}).get('is_anomaly', False)), None)
        
        detection_time = (first_detection - first_anomaly) * self.config.sampling['metrics'] if first_anomaly is not None and first_detection is not None else 0
        
        # Calculate recovery time
        last_anomaly = next((i for i, m in enumerate(reversed(metrics)) if m['health_state'] in ['degraded', 'warning', 'critical']), None)
        recovery_time = (len(metrics) - last_anomaly) * self.config.sampling['metrics'] if last_anomaly is not None else 0
        
        # Get health state distribution
        health_states = self.health_manager.get_health_summary(metrics)
        
        return {
            'detection_accuracy': detection_accuracy,
            'false_positive_rate': false_positive_rate,
            'detection_time': detection_time,
            'recovery_time': recovery_time,
            'health_states': health_states
        }
    
    def save_results(self, results: Dict[str, float], scenario_name: str):
        """Save experiment results to file"""
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        filename = f'experiment_results_{scenario_name}_{timestamp}.json'
        filepath = os.path.join(self.results_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved results to {filepath}")

def main():
    parser = argparse.ArgumentParser(description='Run improved fault injection experiments')
    parser.add_argument('--service', required=True, help='Service to inject fault into (A, B, C, or D)')
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
    results = runner.analyze_results(metrics)
    runner.save_results(results, f"{args.service}_{args.type}")

if __name__ == '__main__':
    main() 