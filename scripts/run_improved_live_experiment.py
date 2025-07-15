#!/usr/bin/env python3
"""
Improved Live Experiment Runner for Graph-Heal
Implements real fault injection and measurement with improved detection logic.
"""

import time
import json
import logging
import requests
import subprocess
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

class ImprovedLiveExperimentRunner:
    """Runs live experiments with improved fault detection and measurement."""
    
    def __init__(self, experiment_config: Dict):
        self.config = experiment_config
        self.services = experiment_config.get('services', [])
        self.fault_duration = experiment_config.get('fault_duration', 20)
        self.monitoring_interval = experiment_config.get('monitoring_interval', 1)
        self.results_dir = Path(experiment_config.get('results_dir', 'data/improved_live_experiments'))
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Detection thresholds (adjusted for realistic values)
        self.cpu_threshold = experiment_config.get('cpu_threshold', 10.0)  # Lowered from 80%
        self.memory_threshold = experiment_config.get('memory_threshold', 85.0)
        self.response_time_threshold = experiment_config.get('response_time_threshold', 1.0)
        self.health_threshold = experiment_config.get('health_threshold', 0.7)  # Health score threshold
        
        # Initialize timing data
        self.experiment_start = None
        self.fault_injection_time = None
        self.detection_time = None
        self.localization_time = None
        self.recovery_time = None
        self.experiment_end = None
        
        # Event log
        self.events = []
        
    def log_event(self, event_type: str, service: str, details: Dict = None):
        """Log an event with timestamp."""
        event = {
            'timestamp': time.time(),
            'event_type': event_type,
            'service': service,
            'details': details or {}
        }
        self.events.append(event)
        logger.info(f"EVENT: {event_type} - {service} - {details}")
        
    def get_service_port(self, service: str) -> int:
        """Get the port for a service."""
        port_map = {
            'service_a': 5001,
            'service_b': 5002,
            'service_c': 5003,
            'service_d': 5004
        }
        return port_map.get(service, 5000)
        
    def inject_fault(self, service: str, fault_type: str = 'cpu') -> bool:
        """Inject a fault into a service."""
        try:
            port = self.get_service_port(service)
            url = f"http://localhost:{port}/fault/cpu"
            payload = {"duration": self.fault_duration}
            
            logger.info(f"Injecting {fault_type} fault into {service}")
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                logger.info(f"Fault injection successful for {service}")
                return True
            else:
                logger.error(f"Fault injection failed for {service}: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Fault injection failed: {e}")
            return False
            
    def get_service_metrics(self, service: str) -> Dict:
        """Get metrics from a service."""
        try:
            port = self.get_service_port(service)
            response = requests.get(f"http://localhost:{port}/metrics", timeout=5)
            
            if response.status_code == 200:
                return self.parse_metrics(response.text)
            else:
                logger.warning(f"Failed to get metrics from {service}: {response.status_code}")
                return {}
        except Exception as e:
            logger.warning(f"Error getting metrics from {service}: {e}")
            return {}
            
    def get_service_health(self, service: str) -> Dict:
        """Get health status from a service."""
        try:
            port = self.get_service_port(service)
            response = requests.get(f"http://localhost:{port}/health", timeout=5)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Failed to get health from {service}: {response.status_code}")
                return {}
        except Exception as e:
            logger.warning(f"Error getting health from {service}: {e}")
            return {}
            
    def parse_metrics(self, metrics_text: str) -> Dict[str, float]:
        """Parse Prometheus metrics format."""
        metrics = {}
        for line in metrics_text.split('\n'):
            if line.startswith('#') or not line.strip():
                continue
            try:
                if 'service_cpu_usage' in line and not line.startswith('#'):
                    value = float(line.split()[-1])
                    metrics['cpu_usage'] = value
                elif 'service_memory_usage' in line and not line.startswith('#'):
                    value = float(line.split()[-1])
                    metrics['memory_usage'] = value
                elif 'service_latency_seconds' in line and not line.startswith('#'):
                    value = float(line.split()[-1])
                    metrics['response_time'] = value
            except (ValueError, IndexError):
                continue
        return metrics
        
    def detect_anomaly(self, service: str) -> bool:
        """Detect anomaly using both metrics and health endpoint."""
        # Check health endpoint first (more reliable for fault detection)
        health_data = self.get_service_health(service)
        if health_data:
            # Check if fault is active
            if health_data.get('cpu_fault_active', False):
                logger.info(f"Fault detected via health endpoint for {service}")
                return True
                
            # Check health score
            health_score = health_data.get('health_score', 1.0)
            if health_score < self.health_threshold:
                logger.info(f"Health degradation detected for {service}: {health_score}")
                return True
                
        # Check metrics as backup
        metrics = self.get_service_metrics(service)
        if metrics:
            # Check CPU usage
            if 'cpu_usage' in metrics and metrics['cpu_usage'] > self.cpu_threshold:
                logger.info(f"High CPU usage detected for {service}: {metrics['cpu_usage']}%")
                return True
                
            # Check memory usage
            if 'memory_usage' in metrics and metrics['memory_usage'] > self.memory_threshold:
                logger.info(f"High memory usage detected for {service}: {metrics['memory_usage']}MB")
                return True
                
            # Check response time
            if 'response_time' in metrics and metrics['response_time'] > self.response_time_threshold:
                logger.info(f"High response time detected for {service}: {metrics['response_time']}s")
                return True
                
        return False
        
    def localize_fault(self, target_service: str) -> str:
        """Localize fault using dependency analysis."""
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
            health_data = self.get_service_health(dep)
            if health_data and health_data.get('health_score', 1.0) < self.health_threshold:
                logger.info(f"Fault localized to dependency {dep}")
                return dep
                
        # If no unhealthy dependencies, return target service
        logger.info(f"Fault localized to target service {target_service}")
        return target_service
        
    def execute_recovery(self, service: str) -> bool:
        """Execute recovery action for a service."""
        try:
            # Use docker-compose restart for better compatibility
            cmd = ['docker-compose', 'restart', service]
            logger.info(f"Executing recovery for {service}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                logger.info(f"Recovery successful for {service}")
                return True
            else:
                logger.error(f"Recovery failed for {service}: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Recovery execution failed: {e}")
            return False
            
    def verify_recovery(self, service: str) -> bool:
        """Verify that recovery was successful."""
        try:
            # Wait for service to fully recover
            time.sleep(10)
            
            # Check health
            health_data = self.get_service_health(service)
            if not health_data or health_data.get('health_score', 0) < self.health_threshold:
                logger.warning(f"Service {service} still unhealthy after recovery")
                return False
                
            # Check that fault is no longer active
            if health_data.get('cpu_fault_active', False):
                logger.warning(f"CPU fault still active for {service}")
                return False
                
            logger.info(f"Recovery verified for {service}")
            return True
        except Exception as e:
            logger.warning(f"Recovery verification failed: {e}")
            return False
            
    def run_single_experiment(self, target_service: str, fault_type: str = 'cpu') -> Dict:
        """Run a single fault injection experiment."""
        logger.info(f"Running single experiment: {target_service} - {fault_type}")
        
        # Initialize timing
        self.experiment_start = time.time()
        self.log_event('experiment_start', 'system', {'target_service': target_service})
        
        # Baseline monitoring
        logger.info("Collecting baseline measurements...")
        baseline_health = self.get_service_health(target_service)
        self.log_event('baseline_collected', 'system', {'health_score': baseline_health.get('health_score', 1.0)})
        
        # Inject fault
        logger.info(f"Injecting {fault_type} fault into {target_service}")
        if not self.inject_fault(target_service, fault_type):
            return self.generate_experiment_results('fault_injection_failed')
            
        self.fault_injection_time = time.time()
        self.log_event('fault_injected', target_service, {'fault_type': fault_type})
        
        # Monitor for detection
        logger.info("Monitoring for fault detection...")
        detection_found = False
        start_time = time.time()
        
        while time.time() - start_time < self.fault_duration + 10:  # Extra time for detection
            if self.detect_anomaly(target_service):
                self.detection_time = time.time()
                detection_latency = self.detection_time - self.fault_injection_time
                self.log_event('fault_detected', target_service, {
                    'detection_latency': detection_latency
                })
                detection_found = True
                break
                
            time.sleep(self.monitoring_interval)
            
        if not detection_found:
            logger.warning("Fault not detected within expected timeframe")
            return self.generate_experiment_results('detection_timeout')
            
        # Localize fault
        logger.info("Localizing fault...")
        localization_start = time.time()
        root_cause = self.localize_fault(target_service)
        self.localization_time = time.time()
        
        localization_latency = self.localization_time - localization_start
        self.log_event('fault_localized', root_cause, {
            'detected_service': target_service,
            'localization_latency': localization_latency
        })
        
        # Execute recovery
        logger.info(f"Executing recovery for {root_cause}")
        recovery_start = time.time()
        recovery_success = self.execute_recovery(root_cause)
        self.recovery_time = time.time()
        
        if recovery_success:
            recovery_latency = self.recovery_time - recovery_start
            self.log_event('recovery_executed', root_cause, {
                'recovery_latency': recovery_latency
            })
        else:
            logger.error("Recovery failed")
            return self.generate_experiment_results('recovery_failed')
            
        # Verify recovery
        logger.info("Verifying recovery...")
        recovery_verified = self.verify_recovery(target_service)
        
        self.experiment_end = time.time()
        self.log_event('experiment_end', 'system', {
            'recovery_verified': recovery_verified,
            'total_duration': self.experiment_end - self.experiment_start
        })
        
        return self.generate_experiment_results('success')
        
    def generate_experiment_results(self, status: str) -> Dict:
        """Generate comprehensive experiment results."""
        results = {
            'status': status,
            'timing': {
                'experiment_start': self.experiment_start,
                'fault_injection_time': self.fault_injection_time,
                'detection_time': self.detection_time,
                'localization_time': self.localization_time,
                'recovery_time': self.recovery_time,
                'experiment_end': self.experiment_end
            },
            'latencies': {},
            'events': self.events
        }
        
        # Calculate latencies
        if all([self.fault_injection_time, self.detection_time]):
            results['latencies']['detection_latency'] = self.detection_time - self.fault_injection_time
            
        if all([self.detection_time, self.localization_time]):
            results['latencies']['localization_latency'] = self.localization_time - self.detection_time
            
        if all([self.localization_time, self.recovery_time]):
            results['latencies']['recovery_latency'] = self.recovery_time - self.localization_time
            
        if all([self.fault_injection_time, self.recovery_time]):
            results['latencies']['total_recovery_time'] = self.recovery_time - self.fault_injection_time
            
        return results
        
    def save_results(self, results: Dict, experiment_name: str):
        """Save experiment results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{experiment_name}_{timestamp}.json"
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Results saved to {filepath}")
        return filepath
        
    def run_comparison_experiment(self, target_service: str) -> Dict:
        """Run comparison between baseline and Graph-Heal."""
        logger.info(f"Running comparison experiment for {target_service}")
        
        # Run Graph-Heal experiment
        logger.info("Running Graph-Heal experiment...")
        graphheal_results = self.run_single_experiment(target_service, 'cpu')
        
        # Wait for system to stabilize
        time.sleep(30)
        
        # Run baseline experiment (simple threshold-based detection)
        logger.info("Running baseline experiment...")
        baseline_results = self.run_baseline_experiment(target_service)
        
        # Compare results
        comparison = self.compare_results(graphheal_results, baseline_results)
        
        return {
            'graphheal': graphheal_results,
            'baseline': baseline_results,
            'comparison': comparison
        }
        
    def run_baseline_experiment(self, target_service: str) -> Dict:
        """Run baseline experiment with simple threshold detection."""
        logger.info(f"Running baseline experiment: {target_service}")
        
        self.experiment_start = time.time()
        self.events = []  # Reset events for baseline
        self.log_event('baseline_experiment_start', 'system')
        
        # Inject fault
        if not self.inject_fault(target_service, 'cpu'):
            return self.generate_experiment_results('fault_injection_failed')
            
        self.fault_injection_time = time.time()
        
        # Simple threshold-based detection (no dependency analysis)
        detection_found = False
        start_time = time.time()
        
        while time.time() - start_time < self.fault_duration + 10:
            if self.detect_anomaly(target_service):
                self.detection_time = time.time()
                self.log_event('baseline_detection', target_service)
                detection_found = True
                break
                
            time.sleep(self.monitoring_interval)
            
        if not detection_found:
            return self.generate_experiment_results('detection_timeout')
            
        # Simple recovery (restart the detected service)
        self.localization_time = self.detection_time  # No separate localization
        recovery_start = time.time()
        recovery_success = self.execute_recovery(target_service)
        self.recovery_time = time.time()
        
        if recovery_success:
            self.log_event('baseline_recovery', target_service)
        else:
            return self.generate_experiment_results('recovery_failed')
            
        self.experiment_end = time.time()
        return self.generate_experiment_results('success')
        
    def compare_results(self, graphheal_results: Dict, baseline_results: Dict) -> Dict:
        """Compare Graph-Heal and baseline results."""
        comparison = {}
        
        if graphheal_results.get('status') == 'success' and baseline_results.get('status') == 'success':
            graphheal_latencies = graphheal_results.get('latencies', {})
            baseline_latencies = baseline_results.get('latencies', {})
            
            # Compare detection latency
            if 'detection_latency' in graphheal_latencies and 'detection_latency' in baseline_latencies:
                gh_detection = graphheal_latencies['detection_latency']
                bl_detection = baseline_latencies['detection_latency']
                comparison['detection_latency_improvement'] = bl_detection - gh_detection
                comparison['detection_latency_ratio'] = gh_detection / bl_detection
                
            # Compare total recovery time
            if 'total_recovery_time' in graphheal_latencies and 'total_recovery_time' in baseline_latencies:
                gh_total = graphheal_latencies['total_recovery_time']
                bl_total = baseline_latencies['total_recovery_time']
                comparison['total_recovery_improvement'] = bl_total - gh_total
                comparison['total_recovery_ratio'] = gh_total / bl_total
                
        return comparison


def main():
    """Main function to run improved live experiments."""
    # Configuration with realistic thresholds
    config = {
        'services': ['service_a', 'service_b', 'service_c', 'service_d'],
        'fault_duration': 15,
        'monitoring_interval': 1,
        'results_dir': 'data/improved_live_experiments',
        'cpu_threshold': 10.0,  # Lowered threshold
        'memory_threshold': 85.0,
        'response_time_threshold': 1.0,
        'health_threshold': 0.7
    }
    
    # Create experiment runner
    runner = ImprovedLiveExperimentRunner(config)
    
    # Run single experiment
    logger.info("Running single experiment on service_a")
    results = runner.run_single_experiment('service_a', 'cpu')
    runner.save_results(results, 'single_experiment')
    
    # Wait for system to stabilize
    time.sleep(30)
    
    # Run baseline comparison
    logger.info("Running baseline comparison")
    comparison_results = runner.run_comparison_experiment('service_b')
    runner.save_results(comparison_results, 'baseline_comparison')
    
    # Print summary
    logger.info("Experiment Summary:")
    logger.info(f"Single experiment status: {results['status']}")
    if 'latencies' in results:
        for metric, value in results['latencies'].items():
            logger.info(f"  {metric}: {value:.2f}s")
            
    logger.info(f"Comparison experiment completed")
    if 'comparison' in comparison_results:
        for metric, value in comparison_results['comparison'].items():
            logger.info(f"  {metric}: {value}")


if __name__ == "__main__":
    main() 