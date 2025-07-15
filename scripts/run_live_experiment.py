#!/usr/bin/env python3
"""
Live Experiment Runner for Graph-Heal
Implements real fault injection and measurement of detection, localization, and recovery performance.
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

class LiveExperimentRunner:
    """Runs live experiments with real fault injection and measurement."""
    
    def __init__(self, experiment_config: Dict):
        self.config = experiment_config
        self.services = experiment_config.get('services', [])
        self.fault_duration = experiment_config.get('fault_duration', 30)
        self.monitoring_interval = experiment_config.get('monitoring_interval', 2)
        self.results_dir = Path(experiment_config.get('results_dir', 'data/live_experiments'))
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
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
        
    def check_service_health(self, service: str) -> bool:
        """Check if a service is healthy."""
        try:
            url = f"http://localhost:{self.get_service_port(service)}/health"
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Health check failed for {service}: {e}")
            return False
            
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
            if fault_type == 'cpu':
                # Use the existing fault injection script
                cmd = [
                    'python', 'scripts/inject_cpu_fault.py',
                    '--service', service,
                    '--duration', str(self.fault_duration)
                ]
                logger.info(f"Injecting CPU fault into {service}")
                subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                return True
            else:
                logger.error(f"Unsupported fault type: {fault_type}")
                return False
        except Exception as e:
            logger.error(f"Fault injection failed: {e}")
            return False
            
    def monitor_services(self) -> Dict[str, Dict]:
        """Monitor all services and return their current state."""
        service_states = {}
        
        for service in self.services:
            try:
                # Get metrics
                metrics_url = f"http://localhost:{self.get_service_port(service)}/metrics"
                response = requests.get(metrics_url, timeout=5)
                
                if response.status_code == 200:
                    metrics = self.parse_metrics(response.text)
                    health = self.check_service_health(service)
                    
                    service_states[service] = {
                        'healthy': health,
                        'metrics': metrics,
                        'timestamp': time.time()
                    }
                else:
                    service_states[service] = {
                        'healthy': False,
                        'metrics': {},
                        'timestamp': time.time()
                    }
            except Exception as e:
                logger.warning(f"Failed to monitor {service}: {e}")
                service_states[service] = {
                    'healthy': False,
                    'metrics': {},
                    'timestamp': time.time()
                }
                
        return service_states
        
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
        
    def detect_anomalies(self, service_states: Dict[str, Dict]) -> List[Dict]:
        """Detect anomalies in service states."""
        anomalies = []
        
        for service, state in service_states.items():
            if not state['healthy']:
                anomalies.append({
                    'service': service,
                    'type': 'health_check_failure',
                    'severity': 'high',
                    'timestamp': state['timestamp']
                })
                continue
                
            metrics = state['metrics']
            if not metrics:
                continue
                
            # Simple threshold-based anomaly detection
            if 'cpu_usage' in metrics and metrics['cpu_usage'] > 80:
                anomalies.append({
                    'service': service,
                    'type': 'high_cpu_usage',
                    'severity': 'medium',
                    'timestamp': state['timestamp'],
                    'value': metrics['cpu_usage']
                })
                
            if 'memory_usage' in metrics and metrics['memory_usage'] > 85:
                anomalies.append({
                    'service': service,
                    'type': 'high_memory_usage',
                    'severity': 'medium',
                    'timestamp': state['timestamp'],
                    'value': metrics['memory_usage']
                })
                
            if 'response_time' in metrics and metrics['response_time'] > 1.0:
                anomalies.append({
                    'service': service,
                    'type': 'high_response_time',
                    'severity': 'medium',
                    'timestamp': state['timestamp'],
                    'value': metrics['response_time']
                })
                
        return anomalies
        
    def localize_fault(self, anomalies: List[Dict], service_states: Dict[str, Dict]) -> Optional[str]:
        """Localize the root cause of faults using dependency analysis."""
        if not anomalies:
            return None
            
        # Simple dependency-based localization
        # In a real implementation, this would use the service graph
        service_dependencies = {
            'service_a': ['service_b', 'service_c'],
            'service_b': ['service_d'],
            'service_c': ['service_d'],
            'service_d': []
        }
        
        # Find the service with the most severe anomaly
        most_severe = max(anomalies, key=lambda x: self.get_severity_score(x))
        
        # Check if this service has dependencies that are also anomalous
        service = most_severe['service']
        dependencies = service_dependencies.get(service, [])
        
        for dep in dependencies:
            if dep in service_states and not service_states[dep]['healthy']:
                # If a dependency is unhealthy, it's likely the root cause
                return dep
                
        # If no unhealthy dependencies, the current service is the root cause
        return service
        
    def get_severity_score(self, anomaly: Dict) -> int:
        """Get a severity score for an anomaly."""
        severity_map = {'low': 1, 'medium': 2, 'high': 3}
        return severity_map.get(anomaly.get('severity', 'low'), 1)
        
    def execute_recovery(self, service: str) -> bool:
        """Execute recovery action for a service."""
        try:
            # Simple restart via Docker
            cmd = ['docker', 'restart', service]
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
            
    def run_experiment(self, target_service: str, fault_type: str = 'cpu') -> Dict:
        """Run a complete experiment with fault injection and measurement."""
        logger.info(f"Starting live experiment on {target_service}")
        
        # Initialize timing
        self.experiment_start = time.time()
        self.log_event('experiment_start', 'system', {'target_service': target_service})
        
        # Baseline monitoring
        logger.info("Collecting baseline measurements...")
        baseline_states = self.monitor_services()
        self.log_event('baseline_collected', 'system', {'services': list(baseline_states.keys())})
        
        # Inject fault
        logger.info(f"Injecting {fault_type} fault into {target_service}")
        if not self.inject_fault(target_service, fault_type):
            logger.error("Fault injection failed")
            return self.generate_experiment_results('failed_injection')
            
        self.fault_injection_time = time.time()
        self.log_event('fault_injected', target_service, {'fault_type': fault_type})
        
        # Monitor for detection
        logger.info("Monitoring for fault detection...")
        detection_found = False
        start_time = time.time()
        
        while time.time() - start_time < self.fault_duration + 30:  # Extra time for detection
            service_states = self.monitor_services()
            anomalies = self.detect_anomalies(service_states)
            
            if anomalies:
                self.detection_time = time.time()
                self.log_event('fault_detected', target_service, {
                    'anomalies': len(anomalies),
                    'detection_latency': self.detection_time - self.fault_injection_time
                })
                detection_found = True
                break
                
            time.sleep(self.monitoring_interval)
            
        if not detection_found:
            logger.warning("Fault not detected within expected timeframe")
            return self.generate_experiment_results('detection_timeout')
            
        # Localize fault
        logger.info("Localizing fault...")
        root_cause = self.localize_fault(anomalies, service_states)
        
        if root_cause:
            self.localization_time = time.time()
            self.log_event('fault_localized', root_cause, {
                'detected_service': target_service,
                'localization_latency': self.localization_time - self.detection_time
            })
        else:
            logger.warning("Failed to localize fault")
            return self.generate_experiment_results('localization_failed')
            
        # Execute recovery
        logger.info(f"Executing recovery for {root_cause}")
        recovery_success = self.execute_recovery(root_cause)
        
        if recovery_success:
            self.recovery_time = time.time()
            self.log_event('recovery_executed', root_cause, {
                'recovery_latency': self.recovery_time - self.localization_time
            })
        else:
            logger.error("Recovery failed")
            return self.generate_experiment_results('recovery_failed')
            
        # Wait for recovery to take effect
        logger.info("Waiting for recovery to take effect...")
        time.sleep(10)
        
        # Final health check
        final_states = self.monitor_services()
        all_healthy = all(state['healthy'] for state in final_states.values())
        
        self.experiment_end = time.time()
        self.log_event('experiment_end', 'system', {
            'all_healthy': all_healthy,
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
        
    def run_baseline_comparison(self, target_service: str) -> Dict:
        """Run comparison between Graph-Heal and baseline detection."""
        logger.info("Running baseline comparison experiment")
        
        # Run Graph-Heal experiment
        logger.info("Running Graph-Heal experiment...")
        graphheal_results = self.run_experiment(target_service, 'cpu')
        
        # Wait for system to stabilize
        time.sleep(30)
        
        # Run baseline experiment (simple threshold-based detection)
        logger.info("Running baseline experiment...")
        baseline_results = self.run_baseline_experiment(target_service)
        
        # Compare results
        comparison = {
            'graphheal': graphheal_results,
            'baseline': baseline_results,
            'comparison': self.compare_results(graphheal_results, baseline_results)
        }
        
        return comparison
        
    def run_baseline_experiment(self, target_service: str) -> Dict:
        """Run baseline experiment with simple threshold detection."""
        # Simplified version of the experiment that only uses basic threshold detection
        # without dependency analysis
        logger.info("Running baseline experiment with simple threshold detection")
        
        self.experiment_start = time.time()
        self.events = []  # Reset events for baseline
        self.log_event('baseline_experiment_start', 'system')
        
        # Inject fault
        if not self.inject_fault(target_service, 'cpu'):
            return self.generate_experiment_results('failed_injection')
            
        self.fault_injection_time = time.time()
        
        # Simple threshold-based detection
        detection_found = False
        start_time = time.time()
        
        while time.time() - start_time < self.fault_duration + 30:
            service_states = self.monitor_services()
            
            # Simple threshold check (no dependency analysis)
            for service, state in service_states.items():
                metrics = state.get('metrics', {})
                if 'cpu_usage' in metrics and metrics['cpu_usage'] > 80:
                    self.detection_time = time.time()
                    self.log_event('baseline_detection', service, {
                        'cpu_usage': metrics['cpu_usage']
                    })
                    detection_found = True
                    break
                    
            if detection_found:
                break
                
            time.sleep(self.monitoring_interval)
            
        if not detection_found:
            return self.generate_experiment_results('detection_timeout')
            
        # Simple recovery (restart the detected service)
        self.localization_time = self.detection_time  # No separate localization
        recovery_success = self.execute_recovery(target_service)
        
        if recovery_success:
            self.recovery_time = time.time()
            self.log_event('baseline_recovery', target_service)
        else:
            return self.generate_experiment_results('recovery_failed')
            
        self.experiment_end = time.time()
        return self.generate_experiment_results('success')
        
    def compare_results(self, graphheal_results: Dict, baseline_results: Dict) -> Dict:
        """Compare Graph-Heal and baseline results."""
        comparison = {}
        
        # Compare detection latency
        gh_detection = graphheal_results.get('latencies', {}).get('detection_latency')
        bl_detection = baseline_results.get('latencies', {}).get('detection_latency')
        
        if gh_detection and bl_detection:
            comparison['detection_latency_improvement'] = bl_detection - gh_detection
            comparison['detection_latency_ratio'] = gh_detection / bl_detection
            
        # Compare total recovery time
        gh_total = graphheal_results.get('latencies', {}).get('total_recovery_time')
        bl_total = baseline_results.get('latencies', {}).get('total_recovery_time')
        
        if gh_total and bl_total:
            comparison['total_recovery_improvement'] = bl_total - gh_total
            comparison['total_recovery_ratio'] = gh_total / bl_total
            
        # Compare accuracy (assuming Graph-Heal is more accurate due to dependency analysis)
        comparison['localization_accuracy'] = {
            'graphheal': 'dependency_aware',
            'baseline': 'threshold_only'
        }
        
        return comparison


def main():
    """Main function to run live experiments."""
    # Configuration
    config = {
        'services': ['service_a', 'service_b', 'service_c', 'service_d'],
        'fault_duration': 30,
        'monitoring_interval': 2,
        'results_dir': 'data/live_experiments'
    }
    
    # Create experiment runner
    runner = LiveExperimentRunner(config)
    
    # Run single experiment
    logger.info("Running single experiment on service_a")
    results = runner.run_experiment('service_a', 'cpu')
    runner.save_results(results, 'single_experiment')
    
    # Run baseline comparison
    logger.info("Running baseline comparison")
    comparison_results = runner.run_baseline_comparison('service_b')
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