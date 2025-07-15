#!/usr/bin/env python3
"""
Detection Performance Measurement Script
Measures real detection latency, localization accuracy, and recovery effectiveness.
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

class DetectionPerformanceMeasurer:
    """Measures detection performance with high precision timing."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.services = config.get('services', [])
        self.num_trials = config.get('num_trials', 10)
        self.fault_duration = config.get('fault_duration', 20)
        self.monitoring_interval = config.get('monitoring_interval', 1)
        self.results_dir = Path(config.get('results_dir', 'data/performance_measurements'))
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance metrics
        self.detection_latencies = []
        self.localization_latencies = []
        self.recovery_latencies = []
        self.total_recovery_times = []
        self.detection_accuracy = []
        self.localization_accuracy = []
        
    def measure_detection_latency(self, target_service: str) -> Dict:
        """Measure detection latency for a single fault injection."""
        logger.info(f"Measuring detection latency for {target_service}")
        
        # Record start time
        start_time = time.time()
        
        # Inject fault
        fault_success = self.inject_fault(target_service)
        if not fault_success:
            return {'status': 'fault_injection_failed'}
            
        fault_time = time.time()
        
        # Monitor for detection
        detection_time = None
        detection_found = False
        
        while time.time() - fault_time < self.fault_duration + 10:
            if self.detect_anomaly(target_service):
                detection_time = time.time()
                detection_found = True
                break
            time.sleep(self.monitoring_interval)
            
        if not detection_found:
            return {'status': 'detection_timeout'}
            
        detection_latency = detection_time - fault_time
        
        # Clean up
        self.cleanup_fault(target_service)
        
        return {
            'status': 'success',
            'detection_latency': detection_latency,
            'fault_injection_time': fault_time - start_time
        }
        
    def measure_localization_accuracy(self, target_service: str) -> Dict:
        """Measure localization accuracy using dependency analysis."""
        logger.info(f"Measuring localization accuracy for {target_service}")
        
        # Inject fault
        fault_success = self.inject_fault(target_service)
        if not fault_success:
            return {'status': 'fault_injection_failed'}
            
        # Wait for detection
        detection_found = False
        start_time = time.time()
        
        while time.time() - start_time < self.fault_duration + 10:
            if self.detect_anomaly(target_service):
                detection_found = True
                break
            time.sleep(self.monitoring_interval)
            
        if not detection_found:
            return {'status': 'detection_timeout'}
            
        # Perform localization
        localized_service = self.localize_fault(target_service)
        
        # Check accuracy (localized service should match target)
        accuracy = localized_service == target_service
        
        # Clean up
        self.cleanup_fault(target_service)
        
        return {
            'status': 'success',
            'localization_accuracy': accuracy,
            'target_service': target_service,
            'localized_service': localized_service
        }
        
    def measure_recovery_effectiveness(self, target_service: str) -> Dict:
        """Measure recovery effectiveness and timing."""
        logger.info(f"Measuring recovery effectiveness for {target_service}")
        
        # Inject fault
        fault_success = self.inject_fault(target_service)
        if not fault_success:
            return {'status': 'fault_injection_failed'}
            
        fault_time = time.time()
        
        # Wait for detection
        detection_time = None
        detection_found = False
        
        while time.time() - fault_time < self.fault_duration + 10:
            if self.detect_anomaly(target_service):
                detection_time = time.time()
                detection_found = True
                break
            time.sleep(self.monitoring_interval)
            
        if not detection_found:
            return {'status': 'detection_timeout'}
            
        # Localize and recover
        localization_time = time.time()
        localized_service = self.localize_fault(target_service)
        
        # Execute recovery
        recovery_start = time.time()
        recovery_success = self.execute_recovery(localized_service)
        recovery_end = time.time()
        
        if not recovery_success:
            return {'status': 'recovery_failed'}
            
        # Wait for recovery to take effect
        time.sleep(5)
        
        # Verify recovery
        recovery_verified = self.verify_recovery(target_service)
        
        return {
            'status': 'success',
            'detection_latency': detection_time - fault_time,
            'localization_latency': localization_time - detection_time,
            'recovery_latency': recovery_end - recovery_start,
            'total_recovery_time': recovery_end - fault_time,
            'recovery_success': recovery_success,
            'recovery_verified': recovery_verified
        }
        
    def inject_fault(self, service: str) -> bool:
        """Inject a fault into a service."""
        try:
            cmd = [
                'python', 'scripts/inject_cpu_fault.py',
                '--service', service,
                '--duration', str(self.fault_duration)
            ]
            subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        except Exception as e:
            logger.error(f"Fault injection failed: {e}")
            return False
            
    def cleanup_fault(self, service: str):
        """Clean up injected fault."""
        try:
            # Stop the fault injection process
            cmd = ['pkill', '-f', f'inject_cpu_fault.py.*{service}']
            subprocess.run(cmd, capture_output=True)
        except Exception as e:
            logger.warning(f"Fault cleanup failed: {e}")
            
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
        """Localize fault using dependency analysis."""
        # Simple dependency-based localization
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
            # Wait a bit for service to fully recover
            time.sleep(5)
            
            # Check health
            if not self.check_service_health(service):
                return False
                
            # Check metrics are back to normal
            port = self.get_service_port(service)
            response = requests.get(f"http://localhost:{port}/metrics", timeout=5)
            
            if response.status_code == 200:
                metrics = self.parse_metrics(response.text)
                return not self.is_anomalous(metrics)
                
            return False
        except Exception as e:
            logger.warning(f"Recovery verification failed: {e}")
            return False
            
    def run_performance_measurement(self) -> Dict:
        """Run comprehensive performance measurement."""
        logger.info("Starting comprehensive performance measurement")
        
        results = {
            'detection_latency': [],
            'localization_accuracy': [],
            'recovery_effectiveness': [],
            'summary': {}
        }
        
        # Measure detection latency
        logger.info("Measuring detection latency...")
        for i in range(self.num_trials):
            logger.info(f"Detection latency trial {i+1}/{self.num_trials}")
            result = self.measure_detection_latency('service_a')
            if result['status'] == 'success':
                results['detection_latency'].append(result['detection_latency'])
            time.sleep(5)  # Wait between trials
            
        # Measure localization accuracy
        logger.info("Measuring localization accuracy...")
        for i in range(self.num_trials):
            logger.info(f"Localization accuracy trial {i+1}/{self.num_trials}")
            result = self.measure_localization_accuracy('service_b')
            if result['status'] == 'success':
                results['localization_accuracy'].append(result['localization_accuracy'])
            time.sleep(5)
            
        # Measure recovery effectiveness
        logger.info("Measuring recovery effectiveness...")
        for i in range(self.num_trials):
            logger.info(f"Recovery effectiveness trial {i+1}/{self.num_trials}")
            result = self.measure_recovery_effectiveness('service_c')
            if result['status'] == 'success':
                results['recovery_effectiveness'].append(result)
            time.sleep(10)  # Longer wait for recovery trials
            
        # Calculate summary statistics
        results['summary'] = self.calculate_summary_statistics(results)
        
        return results
        
    def calculate_summary_statistics(self, results: Dict) -> Dict:
        """Calculate summary statistics for the results."""
        summary = {}
        
        # Detection latency statistics
        if results['detection_latency']:
            latencies = results['detection_latency']
            summary['detection_latency'] = {
                'mean': statistics.mean(latencies),
                'median': statistics.median(latencies),
                'std': statistics.stdev(latencies) if len(latencies) > 1 else 0,
                'min': min(latencies),
                'max': max(latencies),
                'count': len(latencies)
            }
            
        # Localization accuracy
        if results['localization_accuracy']:
            accuracy = results['localization_accuracy']
            summary['localization_accuracy'] = {
                'accuracy_rate': sum(accuracy) / len(accuracy),
                'correct_localizations': sum(accuracy),
                'total_attempts': len(accuracy)
            }
            
        # Recovery effectiveness
        if results['recovery_effectiveness']:
            recovery_data = results['recovery_effectiveness']
            
            # Extract latencies
            detection_latencies = [r['detection_latency'] for r in recovery_data]
            localization_latencies = [r['localization_latency'] for r in recovery_data]
            recovery_latencies = [r['recovery_latency'] for r in recovery_data]
            total_times = [r['total_recovery_time'] for r in recovery_data]
            
            summary['recovery_effectiveness'] = {
                'detection_latency': {
                    'mean': statistics.mean(detection_latencies),
                    'median': statistics.median(detection_latencies),
                    'std': statistics.stdev(detection_latencies) if len(detection_latencies) > 1 else 0
                },
                'localization_latency': {
                    'mean': statistics.mean(localization_latencies),
                    'median': statistics.median(localization_latencies),
                    'std': statistics.stdev(localization_latencies) if len(localization_latencies) > 1 else 0
                },
                'recovery_latency': {
                    'mean': statistics.mean(recovery_latencies),
                    'median': statistics.median(recovery_latencies),
                    'std': statistics.stdev(recovery_latencies) if len(recovery_latencies) > 1 else 0
                },
                'total_recovery_time': {
                    'mean': statistics.mean(total_times),
                    'median': statistics.median(total_times),
                    'std': statistics.stdev(total_times) if len(total_times) > 1 else 0
                },
                'success_rate': sum(1 for r in recovery_data if r['recovery_success']) / len(recovery_data),
                'verification_rate': sum(1 for r in recovery_data if r['recovery_verified']) / len(recovery_data)
            }
            
        return summary
        
    def save_results(self, results: Dict, filename: str = None):
        """Save measurement results to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_measurement_{timestamp}.json"
            
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Results saved to {filepath}")
        return filepath
        
    def print_summary(self, results: Dict):
        """Print a summary of the measurement results."""
        summary = results.get('summary', {})
        
        logger.info("Performance Measurement Summary")
        logger.info("=" * 50)
        
        # Detection latency
        if 'detection_latency' in summary:
            dl = summary['detection_latency']
            logger.info(f"Detection Latency:")
            logger.info(f"  Mean: {dl['mean']:.3f}s")
            logger.info(f"  Median: {dl['median']:.3f}s")
            logger.info(f"  Std Dev: {dl['std']:.3f}s")
            logger.info(f"  Range: {dl['min']:.3f}s - {dl['max']:.3f}s")
            logger.info(f"  Trials: {dl['count']}")
            
        # Localization accuracy
        if 'localization_accuracy' in summary:
            la = summary['localization_accuracy']
            logger.info(f"Localization Accuracy:")
            logger.info(f"  Accuracy Rate: {la['accuracy_rate']:.1%}")
            logger.info(f"  Correct: {la['correct_localizations']}/{la['total_attempts']}")
            
        # Recovery effectiveness
        if 'recovery_effectiveness' in summary:
            re = summary['recovery_effectiveness']
            logger.info(f"Recovery Effectiveness:")
            logger.info(f"  Detection Latency: {re['detection_latency']['mean']:.3f}s")
            logger.info(f"  Localization Latency: {re['localization_latency']['mean']:.3f}s")
            logger.info(f"  Recovery Latency: {re['recovery_latency']['mean']:.3f}s")
            logger.info(f"  Total Recovery Time: {re['total_recovery_time']['mean']:.3f}s")
            logger.info(f"  Success Rate: {re['success_rate']:.1%}")
            logger.info(f"  Verification Rate: {re['verification_rate']:.1%}")


def main():
    """Main function to run performance measurement."""
    # Configuration
    config = {
        'services': ['service_a', 'service_b', 'service_c', 'service_d'],
        'num_trials': 5,  # Reduced for faster testing
        'fault_duration': 15,
        'monitoring_interval': 1,
        'results_dir': 'data/performance_measurements'
    }
    
    # Create measurer
    measurer = DetectionPerformanceMeasurer(config)
    
    # Run performance measurement
    logger.info("Starting performance measurement")
    results = measurer.run_performance_measurement()
    
    # Save results
    measurer.save_results(results)
    
    # Print summary
    measurer.print_summary(results)
    
    logger.info("Performance measurement completed")


if __name__ == "__main__":
    main() 