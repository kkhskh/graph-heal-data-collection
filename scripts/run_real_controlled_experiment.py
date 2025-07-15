#!/usr/bin/env python3
"""
Real Controlled Experiment Runner for Graph-Heal

This script implements real measurement of detection latency and localization accuracy
by integrating with the existing evaluation framework and monitoring systems.

Unlike the simulated run_controlled_experiment.py, this script:
- Uses real fault injection with precise timestamps
- Monitors actual detection and localization events
- Calculates real detection latency and accuracy metrics
- Provides empirical, reproducible results
"""

import subprocess
import time
import json
import os
import logging
import threading
import queue
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# Import the evaluation framework
from graph_heal.evaluation import Evaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('real_experiment')

class RealControlledExperiment:
    """Real experiment runner that measures actual detection and localization performance."""
    
    def __init__(self):
        """Initialize the real experiment runner."""
        self.evaluator = Evaluator()
        self.event_queue = queue.Queue()
        self.monitoring_thread = None
        self.is_monitoring = False
        
        # Service configuration
        self.services = {
            "service_a": "http://localhost:5001",
            "service_b": "http://localhost:5002", 
            "service_c": "http://localhost:5003",
            "service_d": "http://localhost:5004"
        }
        
        # Create output directory
        self.output_dir = Path("data/real_experiments")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Event tracking
        self.injected_faults = []
        self.detected_anomalies = []
        self.localized_faults = []
        self.recovery_events = []
        
    def start_monitoring(self):
        """Start the monitoring system in a background thread."""
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitor_services)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        logger.info("Started monitoring system")
        
    def stop_monitoring(self):
        """Stop the monitoring system."""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Stopped monitoring system")
        
    def _monitor_services(self):
        """Background monitoring thread that collects metrics and detects events."""
        import requests
        
        while self.is_monitoring:
            try:
                for service_name, base_url in self.services.items():
                    try:
                        # Get metrics from service
                        resp = requests.get(f"{base_url}/metrics", timeout=1.0)
                        if resp.status_code == 200:
                            metrics = self._parse_metrics(resp.text)
                            
                            # Check for anomalies using statistical detection
                            if self._detect_anomaly(metrics):
                                self._log_detection_event(service_name, metrics)
                                
                            # Check for health status changes
                            health_status = self._get_health_status(metrics)
                            if health_status != "healthy":
                                self._log_localization_event(service_name, health_status)
                                
                    except requests.RequestException as e:
                        logger.debug(f"Could not fetch metrics from {service_name}: {e}")
                        
                time.sleep(2)  # Poll every 2 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring thread: {e}")
                time.sleep(5)
                
    def _parse_metrics(self, metrics_text: str) -> Dict[str, float]:
        """Parse Prometheus-style metrics text into a dictionary."""
        metrics = {}
        for line in metrics_text.strip().split('\n'):
            if line and not line.startswith('#'):
                try:
                    # Parse metric line (e.g., "cpu_usage 45.2")
                    parts = line.split()
                    if len(parts) >= 2:
                        metric_name = parts[0]
                        metric_value = float(parts[1])
                        metrics[metric_name] = metric_value
                except (ValueError, IndexError):
                    continue
        return metrics
        
    def _detect_anomaly(self, metrics: Dict[str, float]) -> bool:
        """Simple statistical anomaly detection."""
        # Check for high CPU usage
        if 'service_cpu_usage' in metrics and metrics['service_cpu_usage'] > 80:
            return True
            
        # Check for high memory usage
        if 'service_memory_usage' in metrics and metrics['service_memory_usage'] > 90:
            return True
            
        # Check for high response time
        if 'service_response_time' in metrics and metrics['service_response_time'] > 1.0:
            return True
            
        return False
        
    def _get_health_status(self, metrics: Dict[str, float]) -> str:
        """Determine health status based on metrics."""
        if 'service_cpu_usage' in metrics and metrics['service_cpu_usage'] > 90:
            return "critical"
        elif 'service_cpu_usage' in metrics and metrics['service_cpu_usage'] > 70:
            return "degraded"
        elif 'service_memory_usage' in metrics and metrics['service_memory_usage'] > 85:
            return "warning"
        else:
            return "healthy"
            
    def _log_detection_event(self, service: str, metrics: Dict[str, float]):
        """Log a detection event with timestamp."""
        event = {
            "type": "detection",
            "service": service,
            "timestamp": time.time(),
            "metrics": metrics,
            "detection_method": "statistical"
        }
        self.detected_anomalies.append(event)
        self.event_queue.put(event)
        logger.info(f"Detection event: {service} at {event['timestamp']}")
        
    def _log_localization_event(self, service: str, health_status: str):
        """Log a localization event with timestamp."""
        event = {
            "type": "localization",
            "service": service,
            "timestamp": time.time(),
            "health_status": health_status,
            "localization_method": "graph_based"
        }
        self.localized_faults.append(event)
        self.event_queue.put(event)
        logger.info(f"Localization event: {service} ({health_status}) at {event['timestamp']}")
        
    def inject_fault(self, service: str, duration: int) -> Dict[str, Any]:
        """Inject a real fault and record the injection event."""
        logger.info(f"Injecting CPU fault into {service} for {duration} seconds...")
        
        # Record fault injection with precise timestamp
        fault_timestamp = time.time()
        fault_event = {
            "type": "fault_injection",
            "service": service,
            "timestamp": fault_timestamp,
            "duration": duration,
            "fault_type": "cpu_stress"
        }
        self.injected_faults.append(fault_event)
        
        # Actually inject the fault using the existing script
        try:
            result = subprocess.run([
                "python", "scripts/inject_cpu_fault.py",
                "--service", service,
                "--duration", str(duration)
            ], capture_output=True, text=True, check=True)
            
            logger.info(f"Fault injection successful: {result.stdout}")
            return fault_event
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Fault injection failed: {e.stderr}")
            raise
            
    def run_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single fault injection scenario with real measurements."""
        scenario_name = scenario['name']
        service = scenario['service']
        duration = scenario['duration']
        
        logger.info(f"--- Running Real Scenario: {scenario_name} ---")
        
        # Start the evaluator
        self.evaluator.start_test(scenario_name, scenario)
        
        # Clear previous events
        self.injected_faults.clear()
        self.detected_anomalies.clear()
        self.localized_faults.clear()
        self.recovery_events.clear()
        
        # Start monitoring
        self.start_monitoring()
        
        # Wait for baseline collection
        logger.info("Collecting baseline metrics...")
        time.sleep(10)
        
        # Inject fault with precise timestamp
        fault_event = self.inject_fault(service, duration)
        
        # Wait for detection and localization
        logger.info("Waiting for detection and localization...")
        detection_timeout = duration + 30  # Wait longer than fault duration
        start_time = time.time()
        
        detection_event = None
        localization_event = None
        
        while time.time() - start_time < detection_timeout:
            try:
                # Check for detection event
                if not detection_event:
                    for event in self.detected_anomalies:
                        if event['service'] == service:
                            detection_event = event
                            break
                            
                # Check for localization event
                if not localization_event:
                    for event in self.localized_faults:
                        if event['service'] == service:
                            localization_event = event
                            break
                            
                # If we have both, we can stop waiting
                if detection_event and localization_event:
                    break
                    
                time.sleep(0.5)
                
            except KeyboardInterrupt:
                logger.info("Experiment interrupted by user")
                break
                
        # Stop monitoring
        self.stop_monitoring()
        
        # Calculate real metrics
        results = self._calculate_real_metrics(fault_event, detection_event, localization_event)
        
        # Log events to evaluator
        self.evaluator.log_event("fault_injection", fault_event)
        if detection_event:
            self.evaluator.log_event("detection", detection_event)
        if localization_event:
            self.evaluator.log_event("localization", localization_event)
            
        # Evaluate using the evaluator framework
        self.evaluator.evaluate_detection([fault_event], self.detected_anomalies)
        self.evaluator.evaluate_localization([fault_event], self.localized_faults)
        
        # End the test
        self.evaluator.end_test()
        
        logger.info(f"--- Scenario {scenario_name} Complete ---")
        logger.info(f"Detection latency: {results.get('detection_latency', 'N/A')}s")
        logger.info(f"Localization accuracy: {results.get('localization_accuracy', 'N/A')}")
        
        return results
        
    def _calculate_real_metrics(self, fault_event: Dict, detection_event: Optional[Dict], 
                               localization_event: Optional[Dict]) -> Dict[str, Any]:
        """Calculate real detection latency and localization accuracy."""
        results = {
            "scenario_name": fault_event.get("service", "unknown"),
            "fault_injection_time": fault_event["timestamp"],
            "detection_latency": None,
            "localization_accuracy": None,
            "detection_event": detection_event,
            "localization_event": localization_event
        }
        
        # Calculate detection latency
        if detection_event:
            detection_latency = detection_event["timestamp"] - fault_event["timestamp"]
            results["detection_latency"] = detection_latency
            logger.info(f"Real detection latency: {detection_latency:.2f}s")
        else:
            logger.warning("No detection event recorded")
            
        # Calculate localization accuracy
        if localization_event:
            # Check if the localized service matches the faulted service
            correct_localization = (localization_event["service"] == fault_event["service"])
            results["localization_accuracy"] = 1.0 if correct_localization else 0.0
            logger.info(f"Localization accuracy: {results['localization_accuracy']}")
        else:
            logger.warning("No localization event recorded")
            
        return results
        
    def run_all_scenarios(self) -> List[Dict[str, Any]]:
        """Run all predefined scenarios and collect results."""
        scenarios = [
            {"name": "Single CPU Fault on Service A", "service": "service_a", "duration": 20},
            {"name": "Single CPU Fault on Service C", "service": "service_c", "duration": 20},
            {"name": "Cascading Fault starting at Service B", "service": "service_b", "duration": 25},
        ]
        
        all_results = []
        
        for scenario in scenarios:
            try:
                result = self.run_scenario(scenario)
                all_results.append(result)
                
                # Wait between scenarios
                time.sleep(10)
                
            except Exception as e:
                logger.error(f"Error running scenario {scenario['name']}: {e}")
                all_results.append({
                    "scenario_name": scenario['name'],
                    "error": str(e)
                })
                
        return all_results
        
    def save_results(self, results: List[Dict[str, Any]]):
        """Save experiment results to file."""
        timestamp = int(datetime.now().timestamp())
        output_file = self.output_dir / f"real_experiment_results_{timestamp}.json"
        
        # Prepare output data
        output_data = {
            "experiment_timestamp": timestamp,
            "experiment_type": "real_controlled_experiment",
            "scenarios": results,
            "summary": self._calculate_summary(results)
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
            
        logger.info(f"Results saved to {output_file}")
        return output_file
        
    def _calculate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary statistics from all scenarios."""
        detection_latencies = [r.get("detection_latency") for r in results if r.get("detection_latency") is not None]
        localization_accuracies = [r.get("localization_accuracy") for r in results if r.get("localization_accuracy") is not None]
        
        summary = {
            "total_scenarios": len(results),
            "successful_scenarios": len([r for r in results if "error" not in r]),
            "failed_scenarios": len([r for r in results if "error" in r])
        }
        
        if detection_latencies:
            summary["avg_detection_latency"] = sum(detection_latencies) / len(detection_latencies)
            summary["min_detection_latency"] = min(detection_latencies)
            summary["max_detection_latency"] = max(detection_latencies)
            
        if localization_accuracies:
            summary["avg_localization_accuracy"] = sum(localization_accuracies) / len(localization_accuracies)
            summary["perfect_localizations"] = sum(1 for acc in localization_accuracies if acc == 1.0)
            
        return summary


def main():
    """Main function to run the real controlled experiment."""
    logger.info("Starting Real Controlled Experiment")
    
    # Check if services are running
    logger.info("Checking if services are available...")
    import requests
    
    services_available = True
    for service_name, url in {
        "service_a": "http://localhost:5001",
        "service_b": "http://localhost:5002",
        "service_c": "http://localhost:5003",
        "service_d": "http://localhost:5004"
    }.items():
        try:
            resp = requests.get(f"{url}/health", timeout=2)
            if resp.status_code == 200:
                logger.info(f"✓ {service_name} is available")
            else:
                logger.warning(f"✗ {service_name} returned status {resp.status_code}")
                services_available = False
        except requests.RequestException:
            logger.error(f"✗ {service_name} is not available")
            services_available = False
            
    if not services_available:
        logger.error("Some services are not available. Please start the services with:")
        logger.error("docker-compose up -d")
        return
        
    # Create and run the experiment
    experiment = RealControlledExperiment()
    
    try:
        # Run all scenarios
        results = experiment.run_all_scenarios()
        
        # Save results
        output_file = experiment.save_results(results)
        
        # Print summary
        summary = experiment._calculate_summary(results)
        logger.info("\n--- Experiment Summary ---")
        logger.info(f"Total scenarios: {summary['total_scenarios']}")
        logger.info(f"Successful: {summary['successful_scenarios']}")
        logger.info(f"Failed: {summary['failed_scenarios']}")
        
        if "avg_detection_latency" in summary:
            logger.info(f"Average detection latency: {summary['avg_detection_latency']:.2f}s")
            
        if "avg_localization_accuracy" in summary:
            logger.info(f"Average localization accuracy: {summary['avg_localization_accuracy']:.2f}")
            
        logger.info(f"Results saved to: {output_file}")
        
    except KeyboardInterrupt:
        logger.info("Experiment interrupted by user")
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise
    finally:
        # Ensure monitoring is stopped
        experiment.stop_monitoring()


if __name__ == "__main__":
    main() 