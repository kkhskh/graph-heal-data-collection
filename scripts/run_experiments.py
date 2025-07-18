import sys
import os
import subprocess
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import time
import csv
from datetime import datetime
import numpy as np
import pandas as pd
from graph_heal.graph_model import ServiceGraph, RealTimeServiceGraph
from graph_heal.anomaly_detection import StatisticalAnomalyDetector
from graph_heal.fault_localization import GraphBasedFaultLocalizer
from graph_heal.recovery_system import EnhancedRecoverySystem
from graph_heal.recovery.docker_adapter import DockerAdapter
from graph_heal.recovery.kubernetes_adapter import KubernetesAdapter
from graph_heal.service_monitor import ServiceMonitor
import argparse
from typing import Optional, Dict
import logging

# Default service configuration
DEFAULT_SERVICES = {
    'service_a': {'port': 5001, 'dependencies': ['service_b', 'service_c']},
    'service_b': {'port': 5002, 'dependencies': ['service_d']},
    'service_c': {'port': 5003, 'dependencies': ['service_d']},
    'service_d': {'port': 5004, 'dependencies': []}
}

logger = logging.getLogger(__name__)

class ExperimentRunner:
    def _initialize_service_graph(self, services) -> ServiceGraph:
        """Initialize the service graph with nodes and dependencies"""
        graph = ServiceGraph()
        # Add all services as nodes
        for service_name in services:
            graph.add_service(service_name)
        # Add dependencies based on configuration
        for service_name, config in services.items():
            for dep in config['dependencies']:
                graph.add_dependency(service_name, dep)
        return graph

    def __init__(self, services=None, duration_secs=300):
        self.services = services if services is not None else DEFAULT_SERVICES
        self.duration_secs = duration_secs
        self.graph = self._initialize_service_graph(self.services)
        self.graph_analyzer = GraphBasedFaultLocalizer(self.graph)
        self.anomaly_detector = StatisticalAnomalyDetector()
        
        adapter_name = os.getenv("RECOVERY_ADAPTER", "docker").lower()
        if adapter_name == "kubernetes":
            adapter = KubernetesAdapter()
        else:
            adapter = DockerAdapter()

        self.recovery_system = EnhancedRecoverySystem(self.graph, adapter=adapter)
        self.monitor = None  # ServiceMonitor not used with CLI
        self.event_log = []
        self.fault_labels = []
        
        # Initialize metrics storage
        self.metrics = {
            'propagation_metrics': {
                'propagation_detection_accuracy': [],
                'propagation_delay_estimation_error': [],
                'cross_layer_fault_detection': [],
                'cascading_failure_prediction': []
            },
            'localization_metrics': {
                'root_cause_accuracy': [],
                'localization_time': [],
                'false_root_cause_rate': [],
                'multi_hop_localization': []
            }
        }
        
    def _get_docker_stats(self, container_name: str) -> Optional[Dict[str, float]]:
        """Get CPU and Memory usage for a container using `docker stats`."""
        try:
            cmd = ["docker", "stats", "--no-stream", "--format", "{{json .}}", container_name]
            result = subprocess.check_output(cmd, stderr=subprocess.PIPE).decode().strip()
            if not result:
                return None
            
            stats = json.loads(result)
            
            cpu_usage = float(stats.get('CPUPerc', '0.0%').replace('%', ''))
            memory_usage = float(stats.get('MemPerc', '0.0%').replace('%', ''))
            
            return {'cpu_usage': cpu_usage, 'memory_usage': memory_usage}
        except (subprocess.CalledProcessError, json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Could not retrieve or parse stats for {container_name}: {e}")
            return None

    def _docker_update(self, container, cpu_quota=None, mem_limit=None, memswap_limit=None):
        cmd = ["docker", "update"]
        if cpu_quota is not None:
            cmd += [f"--cpu-quota={cpu_quota}"]
        if mem_limit is not None:
            cmd += [f"--memory={mem_limit}"]
        if memswap_limit is not None:
            cmd += [f"--memory-swap={memswap_limit}"]
        cmd.append(container)
        subprocess.run(cmd, check=True)

    def _docker_exec(self, container, exec_cmd):
        cmd = ["docker", "exec", container] + exec_cmd
        subprocess.run(cmd, check=True)

    def _docker_restart(self, container):
        subprocess.run(["docker", "restart", container], check=True)

    def run_experiment(self, experiment_id: int, fault_type: str = 'cpu', target_service: str = None):
        """Run a single experiment and collect metrics"""
        print(f"\nRunning experiment {experiment_id} with {fault_type} fault...")
        
        # Select target service
        if target_service is None:
            target_service = list(self.services.keys())[0]

        # Correct for docker-compose naming convention (e.g., service_a -> service-a)
        target_service = target_service.replace('_', '-')
        
        # Initialize experiment results with proper structure
        results = {
            'experiment_id': experiment_id,
            'fault_type': fault_type,
            'timestamp': time.time(),
            'target_service': target_service,
            'ground_truth': {
                'fault_source': target_service,
                'propagated_services': [],
                'propagation_delays': {},
                'cross_layer_faults': [],
                'cascading_failures': [],
                'root_causes': [target_service],  # Initialize with the target service
                'detection_times': [],
                'localization_times': [],  # Changed to list
                'multi_hop_paths': []
            },
            'graph_predictions': {
                'propagated_services': [],
                'propagation_delays': {},
                'cross_layer_faults': [],
                'cascading_failures': [],
                'root_causes': [],
                'localization_times': [],  # Changed to list
                'multi_hop_paths': []
            },
            'statistical_predictions': {
                'propagated_services': [],
                'propagation_delays': {},
                'cross_layer_faults': [],
                'cascading_failures': [],
                'root_causes': [],
                'localization_times': [],  # Changed to list
                'multi_hop_paths': []
            }
        }
        
        # Record fault injection in labels
        fault_start_time = time.time()
        self.fault_labels.append({
            'timestamp': datetime.fromtimestamp(fault_start_time).isoformat(),
            'service': target_service,
            'fault_type': fault_type,
            'duration': self.duration_secs
        })
        
        # Inject fault using Docker's built-in capabilities
        try:
            container_name = target_service.replace('_', '-')
            if fault_type == 'cpu':
                self._docker_update(container_name, cpu_quota=90000)
            elif fault_type == 'memory':
                self._docker_update(container_name, mem_limit='512m', memswap_limit='512m')
            elif fault_type == 'network':
                self._docker_exec(container_name, ['tc', 'qdisc', 'add', 'dev', 'eth0', 'root', 'netem', 'delay', '100ms'])
        except Exception as e:
            print(f"Error injecting fault: {e}")
            return results
        
        # Monitor propagation and collect metrics
        propagation_start = time.time()
        graph_localization_recorded = False
        stat_localization_recorded = False
        
        while time.time() - propagation_start < self.duration_secs:
            # Get current service states
            service_states = {}
            for service in self.services:
                try:
                    container_name = service.replace('_', '-')
                    inspect = subprocess.check_output(["docker", "inspect", container_name]).decode()
                    info = json.loads(inspect)[0]
                    status = info['State']['Status']
                    health = info['State'].get('Health', {}).get('Status', 'unknown')
                    
                    # Get metrics using docker stats
                    metrics = self._get_docker_stats(container_name) or {}
                    
                    service_states[service] = {
                        'status': status,
                        'health': health,
                        'metrics': {
                            'cpu_usage': metrics.get('cpu_usage'),
                            'memory_usage': metrics.get('memory_usage'),
                            'network_latency': None
                        }
                    }
                except Exception as e:
                    print(f"Error getting state for {service}: {e}")
                    service_states[service] = {
                        'status': 'unknown',
                        'health': 'unknown',
                        'metrics': {
                            'cpu_usage': None,
                            'memory_usage': None,
                            'network_latency': None,
                        },
                        'error': str(e)
                    }
            
            print(f"[DEBUG] service_statuses: {service_states}")
            
            # Detect anomalies
            anomalies = self.anomaly_detector.detect_anomalies(service_states)
            print(f"[DEBUG] StatisticalAnomalyDetector detected {len(anomalies)} anomalies: {anomalies}")
            
            # Localize faults using graph analyzer
            root_causes = self.graph_analyzer.localize_faults(service_states, anomalies)
            
            # Record localization times (only once per experiment)
            current_time = time.time() - propagation_start
            
            if root_causes and not graph_localization_recorded:
                results['graph_predictions']['localization_times'].append(current_time)
                results['graph_predictions']['root_causes'] = [cause['service_id'] for cause in root_causes]
                graph_localization_recorded = True
            
            if anomalies and not stat_localization_recorded:
                results['statistical_predictions']['localization_times'].append(current_time)
                results['statistical_predictions']['root_causes'] = [a['service_id'] for a in anomalies if 'service_id' in a]
                stat_localization_recorded = True
            
            # Update predictions
            graph_predictions = {
                'root_causes': [cause['service_id'] for cause in root_causes],
                'propagated_services': [cause['service_id'] for cause in root_causes]
            }

            stat_predictions = {
                'propagated_services': [a['service_id'] for a in anomalies if 'service_id' in a],
                'root_causes': [a['service_id'] for a in anomalies if 'service_id' in a]
            }
            
            # Update results
            self._update_results(results, service_states, graph_predictions, stat_predictions)
            
            time.sleep(1)
        
        # Calculate final metrics
        self._calculate_metrics(results)
        
        # Save results to file
        self._save_results(results, experiment_id, fault_type)
        
        # Clean up
        try:
            container_name = target_service.replace('_', '-')
            if fault_type == 'cpu':
                self._docker_update(container_name, cpu_quota=0)
            elif fault_type == 'memory':
                self._docker_update(container_name, mem_limit='1g', memswap_limit='1g')
            elif fault_type == 'network':
                self._docker_exec(container_name, ['tc', 'qdisc', 'del', 'dev', 'eth0', 'root'])
        except Exception as e:
            print(f"Error cleaning up: {e}")
        
        return results

    def save_fault_labels(self, output_file='fault_labels.csv'):
        """Save fault injection labels to CSV file"""
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['timestamp', 'service', 'fault_type', 'duration'])
            writer.writeheader()
            writer.writerows(self.fault_labels)
        print(f"Fault labels saved to {output_file}")
    
    def _update_results(self, results: dict, service_states: dict, 
                       graph_predictions: dict, stat_predictions: dict):
        """Update results with current state and predictions"""
        current_time = time.time()
        
        # Update ground truth
        for service, state in service_states.items():
            if state['status'] == 'unhealthy' or state['health'] == 'unhealthy':
                if service not in results['ground_truth']['propagated_services']:
                    results['ground_truth']['propagated_services'].append(service)
                    results['ground_truth']['propagation_delays'][service] = current_time - results['timestamp']
        
        # Update graph predictions
        for service in graph_predictions.get('propagated_services', []):
            if service not in results['graph_predictions']['propagated_services']:
                results['graph_predictions']['propagated_services'].append(service)
                results['graph_predictions']['propagation_delays'][service] = current_time - results['timestamp']
        
        # Update statistical predictions
        for service in stat_predictions.get('propagated_services', []):
            if service not in results['statistical_predictions']['propagated_services']:
                results['statistical_predictions']['propagated_services'].append(service)
                results['statistical_predictions']['propagation_delays'][service] = current_time - results['timestamp']
    
    def _calculate_metrics(self, results: dict):
        """Calculate final performance metrics for the experiment."""
        try:
            # Propagation Detection Accuracy
            graph_accuracy = len(set(results['graph_predictions']['propagated_services']) & 
                               set(results['ground_truth']['propagated_services'])) / \
                            max(len(results['ground_truth']['propagated_services']), 1)
            
            stat_accuracy = len(set(results['statistical_predictions']['propagated_services']) & 
                              set(results['ground_truth']['propagated_services'])) / \
                           max(len(results['ground_truth']['propagated_services']), 1)
            
            self.metrics['propagation_metrics']['propagation_detection_accuracy'].append({
                'graph': graph_accuracy,
                'statistical': stat_accuracy
            })
            
            # Propagation Delay Estimation Error
            graph_delay_error = np.mean([
                abs(results['graph_predictions']['propagation_delays'].get(s, 0) - 
                    results['ground_truth']['propagation_delays'].get(s, 0))
                for s in results['ground_truth']['propagated_services']
            ]) if results['ground_truth']['propagated_services'] else 0
            
            stat_delay_error = np.mean([
                abs(results['statistical_predictions']['propagation_delays'].get(s, 0) - 
                    results['ground_truth']['propagation_delays'].get(s, 0))
                for s in results['ground_truth']['propagated_services']
            ]) if results['ground_truth']['propagated_services'] else 0
            
            self.metrics['propagation_metrics']['propagation_delay_estimation_error'].append({
                'graph': graph_delay_error,
                'statistical': stat_delay_error
            })
            
            # Root Cause Accuracy
            graph_root_cause = results['graph_predictions']['root_causes'][0] if results['graph_predictions']['root_causes'] else None
            stat_root_cause = results['statistical_predictions']['root_causes'][0] if results['statistical_predictions']['root_causes'] else None
            true_root_cause = results['ground_truth']['root_causes'][0] if results['ground_truth']['root_causes'] else None
            
            self.metrics['localization_metrics']['root_cause_accuracy'].append({
                'graph': 1.0 if graph_root_cause == true_root_cause else 0.0,
                'statistical': 1.0 if stat_root_cause == true_root_cause else 0.0
            })
            
            # Localization Time - Now safely accessing lists
            graph_time = results['graph_predictions']['localization_times'][0] if results['graph_predictions']['localization_times'] else float('inf')
            stat_time = results['statistical_predictions']['localization_times'][0] if results['statistical_predictions']['localization_times'] else float('inf')
            
            self.metrics['localization_metrics']['localization_time'].append({
                'graph': graph_time,
                'statistical': stat_time
            })
            
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Error calculating metrics: {e}")
            # Add default metrics to maintain consistency
            self.metrics['propagation_metrics']['propagation_detection_accuracy'].append({
                'graph': 0.0,
                'statistical': 0.0
            })
            self.metrics['propagation_metrics']['propagation_delay_estimation_error'].append({
                'graph': float('inf'),
                'statistical': float('inf')
            })
            self.metrics['localization_metrics']['root_cause_accuracy'].append({
                'graph': 0.0,
                'statistical': 0.0
            })
            self.metrics['localization_metrics']['localization_time'].append({
                'graph': float('inf'),
                'statistical': float('inf')
            })
    
    def _save_results(self, results: dict, experiment_id: int, fault_type: str):
        """Append results to CSV files, creating them if they don't exist."""
        output_dir = "data/fault_injection"
        os.makedirs(output_dir, exist_ok=True)

        # Convert timestamp objects to strings for JSON serialization
        if isinstance(results.get('timestamp'), float):
            results['timestamp'] = datetime.fromtimestamp(results['timestamp']).isoformat()

        # Create results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)
        
        # Save detailed results
        results_filename = f'results/{fault_type}_experiment_{experiment_id}.json'
        with open(results_filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save aggregated metrics
        metrics_filename = f'results/metrics_{fault_type}.json'
        with open(metrics_filename, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        print(f"Results saved to {results_filename}")
        print(f"Metrics saved to {metrics_filename}")

def main():
    parser = argparse.ArgumentParser(description="Run fault injection experiments.")
    parser.add_argument('--duration', type=int, default=300, help='Duration of each experiment in seconds.')
    parser.add_argument('--services', type=str, default='service-a service-b service-c service-d', help='Space-separated list of services to target.')
    parser.add_argument('--start-id', type=int, default=0, help='Starting experiment ID.')
    parser.add_argument('--end-id', type=int, default=99, help='Ending experiment ID.')
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    services_list = args.services.split(' ')
    
    # Map service names back to underscore format for internal use
    service_name_map = {
        'service-a': 'service_a',
        'service-b': 'service_b', 
        'service-c': 'service_c',
        'service-d': 'service_d'
    }
    
    mapped_services = []
    for service in services_list:
        if service in service_name_map:
            mapped_services.append(service_name_map[service])
        elif service in DEFAULT_SERVICES:
            mapped_services.append(service)
    
    services_config = {service: DEFAULT_SERVICES[service] for service in mapped_services if service in DEFAULT_SERVICES}
    
    runner = ExperimentRunner(services=services_config, duration_secs=args.duration)
    
    # Define the sequence of fault types
    fault_types = ['cpu', 'memory', 'network']
    
    # Main experiment loop
    experiment_id = args.start_id
    while experiment_id <= args.end_id:
        for service in mapped_services:
            if experiment_id > args.end_id:
                break
            fault_type = fault_types[experiment_id % len(fault_types)]
            runner.run_experiment(experiment_id, fault_type, service)
            experiment_id += 1

    # Save all collected fault labels at the end
    runner.save_fault_labels('fault_labels.csv')
    print("Experiments complete. Results saved.")

if __name__ == "__main__":
    main()
