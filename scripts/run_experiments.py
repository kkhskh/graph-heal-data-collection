import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import time
from datetime import datetime
import numpy as np
import docker
import pandas as pd
from graph_heal.graph_model import ServiceGraph, RealTimeServiceGraph
from graph_heal.anomaly_detection import AnomalyDetector
from graph_heal.fault_localization import FaultLocalizer
from graph_heal.recovery_system import EnhancedRecoverySystem
from graph_heal.recovery.docker_adapter import DockerAdapter
from graph_heal.recovery.kubernetes_adapter import KubernetesAdapter
from graph_heal.service_monitor import ServiceMonitor

# Service configuration based on existing Docker services
SERVICES = {
    'service_a': {'port': 5001, 'dependencies': ['service_b', 'service_c']},
    'service_b': {'port': 5002, 'dependencies': ['service_d']},
    'service_c': {'port': 5003, 'dependencies': ['service_d']},
    'service_d': {'port': 5004, 'dependencies': []}
}

class ExperimentRunner:
    def _initialize_service_graph(self) -> ServiceGraph:
        """Initialize the service graph with nodes and dependencies"""
        graph = ServiceGraph()
        # Add all services as nodes
        for service_name in SERVICES:
            graph.add_service(service_name)
        # Add dependencies based on configuration
        for service_name, config in SERVICES.items():
            for dep in config['dependencies']:
                graph.add_dependency(service_name, dep)
        return graph

    def __init__(self, services, duration_secs=60):
        self.docker_client = docker.from_env()
        self.graph = self._initialize_service_graph()
        self.graph_analyzer = FaultLocalizer(self.graph)
        self.anomaly_detector = AnomalyDetector(services.keys())
        
        adapter_name = os.getenv("RECOVERY_ADAPTER", "docker").lower()
        if adapter_name == "kubernetes":
            adapter = KubernetesAdapter()
        else:
            adapter = DockerAdapter()

        self.recovery_system = EnhancedRecoverySystem(self.graph, adapter=adapter)
        
        self.monitor = ServiceMonitor(services, self.docker_client)
        self.event_log = []
        
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
        
    def run_experiment(self, experiment_id: int, fault_type: str = 'cpu'):
        """Run a single experiment and collect metrics"""
        print(f"\nRunning experiment {experiment_id} with {fault_type} fault...")
        
        # Initialize experiment results
        results = {
            'experiment_id': experiment_id,
            'fault_type': fault_type,
            'timestamp': datetime.now().isoformat(),
            'ground_truth': {
                'fault_source': 'service_a',
                'propagated_services': [],
                'propagation_delays': {},
                'cross_layer_faults': [],
                'cascading_failures': [],
                'root_causes': [],
                'detection_times': [],
                'multi_hop_paths': []
            },
            'graph_predictions': {
                'propagated_services': [],
                'propagation_delays': {},
                'cross_layer_faults': [],
                'cascading_failures': [],
                'root_causes': [],
                'localization_times': [],
                'multi_hop_paths': []
            },
            'statistical_predictions': {
                'propagated_services': [],
                'propagation_delays': {},
                'cross_layer_faults': [],
                'cascading_failures': [],
                'root_causes': [],
                'localization_times': [],
                'multi_hop_paths': []
            }
        }
        
        # Inject fault and collect ground truth
        fault_service = 'service_a'
        fault_start_time = time.time()
        
        # Inject fault using Docker's built-in capabilities
        try:
            container_name = f"service_{fault_service.split('_')[-1]}"
            container = self.docker_client.containers.get(container_name)
            if fault_type == 'cpu':
                container.update(cpu_quota=90000)  # 90% CPU limit
            elif fault_type == 'memory':
                container.update(mem_limit='512m', memswap_limit='512m')
            elif fault_type == 'network':
                container.exec_run('tc qdisc add dev eth0 root netem delay 100ms')
        except Exception as e:
            print(f"Error injecting fault: {e}")
            return results
        
        # Monitor propagation and collect metrics
        propagation_start = time.time()
        while time.time() - propagation_start < 60:  # Monitor for 60 seconds
            # Get current service states
            service_states = {}
            for service in ['service_a', 'service_b', 'service_c', 'service_d']:
                try:
                    container = self.docker_client.containers.get(service)
                    service_states[service] = {
                        'status': container.status,
                        'health': container.attrs.get('State', {}).get('Health', {}).get('Status', 'unknown'),
                        'cpu_usage': self._get_container_metrics(container, 'cpu'),
                        'memory_usage': self._get_container_metrics(container, 'memory'),
                        'network_latency': self._get_container_metrics(container, 'network')
                    }
                except Exception as e:
                    print(f"Error getting state for {service}: {e}")
                    service_states[service] = {
                        'status': 'unknown',
                        'health': 'unknown',
                        'error': str(e)
                    }
            
            # Update analyzers and get predictions
            for service, metrics in service_states.items():
                self.graph_analyzer.update_metrics(service, metrics)
            
            # Get predictions from both approaches
            graph_predictions = self.graph_analyzer.detect_fault_propagation(
                fault_service,
                [datetime.fromtimestamp(fault_start_time)]
            )
            
            stat_predictions_raw = self.anomaly_detector.detect_anomalies(service_states)
            stat_predictions = {'propagated_services': [a['service_id'] for a in stat_predictions_raw if 'service_id' in a]}
            
            # Update results and calculate metrics
            self._update_results(results, service_states, graph_predictions, stat_predictions)
            
            time.sleep(1)  # Check every second
        
        # Calculate final metrics
        self._calculate_metrics(results)
        
        # Save results
        self._save_results(results, experiment_id, fault_type)
        
        # Clean up
        try:
            container = self.docker_client.containers.get(fault_service)
            if fault_type == 'cpu':
                container.update(cpu_quota=0)  # Reset CPU limit
            elif fault_type == 'memory':
                container.update(mem_limit='1g', memswap_limit='1g')  # Reset memory limit
            elif fault_type == 'network':
                container.exec_run('tc qdisc del dev eth0 root')
        except Exception as e:
            print(f"Error cleaning up: {e}")
        
        return results
    
    def _get_container_metrics(self, container, metric_type):
        """Get container metrics using docker stats"""
        try:
            # Map service names to container names
            container_name = f"service_{container.name.split('_')[-1]}" if container.name.startswith('container_') else container.name
            container = self.docker_client.containers.get(container_name)
            stats = container.stats(stream=False)
            if metric_type == 'cpu':
                cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                           stats['precpu_stats']['cpu_usage']['total_usage']
                system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                             stats['precpu_stats']['system_cpu_usage']
                if system_delta > 0:
                    return (cpu_delta / system_delta) * 100
            elif metric_type == 'memory':
                return (stats['memory_stats']['usage'] / stats['memory_stats']['limit']) * 100
            elif metric_type == 'network':
                # Simple network latency check
                return 0  # Placeholder
        except Exception as e:
            print(f"Error getting metrics for {container_name}: {e}")
            pass
        return 0
    
    def _update_results(self, results: dict, service_states: dict, 
                       graph_predictions: dict, stat_predictions: dict):
        """Update results with current state and predictions"""
        current_time = time.time()
        
        # Update ground truth
        for service, state in service_states.items():
            if state['status'] == 'unhealthy' or state['health'] == 'unhealthy':
                if service not in results['ground_truth']['propagated_services']:
                    results['ground_truth']['propagated_services'].append(service)
                    results['ground_truth']['propagation_delays'][service] = current_time - float(results['timestamp'])
        
        # Update graph predictions
        for service in graph_predictions.get('propagated_services', []):
            if service not in results['graph_predictions']['propagated_services']:
                results['graph_predictions']['propagated_services'].append(service)
                results['graph_predictions']['propagation_delays'][service] = current_time - float(results['timestamp'])
        
        # Update statistical predictions
        for service in stat_predictions.get('propagated_services', []):
            if service not in results['statistical_predictions']['propagated_services']:
                results['statistical_predictions']['propagated_services'].append(service)
                results['statistical_predictions']['propagation_delays'][service] = current_time - float(results['timestamp'])
    
    def _calculate_metrics(self, results: dict):
        """Calculate metrics for comparing graph-based vs statistical approaches"""
        # Propagation Detection Accuracy
        graph_accuracy = len(set(results['graph_predictions']['propagated_services']) & 
                           set(results['ground_truth']['propagated_services'])) / \
                        len(results['ground_truth']['propagated_services']) \
                        if results['ground_truth']['propagated_services'] else 0
        
        stat_accuracy = len(set(results['statistical_predictions']['propagated_services']) & 
                          set(results['ground_truth']['propagated_services'])) / \
                       len(results['ground_truth']['propagated_services']) \
                       if results['ground_truth']['propagated_services'] else 0
        
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
        
        # Localization Time
        self.metrics['localization_metrics']['localization_time'].append({
            'graph': results['graph_predictions']['localization_times'][0] if results['graph_predictions']['localization_times'] else float('inf'),
            'statistical': results['statistical_predictions']['localization_times'][0] if results['statistical_predictions']['localization_times'] else float('inf')
        })
    
    def _save_results(self, results: dict, experiment_id: int, fault_type: str):
        """Save experiment results and metrics"""
        # Create results directory if it doesn't exist
        os.makedirs('results/processed', exist_ok=True)
        
        # Save detailed results
        results_filename = f'results/processed/{fault_type}_experiment_{experiment_id}.json'
        with open(results_filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save aggregated metrics
        metrics_filename = f'results/processed/metrics_{fault_type}.json'
        with open(metrics_filename, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        print(f"Results saved to {results_filename}")
        print(f"Metrics saved to {metrics_filename}")

def main():
    # Initialize experiment runner
    runner = ExperimentRunner(SERVICES)
    
    # Run experiments with different fault types
    fault_types = ['cpu', 'memory', 'network']
    for fault_type in fault_types:
        for i in range(1, 3):  # Run 2 experiments per fault type
            runner.run_experiment(i, fault_type)
    
    print("\nAll experiments completed!")

if __name__ == "__main__":
    main()  
