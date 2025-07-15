import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../graph-heal')))
import json
from datetime import datetime
import matplotlib.pyplot as plt
from graph_heal.graph_analysis import ServiceGraph
from graph_heal.anomaly_detection import StatisticalAnomalyDetector
import re
import requests
import time
from typing import Dict, List, Optional
import docker

# Service configuration based on existing Docker services
SERVICES = {
    'service_a': {'port': 5000, 'dependencies': ['service_b', 'service_c']},
    'service_b': {'port': 5001, 'dependencies': ['service_d']},
    'service_c': {'port': 5002, 'dependencies': ['service_d']},
    'service_d': {'port': 5003, 'dependencies': []}
}

def load_experiment_data(experiment_file: str):
    """Load experiment data from a JSON file"""
    with open(experiment_file) as f:
        data = json.load(f)
    return data

def get_docker_client():
    """Get Docker client"""
    try:
        return docker.from_env()
    except Exception as e:
        print(f"Error connecting to Docker: {e}")
        return None

def check_service_health(service_name: str) -> bool:
    """Check if a service is healthy by checking its metrics endpoint"""
    try:
        port = SERVICES[service_name]['port']
        response = requests.get(f'http://localhost:{port}/metrics', timeout=5)
        return response.status_code == 200
    except:
        return False

def recover_service(service_name: str) -> bool:
    """Attempt to recover a service using Docker"""
    try:
        client = get_docker_client()
        if not client:
            return False

        # Find the container - using the actual container name pattern
        container_name = service_name  # The containers are named directly as service_a, service_b, etc.
        try:
            container = client.containers.get(container_name)
        except docker.errors.NotFound:
            print(f"Container {container_name} not found")
            return False

        # Restart the container
        print(f"[RECOVERY] Restarting container {container_name}...")
        container.restart()
        
        # Wait for service to be healthy
        for _ in range(5):  # Try for 5 seconds
            if check_service_health(service_name):
                return True
            time.sleep(1)
        return False
    except Exception as e:
        print(f"Error recovering {service_name}: {e}")
        return False

def inject_fault(service_name: str, duration: int = 60) -> bool:
    """Inject a CPU fault into a service"""
    try:
        port = SERVICES[service_name]['port']
        response = requests.post(
            f'http://localhost:{port}/inject_fault',
            json={'fault_type': 'cpu', 'duration': duration},
            timeout=5
        )
        return response.status_code == 200
    except Exception as e:
        print(f"Error injecting fault into {service_name}: {e}")
        return False

def create_service_graph(data: dict) -> ServiceGraph:
    """Create a service graph from experiment data with real dependencies"""
    graph = ServiceGraph()
    
    # Add all services from our configuration
    for service_name in SERVICES:
        graph.add_service(service_name)
    
    # Add dependencies based on our configuration
    for service_name, config in SERVICES.items():
        for dep in config['dependencies']:
            graph.add_dependency(service_name, dep)
    
    # Update metrics history
    for metrics in data['metrics']:
        for service_name in SERVICES:
            metric_key = f'service_{service_name}_cpu_usage'
            if metric_key in metrics:
                graph.update_metrics(service_name, {metric_key: metrics[metric_key]})
    
    return graph

def analyze_fault_propagation(graph: ServiceGraph, fault_service: str, fault_timestamps: list):
    """Analyze fault propagation and attempt recovery"""
    # Analyze impact
    impact = graph.analyze_fault_impact(fault_service, fault_timestamps)
    
    print("\nFault Impact Analysis:")
    print(f"Total Services: {impact['total_services']}")
    print(f"Affected Services: {impact['affected_services']}")
    print(f"Impact Percentage: {impact['impact_percentage']:.1f}%")
    
    print("\nCritical Paths:")
    for path in impact['critical_paths']:
        print(" -> ".join(path))
    
    # Check service health
    print("\nService Health Status:")
    for service in SERVICES:
        is_healthy = check_service_health(service)
        print(f"{service}: {'Healthy' if is_healthy else 'Unhealthy'}")
    
    # Attempt recovery for unhealthy services
    print("\nRecovery Attempts:")
    for service in SERVICES:
        if not check_service_health(service):
            if recover_service(service):
                print(f"✓ Successfully recovered {service}")
            else:
                print(f"✗ Failed to recover {service}")
    
    # Visualize graph with fault propagation
    affected_services = set(impact['propagated_faults'].keys())
    graph.visualize_graph(fault_service, affected_services)
    
    return impact

def main():
    # Load experiment data
    experiment_file = "results/processed/cpu_test_cpu_experiment_4.json"
    data = load_experiment_data(experiment_file)
    
    # Create service graph
    graph = create_service_graph(data)
    
    # Get fault timestamps from the experiment
    fault_periods = data['fault_periods']
    fault_timestamps = []
    for period in fault_periods:
        start = datetime.fromisoformat(period['start'])
        end = datetime.fromisoformat(period['end'])
        # Convert to indices in the metrics list
        start_idx = data['timestamps'].index(period['start'])
        end_idx = data['timestamps'].index(period['end'])
        fault_timestamps.extend(range(start_idx, end_idx + 1))
    
    # Analyze fault propagation
    fault_service = 'service_a'  # The service where the fault was injected
    impact = analyze_fault_propagation(graph, fault_service, fault_timestamps)
    
    # Save the graph structure
    graph.save_graph('results/service_graph.json')

    # Calculate propagation metrics
    propagation_metrics = {
        'propagation_detection_accuracy': ...,
        'propagation_delay_estimation_error': ...,
        'cross_layer_fault_detection': ...,
        'cascading_failure_prediction': ...
    }

    # Calculate localization metrics
    localization_metrics = {
        'root_cause_accuracy': ...,
        'localization_time': ...,
        'false_root_cause_rate': ...,
        'multi_hop_localization': ...
    }

if __name__ == "__main__":
    main() 