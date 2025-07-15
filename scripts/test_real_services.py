import pytest
pytest.skip("legacy integration test – requires running services", allow_module_level=True)

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../graph-heal')))

from graph_heal.graph_analysis import ServiceGraph
import requests
from datetime import datetime, timedelta
import time
import json
import re

def parse_prometheus_metrics(metrics_text: str) -> dict:
    """Parse Prometheus metrics text into a dictionary"""
    metrics = {}
    for line in metrics_text.split('\n'):
        if line.startswith('#') or not line.strip():
            continue
        match = re.match(r'(\w+)\s+([\d.]+)', line)
        if match:
            name, value = match.groups()
            metrics[name] = float(value)
    return metrics

def get_service_metrics(service_name: str, port: int) -> dict:
    """Get metrics from a service's metrics endpoint"""
    try:
        response = requests.get(f'http://localhost:{port}/metrics')
        if response.status_code == 200:
            metrics = parse_prometheus_metrics(response.text)
            # Add service_cpu_usage key for compatibility
            if 'cpu_usage' in metrics:
                metrics['service_cpu_usage'] = metrics['cpu_usage']
            metrics['timestamp'] = datetime.now()
            return metrics
    except requests.exceptions.RequestException:
        print(f"Failed to get metrics from {service_name}")
    return {}

def create_multi_layer_service_graph():
    graph = ServiceGraph()
    # Define services, containers, hosts, and network
    services = {
        'service_a': 5001,
        'service_b': 5002,
        'service_c': 5003,
        'service_d': 5004
    }
    # Use actual Docker container names
    containers = {
        'service_a': 'service_a',
        'service_b': 'service_b',
        'service_c': 'service_c',
        'service_d': 'service_d'
    }
    hosts = {
        'host1': ['graph-heal_service_a_1', 'graph-heal_service_b_1'],
        'host2': ['graph-heal_service_c_1', 'graph-heal_service_d_1']
    }
    network = 'graph-heal_default'
    # Add nodes
    for service in services:
        graph.add_service(service)
    for container in containers:
        graph.add_container(container)
    for host in hosts:
        graph.add_host(host)
    graph.add_network(network)
    # Add service-to-service dependencies
    graph.add_dependency('service_a', 'service_b', 1.0, dep_type='logical')
    graph.add_dependency('service_b', 'service_c', 1.0, dep_type='logical')
    graph.add_dependency('service_a', 'service_d', 1.0, dep_type='logical')
    # Add service-to-container, container-to-host, and all to network dependencies
    for container, service in containers.items():
        graph.add_dependency(service, container, 1.0, dep_type='runs_in')
    for host, container_list in hosts.items():
        for container in container_list:
            graph.add_dependency(container, host, 1.0, dep_type='hosted_on')
    for node in list(services.keys()) + list(containers.keys()) + list(hosts.keys()):
        graph.add_dependency(node, network, 1.0, dep_type='networked')
    return graph, services

def inject_fault(service: str, port: int, duration: int = 60):
    """Inject a CPU fault into a service"""
    try:
        response = requests.post(
            f'http://localhost:{port}/inject_fault',
            json={'fault_type': 'cpu', 'duration': duration}
        )
        if response.status_code == 200:
            print(f"Successfully injected CPU fault into {service}")
        else:
            print(f"Failed to inject fault into {service}: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Error injecting fault into {service}: {e}")

def collect_metrics(graph: ServiceGraph, services: dict, duration: int = 60):
    """Collect metrics from all services for a specified duration"""
    print(f"Collecting metrics for {duration} seconds...")
    start_time = datetime.now()
    
    while (datetime.now() - start_time).seconds < duration:
        for service, port in services.items():
            metrics = get_service_metrics(service, port)
            if metrics:
                graph.update_metrics(service, metrics)
        time.sleep(1)  # Collect metrics every second

def test_multi_layer_fault_propagation():
    """Test fault propagation with real services"""
    # Create and populate the graph
    graph, services = create_multi_layer_service_graph()
    
    # Collect initial metrics
    collect_metrics(graph, services, duration=10)
    
    # Inject a fault into service_a
    inject_fault('service_a', services['service_a'], duration=10)
    
    # Collect metrics during fault
    collect_metrics(graph, services, duration=10)
    
    # Analyze fault impact
    fault_time = datetime.now() - timedelta(seconds=10)  # Approximate fault start time
    impact = graph.analyze_fault_impact('service_a', [fault_time])
    
    print("\nFault Impact Analysis:")
    print(f"Total Services: {impact['total_services']}")
    print(f"Affected Services: {impact['affected_services']}")
    print(f"Impact Percentage: {impact['impact_percentage']:.1f}%")
    
    print("\nValidation Metrics:")
    for metric, value in impact['validation_metrics'].items():
        print(f"{metric}: {value:.2f}")
    
    print("\nPropagation Metrics:")
    for metric in impact['propagation_metrics']:
        print(f"\nSource: {metric['source']} -> Target: {metric['target']}")
        print(f"Correlation: {metric['correlation']:.2f}")
        print(f"Delay: {metric['delay']:.2f} seconds")
        print(f"Load Factor: {metric['load_factor']:.2f}")
    
    print("\nCircular Dependencies:")
    cycles = graph.detect_circular_dependencies()
    if cycles:
        for cycle in cycles:
            print(" -> ".join(cycle))
    else:
        print("None detected.")
    
    print("\nDependency Impact Analysis for host1:")
    impact_host = graph.dependency_impact_analysis('host1')
    print(json.dumps(impact_host, indent=2))
    
    print("\nNode Health Scores:")
    for node in ['service_a', 'graph-heal_service_a_1', 'host1', 'graph-heal_default']:
        print(f"{node}: {graph.score_node_health(node):.2f}")
    
    # Get historical summary
    history = graph.summarize_propagation_history()
    print("\nPropagation History Summary:")
    print(json.dumps(history, indent=2))
    
    # Visualize the graph
    graph.visualize_graph(fault_service='service_a', 
                         affected_services=set(impact['propagated_faults'].keys()))

if __name__ == '__main__':
    test_multi_layer_fault_propagation() 