import pytest
pytest.skip("legacy visualization test – skipped in unit CI", allow_module_level=True)

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../graph-heal')))

from graph_heal.graph_analysis import ServiceGraph
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt

def create_test_graph():
    """Create a test service graph with sample metrics"""
    graph = ServiceGraph()
    
    # Add services
    services = ['service_a', 'service_b', 'service_c', 'service_d']
    for service in services:
        graph.add_service(service)
    
    # Add dependencies
    graph.add_dependency('service_a', 'service_b', 1.0)
    graph.add_dependency('service_b', 'service_c', 1.0)
    graph.add_dependency('service_a', 'service_d', 1.0)
    
    # Generate sample metrics
    base_time = datetime.now()
    for i in range(100):
        current_time = base_time + timedelta(seconds=i)
        
        # Generate metrics for each service
        for service in services:
            # Base metrics
            cpu_usage = np.random.normal(50, 10)  # Normal distribution around 50%
            
            # Add correlation between dependent services
            if service == 'service_b':
                cpu_usage += graph.get_service_metrics('service_a')[-1]['service_cpu_usage'] * 0.3
            elif service == 'service_c':
                cpu_usage += graph.get_service_metrics('service_b')[-1]['service_cpu_usage'] * 0.2
            elif service == 'service_d':
                cpu_usage += graph.get_service_metrics('service_a')[-1]['service_cpu_usage'] * 0.4
            
            # Add some noise
            cpu_usage += np.random.normal(0, 5)
            cpu_usage = max(0, min(100, cpu_usage))  # Clamp between 0 and 100
            
            metrics = {
                'timestamp': current_time,
                'service_cpu_usage': cpu_usage
            }
            graph.update_metrics(service, metrics)
    
    return graph

def test_enhanced_analysis():
    """Test the enhanced graph analysis features"""
    # Create and populate the graph
    graph = create_test_graph()
    
    # Simulate a fault in service_a
    fault_time = datetime.now() + timedelta(seconds=50)
    fault_timestamps = [fault_time]
    
    # Analyze fault impact
    impact = graph.analyze_fault_impact('service_a', fault_timestamps)
    
    print("\nFault Impact Analysis:")
    print(f"Total Services: {impact['total_services']}")
    print(f"Affected Services: {impact['affected_services']}")
    print(f"Impact Percentage: {impact['impact_percentage']:.1f}%")
    
    print("\nValidation Metrics:")
    for metric, value in impact['validation_metrics'].items():
        print(f"{metric}: {value:.2f}")
    
    print("\nPath Metrics:")
    for path_metric in impact['path_metrics']:
        print(f"\nPath: {' -> '.join(path_metric['path'])}")
        print(f"Total Delay: {path_metric['total_delay']:.2f} seconds")
        print(f"Correlation: {path_metric['correlation']:.2f}")
    
    # Create and display propagation heatmap
    heatmap = graph.create_propagation_heatmap('service_a')
    plt.figure(figsize=(10, 8))
    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Correlation Strength')
    plt.title('Fault Propagation Heatmap')
    plt.xticks(range(len(graph.graph.nodes())), graph.graph.nodes(), rotation=45)
    plt.yticks(range(len(graph.graph.nodes())), graph.graph.nodes())
    plt.tight_layout()
    plt.show()
    
    # Visualize the graph
    graph.visualize_graph(fault_service='service_a', 
                         affected_services=set(impact['propagated_faults'].keys()))

if __name__ == '__main__':
    test_enhanced_analysis() 