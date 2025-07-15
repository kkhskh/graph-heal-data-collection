import networkx as nx
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
from datetime import datetime, timedelta
import json
from scipy import stats
import pandas as pd

class ServiceGraph:
    def __init__(self, docker_metrics=None):
        """Initialize an empty multi-layer service dependency graph"""
        self.graph = nx.DiGraph()
        self.metrics_history: Dict[str, List[Dict]] = {}
        self.node_types: Dict[str, str] = {}  # Node name -> type/layer
        self.fault_propagation_threshold = 0.7  # Correlation threshold for fault propagation
        self.propagation_delays: Dict[Tuple[str, str], float] = {}  # Store propagation delays between services
        self.service_loads: Dict[str, List[float]] = {}  # Store service load history
        self.propagation_history: List[Dict] = []  # Store historical propagation events
        self.health_scores: Dict[str, float] = {}  # Node health scores
        
        # Docker metrics helper (may be None in unit tests)
        if docker_metrics is None:
            class _NullDockerMetrics:
                """Lightweight stub used in unit/in-process tests."""

                def get_container_health(self, *_, **__):
                    return 1.0

            self.docker_metrics = _NullDockerMetrics()
        else:
            self.docker_metrics = docker_metrics
        
    def add_node(self, node_name: str, node_type: str):
        """Add a node with a type/layer (e.g., service, host, network, library)"""
        self.graph.add_node(node_name, type=node_type)
        self.node_types[node_name] = node_type
        if node_type == 'service':
            self.metrics_history[node_name] = []
            self.service_loads[node_name] = []
        
    def add_service(self, service_name: str):
        self.add_node(service_name, 'service')
        
    def add_host(self, host_name: str):
        self.add_node(host_name, 'host')
        
    def add_container(self, container_name: str):
        self.add_node(container_name, 'container')
        
    def add_network(self, network_name: str):
        self.add_node(network_name, 'network')
        
    def add_library(self, library_name: str):
        self.add_node(library_name, 'library')
        
    def add_dependency(self, source: str, target: str, weight: float = 1.0, dep_type: str = 'logical'):
        """Add a dependency edge between nodes (can be cross-layer)"""
        self.graph.add_edge(source, target, weight=weight, dep_type=dep_type)
        
    def update_metrics(self, service_name: str, metrics: Dict):
        """Update metrics history for a service"""
        if service_name in self.metrics_history:
            self.metrics_history[service_name].append(metrics)
            if 'average_response_time' in metrics:
                self.service_loads[service_name].append(metrics['average_response_time'])
        
    def get_latest_metrics(self, service_name: str) -> Optional[Dict]:
        """Get the most recent metrics for a service"""
        if service_name in self.metrics_history and self.metrics_history[service_name]:
            return self.metrics_history[service_name][-1]
        return None
        
    def get_service_metrics(self, service_name: str) -> List[Dict]:
        """Get metrics history for a service"""
        return self.metrics_history.get(service_name, [])
    
    def get_nodes_by_type(self, node_type: str) -> List[str]:
        return [n for n, t in self.node_types.items() if t == node_type]
    
    def get_node_type(self, node_name: str) -> str:
        return self.node_types.get(node_name, 'unknown')
    
    def calculate_correlation(self, source: str, target: str, window_size: int = 10) -> Tuple[float, float]:
        """Calculate correlation between two services' metrics"""
        source_metrics = self.get_service_metrics(source)
        target_metrics = self.get_service_metrics(target)
        
        if not source_metrics or not target_metrics:
            return 0.0, 1.0
            
        # Get response times
        source_times = [m.get('average_response_time', 0) for m in source_metrics]
        target_times = [m.get('average_response_time', 0) for m in target_metrics]
        
        if not source_times or not target_times:
            return 0.0, 1.0
            
        # Calculate correlation (window_size currently ignored)
        correlation, p_value = stats.pearsonr(source_times, target_times)
        return correlation, p_value
    
    def estimate_propagation_delay(self, source: str, target: str, metric: str = 'service_cpu_usage') -> float:
        """
        Estimate propagation delay between two services using cross-correlation
        Returns estimated delay in seconds
        """
        try:
            metrics1 = [m[metric] for m in self.get_service_metrics(source) if metric in m]
            metrics2 = [m[metric] for m in self.get_service_metrics(target) if metric in m]
        except Exception:
            return 0.0
        if len(metrics1) != len(metrics2) or len(metrics1) < 2:
            return 0.0
        try:
            correlation = np.correlate(metrics1, metrics2, mode='full')
            delay = np.argmax(correlation) - (len(metrics1) - 1)
            return abs(delay)
        except Exception:
            return 0.0
    
    def detect_fault_propagation(self, source_service: str, fault_timestamps: List[datetime], collect_metrics: bool = True) -> Dict[str, List[datetime]]:
        """
        Enhanced fault propagation detection with time-based analysis and service load consideration.
        Collects and stores metrics for each propagation step if collect_metrics is True.
        Returns a dictionary mapping service names to their detected fault timestamps.
        """
        propagated_faults = {source_service: fault_timestamps}
        propagation_metrics = []
        
        # Get all services that could be affected (reachable from source)
        affected_services = nx.descendants(self.graph, source_service)
        
        for target in affected_services:
            # Calculate correlation and p-value
            correlation, p_value = self.calculate_correlation(source_service, target)
            
            # Consider service load
            source_load = np.mean(self.service_loads.get(source_service, [0]))
            target_load = np.mean(self.service_loads.get(target, [0]))
            load_factor = min(source_load, target_load) / 100.0  # Normalize load
            
            # Estimate propagation delay
            delay = self.estimate_propagation_delay(source_service, target)
            self.propagation_delays[(source_service, target)] = delay
            
            # Enhanced propagation detection
            if (abs(correlation) > self.fault_propagation_threshold and 
                p_value < 0.05 and  # Statistical significance
                load_factor > 0.3):  # Minimum load threshold
                
                # Adjust fault timestamps based on propagation delay
                adjusted_timestamps = [
                    ts + timedelta(seconds=int(delay))
                    for ts in fault_timestamps
                ]
                propagated_faults[target] = adjusted_timestamps
                
            # Collect metrics for this propagation step
            if collect_metrics:
                propagation_metrics.append({
                    'source': source_service,
                    'target': target,
                    'correlation': correlation,
                    'p_value': p_value,
                    'source_load': source_load,
                    'target_load': target_load,
                    'load_factor': load_factor,
                    'delay': delay,
                    'propagated': target in propagated_faults
                })
        
        # Store propagation event in history
        if collect_metrics:
            self.propagation_history.append({
                'timestamp': datetime.now().isoformat(),
                'source_service': source_service,
                'fault_timestamps': [ts.isoformat() for ts in fault_timestamps],
                'propagated_faults': {k: [t.isoformat() for t in v] for k, v in propagated_faults.items()},
                'metrics': propagation_metrics
            })
        
        return propagated_faults
    
    def analyze_fault_impact(self, fault_service: str, fault_timestamps: List[datetime], ground_truth: Optional[Dict[str, List[datetime]]] = None) -> Dict:
        """
        Enhanced fault impact analysis with propagation validation, metrics collection, and historical storage.
        Returns detailed metrics about affected services and propagation paths.
        Optionally compares detected propagation to ground truth if provided.
        """
        propagated_faults = self.detect_fault_propagation(fault_service, fault_timestamps, collect_metrics=True)
        
        # Calculate impact metrics
        total_services = len(self.get_nodes_by_type('service'))
        affected_services = len(propagated_faults)
        
        # Find critical paths with propagation delays
        critical_paths = []
        path_metrics = []
        for target in propagated_faults:
            if target != fault_service:
                try:
                    path = nx.shortest_path(self.graph, fault_service, target)
                    total_delay = sum(
                        self.propagation_delays.get((path[i], path[i+1]), 0)
                        for i in range(len(path)-1)
                    )
                    critical_paths.append(path)
                    path_metrics.append({
                        'path': path,
                        'total_delay': total_delay,
                        'correlation': self.calculate_correlation(fault_service, target)[0]
                    })
                except nx.NetworkXNoPath:
                    continue
        
        # Calculate propagation validation metrics
        validation_metrics = {
            'propagation_success_rate': len(propagated_faults) / affected_services if affected_services > 0 else 0,
            'average_propagation_delay': np.mean(list(self.propagation_delays.values())) if self.propagation_delays else 0,
            'max_propagation_delay': max(self.propagation_delays.values()) if self.propagation_delays else 0
        }
        
        # If ground truth is provided, compute precision, recall, F1
        if ground_truth is not None:
            detected = set(propagated_faults.keys())
            actual = set(ground_truth.keys())
            tp = len(detected & actual)
            fp = len(detected - actual)
            fn = len(actual - detected)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            validation_metrics.update({
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })
        
        return {
            'total_services': total_services,
            'affected_services': affected_services,
            'impact_percentage': (affected_services / total_services) * 100 if total_services > 0 else 0,
            'critical_paths': critical_paths,
            'path_metrics': path_metrics,
            'propagated_faults': propagated_faults,
            'validation_metrics': validation_metrics,
            'propagation_metrics': self.propagation_history[-1]['metrics'] if self.propagation_history else []
        }
    
    def create_propagation_heatmap(self, fault_service: str, window_size: int = 10) -> np.ndarray:
        """
        Create a heatmap showing propagation patterns between services
        Returns a numpy array representing the heatmap
        """
        services = list(self.graph.nodes())
        n_services = len(services)
        heatmap = np.zeros((n_services, n_services))
        
        for i, source in enumerate(services):
            for j, target in enumerate(services):
                if source != target:
                    corr, _ = self.calculate_correlation(source, target, window_size=window_size)
                    heatmap[i, j] = abs(corr)
        
        return heatmap
    
    def visualize_graph(self, fault_service: str = None, affected_services: Set[str] = None):
        """Enhanced visualization with real metrics"""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(15, 10))
        pos = nx.spring_layout(self.graph)
        
        # Draw nodes with size based on health score
        node_sizes = []
        node_colors = []
        for node in self.graph.nodes():
            node_type = self.get_node_type(node)
            health_score = self.score_node_health(node)
            node_sizes.append(1000 + health_score * 2000)  # Scale size with health
            
            if node == fault_service:
                node_colors.append('red')
            elif affected_services and node in affected_services:
                node_colors.append('orange')
            elif node_type == 'host':
                node_colors.append('green')
            elif node_type == 'network':
                node_colors.append('purple')
            elif node_type == 'container':
                node_colors.append('yellow')
            elif node_type == 'library':
                node_colors.append('cyan')
            else:
                node_colors.append('lightblue')
                
        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors, 
                             node_size=node_sizes, alpha=0.7)
        
        # Draw edges with width based on dependency strength
        edge_widths = []
        for u, v in self.graph.edges():
            strength = self.dependency_strength(u, v)
            edge_widths.append(strength * 3)
            
        nx.draw_networkx_edges(self.graph, pos, width=edge_widths, 
                             arrows=True, alpha=0.6)
        
        # Add health scores to labels
        labels = {node: f"{node}\n({self.score_node_health(node):.2f})" 
                 for node in self.graph.nodes()}
        nx.draw_networkx_labels(self.graph, pos, labels)
        
        plt.title('Multi-layer Service Dependency Graph with Real Metrics')
        plt.axis('off')
        plt.show()
        
    def save_graph(self, filename: str):
        """Save the graph structure and metrics to a file"""
        graph_data = {
            'nodes': list(self.graph.nodes()),
            'edges': list(self.graph.edges(data=True)),
            'propagation_delays': {str(k): v for k, v in self.propagation_delays.items()},
            'service_loads': self.service_loads
        }
        with open(filename, 'w') as f:
            json.dump(graph_data, f, indent=2)
            
    @classmethod
    def load_graph(cls, filename: str) -> 'ServiceGraph':
        """Load a graph structure and metrics from a file"""
        with open(filename, 'r') as f:
            graph_data = json.load(f)
            
        graph = cls()
        for node in graph_data['nodes']:
            graph.add_node(node, graph_data['nodes'][node]['type'])
        for edge in graph_data['edges']:
            graph.add_dependency(edge[0], edge[1], edge[2].get('weight', 1.0), edge[2].get('dep_type', 'logical'))
            
        # Load additional data if available
        if 'propagation_delays' in graph_data:
            graph.propagation_delays = {
                tuple(eval(k)): v for k, v in graph_data['propagation_delays'].items()
            }
        if 'service_loads' in graph_data:
            graph.service_loads = graph_data['service_loads']
            
        return graph 

    def summarize_propagation_history(self) -> Dict:
        """
        Summarize historical propagation patterns.
        Returns statistics such as most common propagation paths, average delays, etc.
        """
        if not self.propagation_history:
            return {}
        
        all_metrics = [m for event in self.propagation_history for m in event['metrics']]
        if not all_metrics:
            return {}
        
        # Example summary: average correlation, delay, most common targets
        avg_corr = np.mean([m['correlation'] for m in all_metrics])
        avg_delay = np.mean([m['delay'] for m in all_metrics])
        most_common_targets = pd.Series([m['target'] for m in all_metrics]).value_counts().to_dict()
        
        return {
            'average_correlation': avg_corr,
            'average_delay': avg_delay,
            'most_common_targets': most_common_targets,
            'total_propagation_events': len(self.propagation_history)
        }

    def detect_circular_dependencies(self) -> List[List[str]]:
        """Detect cycles (circular dependencies) in the graph"""
        return list(nx.simple_cycles(self.graph))

    def dependency_strength(self, source: str, target: str) -> float:
        """Calculate dependency strength between two services"""
        correlation, p_value = self.calculate_correlation(source, target)
        if p_value > 0.05:  # Not statistically significant
            return 0.0
        return abs(correlation)

    def score_node_health(self, node: str, visited=None) -> float:
        """Aggregate health score for a node using real metrics and propagate health up the stack, avoiding recursion cycles."""
        if visited is None:
            visited = set()
        if node in visited:
            return 1.0  # Prevent infinite recursion, assume healthy if cycle
        visited.add(node)
        node_type = self.get_node_type(node)
        
        if node_type == 'service':
            loads = self.service_loads.get(node, [])
            if loads:
                service_health = 1.0 - (np.mean(loads) / 100.0)
            else:
                service_health = 1.0
            container_name = None
            for n in self.graph.successors(node):
                if self.get_node_type(n) == 'container':
                    container_name = n
                    break
            if container_name:
                container_health = self.score_node_health(container_name, visited)
                host_name = None
                for n2 in self.graph.successors(container_name):
                    if self.get_node_type(n2) == 'host':
                        host_name = n2
                        break
                if host_name:
                    host_health = self.score_node_health(host_name, visited)
                else:
                    host_health = 1.0
            else:
                container_health = 1.0
                host_health = 1.0
            health_score = (
                service_health * 0.5 +
                container_health * 0.3 +
                host_health * 0.2
            )
        elif node_type == 'container':
            container_health = self.docker_metrics.get_container_health(node)
            host_name = None
            for n in self.graph.successors(node):
                if self.get_node_type(n) == 'host':
                    host_name = n
                    break
            if host_name:
                host_health = self.score_node_health(host_name, visited)
            else:
                host_health = 1.0
            health_score = container_health * 0.7 + host_health * 0.3
        elif node_type == 'host':
            host_health = self.docker_metrics.get_host_health()
            hosted_nodes = [n for n in self.graph.predecessors(node) if self.get_node_type(n) in ['container', 'service']]
            if hosted_nodes:
                avg_guest_health = np.mean([self.score_node_health(n, visited) for n in hosted_nodes])
            else:
                avg_guest_health = 1.0
            health_score = host_health * 0.6 + avg_guest_health * 0.4
        elif node_type == 'network':
            connected_nodes = [n for n in self.graph.predecessors(node)]
            if connected_nodes:
                avg_connected_health = np.mean([self.score_node_health(n, visited) for n in connected_nodes])
            else:
                avg_connected_health = 1.0
            health_score = avg_connected_health
        else:
            health_score = 1.0
        self.health_scores[node] = health_score
        return health_score

    def dependency_impact_analysis(self, node: str) -> Dict:
        """Analyze how a fault in this node could impact others using real metrics"""
        impacted = nx.descendants(self.graph, node)
        impact_scores = {}
        
        # Calculate impact scores based on real metrics
        for target in impacted:
            target_type = self.get_node_type(target)
            base_health = self.score_node_health(target)
            
            # Adjust impact based on dependency strength and node type
            dep_strength = self.dependency_strength(node, target)
            if target_type == 'service':
                # Services are more sensitive to upstream issues
                impact_scores[target] = base_health * (1.0 - dep_strength * 0.5)
            elif target_type == 'container':
                # Containers are moderately sensitive
                impact_scores[target] = base_health * (1.0 - dep_strength * 0.3)
            elif target_type == 'host':
                # Hosts are less sensitive to individual service issues
                impact_scores[target] = base_health * (1.0 - dep_strength * 0.2)
            else:
                impact_scores[target] = base_health
        
        return {
            'impacted_nodes': list(impacted),
            'impact_scores': impact_scores,
            'source_health': self.score_node_health(node)
        }

    def get_service_config(self, service_id: str) -> Optional[Dict]:
        """Return the configuration for a given service ID from SERVICES_CONFIG."""
        return SERVICES_CONFIG.get(service_id) 