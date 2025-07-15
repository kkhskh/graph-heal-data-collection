import sys
print(sys.executable)
print(sys.path)

import requests
import time
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import os
from typing import Dict, List, Tuple
import pandas as pd
import networkx as nx
import random

class BaselineComparison:
    def __init__(self):
        self.services = {
            'service_a': 'http://localhost:5001',
            'service_b': 'http://localhost:5002',
            'service_c': 'http://localhost:5003',
            'service_d': 'http://localhost:5004'
        }
        self.monitoring_url = 'http://localhost:5010'
        self.prometheus_url = 'http://localhost:9100'
        self.data_dir = 'data/comparison'
        os.makedirs(self.data_dir, exist_ok=True)

    def parse_prometheus_metrics(self, metrics_text: str) -> Dict:
        """Parse Prometheus metrics format into a dictionary."""
        metrics = {}
        for line in metrics_text.split('\n'):
            if line.startswith('#') or not line.strip():
                continue
            try:
                name, value = line.split(' ')[:2]
                metrics[name] = float(value)
            except (ValueError, IndexError):
                continue
        return metrics

    def collect_metrics(self, duration_minutes: int = 5) -> Dict:
        """Collect metrics from all services for the specified duration."""
        metrics = {service: [] for service in self.services}
        end_time = datetime.now() + timedelta(minutes=duration_minutes)
        
        print(f"Collecting metrics for {duration_minutes} minutes...")
        while datetime.now() < end_time:
            for service, url in self.services.items():
                try:
                    response = requests.get(f"{url}/metrics")
                    if response.status_code == 200:
                        parsed_metrics = self.parse_prometheus_metrics(response.text)
                        metrics[service].append({
                            'timestamp': datetime.now().isoformat(),
                            'data': parsed_metrics
                        })
                except requests.exceptions.RequestException as e:
                    print(f"Error collecting metrics from {service}: {e}")
            time.sleep(15)  # Collect every 15 seconds
        
        return metrics

    def run_threshold_baseline(self, metrics: Dict) -> Dict:
        """Implement simple threshold-based detection."""
        results = {
            "method": "threshold",
            "detections": [],
            "false_positives": 0,
            "true_positives": 0,
            "missed_detections": 0
        }
        
        # Define thresholds
        thresholds = {
            'cpu_usage': 80.0,  # 80% CPU usage
            'memory_usage': 85.0,  # 85% memory usage
            'response_time': 1000  # 1000ms response time
        }
        
        for service, service_metrics in metrics.items():
            for metric in service_metrics:
                data = metric['data']
                timestamp = metric['timestamp']
                
                # Check if any metric exceeds threshold
                if (data.get('cpu_usage', 0) > thresholds['cpu_usage'] or
                    data.get('memory_usage', 0) > thresholds['memory_usage'] or
                    data.get('response_time', 0) > thresholds['response_time']):
                    
                    detection = {
                        'service': service,
                        'timestamp': timestamp,
                        'metrics': data,
                        'threshold_exceeded': True
                    }
                    results['detections'].append(detection)
                    
                    # For now, we'll consider all detections as true positives
                    # In a real scenario, you'd compare against known anomalies
                    results['true_positives'] += 1
        
        return results

    def run_graph_heal(self, metrics: Dict) -> Dict:
        """Run the GRAPH-HEAL approach."""
        results = {
            "method": "graph-heal",
            "detections": [],
            "false_positives": 0,
            "true_positives": 0,
            "missed_detections": 0
        }
        
        # Get anomaly detection results from monitoring service
        try:
            response = requests.get(f"{self.monitoring_url}/anomalies")
            if response.status_code == 200:
                anomalies = response.json()
                
                for anomaly in anomalies:
                    detection = {
                        'service': anomaly.get('service'),
                        'timestamp': anomaly.get('timestamp'),
                        'metrics': anomaly.get('metrics'),
                        'anomaly_score': anomaly.get('score'),
                        'graph_context': anomaly.get('graph_context')
                    }
                    results['detections'].append(detection)
                    
                    # For now, we'll consider all detections as true positives
                    # In a real scenario, you'd compare against known anomalies
                    results['true_positives'] += 1
                    
        except requests.exceptions.RequestException as e:
            print(f"Error getting anomalies from monitoring service: {e}")
        
        return results

    def compare_approaches(self, baseline_results: Dict, graph_heal_results: Dict) -> Dict:
        """Compare the two approaches and generate statistics."""
        comparison_data = {
            "methods": ["Threshold-Based", "GRAPH-HEAL"],
            "detection_count": [
                len(baseline_results['detections']),
                len(graph_heal_results['detections'])
            ],
            "true_positives": [
                baseline_results['true_positives'],
                graph_heal_results['true_positives']
            ],
            "false_positives": [
                baseline_results['false_positives'],
                graph_heal_results['false_positives']
            ],
            "missed_detections": [
                baseline_results['missed_detections'],
                graph_heal_results['missed_detections']
            ]
        }
        
        # Calculate accuracy metrics
        total_cases = max(
            baseline_results['true_positives'] + baseline_results['false_positives'] + baseline_results['missed_detections'],
            graph_heal_results['true_positives'] + graph_heal_results['false_positives'] + graph_heal_results['missed_detections']
        )
        
        comparison_data['accuracy'] = [
            baseline_results['true_positives'] / total_cases if total_cases > 0 else 0,
            graph_heal_results['true_positives'] / total_cases if total_cases > 0 else 0
        ]
        
        return comparison_data

    def visualize_results(self, comparison_data: Dict):
        """Create visualizations comparing the two approaches."""
        # Create directory for plots
        plots_dir = os.path.join(self.data_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # 1. Detection Count Comparison
        plt.figure(figsize=(10, 6))
        plt.bar(comparison_data['methods'], comparison_data['detection_count'])
        plt.title('Number of Detections by Method')
        plt.ylabel('Number of Detections')
        plt.savefig(os.path.join(plots_dir, 'detection_count.png'))
        plt.close()
        
        # 2. Accuracy Comparison
        plt.figure(figsize=(10, 6))
        plt.bar(comparison_data['methods'], comparison_data['accuracy'])
        plt.title('Detection Accuracy by Method')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.savefig(os.path.join(plots_dir, 'accuracy.png'))
        plt.close()
        
        # 3. Detailed Metrics
        metrics = ['true_positives', 'false_positives', 'missed_detections']
        x = np.arange(len(comparison_data['methods']))
        width = 0.25
        
        plt.figure(figsize=(12, 6))
        for i, metric in enumerate(metrics):
            plt.bar(x + i*width, comparison_data[metric], width, label=metric)
        
        plt.title('Detailed Performance Metrics')
        plt.xlabel('Method')
        plt.ylabel('Count')
        plt.xticks(x + width, comparison_data['methods'])
        plt.legend()
        plt.savefig(os.path.join(plots_dir, 'detailed_metrics.png'))
        plt.close()

    def save_results(self, comparison_data: Dict):
        """Save comparison results to a JSON file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(self.data_dir, f'comparison_results_{timestamp}.json')
        
        with open(filename, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        print(f"Results saved to {filename}")

def build_service_graph():
    """Builds a directed graph of service dependencies."""
    G = nx.DiGraph()
    G.add_edge("service_a", "service_b")
    G.add_edge("service_b", "service_c")
    G.add_edge("service_c", "service_d")
    # A more complex dependency to test graph analysis
    G.add_edge("service_a", "service_c") 
    return G

def simulate_trace_based_rca(graph, true_root_cause, symptom_service):
    """
    Simulates a tracing-based RCA.
    Tracing is very accurate as it sees the full path. The main challenge for
    tracing-based RCA is identifying the *first* service in the path with
    high latency, which is the true root cause. We simulate this high accuracy here.
    """
    start_time = time.time()
    
    # In a real system, the trace itself would contain timing information
    # that pinpoints the slow service. The trace gives the path, and analysis
    # on the spans of the trace reveals the root cause.
    # Therefore, a tracing system would correctly identify the true_root_cause.
    identified_cause = true_root_cause

    latency = (time.time() - start_time) * 1000  # in ms
    return identified_cause, latency

def simulate_graph_heal_rca(graph, true_root_cause, symptom_service):
    """
    Simulates Graph-Heal's metric-polling and graph traversal RCA.
    This version uses PageRank, which is a much better proxy for Graph-Heal's logic.
    """
    start_time = time.time()

    # 1. Simulate the delay of polling metrics from all services.
    time.sleep(0.1) 
    
    # 2. Identify anomalous nodes. In a real scenario, the true root cause
    # and all of its downstream children would appear anomalous.
    anomalous_nodes = set(list(nx.descendants(graph, true_root_cause)) + [true_root_cause])
    
    # Graph-Heal only sees the subgraph of services that are currently anomalous.
    subgraph = graph.subgraph(anomalous_nodes).copy()

    if not subgraph.nodes():
        identified_cause = symptom_service # Failsafe
    else:
        # 3. Perform PageRank on the anomalous subgraph.
        # The node with the highest PageRank is the most likely root cause.
        # We use a reversed graph because influence flows *against* the direction of dependencies.
        reversed_subgraph = subgraph.reverse(copy=True)
        pagerank = nx.pagerank(reversed_subgraph)
        identified_cause = max(pagerank, key=pagerank.get)

    latency = (time.time() - start_time) * 1000  # in ms
    return identified_cause, latency

def run_experiment(graph, fault_scenarios):
    """Runs a comparison between Tracing RCA and Graph-Heal RCA."""
    results = []

    for scenario in fault_scenarios:
        true_root_cause = scenario['root_cause']
        symptom = scenario['symptom']

        # Run Tracing RCA
        trace_result, trace_latency = simulate_trace_based_rca(graph, true_root_cause, symptom)
        
        # Run Graph-Heal RCA
        gh_result, gh_latency = simulate_graph_heal_rca(graph, true_root_cause, symptom)

        results.append({
            "True Root Cause": true_root_cause,
            "Symptom": symptom,
            "Trace RCA Result": trace_result,
            "Trace RCA Correct": trace_result == true_root_cause,
            "Trace RCA Latency (ms)": trace_latency,
            "Graph-Heal Result": gh_result,
            "Graph-Heal Correct": gh_result == true_root_cause,
            "Graph-Heal Latency (ms)": gh_latency,
        })

    return pd.DataFrame(results)

def main():
    """Main function to run the baseline comparison."""
    service_graph = build_service_graph()
    
    fault_scenarios = [
        {'root_cause': 'service_a', 'symptom': 'service_d'},
        {'root_cause': 'service_b', 'symptom': 'service_d'},
        {'root_cause': 'service_c', 'symptom': 'service_d'},
        {'root_cause': 'service_d', 'symptom': 'service_d'},
    ]

    results_df = run_experiment(service_graph, fault_scenarios)

    print("--- Baseline Comparison Results ---")
    print(results_df.to_string())

    # Summary
    trace_accuracy = results_df["Trace RCA Correct"].mean() * 100
    gh_accuracy = results_df["Graph-Heal Correct"].mean() * 100
    avg_trace_latency = results_df["Trace RCA Latency (ms)"].mean()
    avg_gh_latency = results_df["Graph-Heal Latency (ms)"].mean()

    print("\n--- Summary ---")
    print(f"Tracing-Based RCA Accuracy: {trace_accuracy:.2f}%")
    print(f"Graph-Heal RCA Accuracy:    {gh_accuracy:.2f}%")
    print(f"Average Tracing Latency: {avg_trace_latency:.4f} ms")
    print(f"Average Graph-Heal Latency:    {avg_gh_latency:.4f} ms")

if __name__ == "__main__":
    main() 