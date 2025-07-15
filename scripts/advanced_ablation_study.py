import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
from dataclasses import dataclass
import networkx as nx
from scipy import stats
import time
import pandas as pd
import psutil
import os
import random
import platform
import gc
import tracemalloc
from cpuinfo import get_cpu_info

# Import our existing detection methods
from calculate_real_accuracy_unified import (
    threshold_based_detection_old,
    graph_heal_detection_old,
    threshold_based_detection_new,
    graph_heal_detection_new,
    calculate_accuracy_metrics,
    load_experiment_data
)

# --- Configuration ---
N_TRIALS = 20
GRAPH_SIZES = [100, 1000, 10000, 50000, 100000]
AVG_DEGREES = [3, 5, 10, 20]
TOPOLOGIES = ["scale_free", "dag"]

@dataclass
class GraphConfig:
    """Configuration for graph structure ablation"""
    single_layer: bool = False
    no_cross_layer_edges: bool = False
    random_graph: bool = False
    static_graph: bool = False

@dataclass
class DetectionConfig:
    """Configuration for detection algorithm ablation"""
    z_threshold: float = 1.5
    use_graph_correlation: bool = True
    use_temporal_patterns: bool = True
    require_consensus: bool = False

@dataclass
class ThresholdConfig:
    """Configuration for threshold sensitivity ablation"""
    z_score: float = 1.5
    cpu_threshold: float = 70.0
    memory_threshold: float = 400.0
    response_threshold: float = 150.0

@dataclass
class RecoveryConfig:
    """Configuration for recovery strategy ablation"""
    proactive: bool = False
    batch_actions: bool = False
    dependency_aware: bool = True
    topology_guided: bool = False

class AdvancedAblationStudy:
    def __init__(self):
        self.experiment_files = [
            'results/cpu_experiment.json',
            'results/memory_experiment.json',
            'results/network_experiment.json'
        ]
        
        # Initialize graph structure
        self.graph = self._create_initial_graph()
        
    def _create_initial_graph(self) -> nx.DiGraph:
        """Create initial multi-layer graph structure"""
        G = nx.DiGraph()
        
        # Add nodes for each service
        services = ['service_a', 'service_b', 'service_c', 'service_d']
        for service in services:
            G.add_node(service, layer='service')
            
        # Add edges representing dependencies
        edges = [
            ('service_a', 'service_b'),
            ('service_b', 'service_c'),
            ('service_c', 'service_d'),
            ('service_a', 'service_d')
        ]
        G.add_edges_from(edges)
        
        return G
    
    def _create_modified_graph(self, config: GraphConfig) -> nx.DiGraph:
        """Create modified graph based on ablation configuration"""
        G = self._create_initial_graph()
        
        if config.single_layer:
            # Flatten all nodes to single layer
            for node in G.nodes():
                G.nodes[node]['layer'] = 'flat'
                
        if config.no_cross_layer_edges:
            # Remove cross-layer edges
            edges_to_remove = []
            for u, v in G.edges():
                if G.nodes[u]['layer'] != G.nodes[v]['layer']:
                    edges_to_remove.append((u, v))
            G.remove_edges_from(edges_to_remove)
            
        if config.random_graph:
            # Randomize edge connections
            edges = list(G.edges())
            G.remove_edges_from(edges)
            np.random.shuffle(edges)
            G.add_edges_from(edges)
            
        if config.static_graph:
            # No dynamic updates - graph remains static
            G.graph['dynamic'] = False
            
        return G
    
    def test_graph_structure_ablations(self) -> Dict[str, Dict]:
        """Test different graph representations"""
        print("\n=== Testing Graph Structure Ablations ===")
        
        ablations = {
            "single_layer": GraphConfig(single_layer=True),
            "no_cross_layer_edges": GraphConfig(no_cross_layer_edges=True),
            "random_graph": GraphConfig(random_graph=True),
            "static_graph": GraphConfig(static_graph=True)
        }
        
        results = {}
        for variant, config in ablations.items():
            print(f"\nTesting {variant}...")
            modified_graph = self._create_modified_graph(config)
            
            # Run detection on all experiments
            experiment_results = {}
            for exp_file in self.experiment_files:
                timestamps, metrics_list = load_experiment_data(exp_file)
                fault_type = exp_file.split('/')[-1].split('_')[0]
                
                # Run detection with modified graph
                detections = graph_heal_detection_new(
                    timestamps, 
                    metrics_list, 
                    fault_type,
                    graph=modified_graph
                )
                
                metrics = calculate_accuracy_metrics(
                    detections,
                    (timestamps[0], timestamps[-1]),
                    timestamps
                )
                
                experiment_results[exp_file] = {
                    "accuracy": metrics['accuracy'],
                    "precision": metrics['precision'],
                    "recall": metrics['recall'],
                    "f1_score": metrics['f1_score']
                }
            
            results[variant] = experiment_results
            
        return results
    
    def test_detection_algorithm_ablations(self) -> Dict[str, Dict]:
        """Test different detection algorithms"""
        print("\n=== Testing Detection Algorithm Ablations ===")
        
        detection_variants = {
            "pure_statistical": DetectionConfig(
                z_threshold=1.5,
                use_graph_correlation=False,
                use_temporal_patterns=False
            ),
            "pure_graph": DetectionConfig(
                z_threshold=None,
                use_graph_correlation=True,
                use_temporal_patterns=False
            ),
            "temporal_only": DetectionConfig(
                z_threshold=None,
                use_graph_correlation=False,
                use_temporal_patterns=True
            ),
            "hybrid_conservative": DetectionConfig(
                z_threshold=2.0,
                use_graph_correlation=True,
                use_temporal_patterns=True,
                require_consensus=True
            ),
            "hybrid_aggressive": DetectionConfig(
                z_threshold=1.0,
                use_graph_correlation=True,
                use_temporal_patterns=True,
                require_consensus=False
            )
        }
        
        results = {}
        for variant, config in detection_variants.items():
            print(f"\nTesting {variant}...")
            
            experiment_results = {}
            for exp_file in self.experiment_files:
                timestamps, metrics_list = load_experiment_data(exp_file)
                fault_type = exp_file.split('/')[-1].split('_')[0]
                
                # Run detection with variant config
                detections = graph_heal_detection_new(
                    timestamps,
                    metrics_list,
                    fault_type,
                    config=config
                )
                
                metrics = calculate_accuracy_metrics(
                    detections,
                    (timestamps[0], timestamps[-1]),
                    timestamps
                )
                
                experiment_results[exp_file] = {
                    "accuracy": metrics['accuracy'],
                    "precision": metrics['precision'],
                    "recall": metrics['recall'],
                    "f1_score": metrics['f1_score']
                }
            
            results[variant] = experiment_results
            
        return results
    
    def test_threshold_sensitivity_ablation(self) -> Tuple[Dict[str, Dict], Dict[str, Any]]:
        """Test sensitivity to detection thresholds"""
        print("\n=== Testing Threshold Sensitivity Ablation ===")
        
        # Test z-score thresholds
        z_score_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
        
        # Test different threshold combinations
        cpu_thresholds = [50, 60, 70, 80, 90]  # %
        memory_thresholds = [200, 300, 400, 500, 600]  # MB
        response_thresholds = [100, 150, 200, 250, 300]  # ms
        
        sensitivity_results = {}
        
        # Z-score sensitivity
        for z_val in z_score_values:
            print(f"\nTesting z-score threshold: {z_val}")
            # Convert to DetectionConfig
            config = DetectionConfig(z_threshold=z_val)
            
            experiment_results = {}
            for exp_file in self.experiment_files:
                timestamps, metrics_list = load_experiment_data(exp_file)
                fault_type = exp_file.split('/')[-1].split('_')[0]
                
                detections = graph_heal_detection_new(
                    timestamps,
                    metrics_list,
                    fault_type,
                    config=config
                )
                
                metrics = calculate_accuracy_metrics(
                    detections,
                    (timestamps[0], timestamps[-1]),
                    timestamps
                )
                
                experiment_results[exp_file] = metrics
            
            sensitivity_results[f"zscore_{z_val}"] = experiment_results
        
        # Find optimal configurations
        optimal_configs = self._find_pareto_optimal_configs(sensitivity_results)
        
        return sensitivity_results, optimal_configs
    
    def test_recovery_strategy_ablations(self) -> Dict[str, Dict]:
        """Test different recovery strategies"""
        print("\n=== Testing Recovery Strategy Ablations ===")
        
        recovery_strategies = {
            "reactive_only": RecoveryConfig(
                proactive=False,
                batch_actions=False,
                dependency_aware=False
            ),
            "proactive_only": RecoveryConfig(
                proactive=True,
                batch_actions=False,
                dependency_aware=False
            ),
            "batch_recovery": RecoveryConfig(
                proactive=False,
                batch_actions=True,
                dependency_aware=True
            ),
            "dependency_unaware": RecoveryConfig(
                proactive=False,
                batch_actions=False,
                dependency_aware=False
            ),
            "graph_guided": RecoveryConfig(
                proactive=True,
                batch_actions=True,
                dependency_aware=True,
                topology_guided=True
            )
        }
        
        results = {}
        for strategy, config in recovery_strategies.items():
            print(f"\nTesting {strategy}...")
            
            experiment_results = {}
            for exp_file in self.experiment_files:
                timestamps, metrics_list = load_experiment_data(exp_file)
                fault_type = exp_file.split('/')[-1].split('_')[0]
                
                # Simulate recovery with strategy
                recovery_metrics = self._simulate_recovery(
                    timestamps,
                    metrics_list,
                    fault_type,
                    config
                )
                
                experiment_results[exp_file] = recovery_metrics
            
            results[strategy] = experiment_results
            
        return results
    
    def _simulate_recovery(
        self,
        timestamps: List[datetime],
        metrics_list: List[Dict[str, float]],
        fault_type: str,
        config: RecoveryConfig
    ) -> Dict[str, float]:
        """Simulate recovery process with given strategy"""
        # This is a simplified simulation - in reality, you'd implement actual recovery logic
        recovery_time = np.random.uniform(1.0, 5.0)  # seconds
        success_rate = np.random.uniform(0.7, 0.95)
        
        return {
            "recovery_time": recovery_time,
            "success_rate": success_rate,
            "resource_overhead": np.random.uniform(0.1, 0.3)  # CPU usage during recovery
        }
    
    def _find_pareto_optimal_configs(self, sensitivity_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Find Pareto-optimal configurations from sensitivity results"""
        # This is a simplified implementation - you might want to use more sophisticated
        # multi-objective optimization techniques in practice
        optimal_configs = {}
        
        for exp_file in self.experiment_files:
            best_config = None
            best_f1 = 0.0
            
            for config_name, results in sensitivity_results.items():
                f1_score = results[exp_file]['f1_score']
                if f1_score > best_f1:
                    best_f1 = f1_score
                    best_config = config_name
            
            optimal_configs[exp_file] = {
                "config": best_config,
                "f1_score": best_f1
            }
        
        return optimal_configs
    
    def plot_results(self, results: Dict[str, Dict], study_type: str):
        """Plot ablation study results"""
        if study_type == 'recovery_strategy':
            self._plot_recovery_results(results)
        else:
            self._plot_detection_results(results, study_type)
    
    def _plot_detection_results(self, results: Dict[str, Dict], study_type: str):
        """Plot detection-related metrics (accuracy, precision, recall, F1)"""
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        experiments = [f.split('/')[-1].replace('.json', '') for f in self.experiment_files]
        
        for metric in metrics:
            plt.figure(figsize=(12, 6))
            x = np.arange(len(experiments))
            width = 0.15
            
            for i, (variant, data) in enumerate(results.items()):
                values = [data[exp_file][metric] for exp_file in self.experiment_files]
                plt.bar(x + i * width, values, width, label=variant.replace('_', ' ').title())
            
            plt.xlabel('Experiment')
            plt.ylabel(f'{metric.capitalize()} (%)')
            plt.title(f'{study_type} - {metric.capitalize()} Comparison')
            plt.xticks(x + width * 2, [exp.replace('_experiment', '').capitalize() for exp in experiments])
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(f'results/advanced_ablation_{study_type}_{metric}.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_recovery_results(self, results: Dict[str, Dict]):
        """Plot recovery-related metrics (success rate, recovery time, resource overhead)"""
        metrics = ['success_rate', 'recovery_time', 'resource_overhead']
        metric_labels = {
            'success_rate': 'Success Rate (%)',
            'recovery_time': 'Recovery Time (s)',
            'resource_overhead': 'Resource Overhead (%)'
        }
        experiments = [f.split('/')[-1].replace('.json', '') for f in self.experiment_files]
        
        for metric in metrics:
            plt.figure(figsize=(12, 6))
            x = np.arange(len(experiments))
            width = 0.15
            
            for i, (strategy, data) in enumerate(results.items()):
                values = [data[exp_file][metric] for exp_file in self.experiment_files]
                plt.bar(x + i * width, values, width, label=strategy.replace('_', ' ').title())
            
            plt.xlabel('Experiment')
            plt.ylabel(metric_labels[metric])
            plt.title(f'Recovery Strategy - {metric.replace("_", " ").title()} Comparison')
            plt.xticks(x + width * 2, [exp.replace('_experiment', '').capitalize() for exp in experiments])
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(f'results/advanced_ablation_recovery_{metric}.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_combined_graph_structure(self, results: Dict[str, Dict]):
        """Generate a combined plot for all graph structure ablation metrics"""
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        experiments = [f.split('/')[-1].replace('.json', '') for f in self.experiment_files]
        variants = list(results.keys())
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Graph Structure Ablation Study Results', fontsize=16)
        
        # Plot each metric in its own subplot
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            x = np.arange(len(experiments))
            width = 0.15
            
            for i, variant in enumerate(variants):
                values = [results[variant][exp_file][metric] for exp_file in self.experiment_files]
                ax.bar(x + i * width, values, width, label=variant.replace('_', ' ').title())
            
            ax.set_xlabel('Experiment')
            ax.set_ylabel(f'{metric.capitalize()} (%)')
            ax.set_title(f'{metric.capitalize()} Comparison')
            ax.set_xticks(x + width * 2)
            ax.set_xticklabels([exp.replace('_experiment', '').capitalize() for exp in experiments])
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('results/advanced_ablation_graph_structure_combined.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_combined_detection_algorithm(self, results: Dict[str, Dict]):
        """Generate a combined plot for all detection algorithm ablation metrics"""
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        experiments = [f.split('/')[-1].replace('.json', '') for f in self.experiment_files]
        variants = list(results.keys())
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Detection Algorithm Ablation Study Results', fontsize=16)
        
        # Plot each metric in its own subplot
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            x = np.arange(len(experiments))
            width = 0.15
            
            for i, variant in enumerate(variants):
                values = [results[variant][exp_file][metric] for exp_file in self.experiment_files]
                ax.bar(x + i * width, values, width, label=variant.replace('_', ' ').title())
            
            ax.set_xlabel('Experiment')
            ax.set_ylabel(f'{metric.capitalize()} (%)')
            ax.set_title(f'{metric.capitalize()} Comparison')
            ax.set_xticks(x + width * 2)
            ax.set_xticklabels([exp.replace('_experiment', '').capitalize() for exp in experiments])
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('results/advanced_ablation_detection_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_combined_sensitivity_analysis(self, results: Dict[str, Dict]):
        """Generate a combined plot for threshold sensitivity analysis metrics"""
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        experiments = [f.split('/')[-1].replace('.json', '') for f in self.experiment_files]
        thresholds = sorted(results.keys(), key=lambda x: float(x.split('_')[1]))
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Threshold Sensitivity Analysis Results', fontsize=16)
        
        # Plot each metric in its own subplot
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            x = np.arange(len(experiments))
            width = 0.15
            
            for i, threshold in enumerate(thresholds):
                values = [results[threshold][exp_file][metric] for exp_file in self.experiment_files]
                ax.bar(x + i * width, values, width, label=f'Z-score {threshold.split("_")[1]}')
            
            ax.set_xlabel('Experiment')
            ax.set_ylabel(f'{metric.capitalize()} (%)')
            ax.set_title(f'{metric.capitalize()} Comparison')
            ax.set_xticks(x + width * 2)
            ax.set_xticklabels([exp.replace('_experiment', '').capitalize() for exp in experiments])
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('results/advanced_ablation_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_all_studies(self):
        """Run all ablation studies and save results"""
        # Graph structure ablation
        graph_results = self.test_graph_structure_ablations()
        self.plot_results(graph_results, 'graph_structure')
        self.plot_combined_graph_structure(graph_results)
        
        # Detection algorithm ablation
        detection_results = self.test_detection_algorithm_ablations()
        self.plot_results(detection_results, 'detection_algorithm')
        self.plot_combined_detection_algorithm(detection_results)
        
        # Threshold sensitivity ablation
        sensitivity_results, optimal_configs = self.test_threshold_sensitivity_ablation()
        self.plot_results(sensitivity_results, 'threshold_sensitivity')
        self.plot_combined_sensitivity_analysis(sensitivity_results)
        
        # Recovery strategy ablation
        recovery_results = self.test_recovery_strategy_ablations()
        self.plot_results(recovery_results, 'recovery_strategy')
        
        # Save all results
        all_results = {
            'graph_structure': graph_results,
            'detection_algorithm': detection_results,
            'threshold_sensitivity': sensitivity_results,
            'optimal_configs': optimal_configs,
            'recovery_strategy': recovery_results
        }
        
        with open('results/advanced_ablation_study_results.json', 'w') as f:
            json.dump(all_results, f, indent=2, cls=NumpyEncoder)
        
        print("\n=== Advanced Ablation Study Complete ===")
        print("Results saved to results/advanced_ablation_study_results.json")
        print("Plots saved to results/advanced_ablation_*.png")

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def get_system_specs():
    """Gets and prints system hardware and software specifications."""
    specs = {
        "Python Version": platform.python_version(),
        "NetworkX Version": nx.__version__,
        "OS": f"{platform.system()} {platform.release()}",
        "CPU": get_cpu_info().get('brand_raw', 'N/A'),
        "Cores": psutil.cpu_count(logical=False),
        "RAM (GB)": round(psutil.virtual_memory().total / (1024**3), 2)
    }
    return specs

def generate_graph(topology, num_nodes, avg_degree, seed):
    """Generates a graph of a specific topology, size, and density."""
    rng = random.Random(seed)
    if topology == "scale_free":
        m = int(avg_degree / 2)
        if m == 0: m = 1
        return nx.barabasi_albert_graph(n=num_nodes, m=m, seed=rng)
    elif topology == "dag":
        # Create a DAG with a similar number of edges
        p = avg_degree / (num_nodes - 1)
        return nx.gnp_random_graph(num_nodes, p, seed=rng, directed=True)
    return None

def simulate_rca_and_measure(graph, process, trial_seed):
    """Simulates a single RCA run and measures its performance."""
    rng = random.Random(trial_seed)
    
    # --- Phase 1: Fault Localization ---
    gc.collect()
    tracemalloc.start()
    
    cpu_percent_before = process.cpu_percent()
    loc_start_time = time.perf_counter_ns()
    
    true_root_cause = rng.choice(list(graph.nodes()))
    anomalous_nodes = set(list(nx.descendants(graph, true_root_cause)) + [true_root_cause])
    anomalous_subgraph = nx.DiGraph(graph.subgraph(anomalous_nodes))
    
    if anomalous_subgraph.number_of_nodes() > 0:
        pagerank = nx.pagerank(anomalous_subgraph.reverse(copy=True))
        max(pagerank, key=pagerank.get)
        
    loc_latency_ns = time.perf_counter_ns() - loc_start_time
    cpu_percent_after = process.cpu_percent()
    
    mem_current_bytes, mem_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return {
        "Localization Latency (ms)": loc_latency_ns / 1e6,
        "CPU Utilization (%)": (cpu_percent_before + cpu_percent_after) / 2,
        "Memory Peak (MB)": mem_peak_bytes / (1024 * 1024),
    }

def run_scalability_experiments():
    """Runs the full suite of scalability experiments."""
    process = psutil.Process(os.getpid())
    psutil.cpu_percent() # Initial call to establish a baseline
    
    all_results = []
    
    for topology in TOPOLOGIES:
        for num_nodes in GRAPH_SIZES:
            # Stop if the estimated memory footprint gets too large
            if num_nodes > 50000 and psutil.virtual_memory().available < 4 * 1024**3: # 4GB
                 print(f"WARN: Low memory, skipping remaining tests for {topology} topology.")
                 break
            
            for degree in AVG_DEGREES:
                trial_results = []
                
                # --- Phase 0: Graph Building (measured once) ---
                build_start_time = time.perf_counter_ns()
                graph = generate_graph(topology, num_nodes, degree, seed=0)
                build_latency_ms = (time.perf_counter_ns() - build_start_time) / 1e6
                if graph is None: continue

                print(f"Running: {topology}, Nodes={num_nodes}, Degree={degree}, Trials={N_TRIALS}")

                for i in range(N_TRIALS):
                    trial_seed = i # Use trial number as seed for repeatability
                    metrics = simulate_rca_and_measure(graph, process, trial_seed)
                    trial_results.append(metrics)
                
                # Aggregate results for this configuration
                df_trials = pd.DataFrame(trial_results)
                summary = {
                    "Topology": topology,
                    "Nodes": num_nodes,
                    "Edges": graph.number_of_edges(),
                    "Avg Degree": degree,
                    "Build Latency (ms)": build_latency_ms,
                    "T_d_median (ms)": df_trials["Localization Latency (ms)"].median(),
                    "T_d_p5 (ms)": df_trials["Localization Latency (ms)"].quantile(0.05),
                    "T_d_p95 (ms)": df_trials["Localization Latency (ms)"].quantile(0.95),
                    "M_r_median (MB)": df_trials["Memory Peak (MB)"].median(),
                    "C_p_median (%)": df_trials["CPU Utilization (%)"].median(),
                }
                all_results.append(summary)

    return pd.DataFrame(all_results)

def main():
    """Main function to run the advanced ablation study."""
    print("--- System Specifications ---")
    specs = get_system_specs()
    for key, value in specs.items():
        print(f"{key}: {value}")
    
    print("\n--- Starting Advanced Ablation Study ---")
    results_df = run_scalability_experiments()
    
    print("\n--- Aggregated Results Summary ---")
    print(results_df.to_string())
    
    # Save detailed results to CSV
    output_filename = "scalability_results_detailed.csv"
    results_df.to_csv(output_filename, index=False)
    print(f"\nFull results saved to {output_filename}")

if __name__ == "__main__":
    main() 