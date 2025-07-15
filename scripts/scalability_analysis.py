import time
import networkx as nx
import pandas as pd
import psutil
import os
import random

def generate_graph(num_nodes, avg_degree):
    """Generates a scale-free graph with a given number of nodes and average degree."""
    # A scale-free graph is a good model for many real-world networks.
    m = int(avg_degree / 2)
    graph = nx.barabasi_albert_graph(n=num_nodes, m=m)
    return graph

def simulate_graph_heal_rca(graph):
    """Simulates Graph-Heal's RCA process on a given graph."""
    
    # 1. Pick a random node as the true root cause of a failure.
    true_root_cause = random.choice(list(graph.nodes()))
    
    # 2. Identify the subgraph of anomalous nodes.
    # In a real scenario, the true root cause and its descendants would be anomalous.
    anomalous_nodes = set(list(nx.descendants(graph, true_root_cause)) + [true_root_cause])
    
    # Create a new graph from the subgraph of anomalous nodes
    anomalous_subgraph = nx.DiGraph(graph.subgraph(anomalous_nodes))
    
    if not anomalous_subgraph.nodes():
        return None # Should not happen

    # 3. Run PageRank on the anomalous subgraph to find the most likely cause.
    # We run PageRank on the reversed graph because influence flows against the direction of dependencies.
    pagerank = nx.pagerank(anomalous_subgraph.reverse(copy=True))
    identified_cause = max(pagerank, key=pagerank.get)
    
    return identified_cause

def run_scalability_test():
    """Runs the scalability test for different graph sizes."""
    
    process = psutil.Process(os.getpid())
    
    node_counts = [100, 1000, 10000, 50000]
    avg_degree = 5
    results = []
    
    print("Starting scalability analysis... (This may take over 5 minutes for 50,000 nodes)")
    
    for num_nodes in node_counts:
        # --- Memory Measurement Start ---
        mem_before = process.memory_info().rss / (1024 * 1024)  # in MB

        # --- Graph Generation ---
        graph = generate_graph(num_nodes, avg_degree)
        
        mem_after_graph = process.memory_info().rss / (1024 * 1024) # in MB
        graph_mem = mem_after_graph - mem_before

        # --- Latency Measurement Start ---
        start_time = time.time()
        
        # --- Run RCA Simulation ---
        simulate_graph_heal_rca(graph)
        
        # --- Latency Measurement End ---
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000

        # --- Memory Measurement End ---
        mem_after_rca = process.memory_info().rss / (1024 * 1024) # in MB
        total_mem = mem_after_rca - mem_before
        
        results.append({
            "Nodes": num_nodes,
            "Edges": graph.number_of_edges(),
            "Memory (MB)": round(total_mem, 2),
            "Detection Latency (ms)": round(latency_ms, 2)
        })
        print(f"Completed test for {num_nodes} nodes.")

    return pd.DataFrame(results)

def main():
    """Main function to run the scalability analysis."""
    print("This script analyzes the scalability of the Graph-Heal RCA algorithm.")
    print("It measures detection latency and memory footprint for growing graph sizes.")
    print("NOTE: You may need to install psutil (`pip install psutil`)")
    
    results_df = run_scalability_test()
    
    print("\n--- Scalability Analysis Results ---")
    print(results_df.to_string())

if __name__ == "__main__":
    main() 