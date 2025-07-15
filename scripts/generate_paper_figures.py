from graphviz import Digraph
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_reactor_service_graph():
    """Generates and saves the reactor service graph diagram."""
    dot = Digraph(comment='Reactor Service Graph')
    dot.attr(rankdir='TB', splines='ortho')

    with dot.subgraph(name='cluster_app') as c:
        c.attr(label='Application Layer', style='filled', color='lightgrey')
        c.node('Temp', 'TempSensorAggregator')
        c.node('Pressure', 'PressureSensorAggregator')
        c.node('Control', 'ReactorControl')
        c.node('Pump', 'CoolantPumpController')
        c.node('Shutdown', 'SafetyShutdown')
        c.node('Analytics', 'AnalyticsEngine')
        c.node('Dashboard', 'OperatorDashboard')

    with dot.subgraph(name='cluster_infra') as c:
        c.attr(label='Infrastructure Layer', style='filled', color='lightgrey')
        c.node('Gateway', 'SensorGateway')

    dot.edge('Gateway', 'Temp')
    dot.edge('Gateway', 'Pressure')
    dot.edge('Temp', 'Control')
    dot.edge('Pressure', 'Control')
    dot.edge('Control', 'Pump')
    dot.edge('Control', 'Shutdown')
    dot.edge('Control', 'Analytics')
    dot.edge('Analytics', 'Dashboard')

    output_path = os.path.join('figures', 'reactor_service_graph')
    dot.render(output_path, format='png', cleanup=True)
    print(f"Saved figure: {output_path}.png")

def generate_dependency_edge_types():
    """Generates and saves the dependency edge types diagram."""
    dot = Digraph(comment='Dependency Edge Types')
    dot.attr(rankdir='TB')

    # Synchronous RPC
    with dot.subgraph(name='cluster_rpc') as c:
        c.attr(label='Synchronous RPC')
        c.node('A1', 'Service A')
        c.node('B1', 'Service B')
        c.edge('A1', 'B1', style='solid')
    
    # Asynchronous Event
    with dot.subgraph(name='cluster_event') as c:
        c.attr(label='Asynchronous Event')
        c.node('A2', 'Service A')
        c.node('B2', 'Service B')
        c.edge('A2', 'B2', style='dashed')
        
    # Periodic Polling
    with dot.subgraph(name='cluster_poll') as c:
        c.attr(label='Periodic Polling')
        c.node('A3', 'Service A')
        c.node('B3', 'Service B')
        c.edge('A3', 'B3', style='dotted')
        
    # Fault Trigger
    with dot.subgraph(name='cluster_fault') as c:
        c.attr(label='Fault Trigger')
        c.node('A4', 'Service A')
        c.node('B4', 'Service B')
        c.edge('A4', 'B4', style='dashdot')
        
    output_path = os.path.join('figures', 'dependency_edge_types')
    dot.render(output_path, format='png', cleanup=True)
    print(f"Saved figure: {output_path}.png")

# --- Configuration ---
RESULTS_FILE = "scalability_results_detailed.csv"
OUTPUT_DIR = "figures"
SCALE_FREE_PLOT_LATENCY = f"{OUTPUT_DIR}/latency_vs_nodes_scale_free.png"
SCALE_FREE_PLOT_MEMORY = f"{OUTPUT_DIR}/memory_vs_nodes_scale_free.png"

def set_plot_style():
    """Sets a professional style for the plots."""
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['axes.titlesize'] = 16

def plot_latency_vs_nodes(df):
    """Plots detection latency vs. number of nodes on a log-log scale."""
    plt.figure()
    
    # Filter for scale-free topology for a clear plot
    scale_free_df = df[df['Topology'] == 'scale_free'].copy()

    # Pivot data to have degrees as columns
    pivot_df = scale_free_df.pivot(index='Nodes', columns='Avg Degree', values='T_d_median (ms)')

    ax = pivot_df.plot(kind='line', marker='o', logx=True, logy=True)
    
    plt.title('Detection Latency vs. System Size (Scale-Free Topology)')
    plt.xlabel('Number of Nodes (|V|)')
    plt.ylabel('Median Detection Latency (ms)')
    plt.grid(True, which="both", ls="--")
    plt.legend(title='Avg. Degree')
    
    # Save the figure
    plt.savefig(SCALE_FREE_PLOT_LATENCY, dpi=300, bbox_inches='tight')
    print(f"Saved latency plot to {SCALE_FREE_PLOT_LATENCY}")
    plt.close()

def plot_memory_vs_nodes(df):
    """Plots memory overhead vs. number of nodes, colored by degree."""
    plt.figure()

    # Filter for scale-free topology
    scale_free_df = df[df['Topology'] == 'scale_free'].copy()

    ax = sns.lineplot(
        data=scale_free_df,
        x="Nodes",
        y="M_r_median (MB)",
        hue="Avg Degree",
        marker="o",
        palette="viridis"
    )

    ax.set_xscale('log')
    ax.set_yscale('log')

    plt.title('Memory Overhead vs. System Size (Scale-Free Topology)')
    plt.xlabel('Number of Nodes (|V|)')
    plt.ylabel('Median Peak Memory (MB)')
    plt.grid(True, which="both", ls="--")
    plt.legend(title='Avg. Degree')

    # Save the figure
    plt.savefig(SCALE_FREE_PLOT_MEMORY, dpi=300, bbox_inches='tight')
    print(f"Saved memory plot to {SCALE_FREE_PLOT_MEMORY}")
    plt.close()

def main():
    """Main function to generate plots from the results CSV."""
    set_plot_style()
    
    try:
        df = pd.read_csv(RESULTS_FILE)
    except FileNotFoundError:
        print(f"ERROR: Results file not found at '{RESULTS_FILE}'")
        print("Please run 'scripts/advanced_ablation_study.py' first.")
        return

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate the plots
    plot_latency_vs_nodes(df)
    plot_memory_vs_nodes(df)

    print("\nFigure generation complete.")

if __name__ == "__main__":
    main() 