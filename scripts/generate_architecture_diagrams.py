import graphviz
import os

def generate_architecture_diagram():
    """Generates the main Graph-Heal system architecture diagram."""
    dot = graphviz.Digraph('Architecture', comment='Graph-Heal System Architecture')
    dot.attr(rankdir='TB', splines='ortho', concentrate='true')
    dot.attr('node', shape='box', style='rounded')

    with dot.subgraph(name='cluster_monitoring') as c:
        c.attr(label='Monitoring & Detection Layer', style='filled', color='lightcyan')
        c.node('prometheus', 'Prometheus\n(Metrics)', shape='cylinder')
        c.node('monitor', 'Central Monitor')
        c.node('detector', 'Anomaly Detector', shape='diamond')

    with dot.subgraph(name='cluster_analysis') as c:
        c.attr(label='Analysis & Localization Layer', style='filled', color='lightyellow')
        c.node('graph_model', 'Service Dependency Graph', shape='box3d')
        c.node('localization', 'Fault Localization', shape='hexagon')

    with dot.subgraph(name='cluster_recovery') as c:
        c.attr(label='Recovery & Adaptation Layer', style='filled', color='lightpink')
        c.node('recovery_system', 'Recovery Orchestrator', shape='ellipse')
        c.node('adapter', 'Pluggable Adapters')

    with dot.subgraph(name='cluster_adapters') as c:
        c.attr(label='Adapter Implementations', style='dotted')
        c.node('docker', 'Docker Adapter', shape='cds')
        c.node('k8s', 'Kubernetes Adapter', shape='cds')
        c.node('opcua', 'OPC-UA Adapter', shape='cds')

    # Edges
    dot.edge('prometheus', 'monitor', label='Metrics Stream')
    dot.edge('monitor', 'detector', label='Time-series Data')
    dot.edge('detector', 'localization', label='Anomalies')
    dot.edge('localization', 'graph_model', label='Updates Health')
    dot.edge('localization', 'recovery_system', label='Root Causes')
    dot.edge('recovery_system', 'adapter', label='Executes Strategy')
    
    dot.edge('adapter', 'docker')
    dot.edge('adapter', 'k8s')
    dot.edge('adapter', 'opcua')
    
    output_path = os.path.join('figures', 'system_architecture')
    dot.render(output_path, format='png', view=False, cleanup=True)
    print(f"Generated {output_path}.png")


def generate_graph_construction_diagram():
    """Generates the service graph construction and enrichment diagram."""
    dot = graphviz.Digraph('GraphConstruction', comment='Service Graph Construction')
    dot.attr(rankdir='LR')
    dot.attr('node', shape='box', style='rounded')

    with dot.subgraph(name='cluster_sources') as c:
        c.attr(label='Discovery Sources')
        c.node('docker_api', 'Docker Engine', shape='component')
        c.node('k8s_api', 'Kubernetes API', shape='component')
        c.node('manual_config', 'Manual Config\n(e.g., OPC-UA)', shape='note')
        
    with dot.subgraph(name='cluster_process') as c:
        c.attr(label='Graph Building Process')
        c.node('builder', 'ServiceGraph Builder')
        c.node('model', 'Initial Graph Model', shape='box3d')
        c.node('enrich', 'Metric/Topology Enrichment')
        c.node('final_graph', 'Attributed Dependency Graph', shape='box3d')

    dot.edge('docker_api', 'builder', label='Service/Network Info')
    dot.edge('k8s_api', 'builder', label='Pod/Service Info')
    dot.edge('manual_config', 'builder', label='Device/Node Info')
    dot.edge('builder', 'model', label='Constructs')
    dot.edge('model', 'enrich', label='Enriches')
    dot.edge('enrich', 'final_graph', label='Produces')

    output_path = os.path.join('figures', 'graph_construction')
    dot.render(output_path, format='png', view=False, cleanup=True)
    print(f"Generated {output_path}.png")


def main():
    os.makedirs('figures', exist_ok=True)
    generate_architecture_diagram()
    generate_graph_construction_diagram()

if __name__ == "__main__":
    main() 