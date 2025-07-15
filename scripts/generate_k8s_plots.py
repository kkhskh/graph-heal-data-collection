import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np

def generate_k8s_timeline_plot(log_file, output_file):
    """
    Parses the Kubernetes experiment log and generates a timeline visualization.
    """
    with open(log_file, 'r') as f:
        log_data = json.load(f)

    events = []
    for entry in log_data:
        # Only process the 'graph-heal' mode for the final plot
        if entry.get('mode') != 'graph-heal':
            continue

        ts = datetime.fromisoformat(entry['timestamp_iso'])
        event_type = entry['event_type']
        
        if event_type == 'fault_injected':
            events.append({
                'time': ts,
                'label': f"Fault Injected\n({entry['service_id']})",
                'type': 'fault',
                'service': entry['service_id']
            })
        elif event_type == 'anomaly_detected':
            events.append({
                'time': ts,
                'label': f"Anomaly Detected\n({entry['anomaly']['service_id']})",
                'type': 'detection',
                'service': entry['anomaly']['service_id']
            })
        elif event_type == 'recovery_action_triggered':
            events.append({
                'time': ts,
                'label': f"Recovery: {entry['action']['action_type']}\n({entry['action']['target_service']})",
                'type': 'recovery',
                'service': entry['action']['target_service']
            })

    if not events:
        print("No 'graph-heal' events found in the log file. Cannot generate plot.")
        return

    # Sort events by time
    events.sort(key=lambda x: x['time'])
    
    services = sorted(list(set(e['service'] for e in events)))
    service_y_map = {name: i for i, name in enumerate(services)}

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot lines connecting events for each service
    for service_name in services:
        service_events = [e for e in events if e['service'] == service_name]
        if not service_events:
            continue
        times = [e['time'] for e in service_events]
        y_vals = [service_y_map[service_name]] * len(times)
        ax.plot(times, y_vals, 'o-', ms=10, alpha=0.7, label=service_name)

    # Add event labels
    for event in events:
        y = service_y_map[event['service']]
        color_map = {'fault': 'red', 'detection': 'orange', 'recovery': 'green'}
        ax.text(event['time'], y + 0.1, event['label'], ha='center', va='bottom', 
                fontsize=9, color=color_map[event['type']], fontweight='bold')

    ax.set_yticks(range(len(services)))
    ax.set_yticklabels(services)
    
    # Format the x-axis to show time nicely
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S.%f'))
    plt.gcf().autofmt_xdate()
    
    plt.title('Kubernetes Fault Detection and Recovery Timeline (Graph-Heal)')
    plt.xlabel('Time')
    plt.ylabel('Service')
    plt.ylim(-0.5, len(services) - 0.5)
    plt.legend(title='Services', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    print(f"Saving plot to {output_file}...")
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    generate_k8s_timeline_plot(
        'results/k8s_experiment_log.json',
        'plots/k8s_recovery_timeline.png'
    ) 