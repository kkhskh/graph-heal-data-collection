import json
from datetime import datetime
import sys
import time
from kubernetes import client, config

filename = 'results/processed/cpu_test_cpu_experiment_4.json'
if len(sys.argv) > 1:
    filename = sys.argv[1]

with open(filename, 'r') as f:
    data = json.load(f)

cpu_values = [m['service_cpu_usage'] for m in data['metrics']]
timestamps = data['timestamps']
fault_periods = data['fault_periods']

print(f"Loaded {filename}")
print(f"First 50 CPU usage values:")
for i in range(min(50, len(cpu_values))):
    print(f"{i:3d} {timestamps[i]}  {cpu_values[i]:.3f}")

print("\nFault periods:")
for period in fault_periods:
    print(f"  Start: {period['start']}  End: {period['end']}  Type: {period['type']}  Pattern: {period.get('pattern', '')}")

# Optionally, print summary stats
print(f"\nCPU usage min: {min(cpu_values):.3f}, max: {max(cpu_values):.3f}, mean: {sum(cpu_values)/len(cpu_values):.3f}") 

def debug_metrics():
    """Connects to Kubernetes and prints raw pod metrics in a loop."""
    try:
        config.load_kube_config()
    except config.ConfigException:
        config.load_incluster_config()
    
    k8s_custom_objects = client.CustomObjectsApi()
    
    print("--- Starting Metric Debugger ---")
    print("Watching for pod metrics every 5 seconds. Press Ctrl+C to stop.")
    
    while True:
        try:
            pod_metrics_list = k8s_custom_objects.list_namespaced_custom_object(
                group="metrics.k8s.io",
                version="v1beta1",
                namespace="default",
                plural="pods"
            )
            
            print(f"\n--- Tick at {time.strftime('%H:%M:%S')} ---")
            if not pod_metrics_list['items']:
                print("No pod metrics found.")
            
            for item in pod_metrics_list['items']:
                pod_name = item['metadata']['name']
                cpu_usage = item['containers'][0]['usage']['cpu']
                mem_usage = item['containers'][0]['usage']['memory']
                print(f"Pod: {pod_name:<40} CPU: {cpu_usage:<10} Memory: {mem_usage}")

        except client.ApiException as e:
            if e.status == 404:
                print("Metrics API not found. Is the Metrics Server installed and running?")
                sys.exit(1)
            print(f"Error querying Metrics API: {e.status} - {e.reason}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            
        time.sleep(5)

if __name__ == "__main__":
    debug_metrics() 