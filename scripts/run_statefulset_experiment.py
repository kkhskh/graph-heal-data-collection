import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import json
import logging
from datetime import datetime
from kubernetes import client, config
from kubernetes.utils import create_from_yaml
from graph_heal.service_graph import ServiceGraph
from graph_heal.anomaly_detection import StatisticalAnomalyDetector
from graph_heal.recovery_system import EnhancedRecoverySystem
from graph_heal.recovery.kubernetes_adapter import KubernetesAdapter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# The service architecture for this experiment is simpler.
# We focus on a single stateful service (redis) and a client (service-a).
SERVICES = {
    'service-a': ['redis'],
    'redis': []
}
STATEFULSET_NAME = "redis"
STATEFULSET_YAML = "k8s/redis-statefulset.yaml"

def inject_statefulset_fault(api_instance, statefulset_name, namespace="default"):
    """Injects a fault by changing the command, causing a crash loop."""
    try:
        stateful_set = api_instance.read_namespaced_stateful_set(statefulset_name, namespace)
        original_command = stateful_set.spec.template.spec.containers[0].command
        
        # A bad command to induce failure
        stateful_set.spec.template.spec.containers[0].command = ["/bin/sh", "-c", "echo 'Failing...' && exit 1"]
        api_instance.patch_namespaced_stateful_set(statefulset_name, namespace, stateful_set)
        
        logger.info(f"Successfully injected fault into StatefulSet {statefulset_name}.")
        return original_command
    except client.ApiException as e:
        logger.error(f"Error injecting fault into StatefulSet {statefulset_name}: {e}")
        return None

def cleanup_statefulset_fault(api_instance, statefulset_name, original_command, namespace="default"):
    """Restores the original command to fix the fault."""
    try:
        stateful_set = api_instance.read_namespaced_stateful_set(statefulset_name, namespace)
        stateful_set.spec.template.spec.containers[0].command = original_command
        api_instance.patch_namespaced_stateful_set(statefulset_name, namespace, stateful_set)
        logger.info(f"Successfully cleaned up fault from StatefulSet {statefulset_name}.")
        return True
    except client.ApiException as e:
        # It might have been deleted, which is fine.
        if e.status == 404:
            logger.warning(f"StatefulSet {statefulset_name} not found during cleanup. It might have been deleted.")
            return True
        logger.error(f"Error cleaning up fault from StatefulSet {statefulset_name}: {e}")
        return False

def log_event(log_data, event):
    """Appends a timestamped event to the log data."""
    event['timestamp_iso'] = datetime.fromtimestamp(event['timestamp']).isoformat()
    log_data.append(event)

def get_statefulset_status(apps_v1_api, statefulset_name, namespace="default"):
    """
    Checks the status of a specific StatefulSet and provides default statuses
    for other known services.
    """
    statuses = {}
    # Provide a default healthy status for all services first
    for service in SERVICES:
        statuses[service] = {"health_status": 1.0, "error_rate": 0.0}

    try:
        sts = apps_v1_api.read_namespaced_stateful_set(statefulset_name, namespace)
        is_ready = sts.status.ready_replicas is not None and sts.status.ready_replicas == sts.spec.replicas
        
        # Overwrite the status for the statefulset if it's not ready
        if not is_ready:
            statuses[statefulset_name] = {
                "health_status": 0.0,
                "error_rate": 1.0,
            }
    except client.ApiException as e:
        logger.error(f"Could not read StatefulSet {statefulset_name}: {e}")
        # If we can't read it, assume it's unhealthy
        statuses[statefulset_name] = {"health_status": 0.0, "error_rate": 1.0}
    return statuses

def run_experiment(mode, log_data, duration=90):
    """Runs a single experiment mode for the StatefulSet."""
    print(f"\n--- Starting StatefulSet experiment: {mode} mode ---")

    k8s_adapter = KubernetesAdapter()
    service_graph = ServiceGraph()
    
    print("Building service graph...")
    for service, dependencies in SERVICES.items():
        service_graph.add_service(service, dependencies=dependencies)
    
    # The detector needs to know about the metric we are using.
    anomaly_detector = StatisticalAnomalyDetector(services=list(SERVICES.keys()), metrics_to_monitor=['error_rate'])
    recovery_system = EnhancedRecoverySystem(service_graph, adapter=k8s_adapter)

    # In baseline mode, we don't prime the detector, just inject and observe.
    if mode == 'baseline':
        print(f"Injecting fault into StatefulSet '{STATEFULSET_NAME}'...")
        original_command = inject_statefulset_fault(k8s_adapter.apps_v1, STATEFULSET_NAME)
        # We need to handle the fault cleanup correctly even if the run is short.
        # The original command can be None, that is not an error.
        print("Observing baseline behavior with fault...")
    # In graph-heal mode, we prime the detector with normal data first.
    else:
        print("Priming anomaly detector with healthy metrics...")
        # Increase priming loop to establish a stronger baseline
        for _ in range(10):
            # Pass all service statuses during priming
            healthy_statuses = {service: {'error_rate': 0.0} for service in SERVICES}
            anomaly_detector.detect_anomalies(healthy_statuses, time.time())
            time.sleep(1)
        
        print(f"Injecting fault into StatefulSet '{STATEFULSET_NAME}'...")
        original_command = inject_statefulset_fault(k8s_adapter.apps_v1, STATEFULSET_NAME)

        print("Waiting for fault to propagate...")
        time.sleep(5) # Give the cluster a moment to reflect the pod's crash loop status

    log_event(log_data, {
        "timestamp": time.time(),
        "event_type": "fault_injected",
        "service_id": STATEFULSET_NAME,
        "mode": mode
    })
    
    start_time = time.time()
    recovery_triggered = False
    while time.time() - start_time < duration:
        print(f"--- Monitoring tick ({mode} mode) ---")
        
        service_statuses = get_statefulset_status(k8s_adapter.apps_v1, STATEFULSET_NAME)

        log_event(log_data, {
            "timestamp": time.time(),
            "event_type": "metrics_tick",
            "service_metrics": service_statuses,
            "mode": mode
        })

        if mode == 'graph-heal' and service_statuses and not recovery_triggered:
            anomalies = anomaly_detector.detect_anomalies(service_statuses, time.time())
            for anom in anomalies:
                # We only care about the redis service for this test
                if anom['service_id'] != STATEFULSET_NAME:
                    continue

                log_event(log_data, {
                    "timestamp": time.time(),
                    "event_type": "anomaly_detected",
                    "anomaly": anom,
                    "mode": mode
                })
                print(f"!!! Anomaly detected: {anom['service_id']} ({anom['type']})")
                
                plan = recovery_system.get_recovery_plan(anom['service_id'], anom['type'])
                for action in plan:
                    # We expect a 'restart_service' action
                    if action.action_type.value == 'restart_service':
                        recovery_triggered = True # Prevent continuous recovery loops

                    log_event(log_data, {
                        "timestamp": time.time(),
                        "event_type": "recovery_action_triggered",
                        "action": {
                            "action_type": action.action_type.value,
                            "target_service": action.target_service,
                        },
                        "mode": mode
                    })
                    print(f">>> Executing recovery action: {action.action_type.value} on {action.target_service}")
                    recovery_system.execute_recovery_action(action)
        
        if mode == 'graph-heal' and recovery_triggered:
            print("Recovery has been triggered. Monitoring for stabilization.")
            # Break early if recovery is done
            if get_statefulset_status(k8s_adapter.apps_v1, STATEFULSET_NAME).get(STATEFULSET_NAME, {}).get('health_status', 0.0) == 1.0:
                print("StatefulSet appears to have recovered.")
                break


        time.sleep(5)

    print(f"\n--- Experiment finished ({mode} mode). Cleaning up fault. ---")
    cleanup_statefulset_fault(k8s_adapter.apps_v1, STATEFULSET_NAME, original_command)

def main():
    """Main function to setup, run, and cleanup the experiment."""
    try:
        config.load_kube_config()
        k8s_client = client.ApiClient()
        apps_v1 = client.AppsV1Api()
        logger.info("Loaded local kubeconfig and initialized Kubernetes client.")
    except config.ConfigException:
        logger.error("Could not load kubeconfig. Is your cluster running and configured?")
        return

    print(f"Applying StatefulSet manifest from {STATEFULSET_YAML}...")
    try:
        create_from_yaml(k8s_client, STATEFULSET_YAML, verbose=True)
        # Wait for the statefulset to be created
        time.sleep(15)
    except Exception as e:
        logger.error(f"Error applying YAML: {e}. It might already exist, continuing...")

    log_data = []
    
    run_experiment('baseline', log_data, duration=60)
    
    print("\n\n--- Waiting 30 seconds for cluster to stabilize between runs ---\n\n")
    time.sleep(30)
    
    run_experiment('graph-heal', log_data)

    log_file = "results/statefulset_experiment_log.json"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=4)
        
    print(f"\nFull experiment log saved to {log_file}")

    print(f"Cleaning up by deleting StatefulSet '{STATEFULSET_NAME}'...")
    try:
        apps_v1.delete_namespaced_stateful_set(STATEFULSET_NAME, "default")
        # Also delete the associated headless service
        core_v1 = client.CoreV1Api()
        core_v1.delete_namespaced_service(STATEFULSET_NAME, "default")
        print("Cleanup complete.")
    except client.ApiException as e:
        if e.status == 404:
            logger.warning("StatefulSet or service not found during cleanup, may have been deleted already.")
        else:
            logger.error(f"Error during cleanup: {e}")


if __name__ == "__main__":
    main() 