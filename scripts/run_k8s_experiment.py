import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import json
import logging
from datetime import datetime
from kubernetes import client, config
from graph_heal.service_graph import ServiceGraph
from graph_heal.anomaly_detection import StatisticalAnomalyDetector
from graph_heal.recovery_system import EnhancedRecoverySystem
from graph_heal.recovery.kubernetes_adapter import KubernetesAdapter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Static definition of the service architecture.
SERVICES = {
    'service-a': ['service-b', 'service-c'],
    'service-b': ['service-d'],
    'service-c': ['service-d'],
    'service-d': []
}

def log_event(log_data, event):
    """Helper to append a timestamped event to the log list."""
    # This is a simplified version; a real implementation would handle non-serializable types better.
    def default_serializer(o):
        return str(o)
    log_data.append(json.loads(json.dumps(event, default=default_serializer)))


def inject_crash_loop_fault(api_instance, deployment_name, namespace="default"):
    """Injects a fault that causes pods to crash loop by setting a bad command."""
    try:
        deployment = api_instance.read_namespaced_deployment(deployment_name, namespace)
        original_command = deployment.spec.template.spec.containers[0].command
        
        deployment.spec.template.spec.containers[0].command = ["/bin/sh", "-c", "echo 'Crashing...' && exit 1"]
        api_instance.patch_namespaced_deployment(deployment_name, namespace, deployment)
        
        logger.info(f"Successfully injected crash loop fault into {deployment_name}.")
        return original_command
    except client.ApiException as e:
        logger.error(f"Error injecting fault into {deployment_name}: {e}")
        return None

def cleanup_fault(api_instance, deployment_name, original_command=None, namespace="default"):
    """Removes the fault condition by resetting the container command."""
    try:
        deployment = api_instance.read_namespaced_deployment(deployment_name, namespace)
        deployment.spec.template.spec.containers[0].command = original_command
        api_instance.patch_namespaced_deployment(deployment_name, namespace, deployment)
        logger.info(f"Successfully cleaned up fault from {deployment_name}.")
        return True
    except client.ApiException as e:
        logger.error(f"Error cleaning up fault from {deployment_name}: {e}")
        return False

def get_service_statuses(custom_objects_api, apps_v1_api, core_v1_api, pod_restart_history, namespace="default"):
    """
    Checks the status and metrics of all deployments by inspecting pod restart counts
    and container readiness, making the check stateful and more robust.
    """
    statuses = {}
    new_pod_restart_history = pod_restart_history.copy()
    try:
        deployments = apps_v1_api.list_namespaced_deployment(namespace)
        for dep in deployments.items:
            service_name = dep.metadata.name.replace('-deployment', '')
            desired_replicas = dep.spec.replicas
            
            selector = dep.spec.selector.match_labels
            pod_list = core_v1_api.list_namespaced_pod(
                namespace, label_selector=','.join([f"{k}={v}" for k, v in selector.items()])
            )
            
            truly_ready_replicas = 0
            has_restarted = False
            for pod in pod_list.items:
                pod_name = pod.metadata.name
                
                if pod.status.container_statuses:
                    current_restarts = pod.status.container_statuses[0].restart_count
                    last_restarts = pod_restart_history.get(pod_name, 0)
                    if current_restarts > last_restarts:
                        logger.warning(f"Detected pod restart for {pod_name}! Marking service as unhealthy.")
                        has_restarted = True
                    new_pod_restart_history[pod_name] = current_restarts

                if pod.status.phase == 'Running' and pod.status.container_statuses and all(cs.ready for cs in pod.status.container_statuses):
                    truly_ready_replicas += 1

            if has_restarted:
                health = 0.0
            else:
                health = (truly_ready_replicas / desired_replicas) if desired_replicas > 0 else 0.0
            
            statuses[service_name] = {"health_status": health, "cpu_usage": 0.0}

        pod_metrics = custom_objects_api.list_namespaced_custom_object(
            "metrics.k8s.io", "v1beta1", namespace, "pods"
        )
        service_metrics = {}
        for item in pod_metrics['items']:
            pod_name = item['metadata']['name']
            service_name = '-'.join(pod_name.split('-')[:-2])
            if service_name in statuses:
                cpu_usage = item['containers'][0]['usage']['cpu']
                if 'n' in cpu_usage:
                    cpu_usage = float(cpu_usage.rstrip('n')) / 1_000_000_000
                else:
                    cpu_usage = float(cpu_usage)
                if service_name not in service_metrics:
                    service_metrics[service_name] = []
                service_metrics[service_name].append(cpu_usage)
        
        for service, health_data in statuses.items():
            cpu = max(service_metrics.get(service, [0.0]))
            statuses[service]["cpu_usage"] = cpu
            if health_data["health_status"] < 1.0:
                 statuses[service]["cpu_usage"] = 1.0

    except client.ApiException as e:
        logger.error(f"Could not list deployments or pod metrics: {e}")
        return {}, pod_restart_history 
    return statuses, new_pod_restart_history


def run_experiment(mode, log_data, duration=120):
    """Runs a single experiment mode (baseline or graph-heal)."""
    print(f"\n--- Starting Kubernetes experiment: {mode} mode ---")

    k8s_adapter = KubernetesAdapter()
    service_graph = ServiceGraph()
    
    try:
        config.load_kube_config()
        core_v1_api = client.CoreV1Api()
    except config.ConfigException:
        logger.error("Could not load kubeconfig for CoreV1Api.")
        return

    pod_restart_history = {}

    print("Building service graph from static definition...")
    for service, dependencies in SERVICES.items():
        service_graph.add_service(service, dependencies=dependencies)
    print("Service graph built.")
    
    anomaly_detector = StatisticalAnomalyDetector()
    recovery_system = EnhancedRecoverySystem(service_graph, adapter=k8s_adapter)

    print("Cleaning up any pre-existing faults...")
    cleanup_fault(k8s_adapter.apps_v1, 'service-a-deployment', original_command=['/bin/sh', '-c', 'sleep 3600'])
    time.sleep(10)
    
    print(f"Injecting crash loop fault into service-a...")
    original_command = inject_crash_loop_fault(k8s_adapter.apps_v1, 'service-a-deployment')
    log_event(log_data, {
        "timestamp": time.time(),
        "event_type": "fault_injected",
        "service_id": "service-a",
        "mode": mode
    })
    
    start_time = time.time()
    
    while time.time() - start_time < duration:
        print(f"--- Monitoring tick ({mode} mode) ---")
        
        service_statuses, pod_restart_history = get_service_statuses(
            k8s_adapter.custom_objects, k8s_adapter.apps_v1, core_v1_api, pod_restart_history
        )
        
        log_event(log_data, {
            "timestamp": time.time(),
            "event_type": "metrics_tick",
            "service_metrics": service_statuses,
            "mode": mode
        })

        if mode == 'graph-heal' and service_statuses:
            anomalies = anomaly_detector.detect_anomalies(service_statuses)
            for anom in anomalies:
                log_event(log_data, {
                    "timestamp": time.time(),
                    "event_type": "anomaly_detected",
                    "anomaly": anom,
                    "mode": mode
                })
                
                plan = recovery_system.get_recovery_plan(anom['service_id'], fault_type='crash_loop')
                for action in plan:
                    log_event(log_data, {
                        "timestamp": time.time(),
                        "event_type": "recovery_action_planned",
                        "action": str(action),
                        "mode": mode
                    })
                    was_successful = recovery_system.execute_recovery_action(action)
                    log_event(log_data, {
                        "timestamp": time.time(),
                        "event_type": "recovery_action_executed",
                        "action": str(action),
                        "success": was_successful,
                        "mode": mode
                    })
        time.sleep(5)

    print(f"\n--- Experiment finished ({mode} mode). Cleaning up fault. ---")
    cleanup_fault(k8s_adapter.apps_v1, 'service-a-deployment', original_command)

def main():
    """Main function to run the experiments and save the log."""
    try:
        config.load_kube_config()
        logger.info("Loaded local kubeconfig.")
    except config.ConfigException:
        logger.error("Could not load kubeconfig. Is your cluster running and configured?")
        return

    log_data = []
    
    # Run baseline first
    run_experiment('baseline', log_data, duration=60)
    
    print("\n\n--- Waiting 30 seconds for cluster to stabilize between runs ---\n\n")
    time.sleep(30)
    
    # Then run with Graph-Heal
    run_experiment('graph-heal', log_data, duration=120)

    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)
    log_file = "results/k8s_experiment_log.json"
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=4)
        
    print(f"\nFull experiment log saved to {log_file}")


if __name__ == "__main__":
    main()
    
    
# import sys
# import os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# import time
# import json
# import logging
# from datetime import datetime
# from kubernetes import client, config
# from graph_heal.service_graph import ServiceGraph
# from graph_heal.anomaly_detection import StatisticalAnomalyDetector
# from graph_heal.recovery_system import EnhancedRecoverySystem
# from graph_heal.recovery.kubernetes_adapter import KubernetesAdapter

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# # Static definition of the service architecture. In a real-world scenario, this
# # might be discovered from an infrastructure-as-code definition or a service mesh.
# SERVICES = {
#     'service-a': ['service-b', 'service-c'],
#     'service-b': ['service-d'],
#     'service-c': ['service-d'],
#     'service-d': []
# }

# def inject_crash_loop_fault(api_instance, deployment_name, namespace="default"):
#     """Injects a fault that causes pods to crash loop by setting a bad command."""
#     try:
#         deployment = api_instance.read_namespaced_deployment(deployment_name, namespace)
#         # Store original command if it exists, to be restored later
#         original_command = deployment.spec.template.spec.containers[0].command
        
#         # Change the command to something that will exit, causing a crash loop
#         deployment.spec.template.spec.containers[0].command = ["/bin/sh", "-c", "echo 'Crashing...' && exit 1"]
#         api_instance.patch_namespaced_deployment(deployment_name, namespace, deployment)
        
#         logger.info(f"Successfully injected crash loop fault into {deployment_name}.")
#         # Return the original command so it can be restored
#         return original_command
#     except client.ApiException as e:
#         logger.error(f"Error injecting fault into {deployment_name}: {e}")
#         return None

# def cleanup_fault(api_instance, deployment_name, original_command=None, namespace="default"):
#     """Removes the fault condition by resetting the container command."""
#     try:
#         deployment = api_instance.read_namespaced_deployment(deployment_name, namespace)
#         # Reset the command to its original state
#         deployment.spec.template.spec.containers[0].command = original_command
#         api_instance.patch_namespaced_deployment(deployment_name, namespace, deployment)
#         logger.info(f"Successfully cleaned up fault from {deployment_name}.")
#         return True
#     except client.ApiException as e:
#         logger.error(f"Error cleaning up fault from {deployment_name}: {e}")
#         return False

# def log_event(log_data, event):
#     """Appends a timestamped event to the log data."""
#     event['timestamp_iso'] = datetime.fromtimestamp(event['timestamp']).isoformat()
#     log_data.append(event)


# def get_service_statuses(custom_objects_api, apps_v1_api, core_v1_api, pod_restart_history, namespace="default"):
#     """
#     Checks the status and metrics of all deployments by inspecting pod restart counts
#     and container readiness, making the check stateful and more robust.
#     """
#     statuses = {}
#     new_pod_restart_history = pod_restart_history.copy()
#     try:
#         deployments = apps_v1_api.list_namespaced_deployment(namespace)
#         for dep in deployments.items:
#             service_name = dep.metadata.name.replace('-deployment', '')
#             desired_replicas = dep.spec.replicas
            
#             selector = dep.spec.selector.match_labels
#             pod_list = core_v1_api.list_namespaced_pod(
#                 namespace, label_selector=','.join([f"{k}={v}" for k, v in selector.items()])
#             )
            
#             truly_ready_replicas = 0
#             has_restarted = False
#             for pod in pod_list.items:
#                 pod_name = pod.metadata.name
                
#                 # Statefully check for pod restarts.
#                 if pod.status.container_statuses:
#                     current_restarts = pod.status.container_statuses[0].restart_count
#                     last_restarts = pod_restart_history.get(pod_name, 0)
#                     if current_restarts > last_restarts:
#                         logger.warning(f"Detected pod restart for {pod_name}! Marking service as unhealthy.")
#                         has_restarted = True
#                     new_pod_restart_history[pod_name] = current_restarts

#                 # Also check current readiness.
#                 if pod.status.phase == 'Running' and pod.status.container_statuses and all(cs.ready for cs in pod.status.container_statuses):
#                     truly_ready_replicas += 1

#             # A restart is a definitive sign of failure.
#             if has_restarted:
#                 health = 0.0
#             else:
#                 health = (truly_ready_replicas / desired_replicas) if desired_replicas > 0 else 0.0
            
#             statuses[service_name] = {"health_status": health, "cpu_usage": 0.0}

#         # --- Metric collection logic remains the same ---
#         pod_metrics = custom_objects_api.list_namespaced_custom_object(
#             "metrics.k8s.io", "v1beta1", namespace, "pods"
#         )
#         service_metrics = {}
#         for item in pod_metrics['items']:
#             pod_name = item['metadata']['name']
#             service_name = '-'.join(pod_name.split('-')[:-2])
#             if service_name in statuses:
#                 cpu_usage = item['containers'][0]['usage']['cpu']
#                 if 'n' in cpu_usage:
#                     cpu_usage = float(cpu_usage.rstrip('n')) / 1_000_000_000
#                 else:
#                     cpu_usage = float(cpu_usage)
#                 if service_name not in service_metrics:
#                     service_metrics[service_name] = []
#                 service_metrics[service_name].append(cpu_usage)
        
#         for service, health_data in statuses.items():
#             cpu = max(service_metrics.get(service, [0.0]))
#             statuses[service]["cpu_usage"] = cpu
#             if health_data["health_status"] < 1.0:
#                  statuses[service]["cpu_usage"] = 1.0

#     except client.ApiException as e:
#         logger.error(f"Could not list deployments or pod metrics: {e}")
#         return {}, pod_restart_history 
#     return statuses, new_pod_restart_history


# def run_experiment(mode, log_data, duration=120):
#     """Runs a single experiment mode (baseline or graph-heal)."""
#     print(f"\n--- Starting Kubernetes experiment: {mode} mode ---")

#     k8s_adapter = KubernetesAdapter()
#     service_graph = ServiceGraph()
    
#     try:
#         config.load_kube_config()
#         core_v1_api = client.CoreV1Api()
#     except config.ConfigException:
#         logger.error("Could not load kubeconfig for CoreV1Api.")
#         return

#     # This dictionary will maintain the state of pod restarts between ticks.
#     pod_restart_history = {}

#     print("Building service graph from static definition...")
#     for service, dependencies in SERVICES.items():
#         service_graph.add_service(service, dependencies=dependencies)
#     print("Service graph built.")
    
#     anomaly_detector = StatisticalAnomalyDetector(list(service_graph.dependencies.keys()))
#     recovery_system = EnhancedRecoverySystem(service_graph, adapter=k8s_adapter)

#     print("Cleaning up any pre-existing faults...")
#     # Ensure cleanup uses a known good state command
#     cleanup_fault(k8s_adapter.apps_v1, 'service-a-deployment', original_command=['/bin/sh', '-c', 'sleep 3600'])
#     time.sleep(10)
    
#     print(f"Injecting crash loop fault into service-a...")
#     original_command = inject_crash_loop_fault(k8s_adapter.apps_v1, 'service-a-deployment')
#     log_event(log_data, {
#         "timestamp": time.time(),
#         "event_type": "fault_injected",
#         "service_id": "service-a",
#         "mode": mode
#     })
    
#     start_time = time.time()
    
#     while time.time() - start_time < duration:
#         print(f"--- Monitoring tick ({mode} mode) ---")
        
#         service_statuses, pod_restart_history = get_service_statuses(
#             k8s_adapter.custom_objects, k8s_adapter.apps_v1, core_v1_api, pod_restart_history
#         )
        
#         if mode == 'graph-heal' and service_statuses:
#             anomalies = anomaly_detector.detect_anomalies(service_statuses)
#             for anom in anomalies:
#                 log_event(log_data, {
#                     "timestamp": time.time(),
#                     "event_type": "anomaly_detected",
#                     "anomaly": anom,
#                     "mode": mode
#                 })
                
#                 plan = recovery_system.get_recovery_plan(anom['service_id'], fault_type='crash_loop')
#                 for action in plan:
#                     log_event(log_data, {
#                         "timestamp": time.time(),
#                         while time.time() - start_time < duration:
#                         print(f"--- Monitoring tick ({mode} mode) ---")
                            
#                         service_statuses, pod_restart_history = get_service_statuses(
#                             k8s_adapter.custom_objects, k8s_adapter.apps_v1, core_v1_api, pod_restart_history
#                         )
#                     was_successful = recovery_system.execute_recovery_action(action)
#                     log_event(log_data, {
#                         "timestamp": time.time(),
#                         "event_type": "recovery_action_executed",
#                         "action": str(action),
#                         "success": was_successful,
#                         "mode": mode
#                     })

#         time.sleep(5)

#     print(f"\n--- Experiment finished ({mode} mode). Cleaning up fault. ---")
#     cleanup_fault(k8s_adapter.apps_v1, 'service-a-deployment', original_command)


# # def get_service_statuses(custom_objects_api, apps_v1_api, core_v1_api, namespace="default"):
# #     """
# #     Checks the status and metrics of all deployments by inspecting the detailed status
# #     of each individual pod.
# #     """
# #     statuses = {}
# #     try:
# #         deployments = apps_v1_api.list_namespaced_deployment(namespace)
# #         for dep in deployments.items:
# #             service_name = dep.metadata.name.replace('-deployment', '')
# #             desired_replicas = dep.spec.replicas
            
# #             selector = dep.spec.selector.match_labels
# #             pod_list = core_v1_api.list_namespaced_pod(
# #                 namespace, label_selector=','.join([f"{k}={v}" for k, v in selector.items()])
# #             )
            
# #             truly_ready_replicas = 0
# #             for pod in pod_list.items:
# #                 if pod.status.phase == 'Running' and pod.status.container_statuses and all(cs.ready for cs in pod.status.container_statuses):
# #                     truly_ready_replicas += 1

# #             health = (truly_ready_replicas / desired_replicas) if desired_replicas > 0 else 0.0
            
# #             statuses[service_name] = {"health_status": health, "cpu_usage": 0.0}

# #         pod_metrics = custom_objects_api.list_namespaced_custom_object(
# #             "metrics.k8s.io", "v1beta1", namespace, "pods"
# #         )
        
# #         service_metrics = {}
# #         for item in pod_metrics['items']:
# #             pod_name = item['metadata']['name']
# #             service_name = '-'.join(pod_name.split('-')[:-2])
            
# #             if service_name in statuses:
# #                 cpu_usage = item['containers'][0]['usage']['cpu']
# #                 if 'n' in cpu_usage:
# #                     cpu_usage = float(cpu_usage.rstrip('n')) / 1_000_000_000
# #                 else:
# #                     cpu_usage = float(cpu_usage)
                
# #                 if service_name not in service_metrics:
# #                     service_metrics[service_name] = []
# #                 service_metrics[service_name].append(cpu_usage)
        
# #         for service, health_data in statuses.items():
# #             cpu = max(service_metrics.get(service, [0.0]))
# #             statuses[service]["cpu_usage"] = cpu
            
# #             if health_data["health_status"] < 1.0:
# #                  statuses[service]["cpu_usage"] = 1.0

# #     except client.ApiException as e:
# #         logger.error(f"Could not list deployments or pod metrics: {e}")
# #         return {}
# #     return statuses

# # def get_service_statuses(custom_objects_api, apps_v1_api, namespace="default"):
# #     """
# #     Checks the status and metrics of all deployments by inspecting individual pod readiness.
# #     """
# #     statuses = {}
# #     try:
# #         deployments = apps_v1_api.list_namespaced_deployment(namespace)
# #         for dep in deployments.items:
# #             service_name = dep.metadata.name.replace('-deployment', '')
# #             desired_replicas = dep.spec.replicas
# #             ready_replicas = dep.status.ready_replicas if dep.status.ready_replicas is not None else 0

# #             # Calculate health as the ratio of ready pods to desired pods.
# #             # This accurately reflects the state of crash-looping pods.
# #             health = (ready_replicas / desired_replicas) if desired_replicas > 0 else 0.0
            
# #             statuses[service_name] = {
# #                 "health_status": health,
# #                 "cpu_usage": 0.0  # Will be updated below
# #             }

# #         # Get metrics from the metrics API
# #         pod_metrics = custom_objects_api.list_namespaced_custom_object(
# #             "metrics.k8s.io", "v1beta1", namespace, "pods"
# #         )
        
# #         # Aggregate metrics by service
# #         service_metrics = {}
# #         for item in pod_metrics['items']:
# #             pod_name = item['metadata']['name']
# #             # This logic correctly handles pod names with random hashes
# #             service_name = '-'.join(pod_name.split('-')[:-2])
            
# #             if service_name in statuses:
# #                 cpu_usage = item['containers'][0]['usage']['cpu']
# #                 if 'n' in cpu_usage: # nanocores
# #                     cpu_usage = float(cpu_usage.rstrip('n')) / 1_000_000_000
# #                 else: # cores
# #                     cpu_usage = float(cpu_usage)
                
# #                 if service_name not in service_metrics:
# #                     service_metrics[service_name] = []
# #                 service_metrics[service_name].append(cpu_usage)
        
# #         # Combine health and metrics
# #         for service, health_data in statuses.items():
# #             # Use max CPU of all pods for the service
# #             cpu = max(service_metrics.get(service, [0.0]))
# #             statuses[service]["cpu_usage"] = cpu
            
# #             # If a service is unhealthy, simulate a high CPU to ensure detection
# #             if health_data["health_status"] < 1.0:
# #                  statuses[service]["cpu_usage"] = 1.0 # 1.0 represents 1 full core

# #     except client.ApiException as e:
# #         logger.error(f"Could not list deployments or pod metrics: {e}")
# #         return {}
# #     return statuses

# def run_experiment(mode, log_data, duration=120):
#     """Runs a single experiment mode (baseline or graph-heal)."""
#     print(f"\n--- Starting Kubernetes experiment: {mode} mode ---")

#     k8s_adapter = KubernetesAdapter()
#     service_graph = ServiceGraph()
    
#     try:
#         config.load_kube_config()
#         core_v1_api = client.CoreV1Api()
#     except config.ConfigException:
#         logger.error("Could not load kubeconfig for CoreV1Api.")
#         return

#     print("Building service graph from static definition...")
#     for service, dependencies in SERVICES.items():
#         service_graph.add_service(service, dependencies=dependencies)
#     print("Service graph built.")
    
#     anomaly_detector = StatisticalAnomalyDetector(list(service_graph.dependencies.keys()))
#     recovery_system = EnhancedRecoverySystem(service_graph, adapter=k8s_adapter)

#     print("Cleaning up any pre-existing faults...")
#     cleanup_fault(k8s_adapter.apps_v1, 'service-a-deployment', original_command=['/bin/sh', '-c', 'sleep 3600'])
#     time.sleep(10)
    
#     print(f"Injecting crash loop fault into service-a...")
#     original_command = inject_crash_loop_fault(k8s_adapter.apps_v1, 'service-a-deployment')
#     log_event(log_data, {
#         "timestamp": time.time(),
#         "event_type": "fault_injected",
#         "service_id": "service-a",
#         "mode": mode
#     })
    
#     start_time = time.time()
    
#     while time.time() - start_time < duration:
#         print(f"--- Monitoring tick ({mode} mode) ---")
        
#         service_statuses = get_service_statuses(
#             k8s_adapter.custom_objects, k8s_adapter.apps_v1, core_v1_api
#         )
        
#         if mode == 'graph-heal' and service_statuses:
#             anomalies = anomaly_detector.detect_anomalies(service_statuses)
#             for anom in anomalies:
#                 log_event(log_data, {
#                     "timestamp": time.time(),
#                     "event_type": "anomaly_detected",
#                     "anomaly": anom,
#                     "mode": mode
#                 })
                
#                 plan = recovery_system.get_recovery_plan(anom['service_id'], fault_type='crash_loop')
#                 for action in plan:
#                     log_event(log_data, {
#                         "timestamp": time.time(),
#                         "event_type": "recovery_action_planned",
#                         "action": str(action),
#                         "mode": mode
#                     })
#                     was_successful = recovery_system.execute_recovery_action(action)
#                     log_event(log_data, {
#                         "timestamp": time.time(),
#                         "event_type": "recovery_action_executed",
#                         "action": str(action),
#                         "success": was_successful,
#                         "mode": mode
#                     })

#         time.sleep(5)

#     print(f"\n--- Experiment finished ({mode} mode). Cleaning up fault. ---")
#     cleanup_fault(k8s_adapter.apps_v1, 'service-a-deployment', original_command)
    
# # def run_experiment(mode, log_data, duration=120):
# #     """Runs a single experiment mode (baseline or graph-heal)."""
# #     print(f"\n--- Starting Kubernetes experiment: {mode} mode ---")

# #     k8s_adapter = KubernetesAdapter()
# #     service_graph = ServiceGraph()
    
# #     # Build the graph from the static definition
# #     print("Building service graph from static definition...")
# #     for service, dependencies in SERVICES.items():
# #         service_graph.add_service(service, dependencies=dependencies)
# #     print("Service graph built.")
    
# #     anomaly_detector = StatisticalAnomalyDetector(list(service_graph.dependencies.keys()))
# #     recovery_system = EnhancedRecoverySystem(service_graph, adapter=k8s_adapter)

# #     print("Cleaning up any pre-existing faults...")
# #     cleanup_fault(k8s_adapter.apps_v1, 'service-a-deployment')
# #     time.sleep(10)
    
# #     print(f"Injecting crash loop fault into service-a...")
# #     original_command = inject_crash_loop_fault(k8s_adapter.apps_v1, 'service-a-deployment')
# #     log_event(log_data, {
# #         "timestamp": time.time(),
# #         "event_type": "fault_injected",
# #         "service_id": "service-a",
# #         "mode": mode
# #     })
    
# #     start_time = time.time()
# #     while time.time() - start_time < duration:
# #         print(f"--- Monitoring tick ({mode} mode) ---")
        
# #         service_statuses = get_service_statuses(k8s_adapter.custom_objects, k8s_adapter.apps_v1)

# #         log_event(log_data, {
# #             "timestamp": time.time(),
# #             "event_type": "metrics_tick",
# #             "service_metrics": service_statuses,
# #             "mode": mode
# #         })

# #         if mode == 'graph-heal' and service_statuses:
# #             anomalies = anomaly_detector.detect_anomalies(service_statuses)
# #             for anom in anomalies:
# #                 log_event(log_data, {
# #                     "timestamp": time.time(),
# #                     "event_type": "anomaly_detected",
# #                     "anomaly": anom,
# #                     "mode": mode
# #                 })
# #                 print(f"!!! Anomaly detected: {anom['service_id']} ({anom['type']})")
                
# #                 plan = recovery_system.get_recovery_plan(anom['service_id'], anom['type'])
# #                 for action in plan:
# #                     action_dict = {
# #                         "action_type": action.action_type.value,
# #                         "target_service": action.target_service,
# #                         "parameters": action.parameters,
# #                     }
# #                     log_event(log_data, {
# #                         "timestamp": time.time(),
# #                         "event_type": "recovery_action_triggered",
# #                         "action": action_dict,
# #                         "mode": mode
# #                     })
# #                     print(f">>> Executing recovery action: {action.action_type.value} on {action.target_service}")
# #                     recovery_system.execute_recovery_action(action)

# #         time.sleep(5)

# #     print(f"\n--- Experiment finished ({mode} mode). Cleaning up fault. ---")
# #     cleanup_fault(k8s_adapter.apps_v1, 'service-a-deployment', original_command)

# def main():
#     """Main function to run the experiments and save the log."""
#     try:
#         config.load_kube_config()
#         logger.info("Loaded local kubeconfig.")
#     except config.ConfigException:
#         logger.error("Could not load kubeconfig. Is your cluster running and configured?")
#         return

#     log_data = []
    
#     run_experiment('baseline', log_data, duration=60)
    
#     print("\n\n--- Waiting 30 seconds for cluster to stabilize between runs ---\n\n")
#     time.sleep(30)
    
#     run_experiment('graph-heal', log_data, duration=120)

#     log_file = "results/k8s_experiment_log.json"
#     with open(log_file, 'w') as f:
#         json.dump(log_data, f, indent=4)
        
#     print(f"\nFull experiment log saved to {log_file}")

# if __name__ == "__main__":
#     main() 