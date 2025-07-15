import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import logging
import yaml
from datetime import datetime
from kubernetes import client, config
from kubernetes.utils import create_from_yaml, FailToCreateError
from graph_heal.recovery.kubernetes_adapter import KubernetesAdapter
from graph_heal.recovery_system import EnhancedRecoverySystem, RecoveryActionType
from graph_heal.service_graph import ServiceGraph

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
CRD_YAML = "k8s/application-health-crd.yaml"
CR_YAML = "k8s/service-a-health-cr.yaml"
SERVICE_A_YAML = "k8s/service-a-deployment.yaml"
TARGET_SERVICE = "service-a"
NAMESPACE = "default"

def setup_cluster(custom_objects_api, apps_v1_api):
    """Applies all necessary Kubernetes manifests."""
    print("--- Setting up cluster resources ---")
    
    # We need a generic client for create_from_yaml
    k8s_client = client.ApiClient()

    # Create CRD
    try:
        print(f"Applying CRD from {CRD_YAML}...")
        create_from_yaml(k8s_client, CRD_YAML, verbose=True)
        # Wait for CRD to be established
        time.sleep(5)
    except FailToCreateError as e:
        if "already exists" in str(e):
            logger.warning("CRD already exists, continuing.")
        else: raise
    except Exception as e:
        logger.error(f"Unexpected error applying CRD: {e}")
        raise

    # Create Service-A Deployment and Service
    try:
        print(f"Applying Service A manifest from {SERVICE_A_YAML}...")
        create_from_yaml(k8s_client, SERVICE_A_YAML, verbose=True)
    except FailToCreateError as e:
        if "already exists" in str(e):
            logger.warning("Service A already exists, continuing.")
        else: raise
    
    # Create ApplicationHealth Custom Resource using the dedicated API
    try:
        print(f"Applying Custom Resource from {CR_YAML}...")
        with open(CR_YAML) as f:
            doc = yaml.safe_load(f)
            custom_objects_api.create_namespaced_custom_object(
                group="graphheal.io",
                version="v1alpha1",
                namespace=NAMESPACE,
                plural="applicationhealths",
                body=doc,
            )
    except client.ApiException as e:
        if e.reason == "Conflict":
             logger.warning("Custom Resource already exists, continuing.")
        else: 
            logger.error(f"Failed to create CR: {e}")
            raise
        
    print("Waiting 15s for service-a to stabilize...")
    time.sleep(15)

def inject_scale_down_fault(apps_v1_api):
    """Injects a fault by scaling the deployment down to 0 replicas."""
    print("--- Injecting fault by scaling down service-a ---")
    deployment_name = f"{TARGET_SERVICE}-deployment"
    try:
        apps_v1_api.patch_namespaced_deployment_scale(
            deployment_name, NAMESPACE, {"spec": {"replicas": 0}}
        )
        logger.info(f"Successfully scaled {deployment_name} to 0 replicas.")
        return True
    except client.ApiException as e:
        logger.error(f"Error injecting scale down fault: {e}")
        return False

def check_service_health(apps_v1_api):
    """Checks if the target service has any available replicas."""
    deployment_name = f"{TARGET_SERVICE}-deployment"
    try:
        dep = apps_v1_api.read_namespaced_deployment_status(deployment_name, NAMESPACE)
        if dep.status.available_replicas is None or dep.status.available_replicas == 0:
            return "Unhealthy"
    except client.ApiException as e:
        logger.error(f"Could not read deployment status for {deployment_name}: {e}")
        return "Unhealthy" # Assume unhealthy if we can't read it
    return "Healthy"

def cleanup_cluster(k8s_client, apps_v1_api, core_v1_api, custom_objects_api):
    """Deletes all resources created by the experiment."""
    print("--- Cleaning up cluster resources ---")
    
    # Delete Deployment and Service
    with open(SERVICE_A_YAML) as f:
        docs = yaml.safe_load_all(f)
        for doc in docs:
            if doc['kind'] == "Deployment":
                apps_v1_api.delete_namespaced_deployment(doc['metadata']['name'], NAMESPACE)
            elif doc['kind'] == "Service":
                core_v1_api.delete_namespaced_service(doc['metadata']['name'], NAMESPACE)
    
    # Delete Custom Resource
    with open(CR_YAML) as f:
        doc = yaml.safe_load(f)
        custom_objects_api.delete_namespaced_custom_object(
            "graphheal.io", "v1alpha1", NAMESPACE, "applicationhealths", doc['metadata']['name']
        )
    
    # CRD deletion should be handled manually if needed
    print("Cleanup complete.")

def main():
    """Runs the full CRD experiment."""
    try:
        config.load_kube_config()
        apps_v1 = client.AppsV1Api()
        core_v1 = client.CoreV1Api()
        custom_objects = client.CustomObjectsApi()
        logger.info("Kubernetes client configured.")
    except config.ConfigException:
        logger.error("Could not load kubeconfig.")
        return

    try:
        setup_cluster(custom_objects, apps_v1)
        
        inject_scale_down_fault(apps_v1)

        print("Waiting 10s for fault to take effect...")
        time.sleep(10)

        # --- Detection and Recovery ---
        print("--- Simulating Detection and Recovery ---")
        health_status = check_service_health(apps_v1)
        print(f"Detected service-a health status: {health_status}")

        if health_status == "Unhealthy":
            print("Service is unhealthy. Generating recovery plan...")
            
            # Initialize Graph-Heal components
            k8s_adapter = KubernetesAdapter()
            service_graph = ServiceGraph()
            service_graph.add_service(TARGET_SERVICE)
            recovery_system = EnhancedRecoverySystem(service_graph, adapter=k8s_adapter)

            # Manually create the recovery action for this experiment
            action = recovery_system.create_recovery_action(
                TARGET_SERVICE,
                RecoveryActionType.UPDATE_CR_STATUS,
                parameters={
                    "status_payload": { "health": "Unhealthy" }
                }
            )
            
            print(f">>> Executing recovery action: {action.action_type.value} on {action.target_service}")
            recovery_system.execute_recovery_action(action)

        # Verify the CR status was updated
        print("\n--- Verifying CR Status ---")
        time.sleep(2) # Give a moment for the patch to apply
        try:
            cr = custom_objects.get_namespaced_custom_object_status(
                "graphheal.io", "v1alpha1", NAMESPACE, "applicationhealths", f"{TARGET_SERVICE}-health"
            )
            final_status = cr.get('status', {})
            print(f"Final status of ApplicationHealth CR: {final_status}")
            if final_status.get('health') == 'Unhealthy':
                print("SUCCESS: CR status was correctly updated.")
            else:
                print("FAILURE: CR status was not updated as expected.")
        except client.ApiException as e:
            logger.error(f"Could not verify CR status: {e}")

    finally:
        cleanup_cluster(None, apps_v1, core_v1, custom_objects)

if __name__ == "__main__":
    main() 