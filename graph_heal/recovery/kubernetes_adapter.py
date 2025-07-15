import logging
from datetime import datetime
from .base import RecoverySystemAdapter
from kubernetes import client, config
from kubernetes.client.rest import ApiException

logger = logging.getLogger(__name__)

class KubernetesAdapter(RecoverySystemAdapter):
    """
    Recovery adapter for interacting with a Kubernetes cluster.
    Implements the recovery actions using the Kubernetes Python client.
    """
    def __init__(self):
        try:
            config.load_incluster_config()
            logger.info("Loaded in-cluster Kubernetes config.")
        except config.ConfigException:
            try:
                config.load_kube_config()
                logger.info("Loaded local kubeconfig.")
            except config.ConfigException:
                logger.error("Could not configure Kubernetes client. No in-cluster or local config found.")
                raise

        self.apps_v1 = client.AppsV1Api()
        self.networking_v1 = client.NetworkingV1Api()
        self.custom_objects = client.CustomObjectsApi()
        self.core_v1 = client.CoreV1Api()

    def _get_deployment_name(self, service_name: str) -> str:
        """Constructs the deployment name from the service name."""
        return f"{service_name}-deployment"

    def restart_service(self, service_name: str, namespace: str = "default", **kwargs):
        """
        Performs a rolling restart of a Kubernetes resource.
        It intelligently handles both Deployments and StatefulSets.
        """
        now = datetime.utcnow().isoformat()
        patch_body = {
            "spec": {
                "template": {
                    "metadata": {
                        "annotations": {
                            "graph-heal.io/restartedAt": now
                        }
                    }
                }
            }
        }

        # Try to restart as a Deployment first
        deployment_name = self._get_deployment_name(service_name)
        try:
            self.apps_v1.read_namespaced_deployment(deployment_name, namespace)
            self.apps_v1.patch_namespaced_deployment(
                name=deployment_name, namespace=namespace, body=patch_body
            )
            logger.info(f"Successfully triggered restart for Deployment '{deployment_name}'.")
            return True
        except ApiException as e:
            if e.status != 404:
                logger.error(f"Error restarting Deployment {deployment_name}: {e}")
                # Don't return yet, fall through to try a StatefulSet
            else:
                logger.info(f"Deployment '{deployment_name}' not found, attempting to restart as a StatefulSet.")

        # If Deployment is not found, try to restart as a StatefulSet
        # We assume the StatefulSet name matches the service_name directly
        try:
            self.apps_v1.read_namespaced_stateful_set(service_name, namespace)
            self.apps_v1.patch_namespaced_stateful_set(
                name=service_name, namespace=namespace, body=patch_body
            )
            logger.info(f"Successfully triggered restart for StatefulSet '{service_name}'.")
            return True
        except ApiException as e:
            if e.status == 404:
                logger.error(f"Could not find a Deployment or StatefulSet for service '{service_name}'.")
            else:
                logger.error(f"Error restarting StatefulSet {service_name}: {e}")
            return False

    def update_custom_resource_status(self, service_name: str, namespace: str = "default", **kwargs):
        """
        Updates the status of the ApplicationHealth custom resource for a service.
        """
        status_payload = kwargs.get("status_payload")
        if not status_payload:
            logger.error("Update CR status action requires 'status_payload' keyword argument.")
            return False
            
        cr_name = f"{service_name}-health"
        group = "graphheal.io"
        version = "v1alpha1"
        plural = "applicationhealths"

        try:
            # Get the current CR to patch it
            self.custom_objects.get_namespaced_custom_object(
                group, version, namespace, plural, cr_name
            )

            # Add a timestamp to the payload
            status_payload['lastUpdateTime'] = datetime.utcnow().isoformat() + "Z"

            patch_body = {
                "status": status_payload
            }

            self.custom_objects.patch_namespaced_custom_object_status(
                group=group,
                version=version,
                namespace=namespace,
                plural=plural,
                name=cr_name,
                body=patch_body,
            )
            logger.info(f"Successfully patched status of CR '{cr_name}' to: {status_payload}")
            return True
        except ApiException as e:
            if e.status == 404:
                logger.error(f"Custom Resource '{cr_name}' not found.")
            else:
                logger.error(f"Failed to update status for CR '{cr_name}': {e}")
            return False

    def isolate_service(self, service_name: str, namespace: str = "default", **kwargs):
        """
        Isolates a service by applying a 'deny-all' NetworkPolicy.
        """
        policy_name = f"graph-heal-isolate-{service_name}"
        deployment_name = self._get_deployment_name(service_name) # Needed for label selector
        
        # Get the labels from the deployment to ensure the NetworkPolicy targets the correct pods.
        try:
            deployment = self.apps_v1.read_namespaced_deployment(deployment_name, namespace)
            pod_labels = deployment.spec.template.metadata.labels
        except ApiException as e:
            logger.error(f"Could not get labels for deployment {deployment_name}: {e}")
            return False

        body = client.V1NetworkPolicy(
            api_version="networking.k8s.io/v1",
            kind="NetworkPolicy",
            metadata=client.V1ObjectMeta(name=policy_name, namespace=namespace),
            spec=client.V1NetworkPolicySpec(
                pod_selector=client.V1LabelSelector(
                    match_labels=pod_labels
                ),
                policy_types=["Ingress"],
                ingress=[] # Deny all ingress traffic
            )
        )
        try:
            self.networking_v1.create_namespaced_network_policy(
                namespace=namespace, body=body
            )
            logger.info(f"Successfully applied isolation NetworkPolicy '{policy_name}' to service '{service_name}'.")
            return True
        except ApiException as e:
            if e.status == 409: # Already exists
                logger.warning(f"Isolation policy for {service_name} already exists.")
                return True
            logger.error(f"Failed to isolate service {service_name}: {e}")
            return False

    def scale_service(self, service_name: str, namespace: str = "default", **kwargs):
        """
        Scales a Kubernetes Deployment to a specific number of replicas.
        """
        replicas = kwargs.get("replicas")
        if replicas is None:
            logger.error("Scale action requires 'replicas' keyword argument.")
            return False

        deployment_name = self._get_deployment_name(service_name)
        scale_body = {
            "spec": {
                "replicas": int(replicas)
            }
        }
        try:
            self.apps_v1.patch_namespaced_deployment_scale(
                name=deployment_name, namespace=namespace, body=scale_body
            )
            logger.info(f"Successfully scaled deployment '{deployment_name}' to {replicas} replicas.")
            return True
        except ApiException as e:
            logger.error(f"Failed to scale service {service_name}: {e}")
            return False

    def degrade_service(self, service_name: str, namespace: str = "default", **kwargs):
        """
        Placeholder for degrading a service. This action is not implemented for Kubernetes.
        """
        logger.warning(f"Degrade action is not implemented for Kubernetes. Service '{service_name}' was not affected.")
        return False
 