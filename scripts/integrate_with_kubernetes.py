#!/usr/bin/env python3
"""
Kubernetes Integration for Graph-Heal
Provides custom resource definitions and operators for Kubernetes environments.
"""

import time
import json
import logging
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class KubernetesIntegration:
    """Integrates Graph-Heal with Kubernetes."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.namespace = config.get('namespace', 'default')
        self.crd_dir = Path(config.get('crd_dir', 'k8s/crds'))
        self.crd_dir.mkdir(parents=True, exist_ok=True)
        self.operator_dir = Path(config.get('operator_dir', 'k8s/operators'))
        self.operator_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_custom_resource_definitions(self):
        """Generate Custom Resource Definitions for Graph-Heal."""
        
        # Fault Detection Policy CRD
        fault_detection_policy_crd = {
            'apiVersion': 'apiextensions.k8s.io/v1',
            'kind': 'CustomResourceDefinition',
            'metadata': {
                'name': 'faultdetectionpolicies.graphheal.io'
            },
            'spec': {
                'group': 'graphheal.io',
                'names': {
                    'kind': 'FaultDetectionPolicy',
                    'listKind': 'FaultDetectionPolicyList',
                    'plural': 'faultdetectionpolicies',
                    'singular': 'faultdetectionpolicy',
                    'shortNames': ['fdp']
                },
                'scope': 'Namespaced',
                'versions': [
                    {
                        'name': 'v1alpha1',
                        'served': True,
                        'storage': True,
                        'schema': {
                            'openAPIV3Schema': {
                                'type': 'object',
                                'properties': {
                                    'spec': {
                                        'type': 'object',
                                        'properties': {
                                            'targetServices': {
                                                'type': 'array',
                                                'items': {'type': 'string'}
                                            },
                                            'detectionThresholds': {
                                                'type': 'object',
                                                'properties': {
                                                    'cpuThreshold': {'type': 'number'},
                                                    'memoryThreshold': {'type': 'number'},
                                                    'responseTimeThreshold': {'type': 'number'}
                                                }
                                            },
                                            'recoveryActions': {
                                                'type': 'array',
                                                'items': {
                                                    'type': 'object',
                                                    'properties': {
                                                        'action': {'type': 'string'},
                                                        'priority': {'type': 'integer'}
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                ]
            }
        }
        
        # Service Health CRD
        service_health_crd = {
            'apiVersion': 'apiextensions.k8s.io/v1',
            'kind': 'CustomResourceDefinition',
            'metadata': {
                'name': 'servicehealths.graphheal.io'
            },
            'spec': {
                'group': 'graphheal.io',
                'names': {
                    'kind': 'ServiceHealth',
                    'listKind': 'ServiceHealthList',
                    'plural': 'servicehealths',
                    'singular': 'servicehealth',
                    'shortNames': ['sh']
                },
                'scope': 'Namespaced',
                'versions': [
                    {
                        'name': 'v1alpha1',
                        'served': True,
                        'storage': True,
                        'schema': {
                            'openAPIV3Schema': {
                                'type': 'object',
                                'properties': {
                                    'spec': {
                                        'type': 'object',
                                        'properties': {
                                            'serviceName': {'type': 'string'},
                                            'healthCheckInterval': {'type': 'integer'},
                                            'failureThreshold': {'type': 'integer'}
                                        }
                                    },
                                    'status': {
                                        'type': 'object',
                                        'properties': {
                                            'healthScore': {'type': 'number'},
                                            'lastCheckTime': {'type': 'string'},
                                            'faults': {
                                                'type': 'array',
                                                'items': {
                                                    'type': 'object',
                                                    'properties': {
                                                        'type': {'type': 'string'},
                                                        'detectedAt': {'type': 'string'},
                                                        'resolvedAt': {'type': 'string'}
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                ]
            }
        }
        
        # Save CRDs
        crd_files = [
            ('fault_detection_policy_crd.yaml', fault_detection_policy_crd),
            ('service_health_crd.yaml', service_health_crd)
        ]
        
        for filename, crd in crd_files:
            filepath = self.crd_dir / filename
            with open(filepath, 'w') as f:
                yaml.dump(crd, f, default_flow_style=False)
            logger.info(f"Generated CRD: {filepath}")
            
    def generate_kubernetes_manifests(self):
        """Generate Kubernetes manifests for Graph-Heal deployment."""
        
        # Graph-Heal Operator Deployment
        operator_deployment = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': 'graphheal-operator',
                'namespace': self.namespace,
                'labels': {
                    'app': 'graphheal-operator'
                }
            },
            'spec': {
                'replicas': 1,
                'selector': {
                    'matchLabels': {
                        'app': 'graphheal-operator'
                    }
                },
                'template': {
                    'metadata': {
                        'labels': {
                            'app': 'graphheal-operator'
                        }
                    },
                    'spec': {
                        'containers': [
                            {
                                'name': 'graphheal-operator',
                                'image': 'graphheal/operator:latest',
                                'ports': [
                                    {'containerPort': 8080}
                                ],
                                'env': [
                                    {'name': 'WATCH_NAMESPACE', 'value': self.namespace},
                                    {'name': 'POD_NAME', 'valueFrom': {'fieldRef': {'fieldPath': 'metadata.name'}}},
                                    {'name': 'OPERATOR_NAME', 'value': 'graphheal-operator'}
                                ],
                                'resources': {
                                    'requests': {
                                        'memory': '64Mi',
                                        'cpu': '250m'
                                    },
                                    'limits': {
                                        'memory': '128Mi',
                                        'cpu': '500m'
                                    }
                                }
                            }
                        ],
                        'serviceAccountName': 'graphheal-operator'
                    }
                }
            }
        }
        
        # Service Account
        service_account = {
            'apiVersion': 'v1',
            'kind': 'ServiceAccount',
            'metadata': {
                'name': 'graphheal-operator',
                'namespace': self.namespace
            }
        }
        
        # Cluster Role
        cluster_role = {
            'apiVersion': 'rbac.authorization.k8s.io/v1',
            'kind': 'ClusterRole',
            'metadata': {
                'name': 'graphheal-operator'
            },
            'rules': [
                {
                    'apiGroups': ['graphheal.io'],
                    'resources': ['faultdetectionpolicies', 'servicehealths'],
                    'verbs': ['get', 'list', 'watch', 'create', 'update', 'patch', 'delete']
                },
                {
                    'apiGroups': ['apps'],
                    'resources': ['deployments', 'statefulsets'],
                    'verbs': ['get', 'list', 'watch', 'update', 'patch']
                },
                {
                    'apiGroups': [''],
                    'resources': ['pods', 'services'],
                    'verbs': ['get', 'list', 'watch']
                }
            ]
        }
        
        # Cluster Role Binding
        cluster_role_binding = {
            'apiVersion': 'rbac.authorization.k8s.io/v1',
            'kind': 'ClusterRoleBinding',
            'metadata': {
                'name': 'graphheal-operator'
            },
            'roleRef': {
                'apiGroup': 'rbac.authorization.k8s.io',
                'kind': 'ClusterRole',
                'name': 'graphheal-operator'
            },
            'subjects': [
                {
                    'kind': 'ServiceAccount',
                    'name': 'graphheal-operator',
                    'namespace': self.namespace
                }
            ]
        }
        
        # Service
        service = {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {
                'name': 'graphheal-operator',
                'namespace': self.namespace
            },
            'spec': {
                'selector': {
                    'app': 'graphheal-operator'
                },
                'ports': [
                    {
                        'protocol': 'TCP',
                        'port': 8080,
                        'targetPort': 8080
                    }
                ]
            }
        }
        
        # Save manifests
        manifest_files = [
            ('operator-deployment.yaml', operator_deployment),
            ('service-account.yaml', service_account),
            ('cluster-role.yaml', cluster_role),
            ('cluster-role-binding.yaml', cluster_role_binding),
            ('service.yaml', service)
        ]
        
        for filename, manifest in manifest_files:
            filepath = self.operator_dir / filename
            with open(filepath, 'w') as f:
                yaml.dump(manifest, f, default_flow_style=False)
            logger.info(f"Generated manifest: {filepath}")
            
    def generate_example_resources(self):
        """Generate example custom resources."""
        
        # Example Fault Detection Policy
        example_policy = {
            'apiVersion': 'graphheal.io/v1alpha1',
            'kind': 'FaultDetectionPolicy',
            'metadata': {
                'name': 'example-fault-policy',
                'namespace': self.namespace
            },
            'spec': {
                'targetServices': ['service-a', 'service-b', 'service-c'],
                'detectionThresholds': {
                    'cpuThreshold': 80.0,
                    'memoryThreshold': 85.0,
                    'responseTimeThreshold': 1.0
                },
                'recoveryActions': [
                    {
                        'action': 'restart',
                        'priority': 1
                    },
                    {
                        'action': 'scale',
                        'priority': 2
                    }
                ]
            }
        }
        
        # Example Service Health
        example_health = {
            'apiVersion': 'graphheal.io/v1alpha1',
            'kind': 'ServiceHealth',
            'metadata': {
                'name': 'service-a-health',
                'namespace': self.namespace
            },
            'spec': {
                'serviceName': 'service-a',
                'healthCheckInterval': 30,
                'failureThreshold': 3
            },
            'status': {
                'healthScore': 1.0,
                'lastCheckTime': datetime.now().isoformat(),
                'faults': []
            }
        }
        
        # Save examples
        example_dir = Path('k8s/examples')
        example_dir.mkdir(parents=True, exist_ok=True)
        
        example_files = [
            ('example-fault-policy.yaml', example_policy),
            ('example-service-health.yaml', example_health)
        ]
        
        for filename, resource in example_files:
            filepath = example_dir / filename
            with open(filepath, 'w') as f:
                yaml.dump(resource, f, default_flow_style=False)
            logger.info(f"Generated example: {filepath}")
            
    def deploy_to_kubernetes(self):
        """Deploy Graph-Heal to Kubernetes cluster."""
        logger.info("Deploying Graph-Heal to Kubernetes")
        
        try:
            # Apply CRDs
            logger.info("Applying Custom Resource Definitions...")
            for crd_file in self.crd_dir.glob('*.yaml'):
                subprocess.run(['kubectl', 'apply', '-f', str(crd_file)], check=True)
                logger.info(f"Applied CRD: {crd_file.name}")
                
            # Apply operator manifests
            logger.info("Applying operator manifests...")
            for manifest_file in self.operator_dir.glob('*.yaml'):
                subprocess.run(['kubectl', 'apply', '-f', str(manifest_file)], check=True)
                logger.info(f"Applied manifest: {manifest_file.name}")
                
            # Wait for operator to be ready
            logger.info("Waiting for operator to be ready...")
            subprocess.run([
                'kubectl', 'wait', '--for=condition=available', 
                '--timeout=300s', 'deployment/graphheal-operator', 
                '-n', self.namespace
            ], check=True)
            
            logger.info("Graph-Heal successfully deployed to Kubernetes")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to deploy to Kubernetes: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during deployment: {e}")
            
    def check_kubernetes_status(self):
        """Check the status of Graph-Heal in Kubernetes."""
        logger.info("Checking Graph-Heal status in Kubernetes")
        
        try:
            # Check operator deployment
            result = subprocess.run([
                'kubectl', 'get', 'deployment', 'graphheal-operator', 
                '-n', self.namespace, '-o', 'json'
            ], capture_output=True, text=True, check=True)
            
            deployment_info = json.loads(result.stdout)
            replicas = deployment_info['status']['replicas']
            available = deployment_info['status']['availableReplicas']
            
            logger.info(f"Operator deployment: {available}/{replicas} replicas available")
            
            # Check custom resources
            result = subprocess.run([
                'kubectl', 'get', 'faultdetectionpolicies', '-n', self.namespace
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Fault Detection Policies:")
                logger.info(result.stdout)
            else:
                logger.info("No Fault Detection Policies found")
                
            # Check service health
            result = subprocess.run([
                'kubectl', 'get', 'servicehealths', '-n', self.namespace
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Service Health Resources:")
                logger.info(result.stdout)
            else:
                logger.info("No Service Health resources found")
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to check Kubernetes status: {e}")
        except Exception as e:
            logger.error(f"Unexpected error checking status: {e}")


def main():
    """Main function to run Kubernetes integration."""
    # Configuration
    config = {
        'namespace': 'default',
        'crd_dir': 'k8s/crds',
        'operator_dir': 'k8s/operators'
    }
    
    # Create integration
    integration = KubernetesIntegration(config)
    
    # Generate all Kubernetes resources
    logger.info("Generating Kubernetes integration resources")
    integration.generate_custom_resource_definitions()
    integration.generate_kubernetes_manifests()
    integration.generate_example_resources()
    
    # Check if kubectl is available
    try:
        subprocess.run(['kubectl', 'version', '--client'], check=True, capture_output=True)
        logger.info("kubectl is available")
        
        # Deploy to Kubernetes (optional)
        deploy = input("Deploy to Kubernetes cluster? (y/n): ").lower().strip()
        if deploy == 'y':
            integration.deploy_to_kubernetes()
            integration.check_kubernetes_status()
        else:
            logger.info("Skipping deployment. Manifests generated in k8s/ directory")
            
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("kubectl not available. Skipping deployment.")
        logger.info("Kubernetes manifests generated in k8s/ directory")


if __name__ == "__main__":
    main() 