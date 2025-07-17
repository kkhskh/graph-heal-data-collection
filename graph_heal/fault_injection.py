import threading
import time
import random
import logging
import docker
import requests
from typing import Dict, List, Any, Optional, Callable
import json
import os
import uuid
import subprocess
import platform
from graph_heal.utils import get_docker_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('fault_injection')

class FaultInjector:
    """Injects faults into services for testing."""
    
    def __init__(self, services_config=None):
        """
        Initialize the fault injector.
        Args:
            services_config (dict): A dictionary containing service configurations,
                                    including their names and ports.
        """
        self.active_faults = {}
        self.is_macos = platform.system() == "Darwin"
        self.services_config = services_config or {}

        # Correct service names to match docker-compose.yml
        self.service_names = ["service-a", "service-b", "service-c", "service-d"]

        # Create data directory if it doesn't exist
        self.data_dir = "data/faults"
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Create pfctl rules directory on macOS
        if self.is_macos:
            self.pfctl_dir = "data/pfctl"
            os.makedirs(self.pfctl_dir, exist_ok=True)
            
    def _get_container_name(self, service_name: str) -> Optional[str]:
        """Find the full container name for a given service."""
        # This function is now simplified as we assume fixed names
        if service_name in self.service_names:
            return service_name
        else:
            # Fallback for old names if needed, but we should standardize
            service_name_hyphenated = service_name.replace('_', '-')
            if service_name_hyphenated in self.service_names:
                return service_name_hyphenated

        logger.error(f"Could not find a matching container for service: {service_name}")
        return None


    def inject_fault(self, fault_type: str, target: str, params: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Inject a fault into a service by executing a command inside its container.
        """
        fault_id = str(uuid.uuid4())
        params = params or {}
        
        if fault_type not in ["latency", "crash", "cpu_stress", "memory_leak"]:
            raise ValueError(f"Invalid fault type: {fault_type}")
        
        # We don't need services_config if we have the fixed list
        # if not self.services_config:
        #     raise ValueError("Service configuration not provided to FaultInjector.")
            
        # if target not in self.services_config:
        #     raise ValueError(f"Unknown service in config: {target}")

        container_name = self._get_container_name(target)
        if not container_name:
            # Error is logged in the helper function
            return None

        try:
            client = get_docker_client()
            container = client.containers.get(container_name)

            if fault_type == "cpu_stress":
                duration = params.get("duration", 30)
                cmd = f"stress-ng --cpu 1 --cpu-load 80 --timeout {duration}s"
                logger.info(f"Executing command in {container_name}: {cmd}")
                container.exec_run(cmd, detach=True)
            else:
                logger.warning(f"Fault type '{fault_type}' is not implemented for container-based injection.")
                return None


            # Record the fault
            fault = {
                "id": fault_id,
                "type": fault_type,
                "target": target,
                "params": params,
                "timestamp": time.time(),
                "status": "active"
            }
            self.active_faults[fault_id] = fault
            self._save_fault(fault)
            
            logger.info(f"Successfully initiated {fault_type} fault in {target} (container: {container_name})")
            return fault_id
            
        except docker.errors.NotFound:
            logger.error(f"Container {container_name} not found for service {target}.")
            return None
        except Exception as e:
            logger.error(f"Failed to inject {fault_type} fault into {target}: {e}")
            return None

    def remove_fault(self, fault_id: str) -> bool:
        """
        Remove an active fault by killing the stress process in the container.
        """
        if fault_id not in self.active_faults:
            return False
        
        fault = self.active_faults[fault_id]
        target = fault["target"]
        
        container_name = self._get_container_name(target)
        if not container_name:
            return False

        try:
            client = get_docker_client()
            container = client.containers.get(container_name)

            if fault["type"] == "cpu_stress":
                # Kill all 'stress-ng' processes in the container
                kill_cmd = "pkill stress-ng"
                logger.info(f"Executing command in {container_name}: {kill_cmd}")
                container.exec_run(kill_cmd)
            
            del self.active_faults[fault_id]
            logger.info(f"Removed {fault['type']} fault from {target}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove fault {fault_id}: {e}")
            return False

    def get_active_faults(self) -> List[Dict[str, Any]]:
        return list(self.active_faults.values())

    def _save_fault(self, fault: Dict[str, Any]) -> None:
        fault_file = os.path.join(self.data_dir, f"fault_{fault['id']}.json")
        with open(fault_file, 'w') as f:
            json.dump(fault, f, indent=2)
