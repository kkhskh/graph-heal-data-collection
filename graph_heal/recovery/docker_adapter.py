from .base import RecoverySystemAdapter
import subprocess
import logging

logger = logging.getLogger(__name__)

class DockerAdapter(RecoverySystemAdapter):
    def __init__(self, timeout=10):
        self.timeout = timeout

    def restart_service(self, service_name: str, **kwargs):
        """Restarts a docker container using the CLI."""
        try:
            container_name = service_name.replace('_', '-')
            subprocess.run(["docker", "restart", container_name], check=True)
            logger.info(f"Successfully restarted service: {container_name}")
            return True
        except subprocess.CalledProcessError:
            logger.error(f"Cannot restart service: container {container_name} not found or failed to restart.")
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred while restarting {container_name}: {e}")
            return False

    def isolate_service(self, service_name: str, **kwargs):
        container_name = service_name.replace('_', '-')
        logger.warning(f"Isolate action for service {container_name} is not fully implemented in DockerAdapter.")
        return False

    def scale_service(self, service_name: str, **kwargs):
        """Scales a docker container's resources using the CLI."""
        try:
            container_name = service_name.replace('_', '-')
            cmd = ["docker", "update"]
            if 'cpu_quota' in kwargs and kwargs['cpu_quota'] is not None:
                cmd.append(f"--cpu-quota={kwargs['cpu_quota']}")
            if 'memory_limit' in kwargs and kwargs['memory_limit'] is not None:
                cmd.append(f"--memory={kwargs['memory_limit']}")
            cmd.append(container_name)
            if len(cmd) > 3:
                subprocess.run(cmd, check=True)
                logger.info(f"Successfully scaled service {container_name} with config: {cmd}")
                return True
            else:
                logger.warning(f"Scale action for {container_name} called with no parameters.")
                return False
        except subprocess.CalledProcessError:
            logger.error(f"Cannot scale service: container {container_name} not found or failed to update.")
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred while scaling {container_name}: {e}")
            return False

    def degrade_service(self, service_name: str, **kwargs):
        container_name = service_name.replace('_', '-')
        logger.warning(f"Degrade action for service {container_name} is not implemented in DockerAdapter.")
        return False 
