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
            subprocess.run(["docker", "restart", service_name], check=True)
            logger.info(f"Successfully restarted service: {service_name}")
            return True
        except subprocess.CalledProcessError:
            logger.error(f"Cannot restart service: container {service_name} not found or failed to restart.")
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred while restarting {service_name}: {e}")
            return False

    def isolate_service(self, service_name: str, **kwargs):
        logger.warning(f"Isolate action for service {service_name} is not fully implemented in DockerAdapter.")
        return False

    def scale_service(self, service_name: str, **kwargs):
        """Scales a docker container's resources using the CLI."""
        try:
            cmd = ["docker", "update"]
            if 'cpu_quota' in kwargs and kwargs['cpu_quota'] is not None:
                cmd.append(f"--cpu-quota={kwargs['cpu_quota']}")
            if 'memory_limit' in kwargs and kwargs['memory_limit'] is not None:
                cmd.append(f"--memory={kwargs['memory_limit']}")
            cmd.append(service_name)
            if len(cmd) > 3:
                subprocess.run(cmd, check=True)
                logger.info(f"Successfully scaled service {service_name} with config: {cmd}")
                return True
            else:
                logger.warning(f"Scale action for {service_name} called with no parameters.")
                return False
        except subprocess.CalledProcessError:
            logger.error(f"Cannot scale service: container {service_name} not found or failed to update.")
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred while scaling {service_name}: {e}")
            return False

    def degrade_service(self, service_name: str, **kwargs):
        logger.warning(f"Degrade action for service {service_name} is not implemented in DockerAdapter.")
        return False 
