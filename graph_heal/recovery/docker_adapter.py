from .base import RecoverySystemAdapter
import docker
import logging

logger = logging.getLogger(__name__)

class DockerAdapter(RecoverySystemAdapter):
    def __init__(self, docker_client: docker.DockerClient = None):
        if docker_client:
            self.client = docker_client
        else:
            try:
                # Let docker-py figure out the correct socket/pipe from your environment
                self.client = docker.from_env()
            except docker.errors.DockerException as e:
                logger.warning(f"docker.from_env() failed: {e}. Falling back to default Unix socket.")
                # Fallback to the default Unix socket explicitly
                self.client = docker.DockerClient(base_url='unix://var/run/docker.sock')

    def restart_service(self, service_name: str, **kwargs):
        """Restarts a docker container."""
        try:
            container = self.client.containers.get(service_name)
            container.restart()
            logger.info(f"Successfully restarted service: {service_name}")
            return True
        except docker.errors.NotFound:
            logger.error(f"Cannot restart service: container {service_name} not found.")
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred while restarting {service_name}: {e}")
            return False

    def isolate_service(self, service_name: str, **kwargs):
        # Implementation for isolating a docker container would go here.
        # e.g., by disconnecting it from its primary network.
        logger.warning(f"Isolate action for service {service_name} is not fully implemented in DockerAdapter.")
        return False

    def scale_service(self, service_name: str, **kwargs):
        """Scales a docker container's resources."""
        try:
            container = self.client.containers.get(service_name)
            
            # docker-py uses resource names like 'cpu_quota', 'mem_limit'
            update_config = {
                'cpu_quota': kwargs.get('cpu_quota'),
                'mem_limit': kwargs.get('memory_limit')
            }
            
            # Filter out None values so we don't overwrite existing settings
            update_config = {k: v for k, v in update_config.items() if v is not None}

            if not update_config:
                logger.warning(f"Scale action for {service_name} called with no parameters.")
                return False

            container.update(**update_config)
            logger.info(f"Successfully scaled service {service_name} with config: {update_config}")
            return True
        except docker.errors.NotFound:
            logger.error(f"Cannot scale service: container {service_name} not found.")
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred while scaling {service_name}: {e}")
            return False

    def degrade_service(self, service_name: str, **kwargs):
        """A placeholder for degrading service functionality."""
        logger.warning(f"Degrade action for service {service_name} is not implemented in DockerAdapter.")
        # In a real scenario, this might modify environment variables,
        # apply traffic shaping, or connect to a limited network.
        return False

    def get_docker_client(self) -> docker.DockerClient:
        """Returns the underlying Docker client."""
        return self.client 