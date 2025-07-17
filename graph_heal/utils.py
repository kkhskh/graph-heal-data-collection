import os
import docker

def get_docker_client():
    """
    Get a Docker client, connecting to the default socket.
    This is a robust way to get a client that avoids issues with
    misconfigured DOCKER_HOST environment variables.
    """
    # Unset DOCKER_HOST to avoid http+docker scheme errors in CI
    os.environ.pop("DOCKER_HOST", None)
    try:
        client = docker.DockerClient(base_url='unix://var/run/docker.sock')
        client.ping()
        return client
    except Exception as e:
        print(f"Warning: Could not connect to Docker socket. Error: {e}")
        raise 
