import docker

def get_docker_client():
    """
    Get a Docker client, connecting to the default socket.
    This is a robust way to get a client that avoids issues with
    misconfigured DOCKER_HOST environment variables.
    """
    try:
        # First, try the default socket connection
        client = docker.DockerClient(base_url='unix://var/run/docker.sock')
        client.ping()
        return client
    except Exception as e:
        # If that fails, fall back to from_env() but with a warning.
        # This might be necessary in some local dev environments.
        print(f"Warning: Could not connect to Docker socket. Falling back to from_env(). Error: {e}")
        return docker.from_env() 
