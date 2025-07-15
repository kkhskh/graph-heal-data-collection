from abc import ABC, abstractmethod
from typing import Any, Dict

class RecoverySystemAdapter(ABC):
    """
    Abstract base class for all recovery system adapters.
    Defines the standard interface for performing recovery actions
    on different types of infrastructure (Docker, Kubernetes, OPC-UA, etc.).
    """

    @abstractmethod
    def restart_service(self, service_name: str, **kwargs: Dict[str, Any]) -> bool:
        """
        Restarts a service.
        This can be a container, a pod, a process, or a physical device.
        """
        pass

    @abstractmethod
    def scale_service(self, service_name: str, **kwargs: Dict[str, Any]) -> bool:
        """
        Scales a service up or down.
        Typically involves changing the number of replicas.
        """
        pass

    @abstractmethod
    def isolate_service(self, service_name: str, **kwargs: Dict[str, Any]) -> bool:
        """
        Isolates a service from the network.
        Prevents it from communicating with other services.
        """
        pass

    @abstractmethod
    def degrade_service(self, service_name: str, **kwargs: Dict[str, Any]) -> bool:
        """
        Degrades a service's performance or functionality.
        Can involve rate limiting, reducing resources, or disabling features.
        """
        pass 