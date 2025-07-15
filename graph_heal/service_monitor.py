import requests
import time
import logging

class ServiceMonitor:
    """A simple, non-threaded service monitor for synchronous metric collection."""
    
    def __init__(self, services, interval=5, timeout=2):
        """
        Initializes the ServiceMonitor.
        Args:
            services (list): A list of dictionaries, where each dictionary represents a service.
                             Each dictionary should have 'id', 'url', and 'health_endpoint'.
            interval (int): The polling interval (not used in this synchronous version but kept for compatibility).
            timeout (int): The timeout for HTTP requests.
        """
        self.services = services
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)

    def _parse_prometheus_metrics(self, text_data: str) -> dict:
        """Parses Prometheus text format into a dictionary."""
        metrics = {}
        for line in text_data.strip().split('\n'):
            if line.startswith('#') or not line.strip():
                continue
            
            parts = line.split()
            if len(parts) >= 2:
                metric_name = parts[0].split('{')[0]
                try:
                    value = float(parts[-1])
                    metrics[metric_name] = value
                except (ValueError, IndexError):
                    pass
        return metrics

    def get_service_status(self, service):
        """Fetches the status and metrics for a single service."""
        try:
            full_url = f"{service['url'].rstrip('/')}/{service['health_endpoint'].lstrip('/')}"
            response = requests.get(full_url, timeout=self.timeout)
            response.raise_for_status()

            metrics_text = response.text
            if not metrics_text.strip():
                 self.logger.warning(f"Metrics response from {service['id']} is empty.")
                 return {"status": "error", "metrics": {}, "timestamp": time.time()}

            return {
                "status": "healthy",
                "metrics": self._parse_prometheus_metrics(metrics_text),
                "timestamp": time.time()
            }
        except requests.exceptions.RequestException:
            return {"status": "unreachable", "metrics": {}, "timestamp": time.time()}
        except Exception as e:
            self.logger.error(f"An unexpected error occurred for service {service['id']}: {e}")
            return {"status": "error", "metrics": {}, "timestamp": time.time()}

    def get_all_services_status(self):
        """Fetches and returns the status of all monitored services."""
        statuses = {}
        for service in self.services:
            statuses[service['id']] = self.get_service_status(service)
        return statuses

    def start(self):
        """Dummy method for compatibility."""
        self.logger.info("Monitoring started (dummy call for simple monitor).")

    def stop(self):
        """Dummy method for compatibility."""
        self.logger.info("Monitoring stopped (dummy call for simple monitor).")

    def start_monitoring(self):
        """Dummy method for compatibility."""
        self.logger.info("Monitoring started (dummy call for simple monitor).")

    def stop_monitoring(self):
        """Dummy method for compatibility."""
        self.logger.info("Monitoring stopped (dummy call for simple monitor).")


# import requests
# import time
# import logging

# class ServiceMonitor:
#     """A simple, non-threaded service monitor for synchronous metric collection."""
    
#     def __init__(self, services, interval=5, timeout=2):
#         """
#         Initializes the ServiceMonitor.
#         Args:
#             services (list): A list of dictionaries, where each dictionary represents a service.
#                              Each dictionary should have 'id', 'url', and 'health_endpoint'.
#             interval (int): The polling interval (not used in this synchronous version but kept for compatibility).
#             timeout (int): The timeout for HTTP requests.
#         """
#         self.services = services
#         self.timeout = timeout
#         self.logger = logging.getLogger(__name__)

#     def _parse_prometheus_metrics(self, text_data: str) -> dict:
#         """Parses Prometheus text format into a dictionary."""
#         metrics = {}
#         for line in text_data.strip().split('\n'):
#             if line.startswith('#') or not line.strip():
#                 continue
            
#             parts = line.split()
#             if len(parts) >= 2:
#                 metric_name = parts[0].split('{')[0]
#                 try:
#                     value = float(parts[-1])
#                     metrics[metric_name] = value
#                 except (ValueError, IndexError):
#                     pass
#         return metrics

#     def get_service_status(self, service):
#         """Fetches the status and metrics for a single service."""
#         try:
#             full_url = f"{service['url'].rstrip('/')}/{service['health_endpoint'].lstrip('/')}"
#             response = requests.get(full_url, timeout=self.timeout)
#             response.raise_for_status()

#             metrics_text = response.text
#             if not metrics_text.strip():
#                  self.logger.warning(f"Metrics response from {service['id']} is empty.")
#                  return {"status": "error", "metrics": {}, "timestamp": time.time()}

#             return {
#                 "status": "healthy",
#                 "metrics": self._parse_prometheus_metrics(metrics_text),
#                 "timestamp": time.time()
#             }
#         except requests.exceptions.RequestException:
#             return {"status": "unreachable", "metrics": {}, "timestamp": time.time()}
#         except Exception as e:
#             self.logger.error(f"An unexpected error occurred for service {service['id']}: {e}")
#             return {"status": "error", "metrics": {}, "timestamp": time.time()}

#     def get_all_services_status(self):
#         """Fetches and returns the status of all monitored services."""
#         statuses = {}
#         for service in self.services:
#             statuses[service['id']] = self.get_service_status(service)
#         return statuses

#     def start(self):
#         """Dummy method for compatibility."""
#         self.logger.info("Monitoring started (dummy call for simple monitor).")

#     def stop(self):
#         """Dummy method for compatibility."""
#         self.logger.info("Monitoring stopped (dummy call for simple monitor).")

#     def start_monitoring(self):
#         """Dummy method for compatibility."""
#         self.logger.info("Monitoring started (dummy call for simple monitor).")

#     def stop_monitoring(self):
#         """Dummy method for compatibility."""
#         self.logger.info("Monitoring stopped (dummy call for simple monitor).") 