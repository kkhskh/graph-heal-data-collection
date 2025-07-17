import docker
import psutil
from typing import Dict, Optional
import time
from datetime import datetime
import logging
from graph_heal.utils import get_docker_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DockerMetrics:
    def __init__(self, services):
        self.services = services
        self.client = get_docker_client()
        self.container_stats = {}
        self.host_stats = {}
        
    def _get_container_name(self, name: str) -> str:
        """Map container names to their actual Docker container names"""
        if name.startswith('container_'):
            return f"service_{name.split('_')[-1]}"
        return name
        
    def get_container_metrics(self, container_name: str) -> Optional[Dict]:
        """Get real-time metrics for a Docker container"""
        try:
            actual_name = self._get_container_name(container_name)
            container = self.client.containers.get(actual_name)
            stats = container.stats(stream=False)
            logger.info(f"Stats for container {actual_name}: {stats}")
            
            # Calculate CPU usage percentage
            cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                       stats['precpu_stats']['cpu_usage']['total_usage']
            system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                          stats['precpu_stats']['system_cpu_usage']
            cpu_percent = 0.0
            if system_delta > 0 and cpu_delta > 0:
                # Get number of CPUs - either from percpu_usage or cpu_stats
                num_cpus = len(stats['cpu_stats'].get('cpu_usage', {}).get('percpu_usage', []))
                if num_cpus == 0:  # If percpu_usage is not available
                    num_cpus = stats['cpu_stats'].get('online_cpus', 1)
                cpu_percent = (cpu_delta / system_delta) * num_cpus * 100.0
            
            # Calculate memory usage
            memory_usage = stats['memory_stats']['usage']
            memory_limit = stats['memory_stats']['limit']
            memory_percent = (memory_usage / memory_limit) * 100.0 if memory_limit > 0 else 0.0
            
            # Get network stats
            network_stats = stats['networks']
            rx_bytes = sum(net['rx_bytes'] for net in network_stats.values())
            tx_bytes = sum(net['tx_bytes'] for net in network_stats.values())
            
            metrics = {
                'container_cpu_usage': cpu_percent,
                'container_memory_usage': memory_percent,
                'container_network_rx': rx_bytes,
                'container_network_tx': tx_bytes,
                'timestamp': datetime.now()
            }
            
            self.container_stats[actual_name] = metrics
            return metrics
            
        except docker.errors.NotFound:
            logger.error(f"Container {actual_name} not found")
            return None
        except Exception as e:
            logger.error(f"Error getting metrics for container {actual_name}: {e}")
            return None
    
    def get_host_metrics(self) -> Dict:
        """Get real-time metrics for the host system"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # Network metrics
            net_io = psutil.net_io_counters()
            
            metrics = {
                'host_cpu_usage': cpu_percent,
                'host_cpu_count': cpu_count,
                'host_memory_usage': memory_percent,
                'host_disk_usage': disk_percent,
                'host_network_rx': net_io.bytes_recv,
                'host_network_tx': net_io.bytes_sent,
                'timestamp': datetime.now()
            }
            
            self.host_stats = metrics
            return metrics
            
        except Exception as e:
            print(f"Error getting host metrics: {e}")
            return {}
    
    def get_all_metrics(self, container_names: list) -> Dict:
        """Get metrics for all containers and the host"""
        metrics = {
            'containers': {},
            'host': self.get_host_metrics()
        }
        
        for container_name in container_names:
            container_metrics = self.get_container_metrics(container_name)
            if container_metrics:
                metrics['containers'][container_name] = container_metrics
        
        return metrics
    
    def get_container_health(self, container_name: str) -> float:
        """Calculate health score for a container based on its metrics"""
        metrics = self.get_container_metrics(container_name)
        if not metrics:
            return 0.0
        
        # Weighted average of different metrics
        cpu_weight = 0.4
        memory_weight = 0.4
        network_weight = 0.2
        
        cpu_health = 1.0 - (metrics['container_cpu_usage'] / 100.0)
        memory_health = 1.0 - (metrics['container_memory_usage'] / 100.0)
        
        # Network health is more complex - could be based on throughput, errors, etc.
        # For now, we'll use a simple placeholder
        network_health = 1.0
        
        health_score = (
            cpu_health * cpu_weight +
            memory_health * memory_weight +
            network_health * network_weight
        )
        
        return max(0.0, min(1.0, health_score))
    
    def get_host_health(self) -> float:
        """Calculate health score for the host based on its metrics"""
        metrics = self.get_host_metrics()
        if not metrics:
            return 0.0
        
        # Weighted average of different metrics
        cpu_weight = 0.3
        memory_weight = 0.3
        disk_weight = 0.2
        network_weight = 0.2
        
        cpu_health = 1.0 - (metrics['host_cpu_usage'] / 100.0)
        memory_health = 1.0 - (metrics['host_memory_usage'] / 100.0)
        disk_health = 1.0 - (metrics['host_disk_usage'] / 100.0)
        
        # Network health is more complex - could be based on throughput, errors, etc.
        # For now, we'll use a simple placeholder
        network_health = 1.0
        
        health_score = (
            cpu_health * cpu_weight +
            memory_health * memory_weight +
            disk_health * disk_weight +
            network_health * network_weight
        )
        
        return max(0.0, min(1.0, health_score)) 
