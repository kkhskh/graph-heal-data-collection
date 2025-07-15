#!/usr/bin/env python3
"""
Prometheus Integration for Graph-Heal
Provides seamless integration with existing Prometheus monitoring stacks.
"""

import time
import json
import logging
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from prometheus_client import start_http_server, Gauge, Counter, Histogram, Info

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PrometheusIntegration:
    """Integrates Graph-Heal with Prometheus monitoring."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.port = config.get('port', 9091)
        self.scrape_interval = config.get('scrape_interval', 15)
        self.services = config.get('services', [])
        self.metrics_dir = Path(config.get('metrics_dir', 'data/prometheus_metrics'))
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Prometheus metrics
        self.graphheal_detection_latency = Histogram(
            'graphheal_detection_latency_seconds',
            'Time taken to detect faults',
            ['service', 'fault_type']
        )
        
        self.graphheal_localization_latency = Histogram(
            'graphheal_localization_latency_seconds',
            'Time taken to localize faults',
            ['service', 'detected_service', 'localized_service']
        )
        
        self.graphheal_recovery_latency = Histogram(
            'graphheal_recovery_latency_seconds',
            'Time taken to recover from faults',
            ['service', 'recovery_action']
        )
        
        self.graphheal_detection_count = Counter(
            'graphheal_detections_total',
            'Total number of fault detections',
            ['service', 'fault_type', 'status']
        )
        
        self.graphheal_localization_accuracy = Gauge(
            'graphheal_localization_accuracy',
            'Accuracy of fault localization',
            ['service']
        )
        
        self.graphheal_recovery_success_rate = Gauge(
            'graphheal_recovery_success_rate',
            'Success rate of recovery actions',
            ['service']
        )
        
        self.graphheal_system_health = Gauge(
            'graphheal_system_health',
            'Overall system health score',
            ['service']
        )
        
        self.graphheal_active_faults = Gauge(
            'graphheal_active_faults',
            'Number of active faults',
            ['service', 'fault_type']
        )
        
        self.graphheal_info = Info(
            'graphheal',
            'Graph-Heal system information'
        )
        
        # Initialize system info
        self.graphheal_info.info({
            'version': '1.0.0',
            'description': 'Graph-Heal Fault Detection and Recovery System',
            'integration_type': 'prometheus'
        })
        
    def start_metrics_server(self):
        """Start the Prometheus metrics server."""
        try:
            start_http_server(self.port)
            logger.info(f"Prometheus metrics server started on port {self.port}")
            logger.info(f"Metrics available at http://localhost:{self.port}/metrics")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
            
    def record_detection_event(self, service: str, fault_type: str, latency: float, status: str = 'success'):
        """Record a detection event."""
        try:
            self.graphheal_detection_latency.labels(service=service, fault_type=fault_type).observe(latency)
            self.graphheal_detection_count.labels(service=service, fault_type=fault_type, status=status).inc()
            logger.info(f"Recorded detection event: {service} - {fault_type} - {latency:.3f}s")
        except Exception as e:
            logger.error(f"Failed to record detection event: {e}")
            
    def record_localization_event(self, service: str, detected_service: str, localized_service: str, latency: float):
        """Record a localization event."""
        try:
            self.graphheal_localization_latency.labels(
                service=service,
                detected_service=detected_service,
                localized_service=localized_service
            ).observe(latency)
            logger.info(f"Recorded localization event: {service} -> {localized_service} - {latency:.3f}s")
        except Exception as e:
            logger.error(f"Failed to record localization event: {e}")
            
    def record_recovery_event(self, service: str, recovery_action: str, latency: float):
        """Record a recovery event."""
        try:
            self.graphheal_recovery_latency.labels(service=service, recovery_action=recovery_action).observe(latency)
            logger.info(f"Recorded recovery event: {service} - {recovery_action} - {latency:.3f}s")
        except Exception as e:
            logger.error(f"Failed to record recovery event: {e}")
            
    def update_localization_accuracy(self, service: str, accuracy: float):
        """Update localization accuracy metric."""
        try:
            self.graphheal_localization_accuracy.labels(service=service).set(accuracy)
            logger.info(f"Updated localization accuracy for {service}: {accuracy:.2f}")
        except Exception as e:
            logger.error(f"Failed to update localization accuracy: {e}")
            
    def update_recovery_success_rate(self, service: str, success_rate: float):
        """Update recovery success rate metric."""
        try:
            self.graphheal_recovery_success_rate.labels(service=service).set(success_rate)
            logger.info(f"Updated recovery success rate for {service}: {success_rate:.2f}")
        except Exception as e:
            logger.error(f"Failed to update recovery success rate: {e}")
            
    def update_system_health(self, service: str, health_score: float):
        """Update system health metric."""
        try:
            self.graphheal_system_health.labels(service=service).set(health_score)
            logger.info(f"Updated system health for {service}: {health_score:.2f}")
        except Exception as e:
            logger.error(f"Failed to update system health: {e}")
            
    def update_active_faults(self, service: str, fault_type: str, count: int):
        """Update active faults metric."""
        try:
            self.graphheal_active_faults.labels(service=service, fault_type=fault_type).set(count)
            logger.info(f"Updated active faults for {service} - {fault_type}: {count}")
        except Exception as e:
            logger.error(f"Failed to update active faults: {e}")
            
    def scrape_service_metrics(self):
        """Scrape metrics from all services and update Prometheus metrics."""
        for service in self.services:
            try:
                # Get service health
                health_data = self.get_service_health(service)
                if health_data:
                    health_score = health_data.get('health_score', 1.0)
                    self.update_system_health(service, health_score)
                    
                    # Check for active faults
                    if health_data.get('cpu_fault_active', False):
                        self.update_active_faults(service, 'cpu', 1)
                    else:
                        self.update_active_faults(service, 'cpu', 0)
                        
            except Exception as e:
                logger.warning(f"Failed to scrape metrics for {service}: {e}")
                
    def get_service_health(self, service: str) -> Optional[Dict]:
        """Get health status from a service."""
        try:
            port = self.get_service_port(service)
            response = requests.get(f"http://localhost:{port}/health", timeout=5)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Failed to get health from {service}: {response.status_code}")
                return None
        except Exception as e:
            logger.warning(f"Error getting health from {service}: {e}")
            return None
            
    def get_service_port(self, service: str) -> int:
        """Get the port for a service."""
        port_map = {
            'service_a': 5001,
            'service_b': 5002,
            'service_c': 5003,
            'service_d': 5004
        }
        return port_map.get(service, 5000)
        
    def generate_prometheus_config(self, output_path: str = None):
        """Generate Prometheus configuration for Graph-Heal."""
        if output_path is None:
            output_path = self.metrics_dir / 'prometheus.yml'
            
        config = {
            'global': {
                'scrape_interval': f'{self.scrape_interval}s',
                'evaluation_interval': f'{self.scrape_interval}s'
            },
            'scrape_configs': [
                {
                    'job_name': 'graphheal',
                    'static_configs': [
                        {
                            'targets': [f'localhost:{self.port}'],
                            'labels': {
                                'service': 'graphheal',
                                'environment': 'production'
                            }
                        }
                    ]
                },
                {
                    'job_name': 'microservices',
                    'static_configs': [
                        {
                            'targets': [
                                'localhost:5001',
                                'localhost:5002',
                                'localhost:5003',
                                'localhost:5004'
                            ],
                            'labels': {
                                'environment': 'production'
                            }
                        }
                    ]
                }
            ]
        }
        
        with open(output_path, 'w') as f:
            import yaml
            yaml.dump(config, f, default_flow_style=False)
            
        logger.info(f"Prometheus configuration generated: {output_path}")
        return output_path
        
    def run_continuous_monitoring(self):
        """Run continuous monitoring and metric collection."""
        logger.info("Starting continuous monitoring")
        
        while True:
            try:
                # Scrape service metrics
                self.scrape_service_metrics()
                
                # Wait for next scrape interval
                time.sleep(self.scrape_interval)
                
            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in continuous monitoring: {e}")
                time.sleep(self.scrape_interval)


def main():
    """Main function to run Prometheus integration."""
    # Configuration
    config = {
        'port': 9091,
        'scrape_interval': 15,
        'services': ['service_a', 'service_b', 'service_c', 'service_d'],
        'metrics_dir': 'data/prometheus_metrics'
    }
    
    # Create integration
    integration = PrometheusIntegration(config)
    
    # Generate Prometheus configuration
    integration.generate_prometheus_config()
    
    # Start metrics server
    integration.start_metrics_server()
    
    # Run continuous monitoring
    integration.run_continuous_monitoring()


if __name__ == "__main__":
    main() 