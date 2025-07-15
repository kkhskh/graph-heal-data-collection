#!/usr/bin/env python3
"""
Observability Integration for Graph-Heal
Provides integration with Jaeger, Grafana, and other observability tools.
"""

import time
import json
import logging
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ObservabilityIntegration:
    """Integrates Graph-Heal with observability tools."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.observability_dir = Path(config.get('observability_dir', 'observability'))
        self.observability_dir.mkdir(parents=True, exist_ok=True)
        self.jaeger_endpoint = config.get('jaeger_endpoint', 'http://localhost:14268/api/traces')
        self.grafana_endpoint = config.get('grafana_endpoint', 'http://localhost:3000')
        
    def generate_jaeger_integration(self):
        """Generate Jaeger tracing integration."""
        
        # Jaeger configuration
        jaeger_config = {
            'jaeger': {
                'endpoint': self.jaeger_endpoint,
                'service_name': 'graph-heal',
                'sampling_rate': 1.0,
                'tags': {
                    'version': '1.0.0',
                    'environment': 'production'
                }
            }
        }
        
        # Save configuration
        config_file = self.observability_dir / 'jaeger-config.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(jaeger_config, f, default_flow_style=False)
        logger.info(f"Generated Jaeger configuration: {config_file}")
        
        # Jaeger tracing decorator
        tracing_decorator = '''import time
import functools
import requests
import json
from typing import Dict, Any

def trace_operation(operation_name: str, tags: Dict[str, str] = None):
    """Decorator to trace operations with Jaeger."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            # Create trace span
            span = {
                'operationName': operation_name,
                'startTime': int(start_time * 1000000),  # Microseconds
                'tags': tags or {},
                'logs': []
            }
            
            try:
                result = func(*args, **kwargs)
                span['tags']['status'] = 'success'
                return result
            except Exception as e:
                span['tags']['status'] = 'error'
                span['tags']['error'] = str(e)
                span['logs'].append({
                    'timestamp': int(time.time() * 1000000),
                    'fields': [{'key': 'error', 'value': str(e)}]
                })
                raise
            finally:
                end_time = time.time()
                span['duration'] = int((end_time - start_time) * 1000000)
                
                # Send span to Jaeger
                self.send_span_to_jaeger(span)
                
        return wrapper
    return decorator

def send_span_to_jaeger(span: Dict[str, Any]):
    """Send span to Jaeger collector."""
    try:
        trace = {
            'data': [{
                'traceID': '1234567890abcdef',
                'spans': [span]
            }]
        }
        
        response = requests.post(
            'http://localhost:14268/api/traces',
            json=trace,
            headers={'Content-Type': 'application/json'}
        )
        response.raise_for_status()
    except Exception as e:
        logger.warning(f"Failed to send span to Jaeger: {e}")'''
        
        # Save tracing decorator
        tracing_file = self.observability_dir / 'tracing.py'
        with open(tracing_file, 'w') as f:
            f.write(tracing_decorator)
        logger.info(f"Generated tracing decorator: {tracing_file}")
        
    def generate_grafana_dashboards(self):
        """Generate Grafana dashboards for Graph-Heal."""
        
        # Main dashboard
        main_dashboard = {
            'dashboard': {
                'id': None,
                'title': 'Graph-Heal Monitoring',
                'tags': ['graph-heal', 'monitoring'],
                'timezone': 'browser',
                'panels': [
                    {
                        'id': 1,
                        'title': 'Detection Latency',
                        'type': 'graph',
                        'targets': [
                            {
                                'expr': 'histogram_quantile(0.95, rate(graphheal_detection_latency_seconds_bucket[5m]))',
                                'legendFormat': '95th percentile'
                            }
                        ],
                        'gridPos': {'h': 8, 'w': 12, 'x': 0, 'y': 0}
                    },
                    {
                        'id': 2,
                        'title': 'Localization Accuracy',
                        'type': 'stat',
                        'targets': [
                            {
                                'expr': 'graphheal_localization_accuracy',
                                'legendFormat': 'Accuracy'
                            }
                        ],
                        'gridPos': {'h': 8, 'w': 12, 'x': 12, 'y': 0}
                    },
                    {
                        'id': 3,
                        'title': 'Recovery Success Rate',
                        'type': 'stat',
                        'targets': [
                            {
                                'expr': 'graphheal_recovery_success_rate',
                                'legendFormat': 'Success Rate'
                            }
                        ],
                        'gridPos': {'h': 8, 'w': 12, 'x': 0, 'y': 8}
                    },
                    {
                        'id': 4,
                        'title': 'Active Faults',
                        'type': 'graph',
                        'targets': [
                            {
                                'expr': 'graphheal_active_faults',
                                'legendFormat': '{{service}} - {{fault_type}}'
                            }
                        ],
                        'gridPos': {'h': 8, 'w': 12, 'x': 12, 'y': 8}
                    }
                ],
                'time': {
                    'from': 'now-1h',
                    'to': 'now'
                },
                'refresh': '30s'
            }
        }
        
        # Performance dashboard
        performance_dashboard = {
            'dashboard': {
                'id': None,
                'title': 'Graph-Heal Performance',
                'tags': ['graph-heal', 'performance'],
                'timezone': 'browser',
                'panels': [
                    {
                        'id': 1,
                        'title': 'Detection Latency Distribution',
                        'type': 'heatmap',
                        'targets': [
                            {
                                'expr': 'rate(graphheal_detection_latency_seconds_bucket[5m])',
                                'legendFormat': '{{le}}'
                            }
                        ],
                        'gridPos': {'h': 8, 'w': 24, 'x': 0, 'y': 0}
                    },
                    {
                        'id': 2,
                        'title': 'Recovery Time by Service',
                        'type': 'bargauge',
                        'targets': [
                            {
                                'expr': 'histogram_quantile(0.95, rate(graphheal_recovery_latency_seconds_bucket[5m]))',
                                'legendFormat': '{{service}}'
                            }
                        ],
                        'gridPos': {'h': 8, 'w': 12, 'x': 0, 'y': 8}
                    },
                    {
                        'id': 3,
                        'title': 'Fault Detection Count',
                        'type': 'timeseries',
                        'targets': [
                            {
                                'expr': 'rate(graphheal_detections_total[5m])',
                                'legendFormat': '{{service}} - {{fault_type}}'
                            }
                        ],
                        'gridPos': {'h': 8, 'w': 12, 'x': 12, 'y': 8}
                    }
                ],
                'time': {
                    'from': 'now-6h',
                    'to': 'now'
                },
                'refresh': '1m'
            }
        }
        
        # Save dashboards
        dashboard_files = [
            ('main-dashboard.json', main_dashboard),
            ('performance-dashboard.json', performance_dashboard)
        ]
        
        for filename, dashboard in dashboard_files:
            filepath = self.observability_dir / filename
            with open(filepath, 'w') as f:
                json.dump(dashboard, f, indent=2)
            logger.info(f"Generated Grafana dashboard: {filepath}")
            
    def generate_prometheus_alerts(self):
        """Generate Prometheus alerting rules."""
        
        alerts = {
            'groups': [
                {
                    'name': 'graphheal.rules',
                    'rules': [
                        {
                            'alert': 'HighDetectionLatency',
                            'expr': 'histogram_quantile(0.95, rate(graphheal_detection_latency_seconds_bucket[5m])) > 1',
                            'for': '2m',
                            'labels': {
                                'severity': 'warning'
                            },
                            'annotations': {
                                'summary': 'High fault detection latency',
                                'description': 'Fault detection latency is above 1 second for {{ $labels.service }}'
                            }
                        },
                        {
                            'alert': 'LowLocalizationAccuracy',
                            'expr': 'graphheal_localization_accuracy < 0.8',
                            'for': '5m',
                            'labels': {
                                'severity': 'critical'
                            },
                            'annotations': {
                                'summary': 'Low fault localization accuracy',
                                'description': 'Fault localization accuracy is below 80% for {{ $labels.service }}'
                            }
                        },
                        {
                            'alert': 'LowRecoverySuccessRate',
                            'expr': 'graphheal_recovery_success_rate < 0.9',
                            'for': '5m',
                            'labels': {
                                'severity': 'critical'
                            },
                            'annotations': {
                                'summary': 'Low recovery success rate',
                                'description': 'Recovery success rate is below 90% for {{ $labels.service }}'
                            }
                        },
                        {
                            'alert': 'ActiveFaults',
                            'expr': 'graphheal_active_faults > 0',
                            'for': '1m',
                            'labels': {
                                'severity': 'warning'
                            },
                            'annotations': {
                                'summary': 'Active faults detected',
                                'description': '{{ $value }} active faults detected in {{ $labels.service }}'
                            }
                        }
                    ]
                }
            ]
        }
        
        filepath = self.observability_dir / 'alerts.yaml'
        with open(filepath, 'w') as f:
            yaml.dump(alerts, f, default_flow_style=False)
        logger.info(f"Generated Prometheus alerts: {filepath}")
        
    def generate_logging_config(self):
        """Generate structured logging configuration."""
        
        logging_config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'structured': {
                    'class': 'pythonjsonlogger.jsonlogger.JsonFormatter',
                    'format': '%(timestamp)s %(level)s %(name)s %(message)s'
                }
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'formatter': 'structured'
                },
                'file': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'filename': '/var/log/graphheal/app.log',
                    'maxBytes': 10485760,  # 10MB
                    'backupCount': 5,
                    'formatter': 'structured'
                }
            },
            'loggers': {
                'graph_heal': {
                    'level': 'INFO',
                    'handlers': ['console', 'file'],
                    'propagate': False
                }
            },
            'root': {
                'level': 'WARNING',
                'handlers': ['console']
            }
        }
        
        filepath = self.observability_dir / 'logging.conf'
        with open(filepath, 'w') as f:
            import json
            json.dump(logging_config, f, indent=2)
        logger.info(f"Generated logging configuration: {filepath}")
        
    def generate_docker_compose_observability(self):
        """Generate Docker Compose for observability stack."""
        
        observability_compose = {
            'version': '3.8',
            'services': {
                'prometheus': {
                    'image': 'prom/prometheus:latest',
                    'ports': ['9090:9090'],
                    'volumes': [
                        './observability/prometheus.yml:/etc/prometheus/prometheus.yml',
                        './observability/alerts.yaml:/etc/prometheus/alerts.yaml'
                    ],
                    'command': [
                        '--config.file=/etc/prometheus/prometheus.yml',
                        '--storage.tsdb.path=/prometheus',
                        '--web.console.libraries=/etc/prometheus/console_libraries',
                        '--web.console.templates=/etc/prometheus/consoles',
                        '--storage.tsdb.retention.time=200h',
                        '--web.enable-lifecycle'
                    ]
                },
                'grafana': {
                    'image': 'grafana/grafana:latest',
                    'ports': ['3000:3000'],
                    'environment': {
                        'GF_SECURITY_ADMIN_PASSWORD': 'admin'
                    },
                    'volumes': [
                        './observability/grafana/provisioning:/etc/grafana/provisioning',
                        './observability/main-dashboard.json:/var/lib/grafana/dashboards/main.json',
                        './observability/performance-dashboard.json:/var/lib/grafana/dashboards/performance.json'
                    ]
                },
                'jaeger': {
                    'image': 'jaegertracing/all-in-one:latest',
                    'ports': ['16686:16686', '14268:14268'],
                    'environment': {
                        'COLLECTOR_OTLP_ENABLED': 'true'
                    }
                },
                'alertmanager': {
                    'image': 'prom/alertmanager:latest',
                    'ports': ['9093:9093'],
                    'volumes': ['./observability/alertmanager.yml:/etc/alertmanager/alertmanager.yml'],
                    'command': ['--config.file=/etc/alertmanager/alertmanager.yml']
                }
            },
            'volumes': {
                'prometheus_data': {}
            }
        }
        
        filepath = self.observability_dir / 'docker-compose.yml'
        with open(filepath, 'w') as f:
            yaml.dump(observability_compose, f, default_flow_style=False)
        logger.info(f"Generated observability Docker Compose: {filepath}")
        
    def setup_observability_stack(self):
        """Set up the observability stack."""
        logger.info("Setting up observability stack")
        
        try:
            # Start observability services
            compose_file = self.observability_dir / 'docker-compose.yml'
            if compose_file.exists():
                import subprocess
                subprocess.run(['docker-compose', '-f', str(compose_file), 'up', '-d'], check=True)
                logger.info("Observability stack started successfully")
                
                # Wait for services to be ready
                time.sleep(10)
                
                # Check service health
                services = {
                    'Prometheus': 'http://localhost:9090/-/healthy',
                    'Grafana': 'http://localhost:3000/api/health',
                    'Jaeger': 'http://localhost:16686/api/services'
                }
                
                for service, url in services.items():
                    try:
                        response = requests.get(url, timeout=5)
                        if response.status_code == 200:
                            logger.info(f"{service} is healthy")
                        else:
                            logger.warning(f"{service} returned status {response.status_code}")
                    except Exception as e:
                        logger.warning(f"Failed to check {service} health: {e}")
                        
            else:
                logger.warning("Docker Compose file not found. Please generate observability stack first.")
                
        except Exception as e:
            logger.error(f"Failed to set up observability stack: {e}")


def main():
    """Main function to run observability integration."""
    # Configuration
    config = {
        'observability_dir': 'observability',
        'jaeger_endpoint': 'http://localhost:14268/api/traces',
        'grafana_endpoint': 'http://localhost:3000'
    }
    
    # Create integration
    integration = ObservabilityIntegration(config)
    
    # Generate all observability configurations
    logger.info("Generating observability integration configurations")
    integration.generate_jaeger_integration()
    integration.generate_grafana_dashboards()
    integration.generate_prometheus_alerts()
    integration.generate_logging_config()
    integration.generate_docker_compose_observability()
    
    # Set up observability stack
    setup = input("Set up observability stack? (y/n): ").lower().strip()
    if setup == 'y':
        integration.setup_observability_stack()
    else:
        logger.info("Skipping observability stack setup. Configurations generated in observability/ directory")
    
    logger.info("Observability integration completed")


if __name__ == "__main__":
    main() 