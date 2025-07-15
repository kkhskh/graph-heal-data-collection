import requests
import time
import logging
import sys
import json
import os
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path
import pandas as pd

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Define the services using localhost and the mapped ports ---
SERVICES = {
    "service_a": "http://localhost:5001",
    "service_b": "http://localhost:5002",
    "service_c": "http://localhost:5003",
    "service_d": "http://localhost:5004",
}

class EnhancedMonitor:
    """Enhanced monitoring system with event logging capabilities."""
    
    def __init__(self):
        """Initialize the enhanced monitor."""
        self.event_log = []
        self.service_history = {}
        self.anomaly_thresholds = {
            'cpu_usage': 80.0,
            'memory_usage': 90.0,
            'response_time': 1.0
        }
        
        # Create event log directory
        self.log_dir = Path("data/monitoring_events")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize service history
        for service_name in SERVICES.keys():
            self.service_history[service_name] = {
                'metrics_history': [],
                'last_health_status': 'healthy',
                'anomaly_count': 0
            }
    
    def parse_metrics(self, metrics_text: str) -> Dict[str, float]:
        """Parse Prometheus-style metrics text into a dictionary."""
        metrics = {}
        for line in metrics_text.strip().split('\n'):
            if line and not line.startswith('#'):
                try:
                    # Parse metric line (e.g., "cpu_usage 45.2" or "cpu_usage{service="a"} 45.2")
                    if '{' in line:
                        # Handle labeled metrics
                        metric_part = line.split('{')[0]
                        value_part = line.split('}')[-1].strip()
                        metric_name = metric_part.strip()
                        metric_value = float(value_part.split()[0])
                    else:
                        # Handle simple metrics
                        parts = line.split()
                        if len(parts) >= 2:
                            metric_name = parts[0]
                            metric_value = float(parts[1])
                        else:
                            continue
                    
                    metrics[metric_name] = metric_value
                except (ValueError, IndexError):
                    continue
        return metrics
    
    def detect_anomaly(self, service_name: str, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Detect anomalies using statistical thresholds."""
        anomalies = []
        
        # Check CPU usage
        if 'service_cpu_usage' in metrics:
            cpu_usage = metrics['service_cpu_usage']
            if cpu_usage > self.anomaly_thresholds['cpu_usage']:
                anomalies.append({
                    'type': 'high_cpu',
                    'value': cpu_usage,
                    'threshold': self.anomaly_thresholds['cpu_usage']
                })
        
        # Check memory usage
        if 'service_memory_usage' in metrics:
            memory_usage = metrics['service_memory_usage']
            if memory_usage > self.anomaly_thresholds['memory_usage']:
                anomalies.append({
                    'type': 'high_memory',
                    'value': memory_usage,
                    'threshold': self.anomaly_thresholds['memory_usage']
                })
        
        # Check response time
        if 'service_response_time' in metrics:
            response_time = metrics['service_response_time']
            if response_time > self.anomaly_thresholds['response_time']:
                anomalies.append({
                    'type': 'high_latency',
                    'value': response_time,
                    'threshold': self.anomaly_thresholds['response_time']
                })
        
        return {
            'service': service_name,
            'timestamp': time.time(),
            'anomalies': anomalies,
            'has_anomaly': len(anomalies) > 0
        }
    
    def determine_health_status(self, metrics: Dict[str, float]) -> str:
        """Determine health status based on metrics."""
        if 'service_cpu_usage' in metrics and metrics['service_cpu_usage'] > 90:
            return "critical"
        elif 'service_cpu_usage' in metrics and metrics['service_cpu_usage'] > 70:
            return "degraded"
        elif 'service_memory_usage' in metrics and metrics['service_memory_usage'] > 85:
            return "warning"
        else:
            return "healthy"
    
    def log_detection_event(self, service_name: str, anomaly_data: Dict[str, Any]):
        """Log a detection event with timestamp."""
        event = {
            "event_type": "detection",
            "service": service_name,
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "anomaly_data": anomaly_data,
            "detection_method": "statistical_threshold"
        }
        
        self.event_log.append(event)
        logging.info(f"🔍 DETECTION: {service_name} - {anomaly_data['anomalies']}")
        
        # Update service history
        self.service_history[service_name]['anomaly_count'] += 1
    
    def log_localization_event(self, service_name: str, health_status: str, metrics: Dict[str, float]):
        """Log a localization event with timestamp."""
        event = {
            "event_type": "localization",
            "service": service_name,
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "health_status": health_status,
            "metrics": metrics,
            "localization_method": "health_status_analysis"
        }
        
        self.event_log.append(event)
        logging.info(f"🎯 LOCALIZATION: {service_name} - {health_status}")
    
    def log_recovery_event(self, service_name: str, action: str, success: bool):
        """Log a recovery event with timestamp."""
        event = {
            "event_type": "recovery",
            "service": service_name,
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "action": action,
            "success": success,
            "recovery_method": "automated_orchestration"
        }
        
        self.event_log.append(event)
        status = "✅" if success else "❌"
        logging.info(f"{status} RECOVERY: {service_name} - {action}")
    
    def log_metric_collection(self, service_name: str, metrics: Dict[str, float]):
        """Log metric collection event."""
        event = {
            "event_type": "metric_collection",
            "service": service_name,
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "metrics": metrics
        }
        
        # Store in service history
        self.service_history[service_name]['metrics_history'].append({
            'timestamp': time.time(),
            'metrics': metrics
        })
        
        # Keep only last 100 entries to prevent memory bloat
        if len(self.service_history[service_name]['metrics_history']) > 100:
            self.service_history[service_name]['metrics_history'] = \
                self.service_history[service_name]['metrics_history'][-100:]
    
    def save_metrics_to_csv(self, output_file: str = 'metric_data.csv'):
        """Extracts and saves all collected metrics to a CSV file."""
        records = []
        # First, find all unique metric names across all services
        all_metric_names = set()
        for service_name, history in self.service_history.items():
            for entry in history['metrics_history']:
                all_metric_names.update(entry['metrics'].keys())
        
        # Now, create a record for each timestamp
        for service_name, history in self.service_history.items():
            for entry in history['metrics_history']:
                record = {'timestamp': entry['timestamp'], 'service_name': service_name}
                # Ensure all columns are present for a consistent CSV structure
                for metric_name in all_metric_names:
                    record[metric_name] = entry['metrics'].get(metric_name)
                records.append(record)
        
        if not records:
            logging.warning("No metric records to save to CSV.")
            return

        df = pd.DataFrame(records)
        # Pivot the table to get services as columns for each metric, which might be a better format
        # For now, keeping a simple flat structure.
        df.to_csv(output_file, index=False)
        logging.info(f"Metrics data saved to {output_file}")


    def save_event_log(self):
        """Save the event log to a file."""
        timestamp = int(datetime.now().timestamp())
        log_file = self.log_dir / f"monitoring_events_{timestamp}.json"
        
        log_data = {
            "monitoring_session": {
                "start_time": datetime.now().isoformat(),
                "services_monitored": list(SERVICES.keys()),
                "total_events": len(self.event_log)
            },
            "events": self.event_log,
            "service_history": self.service_history
        }
        
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        logging.info(f"Event log saved to: {log_file}")
        return log_file
    
    def get_event_stream(self) -> List[Dict[str, Any]]:
        """Get real-time event stream."""
        return self.event_log.copy()
    
    def monitor_services(self):
        """Main monitoring loop with enhanced event logging."""
        logging.info("Starting enhanced monitor. Polling services every 2 seconds.")
        
        try:
            while True:
                for service_name, base_url in SERVICES.items():
                    try:
                        # Fetch metrics
                        resp = requests.get(f"{base_url}/metrics", timeout=1.0)
                        resp.raise_for_status()
                        
                        # Parse metrics
                        metrics = self.parse_metrics(resp.text)
                        
                        # Log metric collection
                        self.log_metric_collection(service_name, metrics)
                        
                        # Detect anomalies
                        anomaly_data = self.detect_anomaly(service_name, metrics)
                        if anomaly_data['has_anomaly']:
                            self.log_detection_event(service_name, anomaly_data)
                        
                        # Determine health status
                        current_health = self.determine_health_status(metrics)
                        previous_health = self.service_history[service_name]['last_health_status']
                        
                        # Log health status changes
                        if current_health != previous_health:
                            self.log_localization_event(service_name, current_health, metrics)
                            self.service_history[service_name]['last_health_status'] = current_health
                        
                        # Display metrics (optional, for debugging)
                        if os.getenv('VERBOSE_MONITORING'):
                            logging.info(f"--- Metrics from {service_name} ---")
                            logging.info(resp.text.strip())
                        
                    except requests.exceptions.RequestException as e:
                        logging.error(f"Could not connect to {service_name}: {e}")
                
                time.sleep(2)
                
        except KeyboardInterrupt:
            logging.info("Monitoring stopped by user.")
        finally:
            self.save_event_log()
            self.save_metrics_to_csv()

def main():
    """Main function to run the enhanced monitor."""
    monitor = EnhancedMonitor()
    monitor.monitor_services()

if __name__ == "__main__":
    main() 