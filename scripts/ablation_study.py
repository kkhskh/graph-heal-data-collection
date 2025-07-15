import sys
import os
import time
import docker
import logging
import subprocess
from typing import Dict, Any, List
import pandas as pd
from collections import defaultdict
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from graph_heal.service_graph import ServiceGraph
from graph_heal.service_monitor import ServiceMonitor
from graph_heal.anomaly_detection import StatisticalAnomalyDetector
from graph_heal.fault_localization import GraphBasedFaultLocalizer, NaiveFaultLocalizer
from graph_heal.recovery_system import EnhancedRecoverySystem
from graph_heal.recovery.base import RecoverySystemAdapter

# --- Dummy Adapter ---
class DummyRecoveryAdapter(RecoverySystemAdapter):
    def restart_service(self, service_id: str) -> bool:
        logger.info(f"[Dummy] Restarting service {service_id}")
        return True
    def scale_service(self, service_id: str, replicas: int) -> bool:
        logger.info(f"[Dummy] Scaling service {service_id} to {replicas} replicas")
        return True
    def isolate_service(self, service_id: str, isolate: bool) -> bool:
        logger.info(f"[Dummy] Setting isolation for service {service_id} to {isolate}")
        return True
    def degrade_service(self, service_id: str) -> bool:
        logger.info(f"[Dummy] Degrading service {service_id}")
        return True
    def get_service_status(self, service_id: str) -> dict:
        return {"status": "dummy"}

# --- Configuration ---
SERVICES = {
    "service-a": {"port": 5001, "dependencies": []},
    "service-b": {"port": 5002, "dependencies": ["service-a"]},
    "service-c": {"port": 5003, "dependencies": ["service-b"]},
    "service-d": {"port": 5004, "dependencies": ["service-c"]},
}
FAULT_TARGET = 'service-d'
FAULT_TYPE = 'cpu_stress'
FAULT_PARAMS = {"duration": 30, "severity": 1}
EXPERIMENT_DURATION_SECS = 120
MONITORING_INTERVAL_SECS = 5

# --- Logging ---
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('AblationStudy')

def setup_service_graph() -> ServiceGraph:
    """Initializes the service graph with known dependencies."""
    g = ServiceGraph()
    # Dependencies based on the docker-compose file
    g.add_dependency('service-a', 'service-b')
    g.add_dependency('service-a', 'service-c')
    g.add_dependency('service-b', 'service-d')
    logger.info(f"SETUP: Graph edges: {list(g.graph.edges())}")
    return g

def calculate_localization_metrics(localized_faults: List[Dict[str, Any]], ground_truth: str) -> Dict[str, float]:
    """Calculates precision and recall for fault localization."""
    if not localized_faults:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    tp = sum(1 for fault in localized_faults if fault.get('service_id') == ground_truth)
    fp = len(localized_faults) - tp
    fn = 1 - (1 if tp > 0 else 0)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {'precision': precision, 'recall': recall, 'f1': f1}

def inject_docker_fault(service_name, fault_type, duration=30):
    """Injects a fault into a container using docker exec."""
    if fault_type != 'cpu_stress':
        raise NotImplementedError(f"Fault type '{fault_type}' not supported by this injector.")
    
    # Find the container ID using docker-compose labels
    try:
        # For debugging, let's see what containers are running
        ps_result = subprocess.run(['docker-compose', 'ps'], capture_output=True, text=True)
        logger.debug(f"docker-compose ps output:\\n{ps_result.stdout}")

        # Use the service label to find the container
        result = subprocess.run(
            ['docker', 'ps', '-q', '--filter', f"label=com.docker.compose.service={service_name}"],
            capture_output=True, text=True
        )

        if result.stdout.strip():
            container_id = result.stdout.strip().splitlines()[0] # Take the first one if multiple
        else:
            logger.error(f"Could not find container for service '{service_name}'.")
            return None

        # Inject the fault
        cmd = [
            "docker", "exec", container_id,
            "stress-ng", "--cpu", "1", "--cpu-load", "80", "--timeout", f"{duration}s"
        ]
        subprocess.Popen(cmd)
        logger.info(f"Injected cpu_stress fault into {service_name} (container: {container_id})")
        return container_id
    except Exception as e:
        logger.error(f"Failed to inject fault into {service_name}: {e}")
        return None

def remove_docker_fault(container_id):
    """Removes a fault by killing the stress process."""
    try:
        # pkill lives in /bin in Alpine & Debian alike – rely on $PATH, and ignore exit-status
        subprocess.run(["docker", "exec", container_id, "pkill", "-f", "stress-ng"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logger.info(f"Attempted to remove cpu_stress fault from container {container_id}")
    except Exception as e:
        logger.error(f"Failed to remove fault from container {container_id}: {e}")

def wait_for_containers(timeout=300):
    """Waits for all containers in the docker-compose setup to be running."""
    logger.info("Waiting for all services to be up and running...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            # Get the status of all services
            result = subprocess.run(['docker-compose', 'ps', '--services'], capture_output=True, text=True, check=True)
            services = [s for s in result.stdout.strip().splitlines() if s]
            
            if not services:
                logger.warning("No services found in docker-compose. Retrying...")
                time.sleep(5)
                continue

            # Check status of each container
            is_all_running = True
            print("--- Health Check Tick ---")
            for service in services:
                res = subprocess.run(['docker-compose', 'ps', '-q', service], capture_output=True, text=True)
                container_id = res.stdout.strip()
                if not container_id:
                    is_all_running = False
                    print(f"DEBUG: Service {service} container not found yet.")
                    break
                    
                # Further check if it's healthy
                res_health = subprocess.run(
                    ['docker', 'inspect', "--format={{json .State.Health}}", container_id],
                    capture_output=True, text=True
                )
                health_status = "unknown"
                try:
                    import json
                    # The health check might not be present if the container doesn't define one.
                    # Handle this gracefully.
                    if res_health.stdout.strip() and res_health.stdout.strip() != 'null':
                        health_info = json.loads(res_health.stdout.strip())
                        health_status = health_info.get("Status", "not_present")
                    else:
                        # If no health check is defined, consider it 'healthy' for our purposes.
                        health_status = 'healthy' 
                except json.JSONDecodeError:
                     # If there's no healthcheck, docker inspect returns 'null' which is not valid JSON.
                     # In this case, we can assume the container is healthy if it's running.
                    health_status = 'healthy'
                except Exception as e:
                    logger.warning(f"Could not parse health status for {service}: {e}. Assuming healthy.")
                    health_status = 'healthy'


                if health_status != 'healthy':
                    is_all_running = False
                    print(f"DEBUG: Service {service} is up but not healthy. Status: {health_status}")
                    break
            
            if is_all_running:
                logger.info("All services are up and healthy.")
                return True
        except Exception as e:
            logger.error(f"Error while waiting for containers: {e}")
        
        time.sleep(10)
        
    logger.error("Timeout waiting for containers to start.")
    return False

def run_single_experiment(localizer_name: str, use_policy_table: bool, service_graph: ServiceGraph, monitor: ServiceMonitor) -> Dict[str, Any]:
    """Runs a single experiment with a specific configuration."""
    logger.info(f"--- Starting Experiment: Localizer={localizer_name}, UsePolicy={use_policy_table} ---")
    logger.info(f"EXPERIMENT_START: Graph edges: {list(service_graph.graph.edges())}")
    
    # 1. Initialization
    recovery_system = EnhancedRecoverySystem(
        service_graph,
        adapter=None, 
        use_policy_table=use_policy_table
    )
    anomaly_detector = StatisticalAnomalyDetector()
    if localizer_name == 'GraphBased':
        fault_localizer = GraphBasedFaultLocalizer(service_graph)
    else: # Naive
        fault_localizer = NaiveFaultLocalizer()

    mttr = float('nan')
    localization_metrics = {}
    
    # 2. Start Monitoring
    monitor.start_monitoring()
    
    # 3. Inject Fault
    fault_injection_time = time.time()
    try:
        inject_docker_fault('service-d', 'cpu_stress', duration=10)
    except Exception as e:
        logger.error(f"Fault injection failed: {e}")
        monitor.stop_monitoring()
        return {'mttr': float('nan'), 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    # 4. Main Loop: Detect, Localize, Recover
    start_time = time.time()
    while time.time() - start_time < 20: # Run for 20 seconds
        time.sleep(5)
        logger.info("--- Monitoring tick ---")
        
        all_statuses = monitor.get_all_services_status()
        
        # Manually update the graph with the latest metrics for each service
        for service_id, status in all_statuses.items():
            if 'timestamp' in status and 'metrics' in status:
                service_graph.add_metrics(service_id, status['metrics'], datetime.fromtimestamp(status['timestamp']))
        
        anomalies = anomaly_detector.detect_anomalies(all_statuses)
        
        if anomalies:
            logger.info(f"Detected {len(anomalies)} anomalies.")
            
            # Localize the fault
            print(f"DEBUG: Anomalies detected for localization: {anomalies}")
            print(f"DEBUG: System status for localization: {all_statuses}")
            localized_faults = fault_localizer.localize_faults(
                service_statuses=all_statuses, 
                anomalies=anomalies
            )
            logger.info(f"Localization raw results (Naive): {localized_faults}")

            if localized_faults and not localization_metrics:
                localization_metrics = calculate_localization_metrics(localized_faults, ground_truth='service-d')
                logger.info(f"Localization metrics: {localization_metrics}")

            # Execute recovery plan if a fault is localized and not already recovered
            if localized_faults and pd.isna(mttr):
                fault_to_recover = localized_faults[0]
                plan = recovery_system.get_recovery_plan(fault_to_recover['service_id'], fault_to_recover.get('type'))
                
                if plan:
                    logger.info(f"Executing recovery plan for {fault_to_recover['service_id']}: {[a.action_type.name for a in plan]}")
                    success = any(recovery_system.execute_recovery_action(action) for action in plan)
                    
                    if success:
                        mttr = time.time() - fault_injection_time
                        logger.info(f"Recovery successful! MTTR: {mttr:.2f}s")
                        # For this study, we break after first recovery to measure MTTR
                        break

    # 5. Cleanup
    logger.info("Cleaning up injected faults...")
    remove_docker_fault('service-d')
    monitor.stop_monitoring()
    
    # Return results, ensuring metrics are not empty
    final_metrics = localization_metrics or {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    final_metrics['mttr'] = mttr if not pd.isna(mttr) else float('inf')
    return final_metrics

def main():
    """Main function to run the ablation study."""
    logger.info("====== Starting Ablation Study ======")
    
    # Define configurations for the ablation study
    configs = {
        "Baseline (Full System)": {"localizer": "GraphBased", "policy": True},
        "No Topological Localizer": {"localizer": "Naive", "policy": True},
        "No Policy Orchestrator": {"localizer": "GraphBased", "policy": False},
    }
    
    results_df = pd.DataFrame(columns=["mttr", "precision", "recall", "f1"])
    
    # Setup services and monitoring once for all experiments
    logger.info("Starting services via docker-compose...")
    try:
        subprocess.run(['docker-compose', 'up', '-d', '--build'], check=True)
        if not wait_for_containers():
            raise RuntimeError("Containers did not become healthy in time.")
        
        service_graph = setup_service_graph()
        
        services_for_monitor = []
        for name, config in SERVICES.items():
            services_for_monitor.append({
                "id": name,
                "url": f"http://localhost:{config['port']}",
                "health_endpoint": "/metrics"
            })
        
        monitor = ServiceMonitor(services_for_monitor)

        for name, config in configs.items():
            experiment_results = run_single_experiment(
                localizer_name=config["localizer"],
                use_policy_table=config["policy"],
                service_graph=service_graph,
                monitor=monitor
            )
            results_df.loc[name] = experiment_results
            
    except Exception as e:
        logger.error(f"An error occurred in the main study loop: {e}", exc_info=True)
    finally:
        logger.info("Stopping all services...")
        subprocess.run(['docker-compose', 'down', '-v', '--remove-orphans'], check=True)

    logger.info("====== Ablation Study Results ======")
    logger.info("\n" + results_df.to_string())
    logger.info("====== Ablation Study Complete ======")

if __name__ == "__main__":
    main()