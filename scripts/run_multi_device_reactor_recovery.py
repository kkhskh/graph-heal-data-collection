import logging
import os
import sys

# Configure logging at the very start of the script
LOG_FILE = "multi_device_reactor_log.txt"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Now, import the rest of the modules
import asyncio
import subprocess
import time
import yaml

from graph_heal.service_graph import ServiceGraph
from graph_heal.anomaly_detection import StatisticalAnomalyDetector
from graph_heal.recovery_system import EnhancedRecoverySystem
from graph_heal.recovery.opcua_adapter import OpcUaAdapter

def start_mock_server():
    """Starts the mock OPC-UA server as a subprocess."""
    logger.info("Starting mock OPC-UA server for multi-device test...")
    python_executable = sys.executable
    server_process = subprocess.Popen(
        [python_executable, "scripts/run_mock_opcua_server.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    time.sleep(4) # Give the server a moment to initialize
    logger.info("Mock OPC-UA server started.")
    return server_process

async def main():
    """
    Runs an end-to-end test with multiple OPC-UA devices to verify
    targeted fault localization and recovery.
    """
    server_process = start_mock_server()

    # --- Setup Graph-Heal components for multiple devices ---
    services_to_monitor = ['pump-1', 'valve-3']
    
    service_graph = ServiceGraph()
    service_graph.add_service('pump-1', dependencies=['valve-3'])
    service_graph.add_service('valve-3', dependencies=[])
    
    anomaly_detector = StatisticalAnomalyDetector(services=services_to_monitor, window_size=10, threshold=0.2)

    opcua_adapter = OpcUaAdapter(config_path='config/opcua_mapping.yaml')
    recovery_system = EnhancedRecoverySystem(service_graph, adapter=opcua_adapter)

    try:
        logger.info("--- Running Multi-Device Recovery Simulation ---")
        
        # Prime the detector with normal data for BOTH devices
        logger.info(f"Priming detector for {services_to_monitor} with normal metrics...")
        for i in range(10):
            normal_metrics = {
                'pump-1': {'cpu_usage': 10 + (i % 3)},
                'valve-3': {'cpu_usage': 20 + (i % 2)}
            }
            anomaly_detector.detect_anomalies(normal_metrics, time.time())
            await asyncio.sleep(0.1)
        
        logger.info("Injecting synthetic fault for 'pump-1' ONLY.")
        faulty_metrics = {
            'pump-1': {'cpu_usage': 80}, # High CPU for the pump
            'valve-3': {'cpu_usage': 21}  # Normal CPU for the valve
        }
        
        anomalies = anomaly_detector.detect_anomalies(faulty_metrics, time.time())

        if not anomalies:
            logger.error("FAILURE: Anomaly was not detected.")
            return

        logger.info(f"SUCCESS: Anomaly detected: {[anom['service_id'] for anom in anomalies]}")

        # --- Verification ---
        if len(anomalies) > 1:
            logger.error(f"FAILURE: Expected only one anomaly, but got {len(anomalies)}. Fault localization is incorrect.")
            return
            
        root_cause_service = anomalies[0]['service_id']
        if root_cause_service != 'pump-1':
            logger.error(f"FAILURE: Incorrect root cause identified. Expected 'pump-1', but got '{root_cause_service}'.")
            return
        
        logger.info(f"Anomaly confirmed for '{root_cause_service}'. Triggering recovery...")
        
        recovery_plan = recovery_system.get_recovery_plan(root_cause_service, fault_type='cpu')
        
        if not recovery_plan:
            logger.error("FAILURE: No recovery plan was generated.")
            return

        logger.info(f"Recovery plan generated: {[action.action_type.value for action in recovery_plan]}")
        
        recovery_successful = await opcua_adapter.restart_service(recovery_plan[0].target_service)

        if recovery_successful:
            logger.info("SUCCESS: End-to-end multi-device OPC-UA recovery verified.")
        else:
            logger.error("FAILURE: OPC-UA recovery action failed.")

    finally:
        logger.info("Shutting down mock OPC-UA server.")
        server_process.terminate()
        try:
            outs, errs = server_process.communicate(timeout=5)
            if outs: logger.info("Server stdout:\n" + outs)
            if errs: logger.error("Server stderr:\n" + errs)
        except subprocess.TimeoutExpired:
            server_process.kill()
        
        logger.info(f"Simulation complete. Check '{LOG_FILE}' for details.")


if __name__ == "__main__":
    asyncio.run(main()) 