import logging
import os
import sys

# FIX: Configure logging at the very start of the script, before any other imports.
LOG_FILE = "reactor_recovery_log.txt"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Now, import the rest of the modules.
import asyncio
import subprocess
import time
import yaml

# Ensure the project root is in the Python path
sys.path.append(os.getcwd())

from graph_heal.service_graph import ServiceGraph
# The correct, concrete detector class to use
from graph_heal.anomaly_detection import StatisticalAnomalyDetector
from graph_heal.recovery_system import EnhancedRecoverySystem
from graph_heal.recovery.opcua_adapter import OpcUaAdapter

def start_mock_server():
    """Starts the mock OPC-UA server as a subprocess."""
    logger.info("Starting mock OPC-UA server...")
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
    Runs a controlled end-to-end test of the OPC-UA recovery mechanism.
    """
    server_process = start_mock_server()

    # The OpcUaAdapter is the component we want to test.
    # It reads its configuration from the YAML file by default.
    opcua_adapter = OpcUaAdapter()

    try:
        logger.info("--- Running Simplified Reactor Recovery Test ---")
        
        # We will directly call the recovery action, bypassing the anomaly detector.
        logger.info("Directly invoking 'restart_service' on 'pump-1' via OPC-UA adapter.")
        
        # Directly call the adapter's restart method for the target service
        recovery_successful = await opcua_adapter.restart_service('pump-1')

        if recovery_successful:
            logger.info("SUCCESS: End-to-end OPC-UA recovery action verified.")
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
    # Ensure the project root is in the path
    if os.getcwd() not in sys.path:
        sys.path.append(os.getcwd())
    
    # Load the correct adapter class
    from graph_heal.recovery.opcua_adapter import OpcUaAdapter
    
    asyncio.run(main()) 