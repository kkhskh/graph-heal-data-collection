import logging
import os
import sys
import subprocess
import time
import json
import paho.mqtt.client as mqtt

# Configure logging
LOG_FILE = "mqtt_recovery_log.txt"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Ensure the project root is in the Python path
sys.path.append(os.getcwd())

from graph_heal.service_graph import ServiceGraph
from graph_heal.recovery_system import EnhancedRecoverySystem
from graph_heal.recovery.mqtt_adapter import MqttAdapter

# --- Configuration ---
BROKER_ADDRESS = "localhost"
DEVICE_ID = "motor-123"
STATUS_TOPIC = f"devices/{DEVICE_ID}/status"

# --- Shared state for monitor ---
fault_detected = False

def on_status_message(client, userdata, msg):
    """Callback for the monitor client when a status message is received."""
    global fault_detected
    try:
        payload = json.loads(msg.payload.decode())
        logger.info(f"Monitor received status: {payload}")
        if payload.get("status") == "faulty":
            logger.warning("FAULT DETECTED in mock device!")
            fault_detected = True
    except json.JSONDecodeError:
        logger.error("Could not decode status payload.")

def start_mock_device():
    """Starts the mock MQTT device as a subprocess."""
    logger.info("Starting mock MQTT device...")
    python_executable = sys.executable
    process = subprocess.Popen(
        [python_executable, "scripts/run_mock_mqtt_device.py"],
        text=True
    )
    time.sleep(2) # Give it a moment to connect
    return process

def main():
    """Runs the end-to-end MQTT recovery experiment."""
    
    # Start the external device
    device_process = start_mock_device()

    # Setup a separate client to monitor the device's status
    monitor_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    monitor_client.on_message = on_status_message
    try:
        monitor_client.connect(BROKER_ADDRESS)
    except ConnectionRefusedError:
        logger.error(f"Connection to MQTT broker at {BROKER_ADDRESS} refused. Is a broker running?")
        device_process.terminate()
        return

    monitor_client.subscribe(STATUS_TOPIC)
    monitor_client.loop_start()

    # Setup Graph-Heal components
    service_graph = ServiceGraph()
    service_graph.add_service(DEVICE_ID)
    mqtt_adapter = MqttAdapter(broker_address=BROKER_ADDRESS)
    recovery_system = EnhancedRecoverySystem(service_graph, adapter=mqtt_adapter)

    try:
        logger.info("--- MQTT E2E Test Running ---")
        
        # 1. Inject a fault by publishing a "faulty" status message
        logger.info("Injecting fault by publishing a 'faulty' status...")
        fault_payload = json.dumps({"status": "faulty", "temperature": 95.0})
        monitor_client.publish(STATUS_TOPIC, fault_payload)
        
        # 2. Wait for the monitor to detect the fault
        time.sleep(1)
        if not fault_detected:
            logger.error("FAILURE: Monitor did not detect the injected fault.")
            return

        # 3. Trigger recovery
        logger.info(f"Fault detected for '{DEVICE_ID}'. Triggering recovery...")
        plan = recovery_system.get_recovery_plan(DEVICE_ID, fault_type="device_fault")
        if not plan:
            logger.error("FAILURE: No recovery plan was generated.")
            return
            
        action = plan[0]
        logger.info(f"Executing recovery action: {action.action_type.value} on {action.target_service}")
        success = recovery_system.execute_recovery_action(action)
        
        if success:
            logger.info("SUCCESS: MQTT-based recovery action executed successfully.")
        else:
            logger.error("FAILURE: MQTT recovery action failed.")
            
    finally:
        logger.info("--- Test Finished. Cleaning up. ---")
        monitor_client.loop_stop()
        monitor_client.disconnect()
        device_process.terminate()
        device_process.wait()
        logger.info(f"Simulation complete. Check '{LOG_FILE}' for details.")

if __name__ == "__main__":
    main() 