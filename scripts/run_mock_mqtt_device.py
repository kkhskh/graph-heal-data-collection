import paho.mqtt.client as mqtt
import time
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
BROKER_ADDRESS = "localhost"
COMMAND_TOPIC = "devices/motor-123/command"
STATUS_TOPIC = "devices/motor-123/status"

# --- Device State ---
device_state = {
    "status": "running",
    "temperature": 25.0
}

def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        logger.info("Connected to MQTT Broker!")
        client.subscribe(COMMAND_TOPIC)
        logger.info(f"Subscribed to command topic: {COMMAND_TOPIC}")
    else:
        logger.error(f"Failed to connect, return code {rc}\n")

def on_message(client, userdata, msg):
    """Callback for when a message is received from the command topic."""
    command = msg.payload.decode()
    logger.info(f"Received command: '{command}' on topic '{msg.topic}'")
    
    global device_state
    if command == "restart":
        logger.info("Restarting device...")
        device_state["status"] = "restarting"
        # Publish the change in status
        client.publish(STATUS_TOPIC, json.dumps(device_state))
        time.sleep(2) # Simulate restart time
        device_state["status"] = "running"
        device_state["temperature"] = 25.0
        logger.info("Device is now running.")
        client.publish(STATUS_TOPIC, json.dumps(device_state))
    else:
        logger.warning(f"Unknown command: {command}")

def main():
    """Main function to run the mock device."""
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    
    try:
        client.connect(BROKER_ADDRESS)
    except ConnectionRefusedError:
        logger.error(f"Connection to MQTT broker at {BROKER_ADDRESS} refused. Is a broker running?")
        return

    # Start the network loop in a non-blocking way
    client.loop_start()
    
    logger.info(f"Mock MQTT device 'motor-123' is running. Publishing status to {STATUS_TOPIC}.")
    
    try:
        while True:
            # Periodically publish the device's status
            # In a real scenario, this would be sensor data
            device_state["temperature"] += 0.5
            client.publish(STATUS_TOPIC, json.dumps(device_state))
            time.sleep(5)
    except KeyboardInterrupt:
        logger.info("Shutting down mock device.")
    finally:
        client.loop_stop()
        client.disconnect()

if __name__ == "__main__":
    main() 