import asyncio
import logging
import yaml
from asyncua import Server, ua

# Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def reset_pump(parent, *variants):
    """A mock OPC-UA method that simulates resetting a pump."""
    logging.info(f"!!! Method called via node: {parent} with variants {variants} !!!")
    return [ua.Variant(0, ua.VariantType.Int64)] # Return a success code

async def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Load the config to get node IDs
    try:
        with open('config/opcua_mapping.yaml', 'r') as f:
            config = yaml.safe_load(f)
        server_url = config['server_url']
        node_map = config['nodes']
        logging.info("Loaded OPC-UA mapping configuration.")
    except Exception as e:
        logging.error(f"Failed to load or parse config/opcua_mapping.yaml: {e}")
        return

    # Setup our server
    server = Server()
    await server.init()
    # Import the DI nodeset to avoid harmless warnings
    try:
        await server.import_xml(
            "/Users/shkh/opt/anaconda3/envs/graph-heal-env/lib/python3.9/site-packages/asyncua/nodesets/Opc.Ua.Di.NodeSet2.xml"
        )
    except FileNotFoundError:
        logging.warning("DI nodeset file not found, skipping import. Harmless warnings may appear.")
    # Force endpoint to 0.0.0.0 to avoid localhost resolution issues
    server.set_endpoint("opc.tcp://0.0.0.0:4840/freeopcua/server/")

    logging.info("Starting OPC-UA server at {server.endpoint}...")
    logging.info("Server is now running. Press Ctrl-C to stop.")

    async with server:
        # Setup our own namespace
        uri = "http://examples.freeopcua.github.io"
        idx = await server.register_namespace(uri)

        # --- Create nodes based on the mapping config ---
        for service_name, node_ids in node_map.items():
            # The object name is the part of the ID before the dot.
            # e.g., "Pump1" from "ns=2;s=Pump1.Reset"
            try:
                string_id = node_ids['restart_method_id'].split('=')[-1]
                object_name = string_id.split('.')[0]
                
                # Create the parent object if it doesn't exist yet
                obj_node_id = f"ns={idx};s={object_name}"
                obj = await server.nodes.objects.add_object(obj_node_id, service_name)
                logging.info(f"Created/found object '{service_name}' with NodeId '{obj_node_id}'")

                # Add the method
                method_id = node_ids['restart_method_id']
                method_name = method_id.split('.')[-1]
                await obj.add_method(idx, method_name, reset_pump, [], [ua.VariantType.Int64])
                logging.info(f"Added method '{method_name}' to object '{service_name}'")

                # Add the variable for isolation
                var_id = node_ids['isolate_variable_id']
                var_name = var_id.split('.')[-1]
                var = await obj.add_variable(var_id, var_name, True)
                await var.set_writable(True)
                logging.info(f"Added variable '{var_name}' with NodeId '{var_id}'")

            except (KeyError, IndexError) as e:
                logging.error(f"Skipping malformed entry for '{service_name}': {e}")
                continue
        
        while True:
            await asyncio.sleep(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Server shutting down due to user interruption.") 