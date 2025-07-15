from .base import RecoverySystemAdapter
import logging
import yaml
from asyncua import Client, ua

logger = logging.getLogger(__name__)

class OpcUaAdapter(RecoverySystemAdapter):
    """
    Recovery adapter for interacting with an OPC-UA server.
    Implements recovery actions by calling methods and writing to variables
    on industrial devices.
    """
    def __init__(self, config_path: str = 'config/opcua_mapping.yaml'):
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.server_url = config['server_url']
            self.node_map = config['nodes']
            logger.info(f"Loaded OPC-UA config from {config_path}")
        except FileNotFoundError:
            logger.error(f"OPC-UA config file not found at {config_path}")
            raise
        except Exception as e:
            logger.error(f"Error parsing OPC-UA config file: {e}")
            raise
        
        # Client will be initialized on demand
        self.client = None

    async def _connect(self):
        """Initializes and connects the OPC-UA client."""
        if self.client:
            return
        try:
            self.client = Client(url=self.server_url)
            await self.client.connect()
            logger.info(f"Successfully connected to OPC-UA server at {self.server_url}")
        except Exception as e:
            logger.error(f"Failed to connect to OPC-UA server: {e}")
            self.client = None # Ensure client is not in a weird state
            raise

    async def _disconnect(self):
        """Disconnects the OPC-UA client if it's active."""
        if self.client:
            try:
                await self.client.disconnect()
                logger.info("Disconnected from OPC-UA server.")
            except Exception as e:
                logger.error(f"Error disconnecting from OPC-UA server: {e}")
            finally:
                self.client = None

    async def restart_service(self, service_name: str, **kwargs):
        logger.info(f"Executing OPC-UA 'restart' on {service_name}")
        if service_name not in self.node_map:
            logger.error(f"Unknown service '{service_name}' in OPC-UA node map.")
            return False
        
        method_id_str = self.node_map[service_name].get('restart_method_id')
        if not method_id_str:
            logger.error(f"'restart_method_id' not defined for '{service_name}'")
            return False

        try:
            await self._connect()
            # FIX: The previous logic was fragile. This is a more robust way to call a method.
            # 1. Split the config string into the parent's ID and the method's browse name.
            #    e.g., parent="ns=2;s=Pump1", bname="Reset" from "ns=2;s=Pump1.Reset"
            parent_id_str, method_bname = method_id_str.rsplit('.', 1)

            # 2. Get the parent node from the server using its full NodeId.
            parent_node = self.client.get_node(parent_id_str)

            # 3. Call the method on the parent node using the "namespace:browsename" format.
            ns_idx = parent_node.nodeid.NamespaceIndex
            await parent_node.call_method(f"{ns_idx}:{method_bname}")

            logger.info(f"Successfully called method '{method_bname}' on '{parent_id_str}'")
            return True
        except Exception as e:
            logger.error(f"Failed to execute restart for {service_name}: {e}")
            return False
        finally:
            await self._disconnect()

    async def isolate_service(self, service_name: str, **kwargs):
        logger.info(f"Executing OPC-UA 'isolate' on {service_name}")
        if service_name not in self.node_map:
            logger.error(f"Unknown service '{service_name}' in OPC-UA node map.")
            return False
        
        variable_id = self.node_map[service_name].get('isolate_variable_id')
        if not variable_id:
            logger.error(f"'isolate_variable_id' not defined for '{service_name}'")
            return False

        try:
            await self._connect()
            # Get the variable node and write to it
            var_node = self.client.get_node(variable_id)
            await var_node.write_value(False)
            logger.info(f"Successfully wrote 'False' to variable '{variable_id}' for service '{service_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to execute isolate for {service_name}: {e}")
            return False
        finally:
            await self._disconnect()

    def scale_service(self, service_name: str, **kwargs):
        logger.warning("OPC-UA 'scale' action is a logical placeholder and is equivalent to 'degrade'.")
        return self.degrade_service(service_name, **kwargs)

    async def degrade_service(self, service_name: str, **kwargs):
        logger.info(f"Executing OPC-UA 'degrade' on {service_name}")
        # TODO: Get node ID and value from kwargs
        # TODO: Write the new value to the 'degrade_variable_id' on the server
        return True 