import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from graph_heal.recovery.opcua_adapter import OpcUaAdapter

async def main():
    """
    A simple script to test the OpcUaAdapter against the mock server.
    
    Instructions:
    1. Run the mock server in one terminal: python scripts/run_mock_opcua_server.py
    2. Run this script in another terminal: python scripts/test_opcua_adapter.py
    """
    print("--- Initializing OPC-UA Adapter Test ---")
    adapter = OpcUaAdapter()

    # Test Case 1: Restart pump-1
    print("\n--> Testing 'restart_service' on 'pump-1'...")
    success = await adapter.restart_service('pump-1')
    if success:
        print("    SUCCESS: restart_service('pump-1') completed.")
    else:
        print("    FAILURE: restart_service('pump-1') failed.")

    # Test Case 2: Isolate valve-3
    print("\n--> Testing 'isolate_service' on 'valve-3'...")
    success = await adapter.isolate_service('valve-3')
    if success:
        print("    SUCCESS: isolate_service('valve-3') completed.")
    else:
        print("    FAILURE: isolate_service('valve-3') failed.")
        
    print("\n--- Test Complete ---")


if __name__ == "__main__":
    asyncio.run(main()) 