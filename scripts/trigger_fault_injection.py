import argparse
import sys
import os
import time

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import inject_cpu_fault

def main():
    parser = argparse.ArgumentParser(description="Trigger fault injections programmatically.")
    
    # Mock the command-line arguments that inject_cpu_fault.py expects
    service_to_target = 'service_a'
    label_file_path = 'fault_labels.csv'
    duration_sec = 30
    
    sys.argv = [
        'scripts/inject_cpu_fault.py',
        '--service', service_to_target,
        '--label-file', label_file_path,
        '--duration', str(duration_sec)
    ]
    
    print(f"Injecting fault into '{service_to_target}' for {duration_sec}s...")
    print(f"Labels will be saved to '{label_file_path}'")
    
    try:
        inject_cpu_fault.main()
        print("\n--- Fault injection completed successfully ---")
    except Exception as e:
        print(f"\n--- An error occurred during fault injection: {e} ---")

if __name__ == "__main__":
    main() 