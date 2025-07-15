#!/usr/bin/env python3
import argparse
import requests
import time
import logging
import os
import csv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('fault_injection')

def inject_cpu_fault(service_url, duration):
    """Injects a CPU fault into a specific service for a given duration."""
    fault_url = f"{service_url}/fault/cpu"
    payload = {"fault_type": "cpu", "duration_seconds": duration}
    try:
        print(f"Injecting CPU fault into {service_url} for {duration} seconds...")
        response = requests.post(fault_url, json=payload)
        response.raise_for_status()
        print("Fault injection successful.")
        return True
    except requests.RequestException as e:
        print(f"Error injecting fault into {service_url}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Inject a CPU fault into a service.")
    parser.add_argument("--service", required=True, help="The name of the service to target (e.g., service_a).")
    parser.add_argument("--duration", type=int, default=30, help="Duration of the fault in seconds.")
    parser.add_argument(
        "--label-file",
        type=str,
        help="Enable label capture by specifying the path to the output CSV file."
    )
    args = parser.parse_args()

    service_ports = {"service_a": 5001, "service_b": 5002, "service_c": 5003, "service_d": 5004}
    
    if args.service.strip() not in service_ports:
        print(f"Error: Unknown service '{args.service}'. Please choose from {list(service_ports.keys())}.")
        return

    service_url = f"http://localhost:{service_ports[args.service.strip()]}"

    if args.label_file:
        start_time = time.time()
        end_time = start_time + args.duration
        
        # Ensure the output directory for the label file exists
        output_dir = os.path.dirname(args.label_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        file_exists = os.path.isfile(args.label_file) and os.path.getsize(args.label_file) > 0
        
        with open(args.label_file, 'a', newline='') as csvfile:
            fieldnames = ['start_time', 'end_time', 'service_name', 'fault_type']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()
            
            writer.writerow({
                'start_time': start_time,
                'end_time': end_time,
                'service_name': args.service,
                'fault_type': 'cpu_stress'
            })
        print(f"Wrote fault label to {args.label_file}")

    inject_cpu_fault(service_url, args.duration)

if __name__ == "__main__":
    main() 