#!/usr/bin/env python3
"""
Automated Fault Injector for 24-Hour Data Collection

This script runs for 24 hours (or user-specified duration), randomly injecting faults
into random services at random intervals and durations, and logs all injections to a CSV file.
"""
import time
import random
import csv
import os
import argparse
import subprocess
from datetime import datetime, timedelta, timezone

SERVICES = ['service_a', 'service_b', 'service_c', 'service_d']
FAULT_TYPES = ['cpu', 'memory']  # Extendable
INJECTION_SCRIPTS = {
    'cpu': 'scripts/inject_cpu_fault.py',
    'memory': 'scripts/inject_memory_fault.py',
}

MIN_FAULT_DURATION = 60      # 1 minute (seconds)
MAX_FAULT_DURATION = 600     # 10 minutes (seconds)
MIN_WAIT = 600               # 10 minutes (seconds)
MAX_WAIT = 1800              # 30 minutes (seconds)

DEFAULT_LABEL_FILE = 'fault_labels.csv'
DEFAULT_TOTAL_HOURS = 24


def log_fault(label_file, start_time, end_time, service, fault_type):
    file_exists = os.path.isfile(label_file) and os.path.getsize(label_file) > 0
    with open(label_file, 'a', newline='') as csvfile:
        fieldnames = ['start_time', 'end_time', 'service_name', 'fault_type']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'start_time': start_time,
            'end_time': end_time,
            'service_name': service,
            'fault_type': fault_type
        })


def main():
    parser = argparse.ArgumentParser(description="Automated Fault Injector for Data Collection")
    parser.add_argument('--label-file', type=str, default=DEFAULT_LABEL_FILE, help='Path to output fault label CSV file')
    parser.add_argument('--hours', type=float, default=DEFAULT_TOTAL_HOURS, help='Total run time in hours (default: 24)')
    args = parser.parse_args()

    end_time = datetime.now(timezone.utc) + timedelta(hours=args.hours)
    print(f"[INFO] Starting automated fault injection for {args.hours} hours. Will finish at {end_time.isoformat()} UTC.")
    print(f"[INFO] Logging all injections to {args.label_file}")

    try:
        while datetime.now(timezone.utc) < end_time:
            # Pick random service, fault type, duration, and wait interval
            service = random.choice(SERVICES)
            fault_type = random.choice(FAULT_TYPES)
            duration = random.randint(MIN_FAULT_DURATION, MAX_FAULT_DURATION)
            wait_time = random.randint(MIN_WAIT, MAX_WAIT)

            # Prepare injection script and args
            script = INJECTION_SCRIPTS[fault_type]
            inject_args = [
                'python', script,
                '--service', service,
                '--duration', str(duration),
                '--label-file', args.label_file
            ]

            # Log and inject
            utc_now = datetime.now(timezone.utc)
            fault_end = utc_now + timedelta(seconds=duration)
            print(f"[INJECT] {utc_now.isoformat()} UTC | Service: {service} | Fault: {fault_type} | Duration: {duration//60}m{duration%60}s")
            try:
                subprocess.run(inject_args, check=True)
                log_fault(args.label_file, utc_now.timestamp(), fault_end.timestamp(), service, fault_type)
                print(f"[LOG] Injection recorded: {service}, {fault_type}, {utc_now.isoformat()} -> {fault_end.isoformat()}")
            except Exception as e:
                print(f"[ERROR] Injection failed for {service} ({fault_type}): {e}")

            # Wait before next injection
            print(f"[WAIT] Waiting {wait_time//60}m{wait_time%60}s before next injection...")
            time.sleep(wait_time)

        print("[DONE] Automated fault injection complete.")
    except KeyboardInterrupt:
        print("[STOP] Interrupted by user. Exiting...")

if __name__ == "__main__":
    main() 