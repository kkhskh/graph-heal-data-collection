#!/usr/bin/env python3
"""
Cloud-ready automated fault injector for Graph-Heal data collection.
Designed to run on servers/cloud platforms for extended periods.
"""

import argparse
import logging
import random
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Configure logging for cloud environments
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cloud_fault_injection.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def inject_cpu_fault(service, duration, label_file):
    """Inject CPU fault into a service."""
    try:
        cmd = [
            'python', 'scripts/inject_cpu_fault.py',
            '--service', service,
            '--duration', str(duration)
        ]
        
        logger.info(f"Executing: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            # Log the injection
            with open(label_file, 'a') as f:
                start_time = datetime.utcnow()
                end_time = start_time + timedelta(seconds=duration)
                f.write(f"{service},cpu,{start_time.isoformat()},{end_time.isoformat()}\n")
            logger.info(f"CPU fault injection successful for {service} ({duration}s)")
            return True
        else:
            logger.error(f"CPU fault injection failed for {service}: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"CPU fault injection timed out for {service}")
        return False
    except Exception as e:
        logger.error(f"CPU fault injection error for {service}: {e}")
        return False

def inject_memory_fault(service, duration, label_file):
    """Inject memory fault into a service."""
    try:
        cmd = [
            'python', 'scripts/inject_memory_fault.py',
            '--service', service,
            '--duration', str(duration)
        ]
        
        logger.info(f"Executing: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            # Log the injection
            with open(label_file, 'a') as f:
                start_time = datetime.utcnow()
                end_time = start_time + timedelta(seconds=duration)
                f.write(f"{service},memory,{start_time.isoformat()},{end_time.isoformat()}\n")
            logger.info(f"Memory fault injection successful for {service} ({duration}s)")
            return True
        else:
            logger.error(f"Memory fault injection failed for {service}: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"Memory fault injection timed out for {service}")
        return False
    except Exception as e:
        logger.error(f"Memory fault injection error for {service}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Cloud-ready automated fault injector')
    parser.add_argument('--hours', type=float, default=24.0, help='Hours to run (default: 24)')
    parser.add_argument('--label-file', default='fault_labels.csv', help='Output file for fault labels')
    parser.add_argument('--services', nargs='+', default=['service_a', 'service_b', 'service_c', 'service_d'], 
                       help='Services to inject faults into')
    parser.add_argument('--min-wait', type=int, default=300, help='Minimum wait between injections (seconds)')
    parser.add_argument('--max-wait', type=int, default=1800, help='Maximum wait between injections (seconds)')
    parser.add_argument('--min-duration', type=int, default=60, help='Minimum fault duration (seconds)')
    parser.add_argument('--max-duration', type=int, default=600, help='Maximum fault duration (seconds)')
    
    args = parser.parse_args()
    
    # Initialize label file
    with open(args.label_file, 'w') as f:
        f.write("service,fault_type,start_time,end_time\n")
    
    end_time = datetime.utcnow() + timedelta(hours=args.hours)
    fault_types = ['cpu', 'memory']
    
    logger.info(f"Starting cloud fault injector for {args.hours} hours")
    logger.info(f"Will finish at: {end_time.isoformat()}")
    logger.info(f"Services: {args.services}")
    logger.info(f"Label file: {args.label_file}")
    
    injection_count = 0
    success_count = 0
    
    try:
        while datetime.utcnow() < end_time:
            # Random service and fault type
            service = random.choice(args.services)
            fault_type = random.choice(fault_types)
            
            # Random duration
            duration = random.randint(args.min_duration, args.max_duration)
            
            logger.info(f"[INJECT] {datetime.utcnow().isoformat()} | Service: {service} | Fault: {fault_type} | Duration: {duration}s")
            
            # Inject fault
            if fault_type == 'cpu':
                success = inject_cpu_fault(service, duration, args.label_file)
            else:
                success = inject_memory_fault(service, duration, args.label_file)
            
            injection_count += 1
            if success:
                success_count += 1
            
            # Random wait before next injection
            wait_time = random.randint(args.min_wait, args.max_wait)
            logger.info(f"[WAIT] Waiting {wait_time}s before next injection...")
            
            # Sleep in smaller chunks to allow for graceful shutdown
            for _ in range(wait_time // 10):
                time.sleep(10)
                if datetime.utcnow() >= end_time:
                    break
            time.sleep(wait_time % 10)
            
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down gracefully...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        logger.info(f"Fault injection completed. Total injections: {injection_count}, Successful: {success_count}")
        logger.info(f"Results saved to: {args.label_file}")

if __name__ == '__main__':
    main() 