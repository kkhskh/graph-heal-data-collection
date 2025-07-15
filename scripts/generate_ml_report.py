#!/usr/bin/env python3
import os
import sys
import time
import logging
import pandas as pd
from typing import List, Dict

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from graph_heal.service_monitor import ServiceMonitor
from graph_heal.fault_injection import FaultInjector
from graph_heal.anomaly_detection import StatisticalAnomalyDetector, AnomalyManager
from graph_heal.ml_detector import MLDetector

# --- Configuration ---
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Service and Fault Configuration ---
SERVICES_CONFIG = [
    {"id": "service_a", "url": "http://localhost:5001", "health_endpoint": "/metrics"},
    {"id": "service_b", "url": "http://localhost:5002", "health_endpoint": "/metrics"},
    {"id": "service_c", "url": "http://localhost:5003", "health_endpoint": "/metrics"},
    {"id": "service_d", "url": "http://localhost:5004", "health_endpoint": "/metrics"},
]

FAULT_SCENARIOS = [
    {'fault_type': 'cpu_stress', 'target': 'service_b', 'params': {'load': 80, 'duration': 60}},
    {'fault_type': 'memory_leak', 'target': 'service_c', 'params': {'memory_mb': 100, 'duration': 60}},
]

def run_experiment(detector, fault_scenario: Dict) -> Dict:
    """Runs a single experiment for a given detector and fault."""
    logging.info(f"--- Running experiment with {detector.__class__.__name__} for fault {fault_scenario['fault_type']} on {fault_scenario['target']} ---")

    # 1. Setup components
    monitor = ServiceMonitor(services=SERVICES_CONFIG, interval=2)
    anomaly_manager = AnomalyManager(detectors=[detector])
    injector = FaultInjector()

    # Store results
    results = {
        'detected_anomalies': [],
        'fault_start_time': None,
        'fault_end_time': None,
        'detection_latency': None
    }

    # 2. Run monitoring loop
    experiment_duration = 90  # seconds
    end_time = time.time() + experiment_duration
    fault_injected = False
    fault_id = None

    while time.time() < end_time:
        # Inject fault after a delay
        if not fault_injected and (time.time() > (end_time - experiment_duration + 15)):
            logging.info(f"Injecting fault: {fault_scenario}")
            fault_id = injector.inject_fault(
                fault_type=fault_scenario['fault_type'],
                target=fault_scenario['target'],
                params=fault_scenario['params']
            )
            results['fault_start_time'] = time.time()
            fault_injected = True

        # Monitor and detect
        service_statuses = monitor.get_all_services_status()
        detected_anomalies = anomaly_manager.detect_anomalies(service_statuses)

        if detected_anomalies:
            logging.info(f"Anomalies detected: {detected_anomalies}")
            results['detected_anomalies'].extend(detected_anomalies)
            if fault_injected and results['detection_latency'] is None:
                detection_time = time.time()
                results['detection_latency'] = detection_time - results['fault_start_time']
                logging.info(f"Detection latency: {results['detection_latency']:.2f}s")


        time.sleep(2)

    # 3. Cleanup
    if fault_id:
        injector.resolve_fault(fault_id)
        results['fault_end_time'] = time.time()
        logging.info("Fault resolved.")

    return results

def calculate_metrics(results: Dict, fault_scenario: Dict) -> Dict:
    """Calculates evaluation metrics from experiment results."""
    
    true_positives = 0
    false_positives = 0

    for anomaly in results['detected_anomalies']:
        anomaly_time = pd.to_datetime(anomaly.get('timestamp')).timestamp()
        
        if results['fault_start_time'] <= anomaly_time <= results['fault_end_time']:
            if anomaly.get('service_id') == fault_scenario['target'] or anomaly.get('type') == 'ml_prediction':
                 true_positives += 1
            else:
                false_positives += 1
        else:
            false_positives += 1

    false_negatives = 1 if true_positives == 0 else 0

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "latency": results.get('detection_latency')
    }


def main():
    """Main function to run the evaluation."""
    model_path = 'models/fault_detection_model.joblib'
    if not os.path.exists(model_path):
        logging.error("Model not found. Please run `scripts/train_model.py` first to create a model.")
        return

    statistical_detector = StatisticalAnomalyDetector(window_size=10, z_score_threshold=3.0)
    ml_detector = MLDetector(model_path=model_path)
    
    detectors = {
        'Statistical': statistical_detector,
        'ML': ml_detector
    }

    all_results = []
    for detector_name, detector_instance in detectors.items():
        for scenario in FAULT_SCENARIOS:
            experiment_results = run_experiment(detector_instance, scenario)
            metrics = calculate_metrics(experiment_results, scenario)
            
            all_results.append({
                'detector': detector_name,
                'fault_type': scenario['fault_type'],
                'target': scenario['target'],
                **metrics
            })
            logging.info(f"Results for {detector_name} on {scenario['fault_type']}: {metrics}")

    results_df = pd.DataFrame(all_results)
    print("\n--- Evaluation Report ---")
    print(results_df.to_string())


if __name__ == "__main__":
    main() 