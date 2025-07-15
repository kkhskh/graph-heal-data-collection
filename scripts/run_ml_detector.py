import time
import requests
import logging
import pandas as pd
from graph_heal.ml_detector import MLDetector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('ml_detector_runner')

# Use the same service URLs and metric parsing as the main monitor
SERVICES = {
    "service_a": "http://localhost:5001/metrics",
    "service_b": "http://localhost:5002/metrics",
    "service_c": "http://localhost:5003/metrics",
    "service_d": "http://localhost:5004/metrics",
}

def parse_metrics(text_data):
    """Parse Prometheus metrics text into a dictionary."""
    metrics = {}
    for line in text_data.split('\n'):
        if line and not line.startswith('#'):
            parts = line.split()
            if len(parts) >= 2:
                metric_name = parts[0].split('{')[0] # Strip labels
                metrics[metric_name] = float(parts[1])
    return metrics

def main():
    logger.info("Initializing ML-based anomaly detector...")
    try:
        detector = MLDetector()
    except FileNotFoundError as e:
        logger.error(f"Could not start detector: {e}")
        logger.error("Please run the training script (scripts/train_model.py) first.")
        return

    logger.info("Starting ML detection loop...")
    while True:
        try:
            all_metrics = {}
            for service, url in SERVICES.items():
                try:
                    resp = requests.get(url, timeout=1.0)
                    resp.raise_for_status()
                    
                    logger.info(f"Raw metrics from {service}: {resp.text[:300]}...") # Log first 300 chars
                    
                    parsed = parse_metrics(resp.text)
                    
                    logger.info(f"Parsed metrics from {service}: {parsed}")
                    
                    for key, value in parsed.items():
                        all_metrics[f"{service}_{key}"] = value
                except requests.RequestException as e:
                    logger.warning(f"Could not fetch metrics from {url}: {e}")
            
            if all_metrics:
                # The model expects a DataFrame with specific columns.
                # We need to ensure `all_metrics` has the same structure as the training data.
                # This is a simplification.
                metrics_df = pd.DataFrame([all_metrics])
                
                # Align columns with the model's training data
                model_cols = detector.model.feature_names_in_
                metrics_df = metrics_df.reindex(columns=model_cols, fill_value=0)

                anomalies = detector.detect(metrics_df)
                if anomalies:
                    logger.warning(f"--- ANOMALY DETECTED ---")
                    for anom in anomalies:
                        logger.warning(anom)
            
            time.sleep(5) # Check for anomalies every 5 seconds

        except KeyboardInterrupt:
            logger.info("Stopping ML detector.")
            break
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}", exc_info=True)
            time.sleep(10)

if __name__ == "__main__":
    main() 