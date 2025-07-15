import numpy as np
import pandas as pd
import time
import logging
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import deque, defaultdict
import json
import os
import datetime
import community as community_louvain
import threading
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression
import joblib
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.Formatter.converter = time.gmtime
logger = logging.getLogger('anomaly_detection')

class BaseDetector(ABC):
    """
    Abstract base class for all anomaly detection engines.

    This class defines the standard interface for a detector. The core
    `detect` method must be implemented by all concrete detector classes.
    """

    @abstractmethod
    def detect(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Analyzes a dictionary of metrics and returns a list of anomalies.
        
        Args:
            metrics: A dictionary where keys are metric names and values are
                     the latest metric readings.

        Returns:
            A list of dictionaries, where each dictionary represents a
            detected anomaly and should contain at least a 'service_name'
            and 'metric_name'.
        """
        pass

class AnomalyDetector:
    """
    Base class for anomaly detection.
    """
    def __init__(self, data_dir: str = "data/anomalies"):
        """
        Initialize the anomaly detector.
        
        Args:
            data_dir: Directory to store anomaly data
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.anomalies: List[Dict[str, Any]] = []
        self.active_anomalies: Dict[str, Dict[str, Any]] = {}
    
    def detect_anomalies(self, *args, **kwargs) -> List[Dict[str, Any]]:
        """
        Detect anomalies in the provided data.
        
        Returns:
            List of anomalies, each as a dictionary
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def log_anomaly(self, anomaly: Dict[str, Any]):
        """
        Log an anomaly to the data directory.
        
        Args:
            anomaly: Anomaly information
        """
        try:
            # Create a unique filename with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            anomaly_id = anomaly.get("id", "unknown")
            filename = f"{self.data_dir}/anomaly_{timestamp}_{anomaly_id}.json"
            
            with open(filename, 'w') as f:
                json.dump(anomaly, f, indent=2)
                
            logger.debug(f"Logged anomaly to {filename}")
            
        except Exception as e:
            logger.error(f"Failed to log anomaly: {e}")
    
    def get_active_anomalies(self) -> List[Dict[str, Any]]:
        """
        Get currently active anomalies.
        
        Returns:
            List of active anomalies
        """
        return list(self.active_anomalies.values())
    
    def get_all_anomalies(self) -> List[Dict[str, Any]]:
        """
        Get all detected anomalies.
        
        Returns:
            List of all anomalies
        """
        return self.anomalies


class StatisticalAnomalyDetector(AnomalyDetector):
    """
    Detects anomalies using statistical methods.
    """
    def __init__(self, window_size: int = 10, z_score_threshold: float = 2.5, data_dir: str = "data/anomalies"):
        """
        Initialize the statistical anomaly detector.
        
        Args:
            window_size: Size of the sliding window for statistics
            z_score_threshold: Threshold for z-score based detection
            data_dir: Directory to store anomaly data
        """
        super().__init__(data_dir)
        self.window_size = window_size
        self.z_score_threshold = z_score_threshold
        self.metrics_history: Dict[str, Dict[str, List[float]]] = {}
    
    def detect_anomalies(self, service_statuses: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect anomalies using statistical methods.
        
        Args:
            service_statuses: Current status of all services
        
        Returns:
            List of detected anomalies
        """
        print("[DEBUG] StatisticalAnomalyDetector.detect_anomalies CALLED")
        print("[DEBUG] service_statuses:", service_statuses)
        current_time = time.time()

        # Fast-path guard: if *every* service lacks numeric metrics or the
        # history window is effectively empty/NaN we skip heavy processing to
        # avoid false positives and divide-by-zero.
        if not service_statuses or all(
            not any(isinstance(v, (int, float)) for v in s.get("metrics", {}).values())
            for s in service_statuses.values()
        ):
            logger.debug("StatisticalAnomalyDetector: empty or non-numeric input – skipping")
            return []

        new_anomalies = []
        
        for service_id, status in service_statuses.items():
            metrics = status.get("metrics", {})
            
            # Update metrics history
            if service_id not in self.metrics_history:
                self.metrics_history[service_id] = {}
            
            for metric_name, metric_value in metrics.items():
                if not isinstance(metric_value, (int, float)):
                    continue
                
                if metric_name not in self.metrics_history[service_id]:
                    self.metrics_history[service_id][metric_name] = []
                
                self.metrics_history[service_id][metric_name].append(metric_value)
                
                # Keep only recent history
                if len(self.metrics_history[service_id][metric_name]) > self.window_size:
                    self.metrics_history[service_id][metric_name] = self.metrics_history[service_id][metric_name][-self.window_size:]

            # --- START: New Trend-Based Detection Logic ---
            # Check for anomalies using trend detection instead of just z-score
            trend_anomalies = self._detect_trends(service_id, self.metrics_history[service_id])
            new_anomalies.extend(trend_anomalies)
            for anomaly in trend_anomalies:
                self.anomalies.append(anomaly)
                self.active_anomalies[anomaly["id"]] = anomaly
                logger.info(f"Detected trend-based anomaly in {service_id}.{anomaly['metric_name']}: "
                            f"slope = {anomaly['details']['slope']:.4f}, r_squared = {anomaly['details']['r_squared']:.3f}")
            # --- END: New Trend-Based Detection Logic ---

        # Update active anomalies
        self._update_active_anomalies()
        
        if new_anomalies:
            print(f"[DEBUG] StatisticalAnomalyDetector detected {len(new_anomalies)} anomalies:", new_anomalies)
        
        return new_anomalies

    def _detect_trends(self, service_id: str, metric_history: Dict[str, List[float]]) -> List[Dict]:
        """Detect anomalies using trend analysis for each metric."""
        anomalies = []
        current_time = time.time()
        
        # Parameters for trend detection (can be tuned)
        trend_threshold = 0.05  # Optimal value from tuning
        sensitivity = 0.01      # Optimal value from tuning
        min_r_squared = 0.4     # Optimal value from tuning
        
        for metric_name, values in metric_history.items():
            if len(values) < self.window_size:
                continue

            window = np.array(values)
            x = np.arange(len(window)).reshape(-1, 1)
            y = window

            try:
                # Fit linear regression
                reg = LinearRegression()
                reg.fit(x, y)

                slope = reg.coef_[0]
                r_squared = reg.score(x, y)

                # Calculate percentage change over the window
                start_val = window[0]
                end_val = window[-1]
                pct_change = abs(end_val - start_val) / start_val if start_val > 0 else 0

                is_anomaly = (
                    abs(slope) > sensitivity and
                    r_squared > min_r_squared and
                    pct_change > trend_threshold
                )

                if is_anomaly:
                    anomaly = {
                        "id": f"anomaly_trend_{service_id}_{metric_name}_{int(current_time)}",
                        "type": "trend_based",
                        "service_id": service_id,
                        "metric_name": metric_name,
                        "value": values[-1],
                        "timestamp": current_time,
                        "severity": "medium",
                        "details": {
                            "slope": slope,
                            "r_squared": r_squared,
                            "pct_change": pct_change,
                            "window_size": self.window_size
                        }
                    }
                    anomalies.append(anomaly)
            except Exception as e:
                logger.error(f"Error during trend detection for {service_id}.{metric_name}: {e}")
                
        return anomalies

    def _update_active_anomalies(self):
        """Update the list of active anomalies."""
        current_time = time.time()
        expired_anomalies = []
        
        for anomaly_id, anomaly in self.active_anomalies.items():
            # Remove anomalies older than 5 minutes
            if current_time - anomaly["timestamp"] > 300:
                expired_anomalies.append(anomaly_id)
        
        for anomaly_id in expired_anomalies:
            del self.active_anomalies[anomaly_id]


class MLDetector(BaseDetector):
    """
    Detects anomalies using a pre-trained machine learning model.
    """
    def __init__(self, model_path: str = "models/fault_detection_model.joblib"):
        """
        Initialize the ML anomaly detector.
        
        Args:
            model_path: Path to the serialized, pre-trained model file.
        """
        self.model = None
        self.features = None
        try:
            logger.info(f"Loading ML model from {model_path}...")
            # The loaded model object contains the classifier and feature list
            loaded_object = joblib.load(model_path)
            if isinstance(loaded_object, tuple) and len(loaded_object) == 2:
                 self.model, self.features = loaded_object
                 logger.info(f"ML model and {len(self.features)} features loaded successfully.")
            else: # Backwards compatibility for models saved without features
                 self.model = loaded_object
                 self.features = self.model.feature_names_in_
                 logger.warning("Loaded a legacy ML model. Feature order may be inconsistent.")

        except FileNotFoundError:
            logger.error(f"ML model file not found at {model_path}. MLDetector will be inactive.")
        except Exception as e:
            logger.error(f"Error loading ML model: {e}. MLDetector will be inactive.")

    def detect(self, service_statuses: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Analyzes metrics using the ML model and returns a list of anomalies.
        
        Args:
            service_statuses: A dictionary containing the latest metrics for all services.

        Returns:
            A list of detected anomalies. An empty list if no anomalies are found
            or if the model is not active.
        """
        if self.model is None or self.features is None or not service_statuses:
            return []

        # --- Feature Engineering: Create a single row with all metrics ---
        live_metrics_row = {}
        for service_id, status in service_statuses.items():
            if "metrics" in status:
                for metric_name, value in status["metrics"].items():
                    # Create the wide-format column name, e.g., 'service_a_cpu_usage'
                    feature_name = f"{service_id}_{metric_name.split('{')[0]}"
                    live_metrics_row[feature_name] = value

        if not live_metrics_row:
            return []
            
        # Create a DataFrame aligned with the model's features
        df = pd.DataFrame([live_metrics_row])
        df_aligned = df.reindex(columns=self.features, fill_value=0)

        # --- Prediction ---
        try:
            prediction = self.model.predict(df_aligned)
            if prediction and prediction[0] != 'normal':
                fault_type = prediction[0]
                logger.info(f"ML model predicted a system-wide fault: {fault_type}")
                
                # The model predicts a single state for the whole system.
                # The fault localizer will need to determine the source.
                anomaly = {
                    "id": f"anomaly_ml_{fault_type}_{int(time.time())}",
                    "type": "ml_based_system",
                    "service_id": "system_wide",
                    "metric_name": "combined_ml",
                    "value": fault_type,
                    "timestamp": time.time(),
                    "severity": "high",
                    "details": f"ML model detected a system-wide '{fault_type}' state."
                }
                return [anomaly]
        except Exception as e:
            logger.error(f"Error during ML prediction: {e}")

        return []


class GraphAnomalyDetector(AnomalyDetector):
    """
    Detects anomalies using graph-based analysis.
    """
    def __init__(self, correlation_threshold: float = 0.7, data_dir: str = "data/anomalies"):
        """
        Initialize the graph anomaly detector.
        
        Args:
            correlation_threshold: Threshold for correlation-based detection
            data_dir: Directory to store anomaly data
        """
        super().__init__(data_dir)
        self.correlation_threshold = correlation_threshold
        self.metrics_history: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        self.prev_communities = None
    
    def detect_anomalies(self, service_statuses: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect anomalies using graph-based analysis.
        
        Args:
            service_statuses: Current status of all services
        
        Returns:
            List of detected anomalies
        """
        current_time = time.time()
        new_anomalies = []
        
        # Update metrics history
        for service_id, status in service_statuses.items():
            metrics = status.get("metrics", {})
            
            if service_id not in self.metrics_history:
                self.metrics_history[service_id] = {}
            
            for metric_name, metric_value in metrics.items():
                if not isinstance(metric_value, (int, float)):
                    continue
                
                if metric_name not in self.metrics_history[service_id]:
                    self.metrics_history[service_id][metric_name] = []
                
                self.metrics_history[service_id][metric_name].append({
                    "timestamp": current_time,
                    "value": metric_value
                })
        
                # Keep only recent history (last 100 points)
                if len(self.metrics_history[service_id][metric_name]) > 100:
                    self.metrics_history[service_id][metric_name] = self.metrics_history[service_id][metric_name][-100:]
        
        # Build the service dependency graph
        G = nx.Graph()
        for service_id, status in service_statuses.items():
            G.add_node(service_id)
            for dep_id in status.get("dependencies", []):
                if dep_id in service_statuses:
                    G.add_edge(service_id, dep_id)
        
        # Community detection using Louvain
        if G.number_of_edges() > 0:
            partition = community_louvain.best_partition(G)
            if self.prev_communities is not None:
                # Compare with previous partition
                changed = sum(1 for n in partition if self.prev_communities.get(n) != partition[n])
                if changed > 0:
                    anomaly = {
                        "id": f"anomaly_community_{int(current_time)}",
                        "type": "community",
                        "description": f"Community structure changed for {changed} nodes",
                        "changed_nodes": [n for n in partition if self.prev_communities.get(n) != partition[n]],
                        "timestamp": current_time,
                        "severity": "medium" if changed < len(partition) // 2 else "high"
                    }
                    new_anomalies.append(anomaly)
                    self.anomalies.append(anomaly)
                    self.active_anomalies[anomaly["id"]] = anomaly
            self.prev_communities = partition
        
        # Check for anomalies in service dependencies
        for service_id, status in service_statuses.items():
            dependencies = status.get("dependencies", [])
            
            for dep_id in dependencies:
                if dep_id not in service_statuses:
                    continue
                
                # Check for correlation between service metrics
                service_metrics = self.metrics_history.get(service_id, {})
                dep_metrics = self.metrics_history.get(dep_id, {})
                
                for s_metric, s_values in service_metrics.items():
                    for d_metric, d_values in dep_metrics.items():
                        if len(s_values) < 2 or len(d_values) < 2:
                            continue
                        
                        # Calculate correlation
                        s_ts = [v["value"] for v in s_values]
                        d_ts = [v["value"] for v in d_values]
                        
                        # Ensure same length
                        min_length = min(len(s_ts), len(d_ts))
                        s_ts = s_ts[-min_length:]
                        d_ts = d_ts[-min_length:]
                        
                        correlation = np.corrcoef(s_ts, d_ts)[0, 1]
                        
                        if abs(correlation) > self.correlation_threshold:
                            # Check for sudden changes in correlation
                            if len(s_ts) >= 3 and len(d_ts) >= 3:
                                recent_corr = np.corrcoef(s_ts[-3:], d_ts[-3:])[0, 1]
        
                                if abs(correlation - recent_corr) > 0.5:  # Significant change
                                    anomaly = {
                                        "id": f"anomaly_{service_id}_{dep_id}_{int(current_time)}",
                                        "type": "graph",
                                        "service_id": service_id,
                                        "dependency_id": dep_id,
                                        "metric_pair": f"{s_metric}-{d_metric}",
                                        "correlation": correlation,
                                        "recent_correlation": recent_corr,
                                        "timestamp": current_time,
                                        "severity": "high" if abs(correlation - recent_corr) > 0.8 else "medium"
                                    }
                    
                                    new_anomalies.append(anomaly)
                                    self.anomalies.append(anomaly)
                                    self.active_anomalies[anomaly["id"]] = anomaly
                    
                                    logger.info(f"Detected graph anomaly between {service_id} and {dep_id}: "
                                              f"correlation change = {abs(correlation - recent_corr):.2f}")
        
        # Update active anomalies
        self._update_active_anomalies()
        
        return new_anomalies
    
    def _update_active_anomalies(self):
        """Update the list of active anomalies."""
        current_time = time.time()
        expired_anomalies = []
        
        for anomaly_id, anomaly in self.active_anomalies.items():
            # Remove anomalies older than 5 minutes
            if current_time - anomaly["timestamp"] > 300:
                expired_anomalies.append(anomaly_id)
        
        for anomaly_id in expired_anomalies:
            del self.active_anomalies[anomaly_id]


class BayesianFaultLocalizer:
    """
    Uses Bayesian reasoning to estimate the probability of each service being the root cause given observed anomalies.
    """
    def __init__(self, prior: Optional[Dict[str, float]] = None):
        # If no prior, use uniform
        self.prior = prior if prior is not None else {}

    def localize(self, anomalies: List[Dict[str, Any]], service_statuses: Dict[str, Dict[str, Any]]) -> List[Tuple[str, float]]:
        # If no prior, use uniform
        if not self.prior:
            all_services = list(service_statuses.keys())
            self.prior = {service: 1.0 / len(all_services) for service in all_services}

        # Calculate likelihoods
        likelihoods = defaultdict(lambda: 1.0)
        for anomaly in anomalies:
            service_id = anomaly['service_id']
            # Simplified likelihood: increase likelihood for anomalous services
            likelihoods[service_id] *= 2.0  # arbitrary factor

        # Calculate posterior probabilities
        posteriors = {}
        for service, prior_prob in self.prior.items():
            likelihood = likelihoods.get(service, 1.0)  # Services with no anomalies have likelihood 1
            posteriors[service] = prior_prob * likelihood

        # Normalize posteriors
        total_posterior = sum(posteriors.values())
        if total_posterior > 0:
            for service in posteriors:
                posteriors[service] /= total_posterior

        return sorted(posteriors.items(), key=lambda item: item[1], reverse=True)


class AnomalyManager:
    """
    Manages multiple anomaly detectors and coordinates detection.
    """
    def __init__(self, detectors: List[AnomalyDetector]):
        """
        Initialize the anomaly manager with a list of detectors.
        
        Args:
            detectors: List of anomaly detector instances
        """
        self.detectors = detectors
        self.all_anomalies = []
        self.save_interval = 60  # seconds
        self.last_save_time = time.time()
        self.monitoring_thread = None
        self.stop_event = threading.Event()
    
    def detect_anomalies(self, system_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Run all configured anomaly detectors on the system state.
        
        Args:
            system_state: Current state of the system, including service statuses
        
        Returns:
            List of all detected anomalies from all detectors
        """
        all_new_anomalies = []
        for detector in self.detectors:
            try:
                # Assuming service_statuses is what detectors need
                service_statuses = system_state.get("service_statuses", {})
                new_anomalies = detector.detect_anomalies(service_statuses)
                all_new_anomalies.extend(new_anomalies)
            except Exception as e:
                logger.error(f"Error in detector {detector.__class__.__name__}: {e}")
        
        if all_new_anomalies:
            self.all_anomalies.extend(all_new_anomalies)
            logger.info(f"Detected {len(all_new_anomalies)} new anomalies.")

        # Periodically save anomalies to disk
        current_time = time.time()
        if current_time - self.last_save_time > self.save_interval:
            self._save_anomalies()
            self.last_save_time = current_time
        
        return all_new_anomalies
    
    def get_active_anomalies(self, max_age_seconds: int = 300) -> List[Dict[str, Any]]:
        """
        Get all active anomalies across all detectors.
        
        Args:
            max_age_seconds: Max age of an anomaly to be considered active
        
        Returns:
            List of active anomalies
        """
        active_anomalies = []
        current_time = time.time()
        for anomaly in self.all_anomalies:
            if current_time - anomaly.get("timestamp", 0) <= max_age_seconds:
                active_anomalies.append(anomaly)
        return active_anomalies
    
    def get_all_anomalies(self) -> List[Dict[str, Any]]:
        """
        Get all detected anomalies.
        
        Returns:
            List of all anomalies
        """
        return self.all_anomalies
    
    def _save_anomalies(self):
        """Save all anomalies to a single file."""
        if not self.all_anomalies:
            return
        
        try:
            filename = os.path.join("data/anomalies", f"anomalies_summary_{int(time.time())}.json")
            with open(filename, 'w') as f:
                json.dump(self.all_anomalies, f, indent=2)
            logger.info(f"Saved {len(self.all_anomalies)} anomalies to {filename}")
        except Exception as e:
            logger.error(f"Failed to save anomalies: {e}")

    def stop_monitoring(self):
        """Stop any background monitoring threads."""
        self.stop_event.set()
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join()