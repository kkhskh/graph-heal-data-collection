import joblib
import pandas as pd
from graph_heal.anomaly_detection import AnomalyDetector
import os

class MLDetector(AnomalyDetector):
    """
    A fault detector that uses a pre-trained machine learning model.
    """
    def __init__(self, model_path='models/fault_detection_model.joblib'):
        """
        Initializes the detector by loading the trained model.

        Args:
            model_path (str): The path to the saved classification model.
        """
        super().__init__()
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")
        
        self.model = joblib.load(model_path)
        # Handle cases where the model might not have feature_names_in_
        if hasattr(self.model, 'feature_names_in_'):
            self.feature_names = self.model.feature_names_in_
        else:
            # This is a fallback. You might need to define the features manually
            # if the model doesn't store them. For now, we'll handle this gracefully.
            self.feature_names = None
            print("Warning: Model does not have 'feature_names_in_'. Feature order might be important.")


    def detect_anomalies(self, service_statuses: dict) -> list:
        """
        Detects anomalies in the given metrics using the loaded ML model.

        Args:
            service_statuses (dict): A dictionary of service statuses, where keys are service IDs.

        Returns:
            list: A list of dictionaries, where each dictionary represents an anomaly.
        """
        if not service_statuses or self.feature_names is None:
            return []

        # Create a single-row DataFrame for prediction
        live_metrics = {}
        for service_id, status in service_statuses.items():
            # In live monitoring, metric names might not have the service_id prefix.
            # We need to construct the feature names as the model expects them.
            # Example: 'service_a_container_cpu_usage_seconds_total'
            for metric_name, value in status.get('metrics', {}).items():
                feature_name = f"{service_id}_{metric_name}"
                if feature_name in self.feature_names:
                    live_metrics[feature_name] = value
        
        # Ensure all features are present, fill missing with 0
        live_df_row = {feature: live_metrics.get(feature, 0) for feature in self.feature_names}
        
        live_df = pd.DataFrame([live_df_row], columns=self.feature_names)

        # Predict faults
        try:
            prediction = self.model.predict(live_df)
        except Exception as e:
            print(f"Error during prediction: {e}")
            return []

        anomalies = []
        if prediction[0] != 'normal':
            fault_type = prediction[0]
            
            # Heuristic to find the root cause service
            root_cause_service = self._find_root_cause_service(fault_type, service_statuses)
            
            anomaly_info = {
                'id': f"ml_anomaly_{root_cause_service}_{pd.Timestamp.now().isoformat()}",
                'service_id': root_cause_service,
                'type': 'ml_prediction',
                'fault_type': fault_type,
                'timestamp': pd.Timestamp.now().isoformat(),
                'description': f"ML model predicted fault of type '{fault_type}'"
            }
            anomalies.append(anomaly_info)

        return anomalies

    def _find_root_cause_service(self, fault_type: str, service_statuses: dict) -> str:
        """
        A heuristic to find the service most likely responsible for a given fault type.
        """
        target_metric = ''
        # This mapping depends on the feature names used during training.
        # These are examples and might need to be adjusted.
        if 'cpu' in fault_type:
            target_metric = 'container_cpu_usage_seconds_total'
        elif 'memory' in fault_type:
            target_metric = 'container_memory_usage_bytes'
        elif 'latency' in fault_type:
            target_metric = 'service_latency_seconds_sum' # Or another relevant latency metric

        if not target_metric:
            # If we can't map fault type to a metric, we can't find the root cause.
            return 'unknown_service'

        max_val = -1
        root_cause_service = 'unknown_service'

        for service_id, status in service_statuses.items():
            val = status.get('metrics', {}).get(target_metric, -1)
            if val > max_val:
                max_val = val
                root_cause_service = service_id
        
        return root_cause_service

    def detect(self, metrics: dict) -> list:
        """
        Placeholder for compatibility if something still calls 'detect'.
        """
        # This method is for compatibility. The main logic is in detect_anomalies.
        print("Warning: MLDetector.detect() called. Please use detect_anomalies().")
        return []