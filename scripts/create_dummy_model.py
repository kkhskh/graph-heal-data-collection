import joblib
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def create_dummy_model():
    """
    Creates and saves a dummy RandomForestClassifier model.
    The model is trained on synthetic data.
    """
    model_dir = 'models'
    model_path = os.path.join(model_dir, 'fault_detection_model.joblib')

    if os.path.exists(model_path):
        print(f"Model already exists at {model_path}. Skipping creation.")
        return

    print("Creating a dummy model...")

    # Define feature names based on what the MLDetector expects
    services = ['service_a', 'service_b', 'service_c', 'service_d']
    metrics = [
        'container_cpu_usage_seconds_total',
        'container_memory_usage_bytes',
        'service_latency_seconds_sum',
        'service_latency_seconds_count',
        'service_latency_seconds_bucket'
    ]
    feature_names = [f"{s}_{m}" for s in services for m in metrics]

    # Generate synthetic data
    X_train = pd.DataFrame(np.random.rand(100, len(feature_names)), columns=feature_names)
    y_train = np.random.choice(['normal', 'cpu_stress', 'memory_leak'], 100)

    # Train a simple model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Save the model
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    joblib.dump(model, model_path)
    
    print(f"Dummy model saved to {model_path}")

if __name__ == "__main__":
    create_dummy_model() 