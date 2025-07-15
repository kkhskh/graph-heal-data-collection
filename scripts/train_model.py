import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import argparse
import os

def train_model(capture_file, label_file, model_output_dir):
    """
    Trains a Random Forest classifier on labeled metric data.
    
    Args:
        capture_file (str): Path to the CSV file containing metric data.
        label_file (str): Path to the CSV file containing fault labels.
        model_output_dir (str): Directory to save the trained model.
    """
    print(f"Loading data from {capture_file} and labels from {label_file}")
    
    # Load data and labels
    try:
        df_metrics = pd.read_csv(capture_file)
        df_labels = pd.read_csv(label_file)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return

    # Convert time columns to numeric
    df_metrics['timestamp'] = pd.to_numeric(df_metrics['timestamp'])
    df_labels['start_time'] = pd.to_numeric(df_labels['start_time'])
    df_labels['end_time'] = pd.to_numeric(df_labels['end_time'])
    
    # Initialize a 'fault' column with a default value (e.g., 'normal')
    df_metrics['fault'] = 'normal'

    # Label metrics based on fault periods
    for _, fault_row in df_labels.iterrows():
        start = fault_row['start_time']
        end = fault_row['end_time']
        fault_type = fault_row['fault_type']
        
        # Find metrics that fall within the fault window and label them
        fault_indices = (df_metrics['timestamp'] >= start) & (df_metrics['timestamp'] <= end)
        df_metrics.loc[fault_indices, 'fault'] = fault_type

    # For simplicity, we drop the timestamp column after using it for labeling
    df = df_metrics.drop(columns=['timestamp'])

    if 'fault' not in df.columns:
        print("Error: 'fault' column could not be created. Check labeling logic.")
        return
        
    if df.empty:
        print("Error: No data to train on after merging. Check timestamps in files.")
        return

    print("Data loaded and merged successfully.")
    
    # --- Feature Engineering ---
    # The data is already in a wide format, so no one-hot encoding is needed.
    # The features are all columns except 'fault'.
    X = df.drop(columns=['fault'])
    y = df['fault']
    
    # Store the feature names in the order the model will see them
    feature_names = X.columns.tolist()

    # Split data for training and validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Training model on {len(X_train)} samples with {len(feature_names)} features...")
    print(f"Features: {feature_names}")

    # Initialize and train the classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    print("Model training completed.")
    
    # Evaluate the model
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy on test set: {accuracy:.4f}")

    # Save the trained model
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)
        
    model_path = os.path.join(model_output_dir, 'fault_detection_model.joblib')
    # Save both the model and the feature list together
    joblib.dump((model, feature_names), model_path)
    
    print(f"Model and feature list saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a fault detection model.")
    parser.add_argument("--capture-file", required=True, help="Path to the captured metrics CSV file.")
    parser.add_argument("--label-file", required=True, help="Path to the fault labels CSV file.")
    parser.add_argument("--model-output-dir", default="models", help="Directory to save the trained model.")
    
    args = parser.parse_args()
    
    train_model(args.capture_file, args.label_file, args.model_output_dir) 