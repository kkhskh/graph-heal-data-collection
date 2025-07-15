import json
from datetime import datetime
import numpy as np
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt

def debug_ground_truth(experiment_file: str) -> List[bool]:
    """Debug ground truth labeling"""
    with open(experiment_file) as f:
        data = json.load(f)
    
    timestamps = [datetime.fromisoformat(ts) for ts in data['timestamps']]
    fault_periods = data['fault_periods']
    
    print(f"\nGround Truth Analysis for {experiment_file}")
    print(f"Total timestamps: {len(timestamps)}")
    print(f"Fault periods: {fault_periods}")
    
    # Convert to labels
    ground_truth = [False] * len(timestamps)
    for period in fault_periods:
        start_time = datetime.fromisoformat(period['start'])
        end_time = datetime.fromisoformat(period['end'])
        print(f"\nFault period: {start_time} to {end_time}")
        
        # Count how many timestamps fall in fault period
        fault_count = 0
        for i, timestamp in enumerate(timestamps):
            if start_time <= timestamp <= end_time:
                ground_truth[i] = True
                fault_count += 1
        
        print(f"Timestamps labeled as faults: {fault_count}")
    
    total_fault_labels = sum(ground_truth)
    print(f"\nTotal fault labels: {total_fault_labels} out of {len(timestamps)}")
    print(f"Fault percentage: {total_fault_labels/len(timestamps)*100:.1f}%")
    
    return ground_truth

def detect_anomalies_rolling(metrics: List[Dict], window_size: int, z_threshold: float, min_duration: int) -> List[bool]:
    cpu_values = np.array([m['service_cpu_usage'] for m in metrics])
    rolling_mean = np.convolve(cpu_values, np.ones(window_size)/window_size, mode='same')
    rolling_std = np.array([
        np.std(cpu_values[max(0, i-window_size//2):min(len(cpu_values), i+window_size//2+1)])
        for i in range(len(cpu_values))
    ])
    z_scores = np.zeros_like(cpu_values)
    for i in range(len(cpu_values)):
        if rolling_std[i] > 0:
            z_scores[i] = (cpu_values[i] - rolling_mean[i]) / rolling_std[i]
    anomalies = np.abs(z_scores) > z_threshold
    for i in range(len(anomalies)):
        if anomalies[i]:
            if i + min_duration <= len(anomalies):
                if not all(anomalies[i:i+min_duration]):
                    anomalies[i] = False
    return anomalies.tolist()

def detect_anomalies_fixed(metrics: List[Dict], pre_fault_points: int, z_threshold: float, min_duration: int) -> List[bool]:
    cpu_values = np.array([m['service_cpu_usage'] for m in metrics])
    baseline_mean = np.mean(cpu_values[:pre_fault_points])
    baseline_std = np.std(cpu_values[:pre_fault_points])
    if baseline_std == 0:
        z_scores = np.zeros_like(cpu_values)
    else:
        z_scores = (cpu_values - baseline_mean) / baseline_std
    anomalies = np.abs(z_scores) > z_threshold
    for i in range(len(anomalies)):
        if anomalies[i]:
            if i + min_duration <= len(anomalies):
                if not all(anomalies[i:i+min_duration]):
                    anomalies[i] = False
    return anomalies.tolist()

def detect_anomalies_hybrid(metrics: List[Dict], window_size: int, z_threshold: float, min_duration: int, fixed_period: int = 60) -> List[bool]:
    cpu_values = np.array([m['service_cpu_usage'] for m in metrics])
    # Rolling window z-score
    rolling_mean = np.convolve(cpu_values, np.ones(window_size)/window_size, mode='same')
    rolling_std = np.array([
        np.std(cpu_values[max(0, i-window_size//2):min(len(cpu_values), i+window_size//2+1)])
        for i in range(len(cpu_values))
    ])
    z_scores_rolling = np.zeros_like(cpu_values)
    for i in range(len(cpu_values)):
        if rolling_std[i] > 0:
            z_scores_rolling[i] = (cpu_values[i] - rolling_mean[i]) / rolling_std[i]
    # Detect first change-point (first time z-score exceeds threshold)
    change_idx = None
    for i in range(len(z_scores_rolling)):
        if abs(z_scores_rolling[i]) > z_threshold:
            change_idx = i
            break
    # Use fixed baseline after change-point
    baseline_mean = np.mean(cpu_values[:window_size])
    baseline_std = np.std(cpu_values[:window_size])
    z_scores_fixed = np.zeros_like(cpu_values)
    if baseline_std > 0:
        z_scores_fixed = (cpu_values - baseline_mean) / baseline_std
    # Build anomaly prediction
    anomalies = np.zeros_like(cpu_values, dtype=bool)
    if change_idx is not None:
        # Use fixed baseline for fixed_period after change-point
        for i in range(change_idx, min(change_idx+fixed_period, len(cpu_values))):
            if abs(z_scores_fixed[i]) > z_threshold:
                anomalies[i] = True
    # Apply min_duration filter
    for i in range(len(anomalies)):
        if anomalies[i]:
            if i + min_duration <= len(anomalies):
                if not all(anomalies[i:i+min_duration]):
                    anomalies[i] = False
    return anomalies.tolist()

def detect_anomalies_cusum(metrics: List[Dict], window_size: int = 60, threshold: float = 5.0, min_duration: int = 5, k: float = 2.0, return_stats: bool = False):
    """Detect anomalies using CUSUM (Cumulative Sum Control Chart) with improved parameters and optional stats return"""
    cpu_values = np.array([m['service_cpu_usage'] for m in metrics])
    
    # Calculate baseline statistics from initial window
    baseline_mean = np.mean(cpu_values[:window_size])
    baseline_std = np.std(cpu_values[:window_size])
    
    if baseline_std == 0:
        if return_stats:
            return [False] * len(cpu_values), np.zeros_like(cpu_values), np.zeros_like(cpu_values)
        return [False] * len(cpu_values)
    
    # Calculate standardized residuals
    residuals = (cpu_values - baseline_mean) / baseline_std
    
    # Initialize CUSUM statistics
    S_plus = np.zeros_like(cpu_values)
    S_minus = np.zeros_like(cpu_values)
    
    for i in range(1, len(cpu_values)):
        S_plus[i] = max(0, S_plus[i-1] + residuals[i] - k)
        S_minus[i] = max(0, S_minus[i-1] - residuals[i] - k)
    
    anomalies = np.zeros_like(cpu_values, dtype=bool)
    for i in range(len(cpu_values)):
        if S_plus[i] > threshold or S_minus[i] > threshold:
            anomalies[i] = True
    
    filtered_anomalies = np.zeros_like(anomalies)
    for i in range(len(anomalies)):
        if i + min_duration <= len(anomalies):
            window = anomalies[i:i+min_duration]
            if np.sum(window) >= 0.7 * min_duration:
                filtered_anomalies[i:i+min_duration] = True
    
    if return_stats:
        return filtered_anomalies.tolist(), S_plus, S_minus
    return filtered_anomalies.tolist()

def grid_search(metrics, ground_truth):
    print("\nGrid search for rolling window:")
    print(f"{'z_thr':>5} {'min_dur':>7} {'recall':>8} {'prec':>8} {'acc':>8} {'pos':>5}")
    for z_thr in [0.8, 1.0, 1.2, 1.5]:
        for min_dur in [2, 3, 5, 10]:
            preds = detect_anomalies_rolling(metrics, window_size=60, z_threshold=z_thr, min_duration=min_dur)
            m = debug_recall_calculation(preds, ground_truth)
            print(f"{z_thr:5.2f} {min_dur:7d} {m['recall']:8.3f} {m['precision']:8.3f} {m['accuracy']:8.3f} {sum(preds):5d}")
    print("\nGrid search for fixed baseline:")
    print(f"{'z_thr':>5} {'min_dur':>7} {'recall':>8} {'prec':>8} {'acc':>8} {'pos':>5}")
    for z_thr in [0.8, 1.0, 1.2, 1.5]:
        for min_dur in [2, 3, 5, 10]:
            preds = detect_anomalies_fixed(metrics, pre_fault_points=60, z_threshold=z_thr, min_duration=min_dur)
            m = debug_recall_calculation(preds, ground_truth)
            print(f"{z_thr:5.2f} {min_dur:7d} {m['recall']:8.3f} {m['precision']:8.3f} {m['accuracy']:8.3f} {sum(preds):5d}")

def debug_predictions(timestamps: List[datetime], metrics: List[Dict], ground_truth: List[bool]) -> List[bool]:
    """Debug prediction generation"""
    predictions = detect_anomalies_rolling(metrics, window_size=60, z_threshold=1.0, min_duration=3)
    
    print(f"\nPrediction Analysis")
    print(f"Total predictions: {len(predictions)}")
    print(f"Positive predictions: {sum(predictions)}")
    print(f"Prediction rate: {sum(predictions)/len(predictions)*100:.1f}%")
    
    # Check alignment
    if len(predictions) != len(ground_truth):
        print(f"ERROR: Length mismatch! Predictions: {len(predictions)}, Ground truth: {len(ground_truth)}")
        return None
    
    return predictions

def debug_recall_calculation(predictions: List[bool], ground_truth: List[bool]) -> Dict[str, float]:
    """Debug recall calculation"""
    # Manual calculation with debugging
    true_positives = 0
    false_negatives = 0
    false_positives = 0
    true_negatives = 0
    
    for i, (pred, gt) in enumerate(zip(predictions, ground_truth)):
        if pred and gt:
            true_positives += 1
        elif not pred and gt:
            false_negatives += 1
        elif pred and not gt:
            false_positives += 1
        else:
            true_negatives += 1
    
    print(f"\nDetailed Metrics")
    print(f"True Positives: {true_positives}")
    print(f"False Negatives: {false_negatives}")
    print(f"False Positives: {false_positives}")
    print(f"True Negatives: {true_negatives}")
    
    total_actual_faults = true_positives + false_negatives
    print(f"Total actual faults: {total_actual_faults}")
    
    if total_actual_faults > 0:
        recall = true_positives / total_actual_faults
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        accuracy = (true_positives + true_negatives) / len(predictions)
        
        print(f"\nCalculated Metrics")
        print(f"Recall: {recall:.3f} ({recall*100:.1f}%)")
        print(f"Precision: {precision:.3f} ({precision*100:.1f}%)")
        print(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    else:
        print("ERROR: No actual faults found in ground truth!")
        recall = precision = accuracy = 0
    
    return {
        'recall': recall,
        'precision': precision,
        'accuracy': accuracy
    }

def plot_results(timestamps, cpu_values, z_scores_rolling, z_scores_fixed, ground_truth, preds, change_idx=None):
    plt.figure(figsize=(15,8))
    t = range(len(cpu_values))
    plt.subplot(3,1,1)
    plt.plot(t, cpu_values, label='CPU Usage')
    plt.title('CPU Usage')
    plt.subplot(3,1,2)
    plt.plot(t, z_scores_rolling, label='Z-score (rolling)', color='blue')
    plt.plot(t, z_scores_fixed, label='Z-score (fixed)', color='orange', alpha=0.7)
    plt.axhline(1.0, color='red', linestyle='--', label='z=1.0')
    plt.axhline(-1.0, color='red', linestyle='--')
    if change_idx is not None:
        plt.axvline(change_idx, color='green', linestyle=':', label='Change-point')
    plt.title('Z-scores')
    plt.legend()
    plt.subplot(3,1,3)
    plt.plot(t, ground_truth, label='Ground Truth', color='black', drawstyle='steps-post')
    plt.plot(t, preds, label='Prediction', color='red', alpha=0.7, drawstyle='steps-post')
    plt.title('Ground Truth and Prediction')
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    experiment_file = "results/processed/cpu_test_cpu_experiment_4.json"
    with open(experiment_file) as f:
        data = json.load(f)
    ground_truth = debug_ground_truth(experiment_file)
    timestamps = [datetime.fromisoformat(ts) for ts in data['timestamps']]
    metrics = data['metrics']
    cpu_values = np.array([m['service_cpu_usage'] for m in metrics])
    
    # Run grid search
    grid_search(metrics, ground_truth)
    
    # Hybrid approach
    print("\nHybrid approach (rolling change-point, then fixed baseline):")
    preds_hybrid = detect_anomalies_hybrid(metrics, window_size=60, z_threshold=1.0, min_duration=3, fixed_period=60)
    m_hybrid = debug_recall_calculation(preds_hybrid, ground_truth)
    print(f"Hybrid: recall={m_hybrid['recall']:.3f}, precision={m_hybrid['precision']:.3f}, accuracy={m_hybrid['accuracy']:.3f}, positives={sum(preds_hybrid)}")
    
    # Improved CUSUM approach with stats
    print("\nImproved CUSUM approach (higher threshold and k, with stats):")
    preds_cusum, S_plus, S_minus = detect_anomalies_cusum(metrics, window_size=60, threshold=5.0, min_duration=5, k=2.0, return_stats=True)
    m_cusum = debug_recall_calculation(preds_cusum, ground_truth)
    print(f"CUSUM: recall={m_cusum['recall']:.3f}, precision={m_cusum['precision']:.3f}, accuracy={m_cusum['accuracy']:.3f}, positives={sum(preds_cusum)}")
    
    # For visualization
    rolling_mean = np.convolve(cpu_values, np.ones(60)/60, mode='same')
    rolling_std = np.array([
        np.std(cpu_values[max(0, i-30):min(len(cpu_values), i+31)]) for i in range(len(cpu_values))
    ])
    z_scores_rolling = np.zeros_like(cpu_values)
    for i in range(len(cpu_values)):
        if rolling_std[i] > 0:
            z_scores_rolling[i] = (cpu_values[i] - rolling_mean[i]) / rolling_std[i]
    
    baseline_mean = np.mean(cpu_values[:60])
    baseline_std = np.std(cpu_values[:60])
    z_scores_fixed = (cpu_values - baseline_mean) / baseline_std if baseline_std > 0 else np.zeros_like(cpu_values)
    
    change_idx = None
    for i in range(len(z_scores_rolling)):
        if abs(z_scores_rolling[i]) > 1.0:
            change_idx = i
            break
    
    # Plot results with both hybrid and CUSUM predictions and CUSUM stats
    plt.figure(figsize=(15,12))
    t = range(len(cpu_values))
    
    plt.subplot(5,1,1)
    plt.plot(t, cpu_values, label='CPU Usage')
    plt.title('CPU Usage')
    
    plt.subplot(5,1,2)
    plt.plot(t, z_scores_rolling, label='Z-score (rolling)', color='blue')
    plt.plot(t, z_scores_fixed, label='Z-score (fixed)', color='orange', alpha=0.7)
    plt.axhline(1.0, color='red', linestyle='--', label='z=1.0')
    plt.axhline(-1.0, color='red', linestyle='--')
    if change_idx is not None:
        plt.axvline(change_idx, color='green', linestyle=':', label='Change-point')
    plt.title('Z-scores')
    plt.legend()
    
    plt.subplot(5,1,3)
    plt.plot(t, S_plus, label='CUSUM S_plus', color='purple')
    plt.plot(t, S_minus, label='CUSUM S_minus', color='brown')
    plt.axhline(5.0, color='red', linestyle='--', label='CUSUM threshold')
    plt.axhline(-5.0, color='red', linestyle='--')
    plt.title('CUSUM Statistics')
    plt.legend()
    
    plt.subplot(5,1,4)
    plt.plot(t, ground_truth, label='Ground Truth', color='black', drawstyle='steps-post')
    plt.plot(t, preds_hybrid, label='Hybrid Prediction', color='red', alpha=0.7, drawstyle='steps-post')
    plt.title('Hybrid Approach')
    plt.legend()
    
    plt.subplot(5,1,5)
    plt.plot(t, ground_truth, label='Ground Truth', color='black', drawstyle='steps-post')
    plt.plot(t, preds_cusum, label='CUSUM Prediction', color='green', alpha=0.7, drawstyle='steps-post')
    plt.title('Improved CUSUM Approach')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main() 