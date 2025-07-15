import json
import numpy as np
from sklearn.model_selection import KFold
from datetime import datetime
import os
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

class CrossValidator:
    def __init__(self, n_splits=5):
        self.n_splits = n_splits
        self.kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
    def load_experiments(self, data_dir: str) -> List[Dict]:
        """Load all experiments from directory"""
        experiments = []
        for filename in os.listdir(data_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(data_dir, filename)
                with open(filepath, 'r') as f:
                    experiments.append(json.load(f))
        return experiments
        
    def calculate_metrics(self, 
                         predictions: List[bool], 
                         actual: List[bool]) -> Dict[str, float]:
        """Calculate detection metrics"""
        tp = sum(1 for p, a in zip(predictions, actual) if p and a)
        fp = sum(1 for p, a in zip(predictions, actual) if p and not a)
        fn = sum(1 for p, a in zip(predictions, actual) if not p and a)
        tn = sum(1 for p, a in zip(predictions, actual) if not p and not a)
        
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
    def detect_anomalies(self, metrics: List[Dict], params: Dict) -> List[bool]:
        window_size = params['window_size']
        z_threshold = params['z_score_threshold']
        min_duration = params['min_anomaly_duration']

        # Extract a single metric (e.g., service_cpu_usage) from the metrics list
        metric_values = np.array([m['service_cpu_usage'] for m in metrics])

        # Calculate rolling statistics using mode='same' to maintain length
        rolling_mean = np.convolve(metric_values, np.ones(window_size)/window_size, mode='same')
        rolling_std = np.array([
            np.std(metric_values[max(0, i-window_size//2):min(len(metric_values), i+window_size//2+1)])
            for i in range(len(metric_values))
        ])

        # Calculate z-scores
        z_scores = np.zeros_like(metric_values)
        for i in range(len(metric_values)):
            if rolling_std[i] > 0:
                z_scores[i] = (metric_values[i] - rolling_mean[i]) / rolling_std[i]

        # Detect anomalies
        anomalies = np.abs(z_scores) > z_threshold

        # Apply minimum duration filter
        for i in range(len(anomalies)):
            if anomalies[i]:
                if i + min_duration <= len(anomalies):
                    if not all(anomalies[i:i+min_duration]):
                        anomalies[i] = False

        return anomalies.tolist()
        
    def perform_cross_validation(self, 
                               experiments: List[Dict],
                               params: Dict) -> Dict[str, List[float]]:
        """Perform k-fold cross validation"""
        metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': []
        }
        
        # Convert experiments to numpy arrays for indexing
        X = np.array(experiments)
        
        # Perform k-fold cross validation
        for train_idx, val_idx in self.kf.split(X):
            # Split data
            train_experiments = X[train_idx]
            val_experiments = X[val_idx]
            
            # Evaluate on validation set
            fold_metrics = []
            for experiment in val_experiments:
                predictions = self.detect_anomalies(
                    experiment['metrics'],
                    params
                )
                experiment_metrics = self.calculate_metrics(
                    predictions,
                    experiment['fault_periods']
                )
                fold_metrics.append(experiment_metrics)
                
            # Calculate average metrics for this fold
            for metric in metrics:
                avg_metric = np.mean([m[metric] for m in fold_metrics])
                metrics[metric].append(avg_metric)
                
        return metrics
        
    def plot_results(self, metrics: Dict[str, List[float]]):
        """Plot cross-validation results"""
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Cross-Validation Results')
        
        # Plot each metric
        for (metric, values), ax in zip(metrics.items(), axes.flat):
            sns.boxplot(data=values, ax=ax)
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.set_ylabel('Score')
            ax.set_xticklabels([f'Fold {i+1}' for i in range(self.n_splits)])
            
        plt.tight_layout()
        plt.savefig('results/cross_validation_results.png')
        plt.close()
        
    def save_results(self, metrics: Dict[str, List[float]]):
        """Save cross-validation results to file"""
        # Calculate summary statistics
        summary = {
            metric: {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
            for metric, values in metrics.items()
        }
        
        output = {
            'timestamp': datetime.now().isoformat(),
            'n_splits': self.n_splits,
            'metrics': metrics,
            'summary': summary
        }
        
        os.makedirs('results/validation', exist_ok=True)
        with open('results/validation/cross_validation_results.json', 'w') as f:
            json.dump(output, f, indent=2)
            
def main():
    # Initialize cross validator
    validator = CrossValidator(n_splits=5)
    
    # Load experiments
    print("Loading experiments...")
    experiments = validator.load_experiments('results/processed')
    
    # Load best parameters from optimization
    with open('results/optimization/parameter_optimization_results.json', 'r') as f:
        optimization_results = json.load(f)
    best_params = optimization_results['best_parameters']
    
    # Perform cross validation
    print("Performing cross validation...")
    metrics = validator.perform_cross_validation(experiments, best_params)
    
    # Plot and save results
    print("Plotting results...")
    validator.plot_results(metrics)
    validator.save_results(metrics)
    
    print("\nCross validation complete!")
    print("Results saved to:")
    print("  - results/validation/cross_validation_results.json")
    print("  - results/cross_validation_results.png")
    
if __name__ == "__main__":
    main() 