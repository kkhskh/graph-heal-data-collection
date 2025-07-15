import json
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import os
import argparse
from datetime import datetime

class FinalEvaluator:
    def __init__(self):
        self.metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': []
        }
        
    def convert_fault_periods_to_timesteps(self, experiment: Dict) -> List[bool]:
        """Convert fault periods to per-timestep ground truth vector"""
        # Get timestamps from experiment
        timestamps = [datetime.fromisoformat(ts) for ts in experiment['timestamps']]
        
        # Initialize ground truth vector
        ground_truth = [False] * len(timestamps)
        
        # For each fault period, mark all timestamps within it as True
        for period in experiment['fault_periods']:
            start_time = datetime.fromisoformat(period['start'])
            end_time = datetime.fromisoformat(period['end'])
            
            for i, ts in enumerate(timestamps):
                if start_time <= ts <= end_time:
                    ground_truth[i] = True
                    
        return ground_truth
        
    def load_experiments(self, data_dir: str, test_only: bool = True) -> List[Dict]:
        """Load test experiments from directory"""
        experiments = []
        for filename in os.listdir(data_dir):
            if filename.endswith('.json'):
                # If test_only, only load experiments 4 and 5
                if test_only and not ('_4.json' in filename or '_5.json' in filename):
                    continue
                filepath = os.path.join(data_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        experiment = json.load(f)
                        
                    # Check if this experiment has the required keys
                    if 'metrics' not in experiment:
                        print(f"Skipping {filename}: missing 'metrics' key")
                        continue
                        
                    if 'fault_periods' not in experiment:
                        print(f"Skipping {filename}: missing 'fault_periods' key")
                        continue
                        
                    experiment['filename'] = filename  # Add filename for debugging
                    experiments.append(experiment)
                    
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
                    continue
                    
        print(f"Loaded {len(experiments)} valid experiments")
        return experiments
        
    def load_best_params(self) -> Dict:
        """Load best parameters from optimization results"""
        with open('results/optimization/parameter_optimization_results.json', 'r') as f:
            results = json.load(f)
        return results['best_parameters']
        
    def calculate_metrics(self, 
                         predictions: List[bool], 
                         ground_truth: List[bool]) -> Dict[str, float]:
        """Calculate detection metrics"""
        tp = sum(1 for p, a in zip(predictions, ground_truth) if p and a)
        fp = sum(1 for p, a in zip(predictions, ground_truth) if p and not a)
        fn = sum(1 for p, a in zip(predictions, ground_truth) if not p and a)
        tn = sum(1 for p, a in zip(predictions, ground_truth) if not p and not a)
        
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
        
    def detect_anomalies(self, metrics: List[Dict], params: Dict, debug: bool = False) -> List[bool]:
        """Detect anomalies using specified parameters. If debug, print z-score, mean, std, and anomaly for first 30 steps."""
        window_size = params['window_size']
        z_threshold = params['z_score_threshold']
        min_duration = params['min_anomaly_duration']
        
        # Extract metric values
        metric_values = np.array([m['service_cpu_usage'] for m in metrics])
        
        # Calculate rolling statistics
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
        
        # Debug output for first 30 time steps
        if debug:
            print("\n[DEBUG] Detection logic for first 30 time steps:")
            print(f"{'i':>3} {'value':>8} {'mean':>8} {'std':>8} {'z':>8} {'anomaly':>8}")
            for i in range(min(30, len(metric_values))):
                print(f"{i:3d} {metric_values[i]:8.3f} {rolling_mean[i]:8.3f} {rolling_std[i]:8.3f} {z_scores[i]:8.3f} {str(anomalies[i]):>8}")
        
        return anomalies.tolist()
        
    def calculate_confidence_interval(self, scores: List[float], confidence: float = 0.95) -> Tuple[float, Tuple[float, float]]:
        """Calculate confidence interval from test results"""
        n = len(scores)
        mean = np.mean(scores)
        std_err = stats.sem(scores)  # Standard error
        
        # t-distribution for small samples
        t_val = stats.t.ppf((1 + confidence) / 2, df=n-1)
        margin_error = t_val * std_err
        
        ci_lower = mean - margin_error
        ci_upper = mean + margin_error
        
        return mean, (ci_lower, ci_upper)
        
    def test_vs_random_baseline(self, scores: List[float], random_baseline: float = 0.5) -> Dict:
        """Test if results are significantly better than random"""
        t_stat, p_value = stats.ttest_1samp(scores, random_baseline)
        
        return {
            'p_value': p_value,
            'significant': p_value < 0.05,
            't_statistic': t_stat
        }
        
    def cohens_d(self, scores: List[float], baseline: float = 0.5) -> float:
        """Calculate Cohen's d effect size"""
        mean_diff = np.mean(scores) - baseline
        pooled_std = np.std(scores, ddof=1)
        
        return mean_diff / pooled_std
        
    def evaluate(self, experiments: List[Dict], params: Dict) -> Dict[str, List[float]]:
        """Evaluate on test data, with debug output for each experiment"""
        results = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': []
        }
        
        for idx, experiment in enumerate(experiments):
            # Debug only for the first experiment
            debug = (idx == 0)
            predictions = self.detect_anomalies(experiment['metrics'], params, debug=debug)
            ground_truth = self.convert_fault_periods_to_timesteps(experiment)
            metrics = self.calculate_metrics(predictions, ground_truth)
            
            # Debug printout
            print(f"\nExperiment {idx+1}: {experiment.get('filename', 'unknown')}")
            print(f"  Experiment name: {experiment.get('experiment_name', 'unknown')}")
            print(f"  Fault type: {experiment.get('fault_type', 'unknown')}")
            print(f"  Pattern: {experiment.get('pattern', 'unknown')}")
            print(f"  Predictions (first 20): {predictions[:20]}")
            print(f"  Ground truth (first 20): {ground_truth[:20]}")
            print(f"  #Predicted positives: {sum(predictions)} / {len(predictions)}")
            print(f"  #Ground truth positives: {sum(ground_truth)} / {len(ground_truth)}")
            print(f"  Metrics: {metrics}")
            print("  ---")
            
            # Print ground truth summary for debug
            if debug:
                print("[DEBUG] Ground truth summary:")
                print(f"  Total samples: {len(ground_truth)}")
                print(f"  Total positives: {sum(ground_truth)}")
                print(f"  First 30 ground truth: {ground_truth[:30]}")
                print(f"  Fault periods: {experiment['fault_periods']}")
            
            for metric, value in metrics.items():
                results[metric].append(value)
        return results
        
    def plot_results(self, results: Dict[str, List[float]]):
        """Plot evaluation results with confidence intervals"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Final Test Evaluation Results')
        
        for (metric, values), ax in zip(results.items(), axes.flat):
            # Calculate confidence interval
            mean, (ci_lower, ci_upper) = self.calculate_confidence_interval(values)
            
            # Plot
            sns.boxplot(data=values, ax=ax)
            ax.axhline(y=mean, color='r', linestyle='--', label='Mean')
            ax.axhline(y=ci_lower, color='g', linestyle=':', label='95% CI')
            ax.axhline(y=ci_upper, color='g', linestyle=':')
            
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.set_ylabel('Score')
            ax.legend()
            
        plt.tight_layout()
        plt.savefig('results/final_evaluation_results.png')
        plt.close()
        
    def save_results(self, results: Dict[str, List[float]]):
        """Save evaluation results with statistical analysis"""
        output = {
            'timestamp': datetime.now().isoformat(),
            'metrics': {},
            'statistical_analysis': {}
        }
        
        for metric, values in results.items():
            # Convert NumPy values to Python native types
            values_list = [float(v) for v in values]
            
            # Calculate confidence interval
            mean, ci = self.calculate_confidence_interval(values_list)
            
            # Test vs random baseline
            significance = self.test_vs_random_baseline(values_list)
            
            # Calculate effect size
            effect_size = self.cohens_d(values_list)
            
            output['metrics'][metric] = {
                'values': values_list,
                'mean': float(mean),
                'confidence_interval': [float(ci[0]), float(ci[1])]
            }
            
            output['statistical_analysis'][metric] = {
                'vs_random_baseline': {
                    'p_value': float(significance['p_value']),
                    'significant': bool(significance['significant']),
                    't_statistic': float(significance['t_statistic'])
                },
                'effect_size': float(effect_size)
            }
            
        os.makedirs('results/final', exist_ok=True)
        with open('results/final/final_evaluation_results.json', 'w') as f:
            json.dump(output, f, indent=2)
            
def main():
    parser = argparse.ArgumentParser(description='Final evaluation of detection algorithm')
    parser.add_argument('--test_only', action='store_true', help='Only evaluate on test experiments')
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = FinalEvaluator()
    
    # Load test experiments
    print("Loading test experiments...")
    experiments = evaluator.load_experiments('results/processed', args.test_only)
    
    # Load best parameters
    print("Loading best parameters...")
    best_params = evaluator.load_best_params()
    
    # Evaluate
    print("Running final evaluation...")
    results = evaluator.evaluate(experiments, best_params)
    
    # Plot and save results
    print("Plotting results...")
    evaluator.plot_results(results)
    evaluator.save_results(results)
    
    # Print summary
    print("\nFinal evaluation complete!")
    print("\nResults summary:")
    for metric, values in results.items():
        mean, ci = evaluator.calculate_confidence_interval(values)
        significance = evaluator.test_vs_random_baseline(values)
        effect_size = evaluator.cohens_d(values)
        
        print(f"\n{metric.replace('_', ' ').title()}:")
        print(f"  Mean: {mean:.3f}")
        print(f"  95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
        print(f"  P-value vs random: {significance['p_value']:.3f}")
        print(f"  Effect size: {effect_size:.3f}")
    
    print("\nResults saved to:")
    print("  - results/final/final_evaluation_results.json")
    print("  - results/final_evaluation_results.png")
    
if __name__ == "__main__":
    main() 