import json
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from datetime import datetime

class ExperimentalValidator:
    def __init__(self, base_dir: str = 'results'):
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / 'raw'
        self.processed_dir = self.base_dir / 'processed'
        self.splits_dir = self.base_dir / 'splits'
        self.plots_dir = self.base_dir / 'plots'
        
        # Create plots directory
        self.plots_dir.mkdir(parents=True, exist_ok=True)
    
    def validate_directory_structure(self) -> bool:
        """Validate the experimental directory structure."""
        required_dirs = [
            self.raw_dir,
            self.processed_dir,
            self.splits_dir,
            self.plots_dir
        ]
        
        for dir_path in required_dirs:
            if not dir_path.exists():
                print(f"Error: Required directory {dir_path} does not exist")
                return False
        
        return True
    
    def validate_split_distribution(self) -> Tuple[bool, Dict]:
        """Validate the distribution of data across splits."""
        # Load split metadata
        split_file = self.splits_dir / 'split_metadata.json'
        if not split_file.exists():
            print("Error: Split metadata file not found")
            return False, {}
        
        with open(split_file, 'r') as f:
            split_metadata = json.load(f)
        
        # Count files in each split
        split_counts = {
            'train': 0,
            'validation': 0,
            'test': 0
        }
        
        for group, splits in split_metadata['experiment_groups'].items():
            for split_name, files in splits.items():
                split_counts[split_name] += len(files)
        
        # Validate distribution
        total = sum(split_counts.values())
        distribution = {
            split: count / total
            for split, count in split_counts.items()
        }
        
        # Check if distribution is reasonable
        is_valid = (
            0.3 <= distribution['train'] <= 0.5 and
            0.2 <= distribution['validation'] <= 0.3 and
            0.2 <= distribution['test'] <= 0.3
        )
        
        return is_valid, distribution
    
    def analyze_metric_distributions(self) -> Dict:
        """Analyze metric distributions across experiments."""
        distributions = {}
        
        # Process each experiment file
        for file_path in self.processed_dir.glob('*.json'):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Check if this is a detailed experiment file with metrics
                if 'experiment_name' not in data:
                    print(f"Skipping {file_path.name}: missing 'experiment_name' key")
                    continue
                    
                if 'metrics' not in data:
                    print(f"Skipping {file_path.name}: missing 'metrics' key")
                    continue
                
                experiment_name = data['experiment_name']
                metrics = data['metrics']
                
                if not metrics:
                    print(f"Skipping {file_path.name}: empty metrics list")
                    continue
                
                # Calculate distributions for each metric
                metric_distributions = {}
                for metric_name in metrics[0].keys():
                    values = [m[metric_name] for m in metrics]
                    metric_distributions[metric_name] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values)
                    }
                
                distributions[experiment_name] = metric_distributions
                
            except Exception as e:
                print(f"Error processing {file_path.name}: {e}")
                continue
        
        return distributions
    
    def plot_metric_distributions(self, distributions: Dict) -> None:
        """Plot metric distributions for visualization."""
        for experiment_name, metric_dist in distributions.items():
            # Create subplot for each metric
            n_metrics = len(metric_dist)
            fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 4 * n_metrics))
            if n_metrics == 1:
                axes = [axes]
            
            for ax, (metric_name, stats) in zip(axes, metric_dist.items()):
                # Plot distribution
                ax.bar(['Min', 'Mean', 'Max'],
                      [stats['min'], stats['mean'], stats['max']])
                ax.set_title(f"{experiment_name} - {metric_name}")
                ax.set_ylabel('Value')
                
                # Add standard deviation as error bars
                ax.errorbar(['Mean'], [stats['mean']],
                          yerr=[stats['std']],
                          fmt='none',
                          color='red',
                          capsize=5)
            
            plt.tight_layout()
            plt.savefig(self.plots_dir / f"{experiment_name}_distributions.png")
            plt.close()
    
    def validate_experimental_setup(self) -> bool:
        """Run all validation checks."""
        print("Validating experimental setup...")
        
        # Check directory structure
        if not self.validate_directory_structure():
            return False
        
        # Validate split distribution
        is_valid, distribution = self.validate_split_distribution()
        if not is_valid:
            print("Warning: Unusual split distribution detected")
            print(f"Distribution: {distribution}")
        else:
            print("Split distribution is valid")
            print(f"Distribution: {distribution}")
        
        # Analyze metric distributions
        print("\nAnalyzing metric distributions...")
        distributions = self.analyze_metric_distributions()
        
        # Plot distributions
        print("Generating distribution plots...")
        self.plot_metric_distributions(distributions)
        
        print("\nValidation complete!")
        print(f"Plots saved in {self.plots_dir}")
        return True

def main():
    # Create validator instance
    validator = ExperimentalValidator()
    
    # Run validation
    validator.validate_experimental_setup()

if __name__ == '__main__':
    main() 