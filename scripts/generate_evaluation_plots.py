import json
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict

class EvaluationPlotGenerator:
    def __init__(self, event_log_file):
        self.log_file = event_log_file
        self.data = self.load_data()
        self.output_dir = os.path.join(os.path.dirname(event_log_file), 'plots')
        os.makedirs(self.output_dir, exist_ok=True)
        self.results = self._process_log()

    def load_data(self):
        with open(self.log_file, 'r') as f:
            return json.load(f)

    def _process_log(self):
        """Process the detailed event log to calculate metrics."""
        results = {
            "baseline": {"tp": 0, "fp": 0, "fn": 0, "latencies": [], "correctly_localized": 0},
            "graph_heal": {"tp": 0, "fp": 0, "fn": 0, "latencies": [], "correctly_localized": 0},
        }
        total_faults = len(self.data)

        for event in self.data:
            gt = event['ground_truth']
            
            # --- Analyze Baseline ---
            baseline_hit = False
            for det in event['baseline_detections']:
                if det['service'] == gt['service']:
                    results['baseline']['tp'] += 1
                    latency = det['detection_time'] - gt['start_time']
                    results['baseline']['latencies'].append(latency)
                    results['baseline']['correctly_localized'] += 1
                    baseline_hit = True
            else:
                    results['baseline']['fp'] += 1
            if not baseline_hit:
                results['baseline']['fn'] += 1

            # --- Analyze Graph-Heal ---
            gh_hit = False
            for det in event['graph_heal_detections']:
                if det['service'] == gt['service']:
                    results['graph_heal']['tp'] += 1
                    latency = det['detection_time'] - gt['start_time']
                    results['graph_heal']['latencies'].append(latency)
                    if det.get('is_root_cause', False):
                        results['graph_heal']['correctly_localized'] += 1
                    gh_hit = True
            else:
                    results['graph_heal']['fp'] += 1
            if not gh_hit:
                results['graph_heal']['fn'] += 1
        
        results['baseline']['total_faults'] = total_faults
        results['graph_heal']['total_faults'] = total_faults
        return results

    def plot_detection_performance(self):
        """Plot Precision, Recall, and F1-Score."""
        labels = ['Baseline', 'Graph-Heal']
        metrics = defaultdict(list)
        
        for method in ['baseline', 'graph_heal']:
            tp = self.results[method]['tp']
            fp = self.results[method]['fp']
            fn = self.results[method]['fn']
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics['Precision'].append(precision)
            metrics['Recall'].append(recall)
            metrics['F1-Score'].append(f1)

        x = np.arange(len(labels))
        width = 0.25
        fig, ax = plt.subplots(figsize=(10, 6))
        
        rects1 = ax.bar(x - width, metrics['Precision'], width, label='Precision')
        rects2 = ax.bar(x, metrics['Recall'], width, label='Recall')
        rects3 = ax.bar(x + width, metrics['F1-Score'], width, label='F1-Score')

        ax.set_ylabel('Scores')
        ax.set_title('Detection Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.set_ylim(0, 1.1)
        fig.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '1_detection_performance.png'))
        plt.close()

    def plot_detection_latency(self):
        """Plot average detection latency."""
        labels = ['Baseline', 'Graph-Heal']
        avg_latencies = [
            np.mean(self.results['baseline']['latencies']),
            np.mean(self.results['graph_heal']['latencies'])
        ]
        
        plt.figure(figsize=(8, 6))
        plt.bar(labels, avg_latencies, color=['skyblue', 'salmon'])
        plt.ylabel('Time (seconds)')
        plt.title('Average Time-to-Detect Fault')
        plt.savefig(os.path.join(self.output_dir, '2_detection_latency.png'))
        plt.close()

    def plot_localization_accuracy(self):
        """Plot fault localization accuracy."""
        labels = ['Baseline', 'Graph-Heal']
        accuracies = [
            self.results['baseline']['correctly_localized'] / self.results['baseline']['total_faults'],
            self.results['graph_heal']['correctly_localized'] / self.results['graph_heal']['total_faults']
        ]

        plt.figure(figsize=(8, 6))
        plt.bar(labels, accuracies, color=['skyblue', 'salmon'])
        plt.ylabel('Accuracy Rate')
        plt.title('Fault Localization Accuracy')
        plt.ylim(0, 1.1)
        plt.savefig(os.path.join(self.output_dir, '3_localization_accuracy.png'))
        plt.close()

    def plot_cascading_failure_timeline(self):
        """Plot a timeline for the cascading failure scenario."""
        cascade_event = next((e for e in self.data if "Cascading" in e['scenario_name']), None)
        if not cascade_event:
            print("Cascading failure scenario not found in log.")
            return

        plt.figure(figsize=(12, 6))
        start_time = cascade_event['ground_truth']['start_time']
        
        # Events to plot: (time, label, color)
        events = []
        events.append((0, "Fault Injected", "red"))
        
        for det in cascade_event['baseline_detections']:
            events.append((det['detection_time'] - start_time, f"Baseline Detects\n({det['service']})", "blue"))
            
        for det in cascade_event['graph_heal_detections']:
            events.append((det['detection_time'] - start_time, f"Graph-Heal Detects\n(Root Cause)", "green"))
            
        if cascade_event['recovery_time']:
            events.append((cascade_event['recovery_time'] - start_time, "Recovery Action", "purple"))

        # Sort events by time
        events.sort()

        times = [e[0] for e in events]
        labels = [e[1] for e in events]
        colors = [e[2] for e in events]

        plt.stem(times, [1] * len(times), linefmt='k--', markerfmt=' ', basefmt=" ")
        
        for i, (time, label, color) in enumerate(events):
            plt.text(time, 1.01 + (i%3 * 0.08), label, ha='center', color=color, weight='bold')
            plt.plot(time, 1, 'o', color=color)

        plt.title("Timeline of Cascading Failure Scenario")
        plt.xlabel("Time Since Fault Injection (seconds)")
        plt.yticks([])
        plt.ylim(0.95, 1.5)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '4_cascading_timeline.png'))
        plt.close()

    def generate_all_plots(self):
        print("Generating plots...")
        self.plot_detection_performance()
        self.plot_detection_latency()
        self.plot_localization_accuracy()
        self.plot_cascading_failure_timeline()
        print(f"All plots saved to {self.output_dir}")

if __name__ == '__main__':
    data_dir = 'data/comparison'
    all_log_files = [
        os.path.join(data_dir, f) 
        for f in os.listdir(data_dir) 
        if f.startswith('experiment_event_log_') and f.endswith('.json')
    ]
    if not all_log_files:
        print("No event log file found. Please run run_controlled_experiment.py first.")
    else:
        latest_log_file = max(all_log_files, key=os.path.getctime)
        print(f"Using latest event log file: {latest_log_file}")
        plotter = EvaluationPlotGenerator(latest_log_file)
        plotter.generate_all_plots() 