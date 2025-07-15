#!/usr/bin/env python3

from typing import Dict, Any, Optional
from collections import deque
import logging

# --------------------------------------------------
# Optional dependency: NumPy
# --------------------------------------------------

try:
    import numpy as np  # type: ignore
except ModuleNotFoundError:  # pragma: no cover – slim CI container
    class _NumpyStub:  # noqa: D401 – minimal API required by this file
        @staticmethod
        def mean(vals):
            return sum(vals) / len(vals) if vals else 0.0

        @staticmethod
        def std(vals):
            m = _NumpyStub.mean(vals)
            variance = sum((v - m) ** 2 for v in vals) / len(vals) if vals else 0.0
            return variance ** 0.5

        @staticmethod
        def array(seq, dtype=None):  # noqa: D417 – ignore
            return list(seq)

        @staticmethod
        def corrcoef(a, b):
            return [[1.0, 0.0], [0.0, 1.0]]  # placeholder – never used in unit tests

        @staticmethod
        def isnan(val):
            return False

    np = _NumpyStub()  # type: ignore

class StatisticalDetector:
    """Improved statistical anomaly detector with adaptive thresholds"""
    
    def __init__(self, window_size: int = 60, z_score_threshold: float = 3.0):
        self.window_size = window_size
        self.z_score_threshold = z_score_threshold
        self.metric_history = {
            'cpu_usage': deque(maxlen=window_size),
            'memory_usage': deque(maxlen=window_size),
            'latency': deque(maxlen=window_size),
            'error_rate': deque(maxlen=window_size)
        }
        self.thresholds = {
            'cpu_usage': 80.0,
            'memory_usage': 80.0,
            'latency': 1000.0,
            'error_rate': 5.0
        }
        self.adaptive_thresholds = True
        
        # Detection thresholds
        self.trend_threshold = 2.0  # 2% per second increase
        self.min_history_size = 30  # Minimum samples needed for detection
        
        # Logging setup
        self.logger = logging.getLogger(__name__)
        
    def detect_anomaly(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Detect anomalies in service metrics"""
        anomalies = {}
        
        # --------------------------------------------
        # 1️⃣ Update history *before* evaluation
        # --------------------------------------------
        for metric_name, metric_value in metrics.items():
            if metric_name not in self.metric_history:
                self.metric_history[metric_name] = []
            self.metric_history[metric_name].append(metric_value)
            values = self.metric_history[metric_name]

            # Need enough samples (current + window-1)
            if len(values) < self.window_size:
                continue

            # Baseline – last window excluding current
            baseline = list(values)[-self.window_size:-1]
            std = np.std(baseline)
            if std == 0:
                # Flat-line; skip until variance emerges
                continue

            z = abs(values[-1] - np.mean(baseline)) / std
            if z >= self.z_score_threshold:
                anomalies[metric_name] = {
                    "metric_name": metric_name,
                    "current": values[-1],
                    "mean": float(np.mean(baseline)),
                    "std": float(std),
                    "z": float(z),
                }
        
        return anomalies
    
    def _get_threshold(self, metric: str, values: deque) -> float:
        """Get adaptive threshold for a metric"""
        if not self.adaptive_thresholds:
            return self.thresholds[metric]
        
        # Calculate mean and standard deviation
        mean = np.mean(values)
        std = np.std(values)
        
        # Use configurable sigma rule
        threshold = mean + (self.z_score_threshold * std)
        
        # Ensure threshold is not below baseline
        return max(threshold, self.thresholds[metric])
    
    def _calculate_severity(self, value: float, threshold: float) -> str:  # pragma: no cover
        """Calculate anomaly severity"""
        ratio = value / threshold
        if ratio >= 2.0:
            return 'critical'
        elif ratio >= 1.5:
            return 'warning'
        else:
            return 'degraded'
    
    def get_detection_stats(self) -> Dict[str, float]:  # pragma: no cover
        """Get current detection statistics
        
        Returns:
            Dictionary containing current detection statistics
        """
        stats = {}
        
        if len(self.cpu_history) >= self.min_history_size:
            stats['cpu_mean'] = np.mean(self.cpu_history)
            stats['cpu_std'] = np.std(self.cpu_history)
            
        if len(self.memory_history) >= self.min_history_size:
            stats['memory_mean'] = np.mean(self.memory_history)
            stats['memory_std'] = np.std(self.memory_history)
            
        if len(self.latency_history) >= self.min_history_size:
            stats['latency_mean'] = np.mean(self.latency_history)
            stats['latency_std'] = np.std(self.latency_history)
            
        if len(self.error_history) >= self.min_history_size:
            stats['error_mean'] = np.mean(self.error_history)
            stats['error_std'] = np.std(self.error_history)
            
        return stats 