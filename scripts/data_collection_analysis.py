#!/usr/bin/env python3
"""
Phase 4: Data Collection & Analysis
Comprehensive data collection, analysis, and insights generation for Graph-Heal.
"""

import time
import json
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, asdict
import sqlite3
from contextlib import contextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class MetricData:
    """Data structure for metric collection."""
    timestamp: float
    service: str
    metric_name: str
    value: float
    unit: str = ""
    labels: Dict = None
    
    def __post_init__(self):
        if self.labels is None:
            self.labels = {}

@dataclass
class AnomalyEvent:
    """Data structure for anomaly events."""
    timestamp: float
    service: str
    anomaly_type: str
    metric_name: str = ""
    value: float = 0.0
    threshold: float = 0.0
    severity: str = "warning"
    affected_services: List[str] = None
    
    def __post_init__(self):
        if self.affected_services is None:
            self.affected_services = []

@dataclass
class RecoveryEvent:
    """Data structure for recovery events."""
    timestamp: float
    service: str
    recovery_action: str
    success: bool
    duration: float
    error_message: str = ""

class DataCollector:
    """Collects metrics and events from all services."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.services = config.get('services', [])
        self.collection_interval = config.get('collection_interval', 30)
        self.data_dir = Path(config.get('data_dir', 'data/analysis'))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self.db_path = self.data_dir / 'metrics.db'
        self.init_database()
        
        # Data storage
        self.metrics_data = []
        self.anomaly_events = []
        self.recovery_events = []
        
    def init_database(self):
        """Initialize SQLite database for data storage."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    service TEXT,
                    metric_name TEXT,
                    value REAL,
                    unit TEXT,
                    labels TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS anomalies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    service TEXT,
                    anomaly_type TEXT,
                    metric_name TEXT,
                    value REAL,
                    threshold REAL,
                    severity TEXT,
                    affected_services TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS recoveries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    service TEXT,
                    recovery_action TEXT,
                    success INTEGER,
                    duration REAL,
                    error_message TEXT
                )
            ''')
            
            # Create indexes for better query performance
            conn.execute('CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_metrics_service ON metrics(service)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_anomalies_timestamp ON anomalies(timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_recoveries_timestamp ON recoveries(timestamp)')
            
        logger.info(f"Database initialized: {self.db_path}")
        
    @contextmanager
    def get_db_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()
            
    def collect_service_metrics(self, service: str) -> List[MetricData]:
        """Collect metrics from a specific service."""
        metrics = []
        try:
            port = self.get_service_port(service)
            
            # Collect health metrics
            health_response = requests.get(f"http://localhost:{port}/health", timeout=5)
            if health_response.status_code == 200:
                health_data = health_response.json()
                
                # Extract metrics from health data
                for metric_name, value in health_data.get('metrics', {}).items():
                    if isinstance(value, (int, float)):
                        metrics.append(MetricData(
                            timestamp=time.time(),
                            service=service,
                            metric_name=metric_name,
                            value=float(value),
                            unit=self.get_metric_unit(metric_name)
                        ))
                        
                # Add health score as a metric
                health_score = health_data.get('health_score', 1.0)
                metrics.append(MetricData(
                    timestamp=time.time(),
                    service=service,
                    metric_name='health_score',
                    value=health_score,
                    unit='score'
                ))
                
            # Collect Prometheus metrics if available
            metrics_response = requests.get(f"http://localhost:{port}/metrics", timeout=5)
            if metrics_response.status_code == 200:
                metrics.extend(self.parse_prometheus_metrics(service, metrics_response.text))
                
        except Exception as e:
            logger.warning(f"Failed to collect metrics from {service}: {e}")
            
        return metrics
        
    def parse_prometheus_metrics(self, service: str, metrics_text: str) -> List[MetricData]:
        """Parse Prometheus format metrics."""
        metrics = []
        timestamp = time.time()
        
        for line in metrics_text.split('\n'):
            if line and not line.startswith('#'):
                try:
                    # Parse Prometheus metric line
                    if '{' in line:
                        # Metric with labels
                        metric_part, value_part = line.split('{', 1)
                        metric_name = metric_part.strip()
                        labels_part, value = value_part.rsplit('}', 1)
                        labels = self.parse_prometheus_labels(labels_part)
                        value = float(value.strip())
                    else:
                        # Simple metric
                        metric_name, value = line.split(' ', 1)
                        labels = {}
                        value = float(value.strip())
                        
                    metrics.append(MetricData(
                        timestamp=timestamp,
                        service=service,
                        metric_name=metric_name,
                        value=value,
                        unit=self.get_metric_unit(metric_name),
                        labels=labels
                    ))
                except Exception as e:
                    logger.debug(f"Failed to parse metric line '{line}': {e}")
                    
        return metrics
        
    def parse_prometheus_labels(self, labels_text: str) -> Dict[str, str]:
        """Parse Prometheus labels."""
        labels = {}
        try:
            # Remove the closing brace and parse
            labels_text = labels_text.rstrip('}')
            for label_pair in labels_text.split(','):
                if '=' in label_pair:
                    key, value = label_pair.split('=', 1)
                    labels[key.strip()] = value.strip().strip('"')
        except Exception as e:
            logger.debug(f"Failed to parse labels '{labels_text}': {e}")
        return labels
        
    def get_service_port(self, service: str) -> int:
        """Get the port for a service."""
        port_map = {
            'service_a': 5001,
            'service_b': 5002,
            'service_c': 5003,
            'service_d': 5004
        }
        return port_map.get(service, 5000)
        
    def get_metric_unit(self, metric_name: str) -> str:
        """Get the unit for a metric."""
        unit_map = {
            'cpu_usage': 'percent',
            'memory_usage': 'bytes',
            'request_latency': 'seconds',
            'request_count': 'count',
            'error_rate': 'percent',
            'temperature': 'celsius',
            'pressure': 'bar',
            'flow_rate': 'lpm',
            'vibration': 'g',
            'health_score': 'score'
        }
        return unit_map.get(metric_name, "")
        
    def collect_all_metrics(self) -> List[MetricData]:
        """Collect metrics from all services."""
        all_metrics = []
        
        for service in self.services:
            metrics = self.collect_service_metrics(service)
            all_metrics.extend(metrics)
            
        # Store in database
        self.store_metrics(all_metrics)
        
        return all_metrics
        
    def store_metrics(self, metrics: List[MetricData]):
        """Store metrics in database."""
        with self.get_db_connection() as conn:
            for metric in metrics:
                conn.execute('''
                    INSERT INTO metrics (timestamp, service, metric_name, value, unit, labels)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    metric.timestamp,
                    metric.service,
                    metric.metric_name,
                    metric.value,
                    metric.unit,
                    json.dumps(metric.labels)
                ))
            conn.commit()
            
    def collect_anomaly_events(self) -> List[AnomalyEvent]:
        """Collect anomaly events from monitoring logs."""
        events = []
        
        # Check recent monitoring logs
        log_files = list(Path('logs').glob('live_run_*.log'))
        if log_files:
            latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
            
            with open(latest_log, 'r') as f:
                for line in f:
                    if 'anomaly' in line.lower():
                        event = self.parse_anomaly_log_line(line)
                        if event:
                            events.append(event)
                            
        return events
        
    def parse_anomaly_log_line(self, line: str) -> Optional[AnomalyEvent]:
        """Parse anomaly event from log line."""
        try:
            if 'metric_anomaly' in line:
                # Parse metric anomaly
                parts = line.split()
                timestamp = float(parts[0].split(',')[0])
                service = parts[2].split('=')[1]
                metric = parts[3].split('=')[1]
                value = float(parts[4].split('=')[1])
                threshold = float(parts[5].split('=')[1])
                
                return AnomalyEvent(
                    timestamp=timestamp,
                    service=service,
                    anomaly_type='metric_anomaly',
                    metric_name=metric,
                    value=value,
                    threshold=threshold
                )
            elif 'dependency_anomaly' in line:
                # Parse dependency anomaly
                parts = line.split()
                timestamp = float(parts[0].split(',')[0])
                service = parts[2].split('=')[1]
                
                return AnomalyEvent(
                    timestamp=timestamp,
                    service=service,
                    anomaly_type='dependency_anomaly'
                )
        except Exception as e:
            logger.debug(f"Failed to parse anomaly line: {e}")
            
        return None
        
    def run_data_collection(self, duration_minutes: int = 60):
        """Run data collection for a specified duration."""
        logger.info(f"Starting data collection for {duration_minutes} minutes")
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        while time.time() < end_time:
            try:
                # Collect metrics
                metrics = self.collect_all_metrics()
                logger.info(f"Collected {len(metrics)} metrics from {len(self.services)} services")
                
                # Collect anomaly events
                anomalies = self.collect_anomaly_events()
                if anomalies:
                    logger.info(f"Collected {len(anomalies)} anomaly events")
                    
                # Wait for next collection interval
                time.sleep(self.collection_interval)
                
            except KeyboardInterrupt:
                logger.info("Data collection stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in data collection: {e}")
                time.sleep(self.collection_interval)
                
        logger.info("Data collection completed")


class DataAnalyzer:
    """Analyzes collected data and generates insights."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.db_path = data_dir / 'metrics.db'
        
    def load_metrics_data(self, start_time: float = None, end_time: float = None) -> pd.DataFrame:
        """Load metrics data from database."""
        query = "SELECT * FROM metrics"
        params = []
        
        if start_time and end_time:
            query += " WHERE timestamp BETWEEN ? AND ?"
            params = [start_time, end_time]
        elif start_time:
            query += " WHERE timestamp >= ?"
            params = [start_time]
        elif end_time:
            query += " WHERE timestamp <= ?"
            params = [end_time]
            
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)
            
        # Convert timestamp to datetime
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        
        return df
        
    def load_anomaly_data(self, start_time: float = None, end_time: float = None) -> pd.DataFrame:
        """Load anomaly data from database."""
        query = "SELECT * FROM anomalies"
        params = []
        
        if start_time and end_time:
            query += " WHERE timestamp BETWEEN ? AND ?"
            params = [start_time, end_time]
        elif start_time:
            query += " WHERE timestamp >= ?"
            params = [start_time]
        elif end_time:
            query += " WHERE timestamp <= ?"
            params = [end_time]
            
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)
            
        # Convert timestamp to datetime
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        
        return df
        
    def analyze_service_health(self, df: pd.DataFrame) -> Dict:
        """Analyze service health patterns."""
        analysis = {}
        
        for service in df['service'].unique():
            service_data = df[df['service'] == service]
            
            # Health score analysis
            health_scores = service_data[service_data['metric_name'] == 'health_score']['value']
            if not health_scores.empty:
                analysis[service] = {
                    'avg_health_score': health_scores.mean(),
                    'min_health_score': health_scores.min(),
                    'max_health_score': health_scores.max(),
                    'health_score_std': health_scores.std(),
                    'health_score_samples': len(health_scores)
                }
                
        return analysis
        
    def analyze_metric_trends(self, df: pd.DataFrame) -> Dict:
        """Analyze metric trends and patterns."""
        trends = {}
        
        for service in df['service'].unique():
            service_data = df[df['service'] == service]
            
            for metric in service_data['metric_name'].unique():
                metric_data = service_data[service_data['metric_name'] == metric]
                
                if len(metric_data) > 1:
                    # Calculate trends
                    values = metric_data['value'].values
                    timestamps = metric_data['timestamp'].values
                    
                    # Linear trend
                    if len(values) > 1:
                        slope = np.polyfit(timestamps, values, 1)[0]
                        
                        trends[f"{service}_{metric}"] = {
                            'mean': values.mean(),
                            'std': values.std(),
                            'min': values.min(),
                            'max': values.max(),
                            'trend_slope': slope,
                            'trend_direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable',
                            'samples': len(values)
                        }
                        
        return trends
        
    def analyze_anomaly_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze anomaly patterns and correlations."""
        patterns = {}
        
        if df.empty:
            return patterns
            
        # Anomaly frequency by service
        service_counts = df['service'].value_counts()
        patterns['anomaly_frequency'] = service_counts.to_dict()
        
        # Anomaly frequency by type
        type_counts = df['anomaly_type'].value_counts()
        patterns['anomaly_types'] = type_counts.to_dict()
        
        # Time-based patterns
        df['hour'] = df['datetime'].dt.hour
        hourly_counts = df['hour'].value_counts().sort_index()
        patterns['hourly_distribution'] = hourly_counts.to_dict()
        
        # Severity analysis
        severity_counts = df['severity'].value_counts()
        patterns['severity_distribution'] = severity_counts.to_dict()
        
        return patterns
        
    def detect_correlations(self, metrics_df: pd.DataFrame, anomalies_df: pd.DataFrame) -> Dict:
        """Detect correlations between metrics and anomalies."""
        correlations = {}
        
        if metrics_df.empty or anomalies_df.empty:
            return correlations
            
        # Group metrics by time windows and correlate with anomalies
        metrics_df['time_window'] = pd.to_datetime(metrics_df['timestamp'], unit='s').dt.floor('5min')
        anomalies_df['time_window'] = pd.to_datetime(anomalies_df['timestamp'], unit='s').dt.floor('5min')
        
        # Count anomalies per time window
        anomaly_counts = anomalies_df.groupby('time_window').size()
        
        # For each metric, calculate correlation with anomaly counts
        for service in metrics_df['service'].unique():
            service_metrics = metrics_df[metrics_df['service'] == service]
            
            for metric in service_metrics['metric_name'].unique():
                metric_data = service_metrics[service_metrics['metric_name'] == metric]
                metric_aggregated = metric_data.groupby('time_window')['value'].mean()
                
                # Align time windows
                common_windows = metric_aggregated.index.intersection(anomaly_counts.index)
                if len(common_windows) > 1:
                    correlation = np.corrcoef(
                        metric_aggregated.loc[common_windows],
                        anomaly_counts.loc[common_windows]
                    )[0, 1]
                    
                    if not np.isnan(correlation):
                        correlations[f"{service}_{metric}"] = {
                            'correlation': correlation,
                            'correlation_strength': 'strong' if abs(correlation) > 0.7 else 'moderate' if abs(correlation) > 0.3 else 'weak'
                        }
                        
        return correlations
        
    def generate_insights_report(self, start_time: float = None, end_time: float = None) -> Dict:
        """Generate comprehensive insights report."""
        logger.info("Generating insights report")
        
        # Load data
        metrics_df = self.load_metrics_data(start_time, end_time)
        anomalies_df = self.load_anomaly_data(start_time, end_time)
        
        # Perform analyses
        health_analysis = self.analyze_service_health(metrics_df)
        metric_trends = self.analyze_metric_trends(metrics_df)
        anomaly_patterns = self.analyze_anomaly_patterns(anomalies_df)
        correlations = self.detect_correlations(metrics_df, anomalies_df)
        
        # Generate insights
        insights = {
            'summary': {
                'total_metrics_collected': len(metrics_df),
                'total_anomalies_detected': len(anomalies_df),
                'data_collection_period': {
                    'start': datetime.fromtimestamp(metrics_df['timestamp'].min()) if not metrics_df.empty else None,
                    'end': datetime.fromtimestamp(metrics_df['timestamp'].max()) if not metrics_df.empty else None
                }
            },
            'service_health': health_analysis,
            'metric_trends': metric_trends,
            'anomaly_patterns': anomaly_patterns,
            'correlations': correlations,
            'recommendations': self.generate_recommendations(health_analysis, metric_trends, anomaly_patterns, correlations)
        }
        
        return insights
        
    def generate_recommendations(self, health_analysis: Dict, metric_trends: Dict, 
                               anomaly_patterns: Dict, correlations: Dict) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        # Health-based recommendations
        for service, health in health_analysis.items():
            if health['avg_health_score'] < 0.8:
                recommendations.append(f"Service {service} has low average health score ({health['avg_health_score']:.2f}). Consider investigating root causes.")
                
        # Trend-based recommendations
        for metric_key, trend in metric_trends.items():
            if abs(trend['trend_slope']) > 0.01:  # Significant trend
                direction = trend['trend_direction']
                recommendations.append(f"Metric {metric_key} shows {direction} trend. Monitor for potential issues.")
                
        # Anomaly-based recommendations
        if anomaly_patterns.get('anomaly_frequency'):
            most_anomalous_service = max(anomaly_patterns['anomaly_frequency'].items(), key=lambda x: x[1])[0]
            recommendations.append(f"Service {most_anomalous_service} has the highest anomaly frequency. Focus monitoring efforts here.")
            
        # Correlation-based recommendations
        strong_correlations = [k for k, v in correlations.items() if v['correlation_strength'] == 'strong']
        if strong_correlations:
            recommendations.append(f"Strong correlations detected for metrics: {', '.join(strong_correlations)}. Consider using these as early warning indicators.")
            
        return recommendations


class ReportGenerator:
    """Generates comprehensive reports and visualizations."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.reports_dir = data_dir / 'reports'
        self.reports_dir.mkdir(exist_ok=True)
        
    def generate_visualizations(self, insights: Dict):
        """Generate visualizations from insights."""
        logger.info("Generating visualizations")
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Service Health Overview
        self.plot_service_health(insights['service_health'])
        
        # 2. Metric Trends
        self.plot_metric_trends(insights['metric_trends'])
        
        # 3. Anomaly Patterns
        self.plot_anomaly_patterns(insights['anomaly_patterns'])
        
        # 4. Correlation Heatmap
        self.plot_correlation_heatmap(insights['correlations'])
        
    def plot_service_health(self, health_data: Dict):
        """Plot service health overview."""
        if not health_data:
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Health scores
        services = list(health_data.keys())
        avg_scores = [health_data[s]['avg_health_score'] for s in services]
        
        bars = ax1.bar(services, avg_scores, color=['green' if score > 0.8 else 'orange' if score > 0.6 else 'red' for score in avg_scores])
        ax1.set_title('Average Health Scores by Service')
        ax1.set_ylabel('Health Score')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, score in zip(bars, avg_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{score:.2f}', ha='center', va='bottom')
        
        # Health score distribution
        all_scores = []
        for service_data in health_data.values():
            all_scores.extend([service_data['min_health_score'], service_data['max_health_score']])
            
        ax2.hist(all_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_title('Health Score Distribution')
        ax2.set_xlabel('Health Score')
        ax2.set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(self.reports_dir / 'service_health_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_metric_trends(self, trends_data: Dict):
        """Plot metric trends."""
        if not trends_data:
            return
            
        # Select top trends by absolute slope
        top_trends = sorted(trends_data.items(), key=lambda x: abs(x[1]['trend_slope']), reverse=True)[:10]
        
        if not top_trends:
            return
            
        fig, ax = plt.subplots(figsize=(12, 8))
        
        metrics = [item[0] for item in top_trends]
        slopes = [item[1]['trend_slope'] for item in top_trends]
        colors = ['red' if slope > 0 else 'blue' for slope in slopes]
        
        bars = ax.barh(metrics, slopes, color=colors, alpha=0.7)
        ax.set_title('Top Metric Trends (by slope magnitude)')
        ax.set_xlabel('Trend Slope')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels
        for bar, slope in zip(bars, slopes):
            ax.text(bar.get_width() + (0.01 if slope > 0 else -0.01), bar.get_y() + bar.get_height()/2,
                   f'{slope:.4f}', ha='left' if slope > 0 else 'right', va='center')
        
        plt.tight_layout()
        plt.savefig(self.reports_dir / 'metric_trends.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_anomaly_patterns(self, patterns_data: Dict):
        """Plot anomaly patterns."""
        if not patterns_data:
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Anomaly frequency by service
        if 'anomaly_frequency' in patterns_data:
            services = list(patterns_data['anomaly_frequency'].keys())
            counts = list(patterns_data['anomaly_frequency'].values())
            
            bars = ax1.bar(services, counts, color='coral')
            ax1.set_title('Anomaly Frequency by Service')
            ax1.set_ylabel('Number of Anomalies')
            ax1.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, count in zip(bars, counts):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(count), ha='center', va='bottom')
        
        # Anomaly types
        if 'anomaly_types' in patterns_data:
            types = list(patterns_data['anomaly_types'].keys())
            counts = list(patterns_data['anomaly_types'].values())
            
            ax2.pie(counts, labels=types, autopct='%1.1f%%', startangle=90)
            ax2.set_title('Anomaly Types Distribution')
        
        # Hourly distribution
        if 'hourly_distribution' in patterns_data:
            hours = sorted(patterns_data['hourly_distribution'].keys())
            counts = [patterns_data['hourly_distribution'][h] for h in hours]
            
            ax3.plot(hours, counts, marker='o', linewidth=2, markersize=6)
            ax3.set_title('Anomaly Distribution by Hour')
            ax3.set_xlabel('Hour of Day')
            ax3.set_ylabel('Number of Anomalies')
            ax3.grid(True, alpha=0.3)
        
        # Severity distribution
        if 'severity_distribution' in patterns_data:
            severities = list(patterns_data['severity_distribution'].keys())
            counts = list(patterns_data['severity_distribution'].values())
            
            colors = ['green', 'orange', 'red']
            bars = ax4.bar(severities, counts, color=colors[:len(severities)])
            ax4.set_title('Anomaly Severity Distribution')
            ax4.set_ylabel('Number of Anomalies')
            
            # Add value labels
            for bar, count in zip(bars, counts):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.reports_dir / 'anomaly_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_correlation_heatmap(self, correlations_data: Dict):
        """Plot correlation heatmap."""
        if not correlations_data:
            return
            
        # Create correlation matrix
        metrics = list(correlations_data.keys())
        correlation_matrix = np.zeros((len(metrics), len(metrics)))
        
        for i, metric1 in enumerate(metrics):
            for j, metric2 in enumerate(metrics):
                if i == j:
                    correlation_matrix[i, j] = 1.0
                else:
                    # Use the correlation value if available
                    key = f"{metric1}_{metric2}"
                    if key in correlations_data:
                        correlation_matrix[i, j] = correlations_data[key]['correlation']
                    else:
                        correlation_matrix[i, j] = 0.0
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create heatmap
        im = ax.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Correlation Coefficient', rotation=-90, va="bottom")
        
        # Set ticks and labels
        ax.set_xticks(range(len(metrics)))
        ax.set_yticks(range(len(metrics)))
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.set_yticklabels(metrics)
        
        # Add text annotations
        for i in range(len(metrics)):
            for j in range(len(metrics)):
                text = ax.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black" if abs(correlation_matrix[i, j]) < 0.5 else "white")
        
        ax.set_title('Metric Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(self.reports_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def generate_html_report(self, insights: Dict):
        """Generate HTML report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Graph-Heal Data Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 30px 0; }}
                .metric {{ background-color: #f9f9f9; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .recommendation {{ background-color: #e8f4fd; padding: 10px; margin: 5px 0; border-left: 4px solid #2196F3; }}
                .visualization {{ text-align: center; margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Graph-Heal Data Analysis Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Summary</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Total Metrics Collected</td><td>{insights['summary']['total_metrics_collected']}</td></tr>
                    <tr><td>Total Anomalies Detected</td><td>{insights['summary']['total_anomalies_detected']}</td></tr>
                    <tr><td>Data Collection Start</td><td>{insights['summary']['data_collection_period']['start']}</td></tr>
                    <tr><td>Data Collection End</td><td>{insights['summary']['data_collection_period']['end']}</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Service Health Analysis</h2>
                {self._generate_health_html(insights['service_health'])}
            </div>
            
            <div class="section">
                <h2>Key Metric Trends</h2>
                {self._generate_trends_html(insights['metric_trends'])}
            </div>
            
            <div class="section">
                <h2>Anomaly Patterns</h2>
                {self._generate_anomaly_html(insights['anomaly_patterns'])}
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                {self._generate_recommendations_html(insights['recommendations'])}
            </div>
            
            <div class="section">
                <h2>Visualizations</h2>
                <div class="visualization">
                    <img src="service_health_overview.png" alt="Service Health Overview" style="max-width: 100%;">
                </div>
                <div class="visualization">
                    <img src="metric_trends.png" alt="Metric Trends" style="max-width: 100%;">
                </div>
                <div class="visualization">
                    <img src="anomaly_patterns.png" alt="Anomaly Patterns" style="max-width: 100%;">
                </div>
                <div class="visualization">
                    <img src="correlation_heatmap.png" alt="Correlation Heatmap" style="max-width: 100%;">
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(self.reports_dir / 'analysis_report.html', 'w') as f:
            f.write(html_content)
            
        logger.info(f"HTML report generated: {self.reports_dir / 'analysis_report.html'}")
        
    def _generate_health_html(self, health_data: Dict) -> str:
        """Generate HTML for health analysis."""
        if not health_data:
            return "<p>No health data available.</p>"
            
        html = "<table><tr><th>Service</th><th>Avg Health Score</th><th>Min Health Score</th><th>Max Health Score</th><th>Std Dev</th></tr>"
        
        for service, health in health_data.items():
            html += f"""
            <tr>
                <td>{service}</td>
                <td>{health['avg_health_score']:.3f}</td>
                <td>{health['min_health_score']:.3f}</td>
                <td>{health['max_health_score']:.3f}</td>
                <td>{health['health_score_std']:.3f}</td>
            </tr>
            """
            
        html += "</table>"
        return html
        
    def _generate_trends_html(self, trends_data: Dict) -> str:
        """Generate HTML for trends analysis."""
        if not trends_data:
            return "<p>No trend data available.</p>"
            
        # Show top 5 trends
        top_trends = sorted(trends_data.items(), key=lambda x: abs(x[1]['trend_slope']), reverse=True)[:5]
        
        html = "<table><tr><th>Metric</th><th>Trend Direction</th><th>Slope</th><th>Mean Value</th></tr>"
        
        for metric, trend in top_trends:
            html += f"""
            <tr>
                <td>{metric}</td>
                <td>{trend['trend_direction']}</td>
                <td>{trend['trend_slope']:.4f}</td>
                <td>{trend['mean']:.3f}</td>
            </tr>
            """
            
        html += "</table>"
        return html
        
    def _generate_anomaly_html(self, patterns_data: Dict) -> str:
        """Generate HTML for anomaly patterns."""
        if not patterns_data:
            return "<p>No anomaly data available.</p>"
            
        html = "<h3>Anomaly Frequency by Service</h3>"
        if 'anomaly_frequency' in patterns_data:
            html += "<table><tr><th>Service</th><th>Anomaly Count</th></tr>"
            for service, count in patterns_data['anomaly_frequency'].items():
                html += f"<tr><td>{service}</td><td>{count}</td></tr>"
            html += "</table>"
            
        html += "<h3>Anomaly Types</h3>"
        if 'anomaly_types' in patterns_data:
            html += "<table><tr><th>Type</th><th>Count</th></tr>"
            for anomaly_type, count in patterns_data['anomaly_types'].items():
                html += f"<tr><td>{anomaly_type}</td><td>{count}</td></tr>"
            html += "</table>"
            
        return html
        
    def _generate_recommendations_html(self, recommendations: List[str]) -> str:
        """Generate HTML for recommendations."""
        if not recommendations:
            return "<p>No recommendations available.</p>"
            
        html = ""
        for i, rec in enumerate(recommendations, 1):
            html += f'<div class="recommendation"><strong>{i}.</strong> {rec}</div>'
            
        return html


def main():
    """Main function to run Phase 4: Data Collection & Analysis."""
    logger.info("Starting Phase 4: Data Collection & Analysis")
    
    # Configuration
    config = {
        'services': ['service_a', 'service_b', 'service_c', 'service_d'],
        'collection_interval': 30,  # seconds
        'data_dir': 'data/analysis'
    }
    
    # Initialize components
    data_dir = Path(config['data_dir'])
    collector = DataCollector(config)
    analyzer = DataAnalyzer(data_dir)
    report_generator = ReportGenerator(data_dir)
    
    # Step 1: Collect data
    logger.info("Step 1: Collecting data for 10 minutes...")
    collector.run_data_collection(duration_minutes=10)
    
    # Step 2: Analyze data
    logger.info("Step 2: Analyzing collected data...")
    insights = analyzer.generate_insights_report()
    
    # Step 3: Generate reports and visualizations
    logger.info("Step 3: Generating reports and visualizations...")
    report_generator.generate_visualizations(insights)
    report_generator.generate_html_report(insights)
    
    # Step 4: Save insights to JSON
    insights_file = data_dir / 'insights.json'
    with open(insights_file, 'w') as f:
        json.dump(insights, f, indent=2, default=str)
        
    logger.info(f"Phase 4 completed! Reports saved to {data_dir}")
    logger.info(f"Insights: {insights_file}")
    logger.info(f"HTML Report: {data_dir}/reports/analysis_report.html")
    logger.info(f"Visualizations: {data_dir}/reports/")
    
    # Print summary
    print("\n" + "="*60)
    print("PHASE 4: DATA COLLECTION & ANALYSIS COMPLETED")
    print("="*60)
    print(f"📊 Total metrics collected: {insights['summary']['total_metrics_collected']}")
    print(f"🚨 Total anomalies detected: {insights['summary']['total_anomalies_detected']}")
    print(f"📈 Key trends identified: {len(insights['metric_trends'])}")
    print(f"💡 Recommendations generated: {len(insights['recommendations'])}")
    print(f"📁 Reports saved to: {data_dir}")
    print("="*60)


if __name__ == "__main__":
    main() 