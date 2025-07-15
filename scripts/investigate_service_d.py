#!/usr/bin/env python3
"""
Service D Health Investigation
Comprehensive diagnostic analysis of Service D's health issues.
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
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import sqlite3

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class HealthDiagnostic:
    """Health diagnostic result."""
    service: str
    timestamp: float
    health_score: float
    metrics: Dict
    dependencies: List[str]
    issues: List[str]
    recommendations: List[str]

class ServiceDInvestigator:
    """Investigates Service D's health issues."""
    
    def __init__(self):
        self.services = ['service_a', 'service_b', 'service_c', 'service_d']
        self.ports = {
            'service_a': 5001,
            'service_b': 5002,
            'service_c': 5003,
            'service_d': 5004
        }
        self.results_dir = Path('data/service_d_investigation')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def get_service_health(self, service: str) -> Optional[Dict]:
        """Get detailed health information from a service."""
        try:
            port = self.ports[service]
            response = requests.get(f"http://localhost:{port}/health", timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Failed to get health from {service}: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Error getting health from {service}: {e}")
            return None
            
    def get_service_metrics(self, service: str) -> Optional[str]:
        """Get Prometheus metrics from a service."""
        try:
            port = self.ports[service]
            response = requests.get(f"http://localhost:{port}/metrics", timeout=10)
            
            if response.status_code == 200:
                return response.text
            else:
                logger.warning(f"Failed to get metrics from {service}: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Error getting metrics from {service}: {e}")
            return None
            
    def analyze_health_comparison(self) -> Dict:
        """Compare health scores across all services."""
        logger.info("Analyzing health comparison across all services")
        
        health_data = {}
        comparison = {
            'services': {},
            'service_d_issues': [],
            'comparison_insights': []
        }
        
        for service in self.services:
            health = self.get_service_health(service)
            if health:
                health_data[service] = health
                comparison['services'][service] = {
                    'health_score': health.get('health_score', 0.0),
                    'status': health.get('status', 'unknown'),
                    'metrics': health.get('metrics', {}),
                    'dependencies': health.get('dependencies', []),
                    'timestamp': time.time()
                }
                
        # Analyze Service D specifically
        if 'service_d' in comparison['services']:
            service_d = comparison['services']['service_d']
            other_services = {k: v for k, v in comparison['services'].items() if k != 'service_d'}
            
            # Compare health scores
            avg_other_health = np.mean([s['health_score'] for s in other_services.values()])
            service_d_health = service_d['health_score']
            
            if service_d_health < avg_other_health:
                comparison['service_d_issues'].append(
                    f"Health score ({service_d_health:.2f}) is below average ({avg_other_health:.2f})"
                )
                
            # Compare metrics
            for metric_name, metric_value in service_d['metrics'].items():
                if isinstance(metric_value, (int, float)):
                    other_values = []
                    for other_service in other_services.values():
                        if metric_name in other_service['metrics']:
                            other_values.append(other_service['metrics'][metric_name])
                            
                    if other_values:
                        avg_other = np.mean(other_values)
                        if abs(metric_value - avg_other) > avg_other * 0.2:  # 20% difference
                            comparison['service_d_issues'].append(
                                f"Metric {metric_name}: {metric_value} vs avg {avg_other:.2f}"
                            )
                            
        return comparison
        
    def analyze_metrics_trends(self) -> Dict:
        """Analyze Service D's metrics trends from the database."""
        logger.info("Analyzing Service D's metrics trends")
        
        db_path = Path('data/analysis/metrics.db')
        if not db_path.exists():
            logger.warning("No metrics database found. Run data collection first.")
            return {}
            
        with sqlite3.connect(db_path) as conn:
            # Get Service D metrics from the last hour
            query = """
            SELECT * FROM metrics 
            WHERE service = 'service_d' 
            AND timestamp >= ? 
            ORDER BY timestamp DESC
            """
            
            one_hour_ago = time.time() - 3600
            df = pd.read_sql_query(query, conn, params=[one_hour_ago])
            
        if df.empty:
            logger.warning("No recent Service D metrics found")
            return {}
            
        # Convert timestamp to datetime
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Analyze trends for each metric
        trends = {}
        for metric_name in df['metric_name'].unique():
            metric_data = df[df['metric_name'] == metric_name]
            
            if len(metric_data) > 1:
                values = metric_data['value'].values
                timestamps = metric_data['timestamp'].values
                
                # Calculate trend
                slope = np.polyfit(timestamps, values, 1)[0]
                
                trends[metric_name] = {
                    'mean': values.mean(),
                    'std': values.std(),
                    'min': values.min(),
                    'max': values.max(),
                    'trend_slope': slope,
                    'trend_direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable',
                    'samples': len(values),
                    'recent_values': values[-5:].tolist()  # Last 5 values
                }
                
        return trends
        
    def check_service_dependencies(self) -> Dict:
        """Check Service D's dependencies and their health."""
        logger.info("Checking Service D's dependencies")
        
        dependencies = {}
        
        # Get Service D's health to see its dependencies
        service_d_health = self.get_service_health('service_d')
        if service_d_health and 'dependencies' in service_d_health:
            service_d_deps = service_d_health['dependencies']
            
            for dep in service_d_deps:
                dep_health = self.get_service_health(dep)
                if dep_health:
                    dependencies[dep] = {
                        'health_score': dep_health.get('health_score', 0.0),
                        'status': dep_health.get('status', 'unknown'),
                        'metrics': dep_health.get('metrics', {}),
                        'is_healthy': dep_health.get('health_score', 0.0) > 0.8
                    }
                else:
                    dependencies[dep] = {
                        'health_score': 0.0,
                        'status': 'unreachable',
                        'metrics': {},
                        'is_healthy': False
                    }
                    
        return dependencies
        
    def analyze_error_patterns(self) -> Dict:
        """Analyze error patterns in Service D."""
        logger.info("Analyzing Service D's error patterns")
        
        # Check recent logs for Service D errors
        log_files = list(Path('logs').glob('*.log'))
        service_d_errors = []
        
        for log_file in log_files:
            try:
                with open(log_file, 'r') as f:
                    for line in f:
                        if 'service_d' in line.lower() and any(error_word in line.lower() for error_word in ['error', 'exception', 'failed', 'timeout']):
                            service_d_errors.append({
                                'log_file': log_file.name,
                                'line': line.strip(),
                                'timestamp': line.split()[0] if line.split() else 'unknown'
                            })
            except Exception as e:
                logger.debug(f"Error reading log file {log_file}: {e}")
                
        return {
            'error_count': len(service_d_errors),
            'errors': service_d_errors[:10],  # Last 10 errors
            'error_patterns': self.analyze_error_patterns_from_errors(service_d_errors)
        }
        
    def analyze_error_patterns_from_errors(self, errors: List[Dict]) -> Dict:
        """Analyze patterns in the collected errors."""
        patterns = {
            'error_types': {},
            'time_distribution': {},
            'common_phrases': {}
        }
        
        for error in errors:
            line = error['line'].lower()
            
            # Categorize error types
            if 'timeout' in line:
                patterns['error_types']['timeout'] = patterns['error_types'].get('timeout', 0) + 1
            elif 'connection' in line:
                patterns['error_types']['connection'] = patterns['error_types'].get('connection', 0) + 1
            elif 'exception' in line:
                patterns['error_types']['exception'] = patterns['error_types'].get('exception', 0) + 1
            elif 'failed' in line:
                patterns['error_types']['failed'] = patterns['error_types'].get('failed', 0) + 1
                
        return patterns
        
    def perform_stress_test(self) -> Dict:
        """Perform a stress test on Service D."""
        logger.info("Performing stress test on Service D")
        
        results = {
            'response_times': [],
            'success_rate': 0,
            'errors': [],
            'health_degradation': False
        }
        
        # Get initial health
        initial_health = self.get_service_health('service_d')
        initial_score = initial_health.get('health_score', 0.0) if initial_health else 0.0
        
        # Send multiple requests
        request_count = 20
        successful_requests = 0
        
        for i in range(request_count):
            try:
                start_time = time.time()
                response = requests.get(f"http://localhost:5004/health", timeout=5)
                end_time = time.time()
                
                response_time = end_time - start_time
                results['response_times'].append(response_time)
                
                if response.status_code == 200:
                    successful_requests += 1
                else:
                    results['errors'].append(f"Request {i+1}: HTTP {response.status_code}")
                    
            except Exception as e:
                results['errors'].append(f"Request {i+1}: {str(e)}")
                
        # Calculate success rate
        results['success_rate'] = successful_requests / request_count if request_count > 0 else 0
        
        # Check if health degraded
        final_health = self.get_service_health('service_d')
        final_score = final_health.get('health_score', 0.0) if final_health else 0.0
        
        if final_score < initial_score:
            results['health_degradation'] = True
            results['health_change'] = final_score - initial_score
            
        return results
        
    def generate_diagnostic_report(self) -> Dict:
        """Generate comprehensive diagnostic report."""
        logger.info("Generating comprehensive diagnostic report")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'service_d_health': self.get_service_health('service_d'),
            'health_comparison': self.analyze_health_comparison(),
            'metrics_trends': self.analyze_metrics_trends(),
            'dependencies': self.check_service_dependencies(),
            'error_patterns': self.analyze_error_patterns(),
            'stress_test': self.perform_stress_test(),
            'diagnosis': {},
            'recommendations': []
        }
        
        # Generate diagnosis
        diagnosis = []
        
        # Check health score
        if report['service_d_health']:
            health_score = report['service_d_health'].get('health_score', 0.0)
            if health_score < 0.8:
                diagnosis.append(f"Low health score: {health_score:.2f} (should be > 0.8)")
                
        # Check dependencies
        unhealthy_deps = [dep for dep, info in report['dependencies'].items() if not info['is_healthy']]
        if unhealthy_deps:
            diagnosis.append(f"Unhealthy dependencies: {', '.join(unhealthy_deps)}")
            
        # Check error patterns
        if report['error_patterns']['error_count'] > 0:
            diagnosis.append(f"Found {report['error_patterns']['error_count']} errors in recent logs")
            
        # Check stress test results
        if report['stress_test']['success_rate'] < 0.9:
            diagnosis.append(f"Low success rate in stress test: {report['stress_test']['success_rate']:.2f}")
            
        if report['stress_test']['health_degradation']:
            diagnosis.append("Health score degraded during stress test")
            
        report['diagnosis'] = diagnosis
        
        # Generate recommendations
        recommendations = []
        
        if health_score < 0.8:
            recommendations.append("Investigate root cause of low health score")
            
        if unhealthy_deps:
            recommendations.append("Fix health issues in dependent services")
            
        if report['error_patterns']['error_count'] > 0:
            recommendations.append("Review and fix error patterns in logs")
            
        if report['stress_test']['success_rate'] < 0.9:
            recommendations.append("Improve service reliability and error handling")
            
        if report['stress_test']['health_degradation']:
            recommendations.append("Implement better resource management and recovery")
            
        report['recommendations'] = recommendations
        
        return report
        
    def create_visualizations(self, report: Dict):
        """Create visualizations for the diagnostic report."""
        logger.info("Creating diagnostic visualizations")
        
        # Set up plotting
        plt.style.use('seaborn-v0_8')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Health Score Comparison
        services = list(report['health_comparison']['services'].keys())
        health_scores = [report['health_comparison']['services'][s]['health_score'] for s in services]
        colors = ['red' if s == 'service_d' else 'green' for s in services]
        
        bars = ax1.bar(services, health_scores, color=colors, alpha=0.7)
        ax1.set_title('Health Score Comparison')
        ax1.set_ylabel('Health Score')
        ax1.set_ylim(0, 1.1)
        
        # Add value labels
        for bar, score in zip(bars, health_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.2f}', ha='center', va='bottom')
        
        # 2. Service D Metrics Trends
        if report['metrics_trends']:
            metrics = list(report['metrics_trends'].keys())[:5]  # Top 5 metrics
            slopes = [report['metrics_trends'][m]['trend_slope'] for m in metrics]
            
            colors = ['red' if slope > 0 else 'blue' for slope in slopes]
            bars = ax2.barh(metrics, slopes, color=colors, alpha=0.7)
            ax2.set_title('Service D Metric Trends')
            ax2.set_xlabel('Trend Slope')
            ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # 3. Dependency Health
        if report['dependencies']:
            deps = list(report['dependencies'].keys())
            dep_health = [report['dependencies'][d]['health_score'] for d in deps]
            colors = ['green' if h > 0.8 else 'orange' if h > 0.6 else 'red' for h in dep_health]
            
            bars = ax3.bar(deps, dep_health, color=colors, alpha=0.7)
            ax3.set_title('Service D Dependencies Health')
            ax3.set_ylabel('Health Score')
            ax3.set_ylim(0, 1.1)
            ax3.tick_params(axis='x', rotation=45)
        
        # 4. Stress Test Results
        if report['stress_test']['response_times']:
            ax4.hist(report['stress_test']['response_times'], bins=10, alpha=0.7, color='skyblue', edgecolor='black')
            ax4.set_title('Service D Response Times (Stress Test)')
            ax4.set_xlabel('Response Time (seconds)')
            ax4.set_ylabel('Frequency')
            ax4.axvline(np.mean(report['stress_test']['response_times']), color='red', linestyle='--', label='Mean')
            ax4.legend()
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'service_d_diagnostic.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def generate_html_report(self, report: Dict):
        """Generate HTML diagnostic report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Service D Health Investigation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 30px 0; }}
                .issue {{ background-color: #ffe6e6; padding: 10px; margin: 5px 0; border-left: 4px solid #ff4444; }}
                .recommendation {{ background-color: #e8f4fd; padding: 10px; margin: 5px 0; border-left: 4px solid #2196F3; }}
                .metric {{ background-color: #f9f9f9; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .visualization {{ text-align: center; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Service D Health Investigation Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <p><strong>Service D Health Score:</strong> {report['service_d_health'].get('health_score', 'N/A') if report['service_d_health'] else 'N/A'}</p>
                <p><strong>Issues Identified:</strong> {len(report['diagnosis'])}</p>
                <p><strong>Recommendations:</strong> {len(report['recommendations'])}</p>
            </div>
            
            <div class="section">
                <h2>Diagnosis</h2>
                {self._generate_diagnosis_html(report['diagnosis'])}
            </div>
            
            <div class="section">
                <h2>Health Comparison</h2>
                {self._generate_health_comparison_html(report['health_comparison'])}
            </div>
            
            <div class="section">
                <h2>Dependencies Analysis</h2>
                {self._generate_dependencies_html(report['dependencies'])}
            </div>
            
            <div class="section">
                <h2>Error Analysis</h2>
                {self._generate_error_analysis_html(report['error_patterns'])}
            </div>
            
            <div class="section">
                <h2>Stress Test Results</h2>
                {self._generate_stress_test_html(report['stress_test'])}
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                {self._generate_recommendations_html(report['recommendations'])}
            </div>
            
            <div class="section">
                <h2>Visualizations</h2>
                <div class="visualization">
                    <img src="service_d_diagnostic.png" alt="Service D Diagnostic" style="max-width: 100%;">
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(self.results_dir / 'service_d_investigation_report.html', 'w') as f:
            f.write(html_content)
            
        logger.info(f"HTML report generated: {self.results_dir / 'service_d_investigation_report.html'}")
        
    def _generate_diagnosis_html(self, diagnosis: List[str]) -> str:
        """Generate HTML for diagnosis."""
        if not diagnosis:
            return "<p>No issues identified.</p>"
            
        html = ""
        for issue in diagnosis:
            html += f'<div class="issue"><strong>Issue:</strong> {issue}</div>'
        return html
        
    def _generate_health_comparison_html(self, comparison: Dict) -> str:
        """Generate HTML for health comparison."""
        html = "<table><tr><th>Service</th><th>Health Score</th><th>Status</th></tr>"
        
        for service, info in comparison['services'].items():
            html += f"""
            <tr>
                <td>{service}</td>
                <td>{info['health_score']:.3f}</td>
                <td>{info['status']}</td>
            </tr>
            """
            
        html += "</table>"
        
        if comparison['service_d_issues']:
            html += "<h3>Service D Issues:</h3>"
            for issue in comparison['service_d_issues']:
                html += f'<div class="issue">{issue}</div>'
                
        return html
        
    def _generate_dependencies_html(self, dependencies: Dict) -> str:
        """Generate HTML for dependencies analysis."""
        if not dependencies:
            return "<p>No dependencies found.</p>"
            
        html = "<table><tr><th>Dependency</th><th>Health Score</th><th>Status</th><th>Healthy</th></tr>"
        
        for dep, info in dependencies.items():
            html += f"""
            <tr>
                <td>{dep}</td>
                <td>{info['health_score']:.3f}</td>
                <td>{info['status']}</td>
                <td>{'✅' if info['is_healthy'] else '❌'}</td>
            </tr>
            """
            
        html += "</table>"
        return html
        
    def _generate_error_analysis_html(self, error_patterns: Dict) -> str:
        """Generate HTML for error analysis."""
        html = f"<p><strong>Total Errors Found:</strong> {error_patterns.get('error_count', 0)}</p>"
        
        if error_patterns.get('error_types'):
            html += "<h3>Error Types:</h3><ul>"
            for error_type, count in error_patterns['error_types'].items():
                html += f"<li>{error_type}: {count}</li>"
            html += "</ul>"
            
        if error_patterns.get('errors'):
            html += "<h3>Recent Errors:</h3>"
            for error in error_patterns['errors'][:5]:  # Show last 5
                html += f'<div class="issue">{error["line"]}</div>'
                
        return html
        
    def _generate_stress_test_html(self, stress_test: Dict) -> str:
        """Generate HTML for stress test results."""
        html = f"""
        <p><strong>Success Rate:</strong> {stress_test['success_rate']:.2%}</p>
        <p><strong>Average Response Time:</strong> {np.mean(stress_test['response_times']):.3f}s</p>
        <p><strong>Health Degradation:</strong> {'Yes' if stress_test['health_degradation'] else 'No'}</p>
        """
        
        if stress_test['errors']:
            html += "<h3>Errors During Stress Test:</h3>"
            for error in stress_test['errors']:
                html += f'<div class="issue">{error}</div>'
                
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
    """Main function to investigate Service D's health issues."""
    logger.info("Starting Service D Health Investigation")
    
    investigator = ServiceDInvestigator()
    
    # Generate comprehensive diagnostic report
    logger.info("Generating diagnostic report...")
    report = investigator.generate_diagnostic_report()
    
    # Create visualizations
    logger.info("Creating visualizations...")
    investigator.create_visualizations(report)
    
    # Generate HTML report
    logger.info("Generating HTML report...")
    investigator.generate_html_report(report)
    
    # Save report to JSON
    report_file = investigator.results_dir / 'service_d_investigation.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
        
    # Print summary
    print("\n" + "="*60)
    print("SERVICE D HEALTH INVESTIGATION COMPLETED")
    print("="*60)
    
    if report['service_d_health']:
        health_score = report['service_d_health'].get('health_score', 0.0)
        print(f"🏥 Service D Health Score: {health_score:.2f}")
    
    print(f"🔍 Issues Identified: {len(report['diagnosis'])}")
    print(f"💡 Recommendations: {len(report['recommendations'])}")
    print(f"📁 Reports saved to: {investigator.results_dir}")
    
    if report['diagnosis']:
        print("\n🚨 Key Issues Found:")
        for issue in report['diagnosis'][:3]:  # Show top 3
            print(f"  • {issue}")
            
    if report['recommendations']:
        print("\n💡 Top Recommendations:")
        for rec in report['recommendations'][:3]:  # Show top 3
            print(f"  • {rec}")
            
    print("="*60)


if __name__ == "__main__":
    main() 