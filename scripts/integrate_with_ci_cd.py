#!/usr/bin/env python3
"""
CI/CD Integration for Graph-Heal
Provides integration with GitHub Actions, GitLab CI, and Jenkins.
"""

import os
import json
import logging
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CICDIntegration:
    """Integrates Graph-Heal with CI/CD pipelines."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.ci_dir = Path(config.get('ci_dir', '.github/workflows'))
        self.ci_dir.mkdir(parents=True, exist_ok=True)
        self.gitlab_dir = Path(config.get('gitlab_dir', '.gitlab-ci'))
        self.gitlab_dir.mkdir(parents=True, exist_ok=True)
        self.jenkins_dir = Path(config.get('jenkins_dir', 'jenkins'))
        self.jenkins_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_github_actions(self):
        """Generate GitHub Actions workflows."""
        
        # Main CI/CD workflow
        main_workflow = {
            'name': 'Graph-Heal CI/CD',
            'on': {
                'push': {
                    'branches': ['main', 'develop']
                },
                'pull_request': {
                    'branches': ['main']
                }
            },
            'jobs': {
                'test': {
                    'runs-on': 'ubuntu-latest',
                    'steps': [
                        {
                            'name': 'Checkout code',
                            'uses': 'actions/checkout@v3'
                        },
                        {
                            'name': 'Set up Python',
                            'uses': 'actions/setup-python@v4',
                            'with': {
                                'python-version': '3.9'
                            }
                        },
                        {
                            'name': 'Install dependencies',
                            'run': 'pip install -r requirements.txt'
                        },
                        {
                            'name': 'Run tests',
                            'run': 'python -m pytest tests/ -v --cov=graph_heal --cov-report=xml'
                        },
                        {
                            'name': 'Upload coverage',
                            'uses': 'codecov/codecov-action@v3',
                            'with': {
                                'file': './coverage.xml'
                            }
                        }
                    ]
                },
                'build': {
                    'runs-on': 'ubuntu-latest',
                    'needs': 'test',
                    'steps': [
                        {
                            'name': 'Checkout code',
                            'uses': 'actions/checkout@v3'
                        },
                        {
                            'name': 'Set up Docker Buildx',
                            'uses': 'docker/setup-buildx-action@v2'
                        },
                        {
                            'name': 'Build and push Docker image',
                            'uses': 'docker/build-push-action@v4',
                            'with': {
                                'context': '.',
                                'push': True,
                                'tags': 'graphheal/graph-heal:latest'
                            }
                        }
                    ]
                },
                'deploy': {
                    'runs-on': 'ubuntu-latest',
                    'needs': 'build',
                    'if': "github.ref == 'refs/heads/main'",
                    'steps': [
                        {
                            'name': 'Deploy to Kubernetes',
                            'run': 'kubectl apply -f k8s/'
                        }
                    ]
                }
            }
        }
        
        # Security scanning workflow
        security_workflow = {
            'name': 'Security Scan',
            'on': {
                'schedule': [{'cron': '0 2 * * *'}],  # Daily at 2 AM
                'workflow_dispatch': {}
            },
            'jobs': {
                'security-scan': {
                    'runs-on': 'ubuntu-latest',
                    'steps': [
                        {
                            'name': 'Checkout code',
                            'uses': 'actions/checkout@v3'
                        },
                        {
                            'name': 'Run Trivy vulnerability scanner',
                            'uses': 'aquasecurity/trivy-action@master',
                            'with': {
                                'scan-type': 'fs',
                                'scan-ref': '.',
                                'format': 'sarif',
                                'output': 'trivy-results.sarif'
                            }
                        },
                        {
                            'name': 'Upload Trivy scan results',
                            'uses': 'github/codeql-action/upload-sarif@v2',
                            'with': {
                                'sarif_file': 'trivy-results.sarif'
                            }
                        }
                    ]
                }
            }
        }
        
        # Performance testing workflow
        performance_workflow = {
            'name': 'Performance Testing',
            'on': {
                'workflow_dispatch': {},
                'schedule': [{'cron': '0 4 * * 0'}]  # Weekly on Sunday
            },
            'jobs': {
                'performance-test': {
                    'runs-on': 'ubuntu-latest',
                    'steps': [
                        {
                            'name': 'Checkout code',
                            'uses': 'actions/checkout@v3'
                        },
                        {
                            'name': 'Set up Python',
                            'uses': 'actions/setup-python@v4',
                            'with': {
                                'python-version': '3.9'
                            }
                        },
                        {
                            'name': 'Install dependencies',
                            'run': 'pip install -r requirements.txt'
                        },
                        {
                            'name': 'Start test environment',
                            'run': 'docker-compose up -d'
                        },
                        {
                            'name': 'Run performance tests',
                            'run': 'python scripts/measure_detection_performance.py'
                        },
                        {
                            'name': 'Generate performance report',
                            'run': 'python scripts/generate_performance_report.py'
                        },
                        {
                            'name': 'Upload performance results',
                            'uses': 'actions/upload-artifact@v3',
                            'with': {
                                'name': 'performance-results',
                                'path': 'data/performance_measurements/'
                            }
                        }
                    ]
                }
            }
        }
        
        # Save workflows
        workflow_files = [
            ('ci-cd.yml', main_workflow),
            ('security-scan.yml', security_workflow),
            ('performance-test.yml', performance_workflow)
        ]
        
        for filename, workflow in workflow_files:
            filepath = self.ci_dir / filename
            with open(filepath, 'w') as f:
                yaml.dump(workflow, f, default_flow_style=False)
            logger.info(f"Generated GitHub Actions workflow: {filepath}")
            
    def generate_gitlab_ci(self):
        """Generate GitLab CI configuration."""
        
        gitlab_ci = {
            'stages': ['test', 'build', 'deploy'],
            'variables': {
                'DOCKER_DRIVER': 'overlay2',
                'DOCKER_TLS_CERTDIR': '/certs'
            },
            'services': ['docker:dind'],
            'test': {
                'stage': 'test',
                'image': 'python:3.9-slim',
                'script': [
                    'pip install -r requirements.txt',
                    'python -m pytest tests/ -v --cov=graph_heal --cov-report=xml'
                ],
                'artifacts': {
                    'reports': {
                        'cobertura': 'coverage.xml'
                    }
                }
            },
            'build': {
                'stage': 'build',
                'image': 'docker:latest',
                'script': [
                    'docker build -t graphheal/graph-heal:$CI_COMMIT_SHA .',
                    'docker tag graphheal/graph-heal:$CI_COMMIT_SHA graphheal/graph-heal:latest'
                ],
                'only': ['main', 'develop']
            },
            'deploy': {
                'stage': 'deploy',
                'image': 'bitnami/kubectl:latest',
                'script': [
                    'kubectl apply -f k8s/'
                ],
                'only': ['main'],
                'environment': {
                    'name': 'production'
                }
            }
        }
        
        filepath = self.gitlab_dir / '.gitlab-ci.yml'
        with open(filepath, 'w') as f:
            yaml.dump(gitlab_ci, f, default_flow_style=False)
        logger.info(f"Generated GitLab CI configuration: {filepath}")
        
    def generate_jenkins_pipeline(self):
        """Generate Jenkins pipeline configuration."""
        
        # Jenkinsfile
        jenkinsfile = '''pipeline {
    agent any
    
    environment {
        DOCKER_IMAGE = 'graphheal/graph-heal'
        DOCKER_TAG = "${env.BUILD_NUMBER}"
    }
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        
        stage('Test') {
            steps {
                sh 'pip install -r requirements.txt'
                sh 'python -m pytest tests/ -v --cov=graph_heal --cov-report=xml'
            }
            post {
                always {
                    publishCobertura coberturaReportFile: 'coverage.xml'
                }
            }
        }
        
        stage('Build') {
            when {
                anyOf {
                    branch 'main'
                    branch 'develop'
                }
            }
            steps {
                script {
                    docker.build("${DOCKER_IMAGE}:${DOCKER_TAG}")
                    docker.tag("${DOCKER_IMAGE}:${DOCKER_TAG}", "${DOCKER_IMAGE}:latest")
                }
            }
        }
        
        stage('Deploy') {
            when {
                branch 'main'
            }
            steps {
                sh 'kubectl apply -f k8s/'
            }
        }
    }
    
    post {
        always {
            cleanWs()
        }
        success {
            echo 'Pipeline completed successfully!'
        }
        failure {
            echo 'Pipeline failed!'
        }
    }
}'''
        
        filepath = self.jenkins_dir / 'Jenkinsfile'
        with open(filepath, 'w') as f:
            f.write(jenkinsfile)
        logger.info(f"Generated Jenkins pipeline: {filepath}")
        
        # Jenkins job configuration
        job_config = {
            'name': 'graphheal-pipeline',
            'description': 'Graph-Heal CI/CD Pipeline',
            'triggers': {
                'githubPush': True,
                'pollSCM': 'H/5 * * * *'  # Poll every 5 minutes
            },
            'pipeline': {
                'definition': {
                    'type': 'pipeline script from SCM',
                    'scm': {
                        'type': 'git',
                        'url': 'https://github.com/your-org/graph-heal.git',
                        'branches': ['main', 'develop']
                    },
                    'scriptPath': 'jenkins/Jenkinsfile'
                }
            }
        }
        
        job_filepath = self.jenkins_dir / 'job-config.xml'
        with open(job_filepath, 'w') as f:
            f.write(f'''<?xml version='1.1' encoding='UTF-8'?>
<project>
    <description>{job_config['description']}</description>
    <triggers>
        <hudson.triggers.SCMTrigger>
            <spec>{job_config['triggers']['pollSCM']}</spec>
        </hudson.triggers.SCMTrigger>
    </triggers>
    <definition class="org.jenkinsci.plugins.workflow.cps.CpsScmFlowDefinition" plugin="workflow-cps@2.92">
        <scm class="hudson.plugins.git.GitSCM" plugin="git@4.10.2">
            <configVersion>2</configVersion>
            <userRemoteConfigs>
                <hudson.plugins.git.UserRemoteConfig>
                    <url>{job_config['pipeline']['scm']['url']}</url>
                </hudson.plugins.git.UserRemoteConfig>
            </userRemoteConfigs>
            <branches>
                <hudson.plugins.git.BranchSpec>
                    <name>*/main</name>
                </hudson.plugins.git.BranchSpec>
                <hudson.plugins.git.BranchSpec>
                    <name>*/develop</name>
                </hudson.plugins.git.BranchSpec>
            </branches>
        </scm>
        <scriptPath>{job_config['pipeline']['definition']['scriptPath']}</scriptPath>
        <lightweight>true</lightweight>
    </definition>
</project>''')
        logger.info(f"Generated Jenkins job configuration: {job_filepath}")
        
    def generate_dockerfile(self):
        """Generate optimized Dockerfile for Graph-Heal."""
        
        dockerfile = '''# Multi-stage build for Graph-Heal
FROM python:3.9-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.9-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY graph_heal/ ./graph_heal/
COPY scripts/ ./scripts/
COPY config/ ./config/

# Create non-root user
RUN useradd --create-home --shell /bin/bash graphheal \\
    && chown -R graphheal:graphheal /app

USER graphheal

# Add local bin to PATH
ENV PATH=/root/.local/bin:$PATH

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8080/health || exit 1

# Expose port
EXPOSE 8080

# Default command
CMD ["python", "-m", "graph_heal"]'''
        
        filepath = Path('Dockerfile')
        with open(filepath, 'w') as f:
            f.write(dockerfile)
        logger.info(f"Generated Dockerfile: {filepath}")
        
    def generate_helm_chart(self):
        """Generate Helm chart for Kubernetes deployment."""
        
        helm_dir = Path('helm/graph-heal')
        helm_dir.mkdir(parents=True, exist_ok=True)
        
        # Chart.yaml
        chart_yaml = {
            'apiVersion': 'v2',
            'name': 'graph-heal',
            'description': 'Graph-Heal Fault Detection and Recovery System',
            'type': 'application',
            'version': '0.1.0',
            'appVersion': '1.0.0'
        }
        
        chart_file = helm_dir / 'Chart.yaml'
        with open(chart_file, 'w') as f:
            yaml.dump(chart_yaml, f, default_flow_style=False)
            
        # values.yaml
        values_yaml = {
            'replicaCount': 1,
            'image': {
                'repository': 'graphheal/graph-heal',
                'tag': 'latest',
                'pullPolicy': 'IfNotPresent'
            },
            'service': {
                'type': 'ClusterIP',
                'port': 8080
            },
            'resources': {
                'limits': {
                    'cpu': '500m',
                    'memory': '512Mi'
                },
                'requests': {
                    'cpu': '250m',
                    'memory': '256Mi'
                }
            },
            'config': {
                'monitoringInterval': 15,
                'detectionThresholds': {
                    'cpu': 80.0,
                    'memory': 85.0,
                    'responseTime': 1.0
                }
            }
        }
        
        values_file = helm_dir / 'values.yaml'
        with open(values_file, 'w') as f:
            yaml.dump(values_yaml, f, default_flow_style=False)
            
        logger.info(f"Generated Helm chart in: {helm_dir}")


def main():
    """Main function to run CI/CD integration."""
    # Configuration
    config = {
        'ci_dir': '.github/workflows',
        'gitlab_dir': '.gitlab-ci',
        'jenkins_dir': 'jenkins'
    }
    
    # Create integration
    integration = CICDIntegration(config)
    
    # Generate all CI/CD configurations
    logger.info("Generating CI/CD integration configurations")
    integration.generate_github_actions()
    integration.generate_gitlab_ci()
    integration.generate_jenkins_pipeline()
    integration.generate_dockerfile()
    integration.generate_helm_chart()
    
    logger.info("CI/CD integration configurations generated successfully")
    logger.info("Files created:")
    logger.info("  - GitHub Actions: .github/workflows/")
    logger.info("  - GitLab CI: .gitlab-ci/.gitlab-ci.yml")
    logger.info("  - Jenkins: jenkins/")
    logger.info("  - Docker: Dockerfile")
    logger.info("  - Helm: helm/graph-heal/")


if __name__ == "__main__":
    main() 