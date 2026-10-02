// Tests and scans only. Deploys run from the Mac with `make deploy`,
// because the playbooks read secrets from gopass.
pipeline {
    agent { label 'linux && gpu && compute' }

    environment {
        SONAR_HOST_URL    = 'http://127.0.0.1:9200'
        SONAR_PROJECT_KEY = 'trading'
    }

    options {
        timeout(time: 30, unit: 'MINUTES')
        disableConcurrentBuilds()
    }

    stages {
        stage('Trivy Security Scan') {
            agent {
                docker {
                    image 'registry.starbluesolutions.net/aquasec/trivy:0.69.3'
                    reuseNode true
                    args '--entrypoint="" -u root --network host'
                }
            }
            steps {
                sh '''trivy fs \
                    --server http://127.0.0.1:4954 \
                    --exit-code 1 \
                    --severity HIGH,CRITICAL \
                    --scanners vuln,secret \
                    --format table \
                    .'''
            }
        }

        stage('Run Tests') {
            agent {
                docker {
                    image 'registry.starbluesolutions.net/astral-sh/uv:python3.13-bookworm-slim'
                    reuseNode true
                    args '-u root --network host'
                }
            }
            steps {
                sh 'uv sync --extra dev --frozen'
                catchError(buildResult: 'UNSTABLE', stageResult: 'UNSTABLE') {
                    sh '.venv/bin/pytest tests/ --tb=short -q --cov=backtest_api --cov-report=xml:coverage.xml'
                }
            }
        }

        stage('SonarQube Analysis') {
            agent {
                docker {
                    image 'registry.starbluesolutions.net/sonarsource/sonar-scanner-cli:latest'
                    reuseNode true
                    args '-u root --network host'
                }
            }
            steps {
                withCredentials([string(credentialsId: 'sonarqube-token-trading', variable: 'SONAR_TOKEN')]) {
                    sh '''sonar-scanner \
                        -Dsonar.projectKey="${SONAR_PROJECT_KEY}" \
                        -Dsonar.sources=backtest_api,paper,scripts \
                        -Dsonar.tests=tests \
                        -Dsonar.python.coverage.reportPaths=coverage.xml \
                        -Dsonar.host.url="${SONAR_HOST_URL}" \
                        -Dsonar.token="${SONAR_TOKEN}"'''
                }
            }
        }
    }

    post {
        always {
            deleteDir()
        }
        success {
            echo 'Pipeline completed successfully.'
        }
        failure {
            echo 'Pipeline failed.'
        }
    }
}
