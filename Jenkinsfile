pipeline {
    agent { label 'linux && gpu && compute' }

    environment {
        SONAR_HOST_URL    = 'http://127.0.0.1:9200'
        SONAR_PROJECT_KEY = 'trading'
        DEPLOY_PLAYBOOK   = 'deploy/ansible-deploy.yml'
    }

    options {
        timeout(time: 30, unit: 'MINUTES')
        disableConcurrentBuilds()
    }

    stages {
        stage('Trivy Security Scan') {
            agent {
                docker {
                    image 'aquasec/trivy:latest'
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
                    image 'ghcr.io/astral-sh/uv:python3.13-bookworm-slim'
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
                    image 'sonarsource/sonar-scanner-cli:latest'
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

        stage('Deploy') {
            steps {
                sh '''ANSIBLE_HOST_KEY_CHECKING=False ansible-playbook \
                    -i "127.0.0.1," \
                    "${DEPLOY_PLAYBOOK}" \
                    --connection local \
                    -e "trading_src_dir=${WORKSPACE}" \
                    -e "build_number=${BUILD_NUMBER}"'''
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
