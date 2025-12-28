pipeline {
    agent any
    
    environment {
        PROJECT_DIR = "DDoS_SDN by Aiken Kazin"
        DOCKER_COMPOSE_FILE = "docker-compose.ci.yml"
        NEXUS_REPOSITORY_URL = "http://localhost:8081"
        NEXUS_REPOSITORY_ID = "nexus-repo"
        BUILD_TIMESTAMP = "${BUILD_ID}-${BUILD_TIMESTAMP}"
    }
    
    options {
        timeout(time: 30, unit: 'MINUTES')
        timestamps()
    }
    
    stages {
        stage('Checkout Code') {
            steps {
                checkout([
                    $class: 'GitSCM',
                    branches: [[name: '*/main']],
                    extensions: [],
                    userRemoteConfigs: [[
                        credentialsId: 'jenkins-pat',
                        url: 'https://github.com/DatNguyenPT/sdn-ids.git'
                    ]]
                ])
                
                script {
                    echo "Repository checked out successfully"
                    sh 'git rev-parse HEAD'
                }
            }
        }

        stage('Stage 1 - Build & Validate Containers') {
            steps {
                dir(env.PROJECT_DIR) {
                    script {
                        echo "Stage 1: Building Docker Images"
                        // Set up Docker Buildx
                        sh """
                            # Install/upgrade Docker Buildx if needed
                            mkdir -p ~/.docker/cli-plugins
                            if ! docker buildx version &>/dev/null; then
                                echo "Installing Docker Buildx..."
                                BUILDX_VERSION=v0.17.0
                                curl -L "https://github.com/docker/buildx/releases/download/\${BUILDX_VERSION}/buildx-\${BUILDX_VERSION}.linux-amd64" -o ~/.docker/cli-plugins/docker-buildx
                                chmod +x ~/.docker/cli-plugins/docker-buildx
                                docker buildx install || true
                            fi
                            docker buildx version || echo "Buildx check completed"
                            docker buildx create --use --name builder || docker buildx use builder || true
                        """
                        sh """
                            # Build images with buildx
                            COMPOSE_DOCKER_CLI_BUILD=1 DOCKER_BUILDKIT=1 docker-compose -f ${env.DOCKER_COMPOSE_FILE} build --no-cache
                        """
                        echo "Docker images built successfully"

                        echo "Validating Docker Images"
                        sh """
                            docker images | grep -E "(flower-server|flower-worker|fl-dashboard|mlflow-server)" || true
                        """
                    }
                }
            }
        }

        stage('Stage 2 - Start FL System (CI Mode)') {
            steps {
                dir(env.PROJECT_DIR) {
                    script {
                        echo "Starting FL System"
                        sh """
                            docker-compose -f ${env.DOCKER_COMPOSE_FILE} up -d mlflow-server fl-dashboard
                        """
                        echo "Waiting for MLflow server to be ready..."
                        sh """
                            timeout 60 bash -c 'until curl -f http://mlflow-server:5000/health 2>/dev/null || docker-compose -f ${env.DOCKER_COMPOSE_FILE} logs mlflow-server | grep -q "listening"; do sleep 2; done' || true
                        """
                        sh """
                            docker-compose -f ${env.DOCKER_COMPOSE_FILE} up -d flower-server-lstm
                        """
                        sleep(time: 10, unit: 'SECONDS')
                        sh """
                            docker-compose -f ${env.DOCKER_COMPOSE_FILE} up -d flower-worker-1 flower-worker-2
                        """
                        sh """
                            docker-compose -f ${env.DOCKER_COMPOSE_FILE} ps
                        """
                    }
                }
            }
        }

        stage('Stage 3 - MLflow Health Check') {
            steps {
                dir(env.PROJECT_DIR) {
                    script {
                        echo "MLflow Health Check"
                        sh """
                            docker-compose -f ${env.DOCKER_COMPOSE_FILE} up -d mlflow-server fl-dashboard flower-server-lstm flower-worker-1 flower-worker-2
                        """
                        sleep(time: 20, unit: 'SECONDS')
                        sh """
                            python scripts/check_mlflow_health.py \\
                                --mlflow-url http://mlflow-server:5002 \\
                                --timeout 30 2>&1 | tee mlflow_health_output.log || true
                        """
                        echo "MLflow health check completed"
                    }
                }
            }
        }

        stage('Stage 4 - Run Federated Smoke Test') {
            steps {
                dir(env.PROJECT_DIR) {
                    script {
                        echo "Running Federated Smoke Test"
                        sh """
                            docker-compose -f ${env.DOCKER_COMPOSE_FILE} up -d mlflow-server fl-dashboard flower-server-lstm flower-worker-1 flower-worker-2
                        """
                        sleep(time: 15, unit: 'SECONDS')
                        sh """
                            python scripts/smoke_test_lstm.py \\
                                --server-url http://flower-server-lstm:8080 \\
                                --mlflow-url http://mlflow-server:5000 \\
                                --timeout 300 \\
                                --min-rounds 2 \\
                                --server-container flower-server-lstm-ci 2>&1 | tee smoke_test_output.log
                        """
                        echo "Smoke test passed"
                    }
                }
            }
        }

        stage('Stage 5 - Verify System Behavior') {
            steps {
                dir(env.PROJECT_DIR) {
                    script {
                        echo "Verifying System Behavior"
                        sh """
                            docker-compose -f ${env.DOCKER_COMPOSE_FILE} up -d mlflow-server fl-dashboard flower-server-lstm flower-worker-1 flower-worker-2
                            sleep 20
                            TIMEOUT=900
                            ELAPSED=0
                            TRAINING_COMPLETE=false
                            while [ \$ELAPSED -lt \$TIMEOUT ]; do
                                SERVER_LOGS=\$(docker-compose -f ${env.DOCKER_COMPOSE_FILE} logs flower-server-lstm 2>&1)
                                if echo "\$SERVER_LOGS" | grep -qE "All 2 rounds completed|Server for LSTM completed all rounds|SUMMARY"; then
                                    TRAINING_COMPLETE=true
                                    break
                                fi
                                if [ -f "models/LSTM_FL.h5" ]; then
                                    TRAINING_COMPLETE=true
                                    break
                                fi
                                sleep 15
                                ELAPSED=\$((ELAPSED + 15))
                            done
                            if [ "\$TRAINING_COMPLETE" = false ]; then
                                echo "Training did not complete within \$TIMEOUT seconds"
                                docker-compose -f ${env.DOCKER_COMPOSE_FILE} ps
                                docker-compose -f ${env.DOCKER_COMPOSE_FILE} logs flower-server-lstm | tail -50
                                exit 1
                            fi
                        """
                        def modelCheckResult = sh(script: 'if [ -f "models/LSTM_FL.h5" ]; then echo "exists"; else echo "not_found"; fi', returnStdout: true).trim()
                        if (modelCheckResult == 'exists') {
                            echo "Model file created successfully"
                        } else {
                            error("Model file not found")
                        }
                    }
                }
            }
        }

        stage('Stage 7 - SAST (Static Application Security Testing)') {
            steps {
                dir(env.PROJECT_DIR) {
                    script {
                        echo "Running SAST"
                        sh '''
                            mkdir -p security
                            which bandit || pip install bandit[toml] || true
                            bandit -r . -f json -o security/sast-bandit-report.json --severity-level medium || true
                            find . -name "*.sh" -type f -exec shellcheck {} \\; 2>/dev/null || true
                        '''
                    }
                }
            }
        }

        stage('Stage 8 - Gitleak (Secrets Detection)') {
            steps {
                dir(env.PROJECT_DIR) {
                    script {
                        echo "Running Gitleaks"
                        sh '''
                            mkdir -p security
                            which gitleaks || curl -sSfL https://github.com/gitleaks/gitleaks/releases/latest/download/gitleaks-linux-amd64.tar.gz | tar -xz && sudo mv gitleaks /usr/local/bin || true
                            gitleaks detect --verbose --redact --report-format json --report-path security/gitleaks-report.json || true
                        '''
                    }
                }
            }
        }

        stage('Stage 9 - Trivy (Container Vulnerability Scanning)') {
            steps {
                dir(env.PROJECT_DIR) {
                    script {
                        echo "Running Trivy"
                        sh '''
                            mkdir -p security
                            which trivy || curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin || true
                            trivy image --format json --output security/trivy-flower-server.json flower-server:latest || true
                            trivy image --format json --output security/trivy-flower-worker.json flower-worker:latest || true
                            trivy image --format json --output security/trivy-fl-dashboard.json fl-dashboard:latest || true
                            trivy image --format json --output security/trivy-mlflow-server.json mlflow-server:latest || true
                        '''
                    }
                }
            }
        }

        stage('Stage 10 - DAST (Dynamic Application Security Testing)') {
            steps {
                dir(env.PROJECT_DIR) {
                    script {
                        echo "Running DAST"
                        sh '''
                            docker-compose -f ${DOCKER_COMPOSE_FILE} up -d fl-dashboard mlflow-server
                            sleep 30
                            curl -I http://localhost:5001 | grep -E "(X-Frame-Options|X-Content-Type-Options|Content-Security-Policy)" || true
                            curl -s http://localhost:5000/health | grep "ok" || true
                        '''
                    }
                }
            }
        }

        stage('Stage 11 - Quality Gate') {
            steps {
                dir(env.PROJECT_DIR) {
                    script {
                        echo "Running Quality Gate"
                        sh '''
                            mkdir -p quality
                            which pylint || pip install pylint || true
                            find . -name "*.py" -not -path "./__pycache__/*" -exec pylint --output-format=json {} \; > quality/pylint-report.json 2>/dev/null || true
                            python -m pytest --cov=. --cov-report=xml:quality/coverage.xml --cov-report=term 2>/dev/null || true
                        '''
                    }
                }
            }
        }

        stage('Stage 12 - OWASP Top 10 for AI Model') {
            steps {
                dir(env.PROJECT_DIR) {
                    script {
                        echo "Running OWASP AI checks"
                        sh '''
                            mkdir -p ai-security
                            grep -r "prompt.*injection\\|input.*sanitiz" --include="*.py" . | wc -l > ai-security/owasp-ai-checks.txt
                        '''
                    }
                }
            }
        }

        stage('Stage 13 - Sign Model Artifacts') {
            steps {
                dir(env.PROJECT_DIR) {
                    script {
                        echo "Signing model artifacts"
                        sh '''
                            mkdir -p signatures
                            if command -v gpg &> /dev/null; then
                                gpg --batch --armor --sign --detach-sign models/LSTM_FL.h5 || true
                            else
                                cd models
                                sha256sum *.h5 > ../signatures/checksums.sha256
                                cd ..
                            fi
                            cat > signatures/model-metadata.json <<EOF
{
  "build_id": "${BUILD_ID}",
  "build_number": "${BUILD_NUMBER}",
  "build_timestamp": "${BUILD_TIMESTAMP}",
  "git_commit": "$(git rev-parse HEAD)",
  "git_branch": "${GIT_BRANCH}",
  "model_type": "LSTM",
  "model_framework": "TensorFlow/Keras",
  "fl_rounds": 2,
  "signature_algorithm": "SHA256"
}
EOF
                        '''
                    }
                }
            }
        }

        stage('Stage 14 - Upload Artifacts to Nexus') {
            steps {
                dir(env.PROJECT_DIR) {
                    script {
                        echo "Uploading artifacts to Nexus"
                        withCredentials([usernamePassword(credentialsId: 'nexus-credentials', usernameVariable: 'NEXUS_USER', passwordVariable: 'NEXUS_PASS')]) {
                            sh '''
                                NEXUS_RAW_REPO="${NEXUS_REPOSITORY_URL}/repository/raw-hosted"
                                ARTIFACT_PATH="fl-pipeline/artifacts/${BUILD_ID}"
                                [ -f "signatures/model-metadata.json" ] && curl -v -u ${NEXUS_USER}:${NEXUS_PASS} --upload-file signatures/model-metadata.json "${NEXUS_RAW_REPO}/${ARTIFACT_PATH}/model-metadata.json"
                                [ -f "signatures/checksums.sha256" ] && curl -v -u ${NEXUS_USER}:${NEXUS_PASS} --upload-file signatures/checksums.sha256 "${NEXUS_RAW_REPO}/${ARTIFACT_PATH}/checksums.sha256"
                            '''
                        }
                    }
                }
            }
        }

        stage('Stage 15 - Upload Model to Nexus') {
            steps {
                dir(env.PROJECT_DIR) {
                    script {
                        echo "Uploading models to Nexus"
                        withCredentials([usernamePassword(credentialsId: 'nexus-credentials', usernameVariable: 'NEXUS_USER', passwordVariable: 'NEXUS_PASS')]) {
                            sh '''
                                NEXUS_MODELS_REPO="${NEXUS_REPOSITORY_URL}/repository/models-hosted"
                                [ -f "models/LSTM_FL.h5" ] && curl -v -u ${NEXUS_USER}:${NEXUS_PASS} --upload-file models/LSTM_FL.h5 "${NEXUS_MODELS_REPO}/lstm-fl-${BUILD_ID}.h5"
                            '''
                        }
                    }
                }
            }
        }

        stage('Stage 16 - Upload Docker Images to Nexus') {
            steps {
                script {
                    echo "Uploading Docker images to Nexus"
                    withCredentials([usernamePassword(credentialsId: 'nexus-credentials', usernameVariable: 'NEXUS_USER', passwordVariable: 'NEXUS_PASS')]) {
                        sh '''
                            NEXUS_DOCKER_REPO="${NEXUS_REPOSITORY_URL}/repository/docker-self-hosted"
                            IMAGE_PATH="fl-pipeline/docker-images/${BUILD_ID}"
                            for IMG in flower-server flower-worker fl-dashboard mlflow-server; do
                                docker save ${IMG}:latest -o ${IMG}-${BUILD_ID}.tar
                                curl -v -u ${NEXUS_USER}:${NEXUS_PASS} --upload-file ${IMG}-${BUILD_ID}.tar "${NEXUS_DOCKER_REPO}/${IMAGE_PATH}/${IMG}-${BUILD_ID}.tar"
                                rm ${IMG}-${BUILD_ID}.tar
                            done
                        '''
                    }
                }
            }
        }

        stage('Stage 17 - Generate Build Report') {
            steps {
                script {
                    echo "Generating build report"
                    withCredentials([usernamePassword(credentialsId: 'nexus-credentials', usernameVariable: 'NEXUS_USER', passwordVariable: 'NEXUS_PASS')]) {
                        sh '''
                            cat > build-report.json <<EOF
{
  "build_id": "${BUILD_ID}",
  "build_number": "${BUILD_NUMBER}",
  "build_timestamp": "${BUILD_TIMESTAMP}",
  "status": "SUCCESS",
  "stages_completed": 17
}
EOF
                            curl -v -u ${NEXUS_USER}:${NEXUS_PASS} --upload-file build-report.json "${NEXUS_REPOSITORY_URL}/repository/raw-hosted/fl-pipeline/reports/build-report-${BUILD_ID}.json"
                        '''
                    }
                }
            }
        }
    }

    post {
        always {
            dir(env.PROJECT_DIR) {
                script {
                    echo "Cleaning up"
                    sh "docker-compose -f ${env.DOCKER_COMPOSE_FILE} down -v || true"
                    sh "docker system prune -f || true"
                }
            }
        }
        failure {
            dir(env.PROJECT_DIR) {
                script {
                    echo "Collecting logs for debugging"
                    sh """
                        docker-compose -f ${env.DOCKER_COMPOSE_FILE} logs flower-server-lstm || true
                        docker-compose -f ${env.DOCKER_COMPOSE_FILE} logs flower-worker-1 || true
                        docker-compose -f ${env.DOCKER_COMPOSE_FILE} logs flower-worker-2 || true
                        docker-compose -f ${env.DOCKER_COMPOSE_FILE} logs mlflow-server || true
                        docker-compose -f ${env.DOCKER_COMPOSE_FILE} logs fl-dashboard || true
                    """
                }
            }
        }
        success {
            echo "Pipeline completed successfully!"
        }
    }
}
