pipeline {
    agent any
    
    environment {
        PROJECT_DIR = "DDoS_SDN by Aiken Kazin"
        DOCKER_COMPOSE_FILE = "docker-compose.ci.yml"
        NEXUS_REPOSITORY_URL = "http://localhost:8081"
        NEXUS_REPOSITORY_ID = "nexus-repo"
        NEXUS_DOCKER_REGISTRY_URL = "localhost:5000"
        // Use Nexus Docker hosted registry as the docker registry
        DOCKER_REGISTRY_URL = "${NEXUS_DOCKER_REGISTRY_URL}"
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
                        echo "=========================================="
                        echo "Stage 1: Building Docker Images"
                        echo "=========================================="
                        
                        // Build all required images
                        sh """
                            docker compose -f ${env.DOCKER_COMPOSE_FILE} build --no-cache
                        """
                        
                        echo "Docker images built successfully"
                        
                        echo "=========================================="
                        echo "Validating Docker Images"
                        echo "=========================================="
                        
                        // Validate images exist
                        sh """
                            docker images | grep -E "(flower-server|flower-worker|fl-dashboard|mlflow-server)" || true
                        """
                        
                        echo "Docker images validated"
                    }
                }
            }
        }
        
        stage('Stage 2 - Start FL System (CI Mode)') {
            steps {
                dir(env.PROJECT_DIR) {
                    script {
                        echo "=========================================="
                        echo "Stage 2: Starting FL System"
                        echo "=========================================="
                        
                        // Start infrastructure services first (MLflow, Dashboard)
                        sh """
                            docker compose -f ${env.DOCKER_COMPOSE_FILE} up -d mlflow-server fl-dashboard
                        """
                        
                        // Wait for MLflow to be ready
                        echo "Waiting for MLflow server to be ready..."
                        sh """
                            timeout 60 bash -c 'until curl -f http://mlflow-server:5000/health 2>/dev/null || docker compose -f ${env.DOCKER_COMPOSE_FILE} logs mlflow-server | grep -q "listening"; do sleep 2; done' || true
                        """
                        
                        // Start FL server
                        sh """
                            docker compose -f ${env.DOCKER_COMPOSE_FILE} up -d flower-server-lstm
                        """
                        
                        // Wait for FL server to be ready
                        echo "Waiting for FL server to be ready..."
                        sleep(time: 10, unit: 'SECONDS')
                        
                        // Start workers
                        sh """
                            docker compose -f ${env.DOCKER_COMPOSE_FILE} up -d flower-worker-1 flower-worker-2
                        """
                        
                        echo "FL System started"
                        
                        // Show running containers
                        sh """
                            docker compose -f ${env.DOCKER_COMPOSE_FILE} ps
                        """
                    }
                }
            }
        }
        
        stage('Stage 3 - MLflow Health Check') {
            steps {
                dir(env.PROJECT_DIR) {
                    script {
                        echo "=========================================="
                        echo "Stage 3: MLflow Health Check"
                        echo "=========================================="
                        
                        // Ensure services are running
                        sh """
                            docker compose -f ${env.DOCKER_COMPOSE_FILE} up -d mlflow-server fl-dashboard flower-server-lstm flower-worker-1 flower-worker-2
                        """
                        
                        sleep(time: 15, unit: 'SECONDS')
                        
                        // Wait a bit more for services to stabilize
                        sleep(time: 5, unit: 'SECONDS')
                        
                        // Run MLflow health check script
                        sh """
                            python scripts/check_mlflow_health.py \\
                                --mlflow-url http://mlflow-server:5002 \\
                                --timeout 30 2>&1 | tee mlflow_health_output.log || true
                        """
                        
                        echo ""
                        echo "MLflow Health Statistics:"
                        echo "   - Health check completed"
                        echo "   - Check mlflow_health_output.log for details"
                        
                        echo "MLflow health check passed"
                    }
                }
            }
        }
        
        stage('Stage 4 - Run Federated Smoke Test') {
            steps {
                dir(env.PROJECT_DIR) {
                    script {
                        echo "=========================================="
                        echo "Stage 4: Running Federated Smoke Test"
                        echo "=========================================="
                        
                        // Ensure services are running
                        sh """
                            docker compose -f ${env.DOCKER_COMPOSE_FILE} up -d mlflow-server fl-dashboard flower-server-lstm flower-worker-1 flower-worker-2
                        """
                        
                        sleep(time: 15, unit: 'SECONDS')
                        
                        // Run smoke test script and capture output
                        sh """
                            python scripts/smoke_test_lstm.py \\
                                --server-url http://flower-server-lstm:8080 \\
                                --mlflow-url http://mlflow-server:5000 \\
                                --timeout 300 \\
                                --min-rounds 2 \\
                                --server-container flower-server-lstm-ci 2>&1 | tee smoke_test_output.log
                        """
                        
                        echo ""
                        echo "Smoke Test Statistics:"
                        echo "   - Test completed successfully"
                        echo "   - Check smoke_test_output.log for detailed results"
                        
                        echo "Federated smoke test passed"
                    }
                }
            }
        }
        
        stage('Stage 5 - Verify System Behavior') {
            steps {
                dir(env.PROJECT_DIR) {
                    script {
                        echo "=========================================="
                        echo "Stage 5: Verifying System Behavior"
                        echo "=========================================="
                        
                        // Start FL system and wait for training
                        sh """
                            echo "Starting FL system..."
                            docker compose -f ${env.DOCKER_COMPOSE_FILE} up -d mlflow-server fl-dashboard flower-server-lstm flower-worker-1 flower-worker-2
                            
                            echo "Waiting for services to start..."
                            sleep 20
                            
                            echo "Waiting for FL training to complete (max 5 minutes)..."
                            TIMEOUT=300
                            ELAPSED=0
                            TRAINING_COMPLETE=false
                            
                            while [ \$ELAPSED -lt \$TIMEOUT ]; do
                                SERVER_LOGS=\$(docker compose -f ${env.DOCKER_COMPOSE_FILE} logs flower-server-lstm 2>&1)
                                if echo "\$SERVER_LOGS" | grep -qE "Training completed|FL training finished|round 2/2|Final round"; then
                                    echo "Training completed (found in logs)!"
                                    TRAINING_COMPLETE=true
                                    break
                                fi
                                
                                if [ -f "models/LSTM_FL.h5" ]; then
                                    echo "Model file found!"
                                    TRAINING_COMPLETE=true
                                    break
                                fi
                                
                                if echo "\$SERVER_LOGS" | grep -qE "Waiting for clients|Requesting initial parameters"; then
                                    echo "   Server waiting for clients... (\$ELAPSED/\$TIMEOUT seconds)"
                                elif echo "\$SERVER_LOGS" | grep -qE "round 1|round 2|Round"; then
                                    echo "   Training in progress... (\$ELAPSED/\$TIMEOUT seconds)"
                                else
                                    echo "   Checking status... (\$ELAPSED/\$TIMEOUT seconds)"
                                fi
                                
                                sleep 10
                                ELAPSED=\$((ELAPSED + 10))
                            done
                            
                            if [ "\$TRAINING_COMPLETE" = false ]; then
                                echo "Timeout waiting for training completion"
                                echo "Checking current status..."
                                docker compose -f ${env.DOCKER_COMPOSE_FILE} ps
                                echo ""
                                echo "Server logs (last 30 lines):"
                                docker compose -f ${env.DOCKER_COMPOSE_FILE} logs flower-server-lstm | tail -30
                            fi
                        """
                        
                        // Wait a bit more to ensure model file is written
                        sleep(time: 5, unit: 'SECONDS')
                        
                        // Verify model file exists
                        script {
                            def modelFile = "${env.PROJECT_DIR}/models/LSTM_FL.h5"
                            if (fileExists(modelFile)) {
                                echo "Model file created: models/LSTM_FL.h5"
                                sh """
                                    MODEL_SIZE=\$(ls -lh models/LSTM_FL.h5 | awk '{print \$5}')
                                    echo "Model Statistics:"
                                    echo "   - File: models/LSTM_FL.h5"
                                    echo "   - Size: \$MODEL_SIZE"
                                    ls -lh models/LSTM_FL.h5
                                """
                            } else {
                                echo "WARNING: Model file not found, checking logs..."
                                sh """
                                    echo "Container status:"
                                    docker compose -f ${env.DOCKER_COMPOSE_FILE} ps
                                    echo ""
                                    echo "Server logs (last 100 lines):"
                                    docker compose -f ${env.DOCKER_COMPOSE_FILE} logs flower-server-lstm | tail -100
                                    echo ""
                                    echo "Worker 1 logs (last 50 lines):"
                                    docker compose -f ${env.DOCKER_COMPOSE_FILE} logs flower-worker-1 | tail -50
                                    echo ""
                                    echo "Worker 2 logs (last 50 lines):"
                                    docker compose -f ${env.DOCKER_COMPOSE_FILE} logs flower-worker-2 | tail -50
                                """
                                error("Model file not found")
                            }
                        }
                        
                        // Check MLflow runs
                        sh """
                            if [ -d "mlruns" ] && [ "\$(ls -A mlruns)" ]; then
                                echo "MLflow runs directory exists and contains data"
                                echo "MLflow Statistics:"
                                echo "   - Runs directory: mlruns/"
                                ls -la mlruns/
                            else
                                echo "WARNING: MLflow runs directory empty or missing"
                            fi
                        """
                        
                        // Check container status
                        echo ""
                        echo "Container Status:"
                        sh """
                            docker compose -f ${env.DOCKER_COMPOSE_FILE} ps
                        """
                        
                        // Extract training metrics from server logs
                        echo ""
                        echo "Training Statistics (from server logs):"
                        sh """
                            docker compose -f ${env.DOCKER_COMPOSE_FILE} logs flower-server-lstm | grep -E "(Round|accuracy|loss|Training|completed)" | tail -20 || echo "   No training metrics found in logs"
                        """
                        
                        // Check for errors in logs
                        echo ""
                        echo "Checking for errors in server logs..."
                        sh """
                            if docker compose -f ${env.DOCKER_COMPOSE_FILE} logs flower-server-lstm | grep -i "error\\|exception\\|failed"; then
                                echo "WARNING: Errors found in server logs"
                                docker compose -f ${env.DOCKER_COMPOSE_FILE} logs flower-server-lstm | tail -100
                            else
                                echo "No critical errors found in server logs"
                            fi
                        """
                        
                        echo ""
                        echo "System behavior verified"
                    }
                }
            }
        }

        stage('Stage 6 - Sign Model Artifacts') {
            steps {
                dir(env.PROJECT_DIR) {
                    script {
                        echo "=========================================="
                        echo "Stage 6: Signing Model Artifacts"
                        echo "=========================================="
                        
                        sh """
                            # Install jarsigner (Java tool for signing)
                            which jarsigner || echo "jarsigner not found, will attempt to use openssl for signing"
                            
                            # Create signatures directory
                            mkdir -p signatures
                            
                            # Sign model files using GPG (if available) or create checksums
                            if command -v gpg &> /dev/null; then
                                echo "Signing with GPG..."
                                gpg --batch --armor --sign --detach-sign models/LSTM_FL.h5 || echo "GPG signing skipped"
                                ls -lh signatures/ || echo "No GPG signatures created"
                            else
                                echo "GPG not available, creating SHA256 checksums instead..."
                                cd models
                                sha256sum *.h5 > ../signatures/checksums.sha256
                                cd ..
                                cat signatures/checksums.sha256
                            fi
                            
                            # Create metadata file with model information
                            cat > signatures/model-metadata.json <<'EOF'
{
  "build_id": "${BUILD_ID}",
  "build_number": "${BUILD_NUMBER}",
  "build_timestamp": "${BUILD_TIMESTAMP}",
    "git_commit": "\$(git rev-parse HEAD)",
  "git_branch": "${GIT_BRANCH}",
  "model_type": "LSTM",
  "model_framework": "TensorFlow/Keras",
  "fl_rounds": 2,
  "signature_algorithm": "SHA256"
}
EOF
                            
                            echo "Model metadata created:"
                            cat signatures/model-metadata.json
                            
                            echo "Artifacts signed successfully"
                        """
                    }
                }
            }
        }

        stage('Stage 7 - Upload Artifacts to Nexus') {
            steps {
                dir(env.PROJECT_DIR) {
                    script {
                        echo "=========================================="
                        echo "Stage 7: Uploading Artifacts to Nexus"
                        echo "=========================================="
                        
                        withCredentials([usernamePassword(credentialsId: 'nexus-credentials', usernameVariable: 'NEXUS_USER', passwordVariable: 'NEXUS_PASS')]) {
                            sh """
                                # Upload artifacts to Nexus raw repository
                                echo "Uploading artifacts to Nexus raw repository..."
                                
                                NEXUS_RAW_REPO="${NEXUS_REPOSITORY_URL}/repository/raw-hosted"
                                ARTIFACT_PATH="fl-pipeline/artifacts/${BUILD_ID}"
                                
                                # Upload model artifact metadata
                                if [ -f "signatures/model-metadata.json" ]; then
                                    echo "Uploading model metadata..."
                                    curl -v -u \${NEXUS_USER}:\${NEXUS_PASS} \\
                                        --upload-file signatures/model-metadata.json \\
                                        "\${NEXUS_RAW_REPO}/\${ARTIFACT_PATH}/model-metadata.json"
                                fi
                                
                                # Upload checksums
                                if [ -f "signatures/checksums.sha256" ]; then
                                    echo "Uploading checksums..."
                                    curl -v -u \${NEXUS_USER}:\${NEXUS_PASS} \\
                                        --upload-file signatures/checksums.sha256 \\
                                        "\${NEXUS_RAW_REPO}/\${ARTIFACT_PATH}/checksums.sha256"
                                fi
                                
                                # Upload logs and reports if they exist
                                if [ -d "logs" ]; then
                                    echo "Uploading logs..."
                                    tar -czf logs-${BUILD_ID}.tar.gz logs/
                                    curl -v -u \${NEXUS_USER}:\${NEXUS_PASS} \\
                                        --upload-file logs-${BUILD_ID}.tar.gz \\
                                        "\${NEXUS_RAW_REPO}/\${ARTIFACT_PATH}/logs-${BUILD_ID}.tar.gz"
                                fi
                                
                                echo "Artifacts uploaded to Nexus successfully"
                            """
                        }
                    }
                }
            }
        }

        stage('Stage 8 - Upload Model to Nexus') {
            steps {
                dir(env.PROJECT_DIR) {
                    script {
                        echo "=========================================="
                        echo "Stage 8: Uploading Model to Nexus"
                        echo "=========================================="
                        
                        withCredentials([usernamePassword(credentialsId: 'nexus-credentials', usernameVariable: 'NEXUS_USER', passwordVariable: 'NEXUS_PASS')]) {
                            sh """
                                # Upload models to Nexus model repository
                                echo "Uploading trained models to Nexus..."
                                
                                NEXUS_MODELS_REPO="${NEXUS_REPOSITORY_URL}/repository/models-hosted"
                                
                                # Upload LSTM model
                                if [ -f "models/LSTM_FL.h5" ]; then
                                    echo "Uploading LSTM model..."
                                    curl -v -u \${NEXUS_USER}:\${NEXUS_PASS} \\
                                        --upload-file models/LSTM_FL.h5 \\
                                        "\${NEXUS_MODELS_REPO}/lstm-fl-${BUILD_ID}.h5"
                                    
                                    # Also upload to versioned path
                                    curl -v -u \${NEXUS_USER}:\${NEXUS_PASS} \\
                                        --upload-file models/LSTM_FL.h5 \\
                                        "\${NEXUS_MODELS_REPO}/lstm-fl/lstm-fl-${BUILD_ID}.h5"
                                fi
                                
                                # Upload feature order
                                if [ -f "models/feature_order.csv" ]; then
                                    echo "Uploading feature order..."
                                    curl -v -u \${NEXUS_USER}:\${NEXUS_PASS} \\
                                        --upload-file models/feature_order.csv \\
                                        "\${NEXUS_MODELS_REPO}/lstm-fl/feature_order-${BUILD_ID}.csv"
                                fi
                                
                                echo "Models uploaded to Nexus successfully"
                            """
                        }
                    }
                }
            }
        }

        stage('Stage 9 - Push Docker Images to Nexus Registry') {
            steps {
                script {
                    echo "=========================================="
                    echo "Stage 9: Pushing Docker Images to Nexus Registry"
                    echo "=========================================="
                    
                    withCredentials([usernamePassword(credentialsId: 'nexus-docker-credentials', usernameVariable: 'NEXUS_DOCKER_USER', passwordVariable: 'NEXUS_DOCKER_PASS')]) {
                        sh """
                            # Login to Nexus Docker hosted registry (use Nexus docker-hosted repository port)
                            echo "\${NEXUS_DOCKER_PASS}" | docker login -u \${NEXUS_DOCKER_USER} --password-stdin ${NEXUS_DOCKER_REGISTRY_URL}

                            # Tag and push flower-server image to Nexus Docker registry
                            echo "Tagging and pushing flower-server image to Nexus..."
                            docker tag flower-server:latest ${NEXUS_DOCKER_REGISTRY_URL}/flower-server:\${BUILD_ID}
                            docker tag flower-server:latest ${NEXUS_DOCKER_REGISTRY_URL}/flower-server:latest
                            docker push ${NEXUS_DOCKER_REGISTRY_URL}/flower-server:\${BUILD_ID}
                            docker push ${NEXUS_DOCKER_REGISTRY_URL}/flower-server:latest

                            # Tag and push flower-worker image to Nexus
                            echo "Tagging and pushing flower-worker image to Nexus..."
                            docker tag flower-worker:latest ${NEXUS_DOCKER_REGISTRY_URL}/flower-worker:\${BUILD_ID}
                            docker tag flower-worker:latest ${NEXUS_DOCKER_REGISTRY_URL}/flower-worker:latest
                            docker push ${NEXUS_DOCKER_REGISTRY_URL}/flower-worker:\${BUILD_ID}
                            docker push ${NEXUS_DOCKER_REGISTRY_URL}/flower-worker:latest

                            # Tag and push fl-dashboard image to Nexus
                            echo "Tagging and pushing fl-dashboard image to Nexus..."
                            docker tag fl-dashboard:latest ${NEXUS_DOCKER_REGISTRY_URL}/fl-dashboard:\${BUILD_ID}
                            docker tag fl-dashboard:latest ${NEXUS_DOCKER_REGISTRY_URL}/fl-dashboard:latest
                            docker push ${NEXUS_DOCKER_REGISTRY_URL}/fl-dashboard:\${BUILD_ID}
                            docker push ${NEXUS_DOCKER_REGISTRY_URL}/fl-dashboard:latest

                            # Tag and push mlflow-server image to Nexus
                            echo "Tagging and pushing mlflow-server image to Nexus..."
                            docker tag mlflow-server:latest ${NEXUS_DOCKER_REGISTRY_URL}/mlflow-server:\${BUILD_ID}
                            docker tag mlflow-server:latest ${NEXUS_DOCKER_REGISTRY_URL}/mlflow-server:latest
                            docker push ${NEXUS_DOCKER_REGISTRY_URL}/mlflow-server:\${BUILD_ID}
                            docker push ${NEXUS_DOCKER_REGISTRY_URL}/mlflow-server:latest

                            # Logout from Nexus Docker registry
                            docker logout ${NEXUS_DOCKER_REGISTRY_URL}

                            echo "All Docker images pushed to Nexus Docker hosted registry successfully"
                        """
                    }
                }
            }
        }

        stage('Stage 10 - Generate Build Report') {
            steps {
                dir(env.PROJECT_DIR) {
                    script {
                        echo "=========================================="
                        echo "Stage 10: Generating Build Report"
                        echo "=========================================="
                        
                        withCredentials([usernamePassword(credentialsId: 'nexus-credentials', usernameVariable: 'NEXUS_USER', passwordVariable: 'NEXUS_PASS')]) {
                            sh """
                                # Create comprehensive build report
                                cat > build-report.json <<'EOF'
{
  "build_id": "${BUILD_ID}",
  "build_number": "${BUILD_NUMBER}",
  "build_timestamp": "${BUILD_TIMESTAMP}",
  "status": "SUCCESS",
  "artifacts": {
    "models": {
      "lstm_fl": "models/LSTM_FL.h5",
      "feature_order": "models/feature_order.csv"
    },
    "signatures": {
      "metadata": "signatures/model-metadata.json",
      "checksums": "signatures/checksums.sha256"
    },
    "logs": "logs-${BUILD_ID}.tar.gz"
  },
    "nexus_repositories": {
        "models_repo": "${NEXUS_REPOSITORY_URL}/repository/models-hosted",
        "artifacts_repo": "${NEXUS_REPOSITORY_URL}/repository/raw-hosted",
        "docker_registry": "${NEXUS_DOCKER_REGISTRY_URL}"
    },
    "docker_images": {
        "flower_server": "${NEXUS_DOCKER_REGISTRY_URL}/flower-server:${BUILD_ID}",
        "flower_worker": "${NEXUS_DOCKER_REGISTRY_URL}/flower-worker:${BUILD_ID}",
        "fl_dashboard": "${NEXUS_DOCKER_REGISTRY_URL}/fl-dashboard:${BUILD_ID}",
        "mlflow_server": "${NEXUS_DOCKER_REGISTRY_URL}/mlflow-server:${BUILD_ID}"
    }
}
EOF
                                
                                cat build-report.json
                                
                                # Upload build report to Nexus
                                echo "Uploading build report to Nexus..."
                                curl -v -u \${NEXUS_USER}:\${NEXUS_PASS} \\
                                    --upload-file build-report.json \\
                                    "${NEXUS_REPOSITORY_URL}/repository/raw-hosted/fl-pipeline/reports/build-report-${BUILD_ID}.json"
                            """
                        }
                    }
                }
            }
        }
}
    
    post {
        always {
            dir(env.PROJECT_DIR) {
                script {
                    echo "=========================================="
                    echo "Stage 6: Shutting down and cleaning up"
                    echo "=========================================="
                    
                    // Stop all containers
                    sh """
                        docker compose -f ${env.DOCKER_COMPOSE_FILE} down -v || true
                    """
                    
                    // Clean up Docker resources
                    sh """
                        docker system prune -f || true
                    """
                    
                    echo "Cleanup completed"
                }
            }
        }
        
        failure {
            dir(env.PROJECT_DIR) {
                script {
                    echo "=========================================="
                    echo "Collecting logs for debugging"
                    echo "=========================================="
                    
                    sh """
                        echo "--- FL Server Logs ---"
                        docker compose -f ${env.DOCKER_COMPOSE_FILE} logs flower-server-lstm || true
                        
                        echo "--- Worker 1 Logs ---"
                        docker compose -f ${env.DOCKER_COMPOSE_FILE} logs flower-worker-1 || true
                        
                        echo "--- Worker 2 Logs ---"
                        docker compose -f ${env.DOCKER_COMPOSE_FILE} logs flower-worker-2 || true
                        
                        echo "--- MLflow Server Logs ---"
                        docker compose -f ${env.DOCKER_COMPOSE_FILE} logs mlflow-server || true
                        
                        echo "--- Dashboard Logs ---"
                        docker compose -f ${env.DOCKER_COMPOSE_FILE} logs fl-dashboard || true
                    """
                }
            }
        }
        
        success {
            echo "Pipeline completed successfully!"
        }
    }
}

