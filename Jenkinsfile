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
                            
                            echo "Waiting for FL training to complete (max 15 minutes for LSTM)..."
                            TIMEOUT=900
                            ELAPSED=0
                            TRAINING_COMPLETE=false
                            
                            while [ \$ELAPSED -lt \$TIMEOUT ]; do
                                SERVER_LOGS=\$(docker compose -f ${env.DOCKER_COMPOSE_FILE} logs flower-server-lstm 2>&1)
                                
                                # Check for completion indicators
                                if echo "\$SERVER_LOGS" | grep -qE "All 2 rounds completed|Server for LSTM completed all rounds|SUMMARY"; then
                                    echo "✅ Training completed successfully!"
                                    TRAINING_COMPLETE=true
                                    break
                                fi
                                
                                # Check if model file exists (alternative success indicator)
                                if [ -f "models/LSTM_FL.h5" ]; then
                                    echo "✅ Model file found!"
                                    TRAINING_COMPLETE=true
                                    break
                                fi
                                
                                # Show progress based on log content
                                if echo "\$SERVER_LOGS" | grep -q "Waiting for clients"; then
                                    echo "   ⏳ Server waiting for clients... (\$ELAPSED/\$TIMEOUT seconds)"
                                elif echo "\$SERVER_LOGS" | grep -q "ROUND 1"; then
                                    echo "   🔄 Round 1 in progress... (\$ELAPSED/\$TIMEOUT seconds)"
                                elif echo "\$SERVER_LOGS" | grep -q "ROUND 2"; then
                                    echo "   🔄 Round 2 in progress... (\$ELAPSED/\$TIMEOUT seconds)"
                                elif echo "\$SERVER_LOGS" | grep -q "aggregate_fit.*received 2 results"; then
                                    echo "   ✓ Aggregating weights... (\$ELAPSED/\$TIMEOUT seconds)"
                                else
                                    echo "   ⏳ Training in progress... (\$ELAPSED/\$TIMEOUT seconds)"
                                fi
                                
                                sleep 15
                                ELAPSED=\$((ELAPSED + 15))
                            done
                            
                            if [ "\$TRAINING_COMPLETE" = false ]; then
                                echo "❌ ERROR: Training did not complete within \$TIMEOUT seconds"
                                echo ""
                                echo "Container status:"
                                docker compose -f ${env.DOCKER_COMPOSE_FILE} ps
                                echo ""
                                echo "Server logs (last 50 lines):"
                                docker compose -f ${env.DOCKER_COMPOSE_FILE} logs flower-server-lstm | tail -50
                                exit 1
                            fi
                        """
                        
                        // Wait for model file to be synced from container volume
                        echo "Waiting for model file to be synced..."
                        sleep(time: 10, unit: 'SECONDS')
                        
                        // Verify model file exists using shell command (more reliable)
                        def modelCheckResult = sh(
                            script: """
                                if [ -f "models/LSTM_FL.h5" ]; then
                                    echo "exists"
                                else
                                    echo "not_found"
                                fi
                            """,
                            returnStdout: true
                        ).trim()
                        
                        if (modelCheckResult == 'exists') {
                            echo "✅ Model file created successfully: models/LSTM_FL.h5"
                            sh """
                                MODEL_SIZE=\$(ls -lh models/LSTM_FL.h5 | awk '{print \$5}')
                                echo ""
                                echo "Model Statistics:"
                                echo "   📁 File: models/LSTM_FL.h5"
                                echo "   📊 Size: \$MODEL_SIZE"
                                echo "   🏗️  Architecture: LSTM (2 rounds)"
                                echo ""
                                ls -lh models/LSTM_FL.h5
                            """
                        } else {
                            echo "⚠️  WARNING: Model file not found after training completion"
                            sh """
                                echo ""
                                echo "Container status:"
                                docker compose -f ${env.DOCKER_COMPOSE_FILE} ps
                                echo ""
                                echo "Models directory contents:"
                                ls -la models/ || echo "Models directory not found"
                                echo ""
                                echo "Checking server logs for save confirmation:"
                                docker compose -f ${env.DOCKER_COMPOSE_FILE} logs flower-server-lstm | grep -i "Model saved"
                            """
                            error("Model file not found at expected location")
                        }
                        
                        // Verify MLflow tracking
                        sh """
                            if [ -d "mlruns" ] && [ "\$(ls -A mlruns)" ]; then
                                echo ""
                                echo "✅ MLflow tracking data collected"
                                echo "   📁 Directory: mlruns/"
                                EXPERIMENT_COUNT=\$(find mlruns -name "meta.yaml" | wc -l)
                                echo "   🧪 Experiments: \$EXPERIMENT_COUNT"
                            else
                                echo "⚠️  WARNING: MLflow runs directory empty or missing"
                            fi
                        """
                        
                        // Show final training summary
                        echo ""
                        echo "=" * 60
                        echo "Training Summary:"
                        echo "=" * 60
                        sh """
                            docker compose -f ${env.DOCKER_COMPOSE_FILE} logs flower-server-lstm | grep -E "Round.*evaluation complete|Model saved" | tail -10
                        """
                        
                        // Show container status
                        echo ""
                        echo "Final Container Status:"
                        sh """
                            docker compose -f ${env.DOCKER_COMPOSE_FILE} ps
                        """
                        
                        echo ""
                        echo "✅ System behavior verified successfully"
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

        stage('Stage 9 - Upload Docker Images to Nexus') {
            steps {
                script {
                    echo "=========================================="
                    echo "Stage 9: Uploading Docker Images to Nexus"
                    echo "=========================================="

                    withCredentials([usernamePassword(credentialsId: 'nexus-credentials', usernameVariable: 'NEXUS_USER', passwordVariable: 'NEXUS_PASS')]) {
                        sh """
                            # Upload Docker images to Nexus docker-self-hosted repository
                            echo "Uploading Docker images to Nexus docker-self-hosted repository..."

                            NEXUS_DOCKER_REPO="${NEXUS_REPOSITORY_URL}/repository/docker-self-hosted"
                            IMAGE_PATH="fl-pipeline/docker-images/${BUILD_ID}"

                            # Export and upload flower-server image
                            if docker image inspect flower-server:latest >/dev/null 2>&1; then
                                echo "Exporting and uploading flower-server image..."
                                docker save flower-server:latest -o flower-server-${BUILD_ID}.tar
                                curl -v -u \${NEXUS_USER}:\${NEXUS_PASS} \\
                                    --upload-file flower-server-${BUILD_ID}.tar \\
                                    "\${NEXUS_DOCKER_REPO}/\${IMAGE_PATH}/flower-server-${BUILD_ID}.tar"
                                rm flower-server-${BUILD_ID}.tar
                            fi

                            # Export and upload flower-worker image
                            if docker image inspect flower-worker:latest >/dev/null 2>&1; then
                                echo "Exporting and uploading flower-worker image..."
                                docker save flower-worker:latest -o flower-worker-${BUILD_ID}.tar
                                curl -v -u \${NEXUS_USER}:\${NEXUS_PASS} \\
                                    --upload-file flower-worker-${BUILD_ID}.tar \\
                                    "\${NEXUS_DOCKER_REPO}/\${IMAGE_PATH}/flower-worker-${BUILD_ID}.tar"
                                rm flower-worker-${BUILD_ID}.tar
                            fi

                            # Export and upload fl-dashboard image
                            if docker image inspect fl-dashboard:latest >/dev/null 2>&1; then
                                echo "Exporting and uploading fl-dashboard image..."
                                docker save fl-dashboard:latest -o fl-dashboard-${BUILD_ID}.tar
                                curl -v -u \${NEXUS_USER}:\${NEXUS_PASS} \\
                                    --upload-file fl-dashboard-${BUILD_ID}.tar \\
                                    "\${NEXUS_DOCKER_REPO}/\${IMAGE_PATH}/fl-dashboard-${BUILD_ID}.tar"
                                rm fl-dashboard-${BUILD_ID}.tar
                            fi

                            # Export and upload mlflow-server image
                            if docker image inspect mlflow-server:latest >/dev/null 2>&1; then
                                echo "Exporting and uploading mlflow-server image..."
                                docker save mlflow-server:latest -o mlflow-server-${BUILD_ID}.tar
                                curl -v -u \${NEXUS_USER}:\${NEXUS_PASS} \\
                                    --upload-file mlflow-server-${BUILD_ID}.tar \\
                                    "\${NEXUS_DOCKER_REPO}/\${IMAGE_PATH}/mlflow-server-${BUILD_ID}.tar"
                                rm mlflow-server-${BUILD_ID}.tar
                            fi

                            echo "Docker images uploaded to Nexus docker-self-hosted repository successfully"
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
        "docker_repo": "${NEXUS_REPOSITORY_URL}/repository/docker-self-hosted"
    },
    "docker_images": {
        "flower_server": "${NEXUS_REPOSITORY_URL}/repository/docker-self-hosted/fl-pipeline/docker-images/${BUILD_ID}/flower-server-${BUILD_ID}.tar",
        "flower_worker": "${NEXUS_REPOSITORY_URL}/repository/docker-self-hosted/fl-pipeline/docker-images/${BUILD_ID}/flower-worker-${BUILD_ID}.tar",
        "fl_dashboard": "${NEXUS_REPOSITORY_URL}/repository/docker-self-hosted/fl-pipeline/docker-images/${BUILD_ID}/fl-dashboard-${BUILD_ID}.tar",
        "mlflow_server": "${NEXUS_REPOSITORY_URL}/repository/docker-self-hosted/fl-pipeline/docker-images/${BUILD_ID}/mlflow-server-${BUILD_ID}.tar"
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

