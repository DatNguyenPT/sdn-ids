# CI/CD Pipeline Documentation

## Overview

This project uses GitHub Actions for Continuous Integration (CI) of the Federated Learning (FL) system. The CI pipeline focuses on testing the **LSTM model** only, optimized for fast feedback in research workflows.

## Pipeline Stages

```
Code Change
   ↓
GitHub Actions Trigger
   ↓
Build & Validate Containers
   ↓
Start FL System (CI Mode)
   ↓
Run Federated Smoke Test
   ↓
Verify System Behavior
   ↓
Shutdown & Cleanup
```

## Files Structure

### CI Configuration Files

- **`.github/workflows/ci.yml`** - GitHub Actions workflow definition
- **`docker-compose.ci.yml`** - Docker Compose configuration for CI (LSTM only)
- **`scripts/smoke_test_lstm.py`** - Smoke test script for LSTM FL
- **`scripts/check_mlflow_health.py`** - MLflow health check script

## CI Profile (`docker-compose.ci.yml`)

The CI profile is optimized for speed and resource efficiency:

- **Model**: LSTM only (research focus)
- **Rounds**: 2 rounds (reduced from 5 for speed)
- **Epochs per round**: 2 (reduced from 5 for speed)
- **Workers**: 2 workers (minimum for FL)
- **Services**: 
  - FL Server (LSTM)
  - 2 Workers (LSTM)
  - MLflow Server
  - FL Dashboard

## Pipeline Stages Explained

### Stage 1: Build & Validate Containers

- Builds all Docker images required for the FL system
- Validates that images were created successfully
- Uses `docker-compose.ci.yml` for CI-specific configuration

### Stage 2: Start FL System (CI Mode)

- Starts infrastructure services (MLflow, Dashboard)
- Waits for MLflow to be ready
- Starts FL server for LSTM model
- Starts 2 workers for federated training

### Stage 3: MLflow Health Check

- Verifies MLflow server is accessible
- Checks MLflow REST API functionality
- Ensures experiment tracking is ready

### Stage 4: Run Federated Smoke Test

- Monitors FL training progress
- Verifies training completes successfully
- Checks that model file is created
- Validates MLflow integration

### Stage 5: Verify System Behavior

- Confirms model file exists (`models/LSTM_FL.h5`)
- Checks MLflow runs directory
- Verifies container status
- Checks logs for errors

### Stage 6: Shutdown & Cleanup

- Stops all containers
- Cleans up Docker resources
- Ensures no resource leaks

## Running CI Locally

You can test the CI pipeline locally using Docker Compose:

```bash
# Build images
docker-compose -f docker-compose.ci.yml build

# Start system
docker-compose -f docker-compose.ci.yml up -d

# Run smoke test
python scripts/smoke_test_lstm.py \
  --server-url http://localhost:8080 \
  --mlflow-url http://localhost:5002

# Check MLflow health
python scripts/check_mlflow_health.py --mlflow-url http://localhost:5002

# Stop and cleanup
docker-compose -f docker-compose.ci.yml down -v
```

## Scripts

### `scripts/smoke_test_lstm.py`

Performs comprehensive smoke testing of the FL system:

**Usage:**
```bash
python scripts/smoke_test_lstm.py \
  --server-url http://localhost:8080 \
  --mlflow-url http://localhost:5002 \
  --timeout 300 \
  --min-rounds 2
```

**What it checks:**
- FL server accessibility
- Training completion
- Model file creation
- MLflow integration

### `scripts/check_mlflow_health.py`

Checks MLflow server health and API:

**Usage:**
```bash
python scripts/check_mlflow_health.py \
  --mlflow-url http://localhost:5002 \
  --timeout 30 \
  --wait  # Wait for server to become available
```

**What it checks:**
- MLflow server connectivity
- REST API functionality
- Experiment tracking readiness

## GitHub Actions Workflow

The workflow triggers on:
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop`
- Manual trigger via `workflow_dispatch`

## CI vs Production Differences

| Aspect | CI Mode | Production Mode |
|--------|---------|----------------|
| Models | LSTM only | All 4 models |
| FL Rounds | 2 | 5 |
| Epochs/Round | 2 | 5 |
| Purpose | Fast validation | Full training |
| Time | ~5-10 minutes | ~30-60 minutes |

## Troubleshooting

### CI Pipeline Fails

1. **Check GitHub Actions logs** - Full logs are available in the Actions tab
2. **Verify Docker images build** - Check Stage 1 logs
3. **Check container logs** - Stage 5 collects logs on failure
4. **Verify dataset file** - Ensure `dataset_sdn.csv` exists

### Common Issues

**MLflow not ready:**
- Increase wait time in workflow
- Check MLflow container logs

**Training timeout:**
- Increase timeout in smoke test
- Check worker logs for errors

**Model not created:**
- Check server logs for errors
- Verify workers completed training
- Check file permissions

## Requirements

- Docker and Docker Compose
- Python 3.9+
- Required Python packages (see `requirements-server.txt` and `requirements-worker.txt`)

## Next Steps

- Add more comprehensive tests
- Add performance benchmarks
- Add model quality checks
- Integrate with model registry
- Add deployment automation

